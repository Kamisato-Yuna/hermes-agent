from __future__ import annotations

import json
import logging
import os
import queue
import subprocess
import threading
import time
import uuid
from pathlib import Path
from types import TracebackType
from typing import Any, Dict, Optional

from hermes_cli._subprocess_compat import windows_hide_flags

logger = logging.getLogger(__name__)


class MLXWhisperWorkerError(RuntimeError):
    """The persistent worker could not start or lost its JSONL transport."""


class MLXWhisperWorker:
    """Lifecycle-managed client for a persistent MLX Whisper JSONL child."""

    def __init__(
        self,
        *,
        python_path: str,
        model: str,
        runner_path: str,
        mlx_config: Optional[dict[str, Any]] = None,
        timeout_seconds: float = 300.0,
    ) -> None:
        self.python_path = python_path
        self.model = model
        self.runner_path = runner_path
        self.mlx_config = dict(mlx_config or {})
        self.timeout_seconds = max(1.0, float(timeout_seconds))
        self._process: Optional[subprocess.Popen] = None
        self._messages: queue.Queue[dict[str, Any]] = queue.Queue()
        self._request_lock = threading.Lock()
        self._lifecycle_lock = threading.Lock()
        self._reader_threads: list[threading.Thread] = []
        self._stderr_lines: list[str] = []
        self._started = False

    @property
    def pid(self) -> Optional[int]:
        process = self._process
        return getattr(process, "pid", None) if process is not None else None

    @property
    def is_running(self) -> bool:
        process = self._process
        if process is None:
            return False
        try:
            return process.poll() is None
        except AttributeError:
            return getattr(process, "returncode", None) is None

    @property
    def stderr_tail(self) -> str:
        return "\n".join(self._stderr_lines[-20:])

    def _child_env(self) -> dict[str, str]:
        from tools.environments.local import served_profile_child_env

        env = served_profile_child_env(inherit_credentials=False)
        for config_key, env_key in (
            ("hf_home", "HF_HOME"),
            ("hf_endpoint", "HF_ENDPOINT"),
            ("hf_api_endpoint", "HF_API_ENDPOINT"),
            ("hf_xet_high_performance", "HF_XET_HIGH_PERFORMANCE"),
        ):
            value = self.mlx_config.get(config_key)
            if isinstance(value, str) and value.strip():
                env[env_key] = value
        return env

    def _command(self) -> list[str]:
        return [
            self.python_path,
            self.runner_path,
            "--model",
            self.model,
        ]

    def _read_stdout(self, stream: Any) -> None:
        try:
            for raw_line in iter(stream.readline, ""):
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    message = json.loads(line)
                    if isinstance(message, dict):
                        self._messages.put(message)
                    else:
                        self._messages.put({
                            "success": False,
                            "error_type": "protocol",
                            "error": "worker emitted a non-object JSON value",
                        })
                except json.JSONDecodeError as exc:
                    self._messages.put({
                        "success": False,
                        "error_type": "protocol",
                        "error": f"worker emitted invalid JSON: {exc}",
                    })
        except (OSError, ValueError) as exc:
            self._messages.put({
                "success": False,
                "error_type": "worker_transport",
                "error": f"worker stdout reader failed: {exc}",
            })

    def _read_stderr(self, stream: Any) -> None:
        try:
            for raw_line in iter(stream.readline, ""):
                line = raw_line.strip()
                if line:
                    self._stderr_lines.append(line)
                    del self._stderr_lines[:-100]
        except (OSError, ValueError):
            return

    def _terminate_process(self, process: Any) -> None:
        try:
            running = process.poll() is None
        except AttributeError:
            running = getattr(process, "returncode", None) is None
        pid = getattr(process, "pid", None)
        if os.name == "posix" and isinstance(pid, int):
            from hermes_cli.process_identity import stop_owned_group_survivors

            if stop_owned_group_survivors(pid, proc=process, grace=1.0) or not running:
                return
        if not running:
            return
        try:
            process.terminate()
        except (AttributeError, OSError):
            return
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            try:
                process.kill()
            except (AttributeError, OSError):
                return
            process.wait()

    def start(self) -> MLXWhisperWorker:
        # A crashed process can leave its last response in the queue. Reap
        # that generation before spawning the replacement so a stale response
        # cannot satisfy a request from the new generation.
        if self._process is not None and not self.is_running:
            self.stop()
        with self._lifecycle_lock:
            if self.is_running:
                return self
            if not Path(self.python_path).is_file() or not os.access(self.python_path, os.X_OK):
                raise MLXWhisperWorkerError(
                    f"MLX worker Python is missing or not executable: {self.python_path}"
                )
            if not Path(self.runner_path).is_file():
                raise MLXWhisperWorkerError(f"MLX worker script is missing: {self.runner_path}")
            try:
                from hermes_cli.process_identity import spawn_owned

                process = spawn_owned(
                    self._command(),
                    purpose="mlx-whisper-worker",
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    env=self._child_env(),
                    creationflags=windows_hide_flags(),
                )
            except OSError as exc:
                raise MLXWhisperWorkerError(f"MLX worker failed to start: {exc}") from exc
            self._process = process
            self._started = False
            self._stderr_lines.clear()
            self._messages = queue.Queue()
            self._reader_threads = []
            if process.stdout is not None:
                from agent.memory_provider import spawn_context_thread

                thread = spawn_context_thread(
                    self._read_stdout,
                    name="hermes-mlx-worker-stdout",
                    daemon=True,
                    args=(process.stdout,),
                )
                thread.start()
                self._reader_threads.append(thread)
            if process.stderr is not None:
                from agent.memory_provider import spawn_context_thread

                thread = spawn_context_thread(
                    self._read_stderr,
                    name="hermes-mlx-worker-stderr",
                    daemon=True,
                    args=(process.stderr,),
                )
                thread.start()
                self._reader_threads.append(thread)

        message = self._wait_for_message(expected_event="ready", timeout=self.timeout_seconds)
        if not message.get("success"):
            self.stop()
            raise MLXWhisperWorkerError(
                f"MLX worker readiness failed: {message.get('error', self.stderr_tail or 'unknown error')}"
            )
        self._started = True
        return self

    def _wait_for_message(
        self,
        *,
        expected_id: Optional[str] = None,
        expected_event: Optional[str] = None,
        timeout: Optional[float] = None,
        cancel_event: Optional[threading.Event] = None,
    ) -> dict[str, Any]:
        deadline = time.monotonic() + (self.timeout_seconds if timeout is None else timeout)
        while True:
            if cancel_event is not None and cancel_event.is_set():
                self.stop()
                return {
                    "success": False,
                    "error_type": "cancelled",
                    "error": "MLX worker request was cancelled",
                }
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self.stop()
                return {
                    "success": False,
                    "error_type": "timeout",
                    "error": f"MLX worker timed out after {self.timeout_seconds:.0f}s",
                }
            try:
                message = self._messages.get(timeout=min(0.1, remaining))
            except queue.Empty:
                process = self._process
                if process is not None:
                    try:
                        if process.poll() is not None:
                            details = self.stderr_tail or "worker exited unexpectedly"
                            self.stop()
                            raise MLXWhisperWorkerError(details)
                    except AttributeError:
                        pass
                continue
            if expected_id is not None and message.get("id") != expected_id:
                if message.get("error_type"):
                    raise MLXWhisperWorkerError(str(message.get("error")))
                continue
            if expected_event is not None and message.get("event") != expected_event:
                continue
            return message

    def _send(
        self,
        payload: dict[str, Any],
        *,
        cancel_event: Optional[threading.Event] = None,
    ) -> dict[str, Any]:
        process = self._process
        if not self._started or process is None or not self.is_running or process.stdin is None:
            raise MLXWhisperWorkerError("MLX worker is not running")
        try:
            process.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
            process.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            self.stop()
            raise MLXWhisperWorkerError(f"MLX worker stdin failed: {exc}") from exc
        return self._wait_for_message(
            expected_id=str(payload["id"]),
            cancel_event=cancel_event,
        )

    def request(
        self,
        *,
        audio_path: str,
        operation: str = "transcribe",
        cancel_event: Optional[threading.Event] = None,
    ) -> dict[str, Any]:
        if not self.is_running:
            self.start()
        request_id = uuid.uuid4().hex
        payload: dict[str, Any] = {
            "id": request_id,
            "op": operation,
            "audio": audio_path,
            "language": "zh",
            "task": "transcribe",
            "no_speech_threshold": self.mlx_config.get("no_speech_threshold", 0.6),
            "logprob_threshold": self.mlx_config.get("logprob_threshold", -1.0),
            "compression_ratio_threshold": self.mlx_config.get("compression_ratio_threshold", 2.4),
            "silence_rms_threshold": self.mlx_config.get("silence_rms_threshold", 32),
            "silence_peak_threshold": self.mlx_config.get("silence_peak_threshold", 256),
        }
        prompt = self.mlx_config.get("initial_prompt")
        if isinstance(prompt, str) and prompt.strip():
            payload["initial_prompt"] = prompt
        with self._request_lock:
            if cancel_event is not None and cancel_event.is_set():
                self.stop()
                return {"success": False, "error_type": "cancelled", "error": "MLX worker request was cancelled"}
            try:
                return self._send(payload, cancel_event=cancel_event)
            except MLXWhisperWorkerError:
                self.stop()
                raise

    def warmup(self, audio_path: str) -> dict[str, Any]:
        return self.request(audio_path=audio_path, operation="warmup")

    def transcribe(
        self,
        audio_path: str,
        *,
        cancel_event: Optional[threading.Event] = None,
    ) -> dict[str, Any]:
        return self.request(audio_path=audio_path, cancel_event=cancel_event)

    def stop(self) -> None:
        with self._lifecycle_lock:
            process = self._process
            self._process = None
            self._started = False
        if process is None:
            return
        try:
            if process.stdin is not None:
                process.stdin.close()
        except (OSError, ValueError):
            logger.debug("MLX worker stdin was already closed", exc_info=True)
        self._terminate_process(process)
        for thread in self._reader_threads:
            thread.join(timeout=0.5)
        self._reader_threads = []

    close = stop

    def __enter__(self) -> MLXWhisperWorker:
        return self.start()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.stop()
