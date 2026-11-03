"""Adapter for speech-to-text in a separate Apple MLX Whisper environment."""
from __future__ import annotations

import json
import logging
import math
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import wave
from pathlib import Path
from typing import Any, Dict, Optional

from hermes_cli._subprocess_compat import windows_hide_flags
from tools.transcription_audio import _probe_audio_duration

logger = logging.getLogger(__name__)


def _transcribe_with_worker(
    mlx_worker: Any,
    file_path: str,
    language: str,
    cancel_event: Optional[threading.Event],
) -> Optional[dict[str, Any]]:
    from tools.mlx_whisper_client import MLXWhisperWorkerError

    try:
        result = mlx_worker.transcribe(file_path, cancel_event=cancel_event)
    except MLXWhisperWorkerError as exc:
        logger.warning("MLX worker transport failed; falling back to one-shot runner: %s", exc)
        return None
    if not isinstance(result, dict):
        return None
    result.setdefault("provider", "local_mlx")
    result.setdefault("language", language)
    result.setdefault("task", "transcribe")
    return result


def _transcribe_local_mlx(
    file_path: str,
    model_name: str,
    mlx_cfg: dict[str, Any],
    *,
    language: Optional[str] = None,
    prompt: Optional[str] = None,
    cancel_event: Optional[threading.Event] = None,
    mlx_worker: Optional[Any] = None,
) -> dict[str, Any]:
    """Transcribe through the user's separate MLX Whisper Python environment."""
    python_path = str(mlx_cfg.get("python") or "").strip()
    runner = Path(__file__).with_name("mlx_whisper_runner.py")
    language = str(language or mlx_cfg.get("language") or "zh").strip().lower()
    configured_task = str(mlx_cfg.get("task") or "transcribe").strip().lower()
    if configured_task != "transcribe":
        return _mlx_configuration_error(
            "stt.mlx.task must be transcribe; translation is unsupported"
        )
    if language != "zh":
        return _mlx_configuration_error(
            "stt.mlx.language must be zh for this Chinese voice backend"
        )
    numeric_errors = _mlx_numeric_config_errors(mlx_cfg)
    if numeric_errors:
        return _mlx_configuration_error("; ".join(numeric_errors))
    if not python_path:
        return _mlx_configuration_error("stt.mlx.python is not configured")
    if not Path(python_path).is_file() or not os.access(python_path, os.X_OK):
        return _mlx_configuration_error(
            f"MLX Python is missing or not executable: {python_path}"
        )
    if not runner.is_file():
        return _mlx_configuration_error(f"MLX runner is missing: {runner}")

    if mlx_worker is not None:
        worker_result = _transcribe_with_worker(mlx_worker, file_path, language, cancel_event)
        if worker_result is not None:
            return worker_result

    timeout = max(
        1.0,
        _mlx_float_config(mlx_cfg, "timeout_seconds", DEFAULT_MLX_TIMEOUT_SECONDS),
    )

    command = [
        python_path,
        str(runner),
        file_path,
        "--model",
        model_name,
        "--language",
        language,
        "--task",
        "transcribe",
        "--no-speech-threshold",
        str(_mlx_float_config(mlx_cfg, "no_speech_threshold", DEFAULT_MLX_NO_SPEECH_THRESHOLD)),
        "--logprob-threshold",
        str(_mlx_float_config(mlx_cfg, "logprob_threshold", DEFAULT_MLX_LOGPROB_THRESHOLD)),
        "--compression-ratio-threshold",
        str(_mlx_float_config(
            mlx_cfg, "compression_ratio_threshold", DEFAULT_MLX_COMPRESSION_RATIO_THRESHOLD)),
        "--silence-rms-threshold",
        str(_mlx_float_config(mlx_cfg, "silence_rms_threshold", DEFAULT_MLX_SILENCE_RMS_THRESHOLD)),
        "--silence-peak-threshold",
        str(_mlx_float_config(
            mlx_cfg, "silence_peak_threshold", DEFAULT_MLX_SILENCE_PEAK_THRESHOLD)),
    ]
    initial_prompt = mlx_cfg.get("initial_prompt")
    if not isinstance(initial_prompt, str) or not initial_prompt.strip():
        initial_prompt = prompt
    if isinstance(initial_prompt, str) and initial_prompt.strip():
        command.extend(["--initial-prompt", initial_prompt])

    child_env = _mlx_child_env(mlx_cfg)

    try:
        if cancel_event is None:
            completed = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout,
                stdin=subprocess.DEVNULL,
                env=child_env,
                creationflags=windows_hide_flags(),
            )
        else:
            completed = _run_local_mlx_cancelable(
                command,
                timeout=timeout,
                env=child_env,
                cancel_event=cancel_event,
            )
        if completed.returncode != 0:
            details = completed.stderr.strip() or completed.stdout.strip()
            return {
                "success": False,
                "transcript": "",
                "error_type": "runtime",
                "error": f"MLX Whisper failed: {details or completed.returncode}",
                "provider": "local_mlx",
            }
        lines = completed.stdout.strip().splitlines()
        if not lines:
            raise ValueError("MLX Whisper runner returned no JSON")
        payload = json.loads(lines[-1])
        if not isinstance(payload, dict):
            raise ValueError("MLX Whisper runner returned a non-object JSON value")
        text = payload.get("transcript", payload.get("text", ""))
        if not isinstance(text, str):
            text = str(text)
        payload["transcript"] = text.strip()
        payload.setdefault("text", payload["transcript"])
        payload.setdefault("provider", "local_mlx")
        payload.setdefault("language", language)
        payload.setdefault("task", "transcribe")
        return payload
    except _MLXTranscriptionCancelled:
        return {
            "success": False,
            "transcript": "",
            "error_type": "cancelled",
            "error": "MLX Whisper transcription was cancelled",
            "provider": "local_mlx",
        }
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "transcript": "",
            "error_type": "timeout",
            "error": f"MLX Whisper timed out after {timeout:.0f}s",
            "provider": "local_mlx",
        }
    except json.JSONDecodeError as exc:
        return {
            "success": False,
            "transcript": "",
            "error_type": "protocol",
            "error": f"MLX Whisper returned invalid JSON: {exc}",
            "provider": "local_mlx",
        }
    except (OSError, ValueError) as exc:
        return {
            "success": False,
            "transcript": "",
            "error_type": "runtime",
            "error": f"MLX Whisper unavailable: {exc}",
            "provider": "local_mlx",
        }


DEFAULT_MLX_MODEL = "mlx-community/whisper-large-v3-mlx"
DEFAULT_MLX_TIMEOUT_SECONDS = 300.0
DEFAULT_MLX_NO_SPEECH_THRESHOLD = 0.6
DEFAULT_MLX_LOGPROB_THRESHOLD = -1.0
DEFAULT_MLX_COMPRESSION_RATIO_THRESHOLD = 2.4
DEFAULT_MLX_SILENCE_RMS_THRESHOLD = 32.0
DEFAULT_MLX_SILENCE_PEAK_THRESHOLD = 256.0


def _mlx_float_config(config: dict[str, Any], key: str, default: float) -> float:
    """Read a finite numeric MLX option safely."""
    raw = config.get(key, default)
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) else default


def _mlx_numeric_config_errors(config: dict[str, Any]) -> list[str]:
    """Validate MLX numeric options before they reach the child process."""
    constraints = {
        "timeout_seconds": (0.0, None),
        "no_speech_threshold": (0.0, 1.0),
        "compression_ratio_threshold": (0.0, None),
        "silence_rms_threshold": (0.0, None),
        "silence_peak_threshold": (0.0, None),
        "max_segment_seconds": (0.0, None),
    }
    errors: list[str] = []
    for key, (minimum, maximum) in constraints.items():
        if key not in config:
            continue
        raw = config.get(key)
        if raw is None or isinstance(raw, bool):
            errors.append(f"stt.mlx.{key} must be numeric")
            continue
        try:
            value = float(raw)
        except (TypeError, ValueError):
            errors.append(f"stt.mlx.{key} must be numeric")
            continue
        if not math.isfinite(value) or value < minimum or (maximum is not None and value > maximum):
            bound = f"between {minimum} and {maximum}" if maximum is not None else f">= {minimum}"
            errors.append(f"stt.mlx.{key} must be {bound}")
    if "logprob_threshold" in config:
        raw = config.get("logprob_threshold")
        if raw is None or isinstance(raw, bool):
            errors.append("stt.mlx.logprob_threshold must be numeric")
        else:
            try:
                if not math.isfinite(float(raw)):
                    raise ValueError
            except (TypeError, ValueError):
                errors.append("stt.mlx.logprob_threshold must be numeric")
    return errors


def _mlx_child_env(mlx_cfg: dict[str, Any]) -> dict[str, str]:
    """Build the child environment without inheriting Hermes credentials."""
    from tools.environments.local import served_profile_child_env

    child_env = served_profile_child_env(inherit_credentials=False)
    for key in ("HF_HOME", "HF_ENDPOINT", "HF_API_ENDPOINT", "HF_XET_HIGH_PERFORMANCE"):
        value = mlx_cfg.get(key.lower()) or mlx_cfg.get(key)
        if value:
            child_env[key] = str(value)
    return child_env


def _mlx_model_cached(model_name: str, hf_home: Optional[str]) -> bool:
    """Return whether a local Hugging Face model snapshot appears to exist."""
    if Path(model_name).expanduser().is_dir():
        return True
    if not hf_home:
        return False
    root = Path(hf_home).expanduser()
    slug = f"models--{model_name.replace('/', '--')}"
    for model_root in (root / "hub" / slug, root / slug):
        snapshots = model_root / "snapshots"
        if snapshots.is_dir() and any(snapshot.is_dir() for snapshot in snapshots.iterdir()):
            return True
    return False


def check_local_mlx_requirements(
    stt_config: Optional[dict[str, Any]] = None,
    *,
    require_model_cache: bool = False,
    import_timeout: float = 10.0,
) -> dict[str, Any]:
    """Check the isolated MLX runtime without loading or downloading the model."""
    if stt_config is None:
        from tools.transcription_tools import _load_stt_config
        stt_config = _load_stt_config()
    stt_config = stt_config if isinstance(stt_config, dict) else {}
    raw_mlx_cfg = stt_config.get("mlx")
    mlx_cfg = raw_mlx_cfg if isinstance(raw_mlx_cfg, dict) else {}
    python_path = str(mlx_cfg.get("python") or "").strip()
    runner = Path(__file__).with_name("mlx_whisper_runner.py")
    model_name = str(mlx_cfg.get("model") or DEFAULT_MLX_MODEL).strip()
    language = str(mlx_cfg.get("language") or "zh").strip().lower()
    task = str(mlx_cfg.get("task") or "transcribe").strip().lower()
    hf_home = str(mlx_cfg.get("hf_home") or mlx_cfg.get("HF_HOME") or "").strip()
    hf_home = hf_home or os.environ.get("HF_HOME", "").strip()
    errors: list[str] = []
    warnings: list[str] = []
    import_ok = False

    if not python_path:
        errors.append("stt.mlx.python is not configured")
    elif not Path(python_path).is_file() or not os.access(python_path, os.X_OK):
        errors.append(f"MLX Python is missing or not executable: {python_path}")
    if not runner.is_file():
        errors.append(f"MLX runner is missing: {runner}")
    if task != "transcribe":
        errors.append("stt.mlx.task must be transcribe; translation is unsupported")
    if language != "zh":
        errors.append("stt.mlx.language must be zh for this Chinese voice backend")
    errors.extend(_mlx_numeric_config_errors(mlx_cfg))

    if python_path and Path(python_path).is_file() and os.access(python_path, os.X_OK):
        try:
            probe = subprocess.run(
                [python_path, "-c", "import mlx_whisper"],
                check=False,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=max(1.0, float(import_timeout)),
                stdin=subprocess.DEVNULL,
                env=_mlx_child_env(mlx_cfg),
                creationflags=windows_hide_flags(),
            )
            if probe.returncode == 0:
                import_ok = True
            else:
                details = (probe.stderr or probe.stdout or "import failed").strip()
                errors.append(f"MLX Python cannot import mlx_whisper: {details}")
        except subprocess.TimeoutExpired:
            errors.append("MLX Python import probe timed out")
        except OSError as exc:
            errors.append(f"MLX Python import probe failed: {exc}")

    model_cached = _mlx_model_cached(model_name, hf_home)
    if not hf_home:
        warnings.append("HF_HOME is not configured; the model may use a second default cache")
    elif not Path(hf_home).expanduser().is_dir():
        warnings.append(f"HF_HOME does not exist yet: {hf_home}")
    if not model_cached:
        warnings.append(f"MLX model is not cached; first use may download: {model_name}")
        if require_model_cache:
            errors.append(f"MLX model is not cached: {model_name}")

    return {
        "ready": not errors,
        "python": python_path,
        "runner": str(runner),
        "model": model_name,
        "language": language,
        "task": task,
        "import_ok": import_ok,
        "model_cached": model_cached,
        "hf_home": hf_home,
        "errors": errors,
        "warnings": warnings,
    }


def _create_mlx_warmup_wav() -> str:
    """Create a short non-silent WAV used only to load the MLX model."""
    fd, warmup_path = tempfile.mkstemp(prefix="hermes-mlx-warmup-", suffix=".wav")
    os.close(fd)
    try:
        sample_rate = 16000
        sample_count = int(sample_rate * 0.4)
        amplitude = 3000
        with wave.open(warmup_path, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            frames = bytearray()
            for index in range(sample_count):
                sample = int(amplitude * math.sin(2 * math.pi * 440 * index / sample_rate + math.pi / 2))
                frames.extend(sample.to_bytes(2, "little", signed=True))
            wav_file.writeframes(frames)
        return warmup_path
    except Exception:
        try:
            os.unlink(warmup_path)
        except OSError:
            pass
        raise


def prewarm_local_mlx(stt_config: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    """Load the MLX model with a temporary tone before voice capture begins."""
    if stt_config is None:
        from tools.transcription_tools import _load_stt_config
        stt_config = _load_stt_config()
    stt_config = stt_config if isinstance(stt_config, dict) else {}
    readiness = check_local_mlx_requirements(stt_config)
    if not readiness.get("ready"):
        errors = "; ".join(readiness.get("errors") or ["MLX runtime is not ready"])
        return {
            "success": False, "transcript": "", "provider": "local_mlx",
            "prewarm": True, "error_type": "readiness", "error": errors,
        }

    raw_mlx_cfg = stt_config.get("mlx")
    mlx_cfg = raw_mlx_cfg if isinstance(raw_mlx_cfg, dict) else {}
    model_name = str(mlx_cfg.get("model") or DEFAULT_MLX_MODEL)
    warmup_path = _create_mlx_warmup_wav()
    try:
        result = _transcribe_local_mlx(warmup_path, model_name, mlx_cfg)
        result = dict(result)
        result.update(prewarm=True, transcript="", text="")
        return result
    except Exception as exc:
        logger.exception("MLX prewarm failed")
        return {
            "success": False, "transcript": "", "provider": "local_mlx",
            "prewarm": True, "error_type": "runtime", "error": f"MLX prewarm failed: {exc}",
        }
    finally:
        try:
            os.unlink(warmup_path)
        except OSError:
            pass


def start_local_mlx_worker(
    stt_config: Optional[dict[str, Any]] = None,
    *,
    prewarm: bool = True,
) -> tuple[Optional[Any], dict[str, Any]]:
    """Start a Voice Mode scoped MLX worker and optionally load its model."""
    if stt_config is None:
        from tools.transcription_tools import _load_stt_config
        stt_config = _load_stt_config()
    stt_config = stt_config if isinstance(stt_config, dict) else {}
    readiness = check_local_mlx_requirements(stt_config)
    if not readiness.get("ready"):
        errors = "; ".join(readiness.get("errors") or ["MLX runtime is not ready"])
        return None, {
            "success": False,
            "provider": "local_mlx",
            "error_type": "readiness",
            "error": errors,
        }

    raw_mlx_cfg = stt_config.get("mlx")
    mlx_cfg = raw_mlx_cfg if isinstance(raw_mlx_cfg, dict) else {}
    worker_script = Path(__file__).with_name("mlx_whisper_worker.py")
    if not worker_script.is_file():
        return None, {
            "success": False,
            "provider": "local_mlx",
            "error_type": "configuration",
            "error": f"MLX worker script is missing: {worker_script}",
        }

    from tools.mlx_whisper_client import MLXWhisperWorker, MLXWhisperWorkerError

    worker = None
    try:
        worker = MLXWhisperWorker(
            python_path=readiness["python"],
            model=readiness["model"],
            runner_path=str(worker_script),
            mlx_config=mlx_cfg,
            timeout_seconds=_mlx_float_config(
                mlx_cfg, "timeout_seconds", DEFAULT_MLX_TIMEOUT_SECONDS,
            ),
        )
        worker.start()
        warmup_result: dict[str, Any] = {"success": True}
        if prewarm:
            warmup_path = _create_mlx_warmup_wav()
            try:
                warmup_result = worker.warmup(warmup_path)
            finally:
                try:
                    os.unlink(warmup_path)
                except OSError:
                    pass
            if not warmup_result.get("success"):
                worker.stop()
                return None, {
                    "success": False,
                    "provider": "local_mlx",
                    "error_type": warmup_result.get("error_type", "runtime"),
                    "error": warmup_result.get("error", "MLX worker prewarm failed"),
                }
        return worker, {
            "success": True,
            "provider": "local_mlx",
            "worker": True,
            "worker_pid": worker.pid,
            "prewarm": prewarm,
            "prewarm_result": warmup_result,
        }
    except (MLXWhisperWorkerError, OSError, ValueError) as exc:
        if worker is not None:
            worker.stop()
        return None, {
            "success": False,
            "provider": "local_mlx",
            "error_type": "worker",
            "error": f"MLX worker failed: {exc}",
        }


def stop_local_mlx_worker(worker: Any) -> None:
    """Stop a Voice Mode MLX worker without allowing cleanup errors to escape."""
    if worker is None:
        return
    try:
        worker.stop()
    except Exception:
        logger.debug("MLX worker cleanup failed", exc_info=True)


class _MLXTranscriptionCancelled(Exception):
    """Internal signal for a caller-cancelled MLX subprocess."""


def _terminate_local_mlx_process(proc: Any) -> None:
    """Terminate the MLX process group, then fall back to the child process."""
    try:
        running = proc.poll() is None
    except AttributeError:
        running = getattr(proc, "returncode", None) is None
    pid = getattr(proc, "pid", None)
    if os.name == "posix" and isinstance(pid, int):
        from hermes_cli.process_identity import stop_owned_group_survivors

        if stop_owned_group_survivors(pid, proc=proc, grace=1.0) or not running:
            return
    if not running:
        return
    try:
        proc.terminate()
    except (AttributeError, OSError):
        return
    try:
        proc.wait(timeout=2)
    except subprocess.TimeoutExpired:
        try:
            proc.kill()
        except (AttributeError, OSError):
            return
        proc.wait()


def _run_local_mlx_cancelable(
    command: list[str],
    *,
    timeout: float,
    env: dict[str, str],
    cancel_event: threading.Event,
) -> subprocess.CompletedProcess:
    """Run the MLX process with bounded timeout and cooperative cancellation."""
    if cancel_event.is_set():
        raise _MLXTranscriptionCancelled
    popen_kwargs: dict[str, Any] = {
        "stdin": subprocess.DEVNULL,
        "stdout": subprocess.PIPE,
        "stderr": subprocess.PIPE,
        "text": True,
        "encoding": "utf-8",
        "errors": "replace",
        "env": env,
        "creationflags": windows_hide_flags(),
    }
    from hermes_cli.process_identity import release_process_group_once_empty, spawn_owned

    proc = spawn_owned(command, purpose="mlx-whisper-cancelable", **popen_kwargs)
    deadline = time.monotonic() + timeout
    while True:
        if cancel_event.is_set():
            _terminate_local_mlx_process(proc)
            raise _MLXTranscriptionCancelled
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            _terminate_local_mlx_process(proc)
            raise subprocess.TimeoutExpired(command, timeout)
        try:
            stdout, stderr = proc.communicate(timeout=min(0.1, remaining))
            completed = subprocess.CompletedProcess(command, proc.returncode, stdout, stderr)
            release_process_group_once_empty(proc.pid, proc=proc)
            return completed
        except subprocess.TimeoutExpired:
            continue


def _mlx_configuration_error(message: str) -> dict[str, Any]:
    return {
        "success": False,
        "transcript": "",
        "provider": "local_mlx",
        "error_type": "configuration",
        "error": message,
    }


def _wav_duration_and_params(file_path: str) -> Optional[tuple[float, Any]]:
    """Return WAV duration and parameters for local segmentation, if available."""
    if Path(file_path).suffix.lower() != ".wav":
        return None
    try:
        with wave.open(file_path, "rb") as wav_file:
            params = wav_file.getparams()
            rate = wav_file.getframerate()
            if rate <= 0:
                return None
            return wav_file.getnframes() / rate, params
    except (OSError, EOFError, wave.Error, ValueError):
        return None


def _split_wav_for_local_mlx(
    file_path: str,
    *,
    max_seconds: float,
) -> tuple[str, list[tuple[str, float, float]]]:
    """Split PCM WAV audio into temporary chunks and retain their timeline offsets."""
    duration_params = _wav_duration_and_params(file_path)
    if duration_params is None:
        return "", []
    duration, params = duration_params
    if duration <= max_seconds:
        return "", [(file_path, 0.0, duration)]
    frames_per_chunk = max(1, int(params.framerate * max_seconds))
    work_dir = tempfile.mkdtemp(prefix="hermes-mlx-segments-")
    chunks: list[tuple[str, float, float]] = []
    try:
        with wave.open(file_path, "rb") as source:
            index = 0
            offset = 0.0
            while True:
                raw = source.readframes(frames_per_chunk)
                if not raw:
                    break
                chunk_path = os.path.join(work_dir, f"chunk-{index:04d}.wav")
                with wave.open(chunk_path, "wb") as target:
                    target.setparams(params)
                    target.writeframes(raw)
                chunk_duration = len(raw) / max(
                    1, params.nchannels * params.sampwidth * params.framerate,
                )
                if chunk_duration <= 0:
                    chunk_duration = min(max_seconds, duration - offset)
                chunks.append((chunk_path, offset, chunk_duration))
                offset += chunk_duration
                index += 1
        return work_dir, chunks
    except Exception:
        shutil.rmtree(work_dir, ignore_errors=True)
        raise


def _transcribe_local_mlx_with_segments(
    file_path: str,
    model_name: str,
    mlx_cfg: dict[str, Any],
    *,
    language: Optional[str] = None,
    prompt: Optional[str] = None,
    cancel_event: Optional[threading.Event] = None,
    mlx_worker: Optional[Any] = None,
) -> dict[str, Any]:
    """Transcribe long WAVs in bounded MLX chunks and merge their text."""
    max_seconds = _mlx_float_config(mlx_cfg, "max_segment_seconds", 60.0)
    if max_seconds <= 0:
        return _transcribe_local_mlx(
            file_path, model_name, mlx_cfg, language=language, prompt=prompt,
            cancel_event=cancel_event, mlx_worker=mlx_worker,
        )
    duration_params = _wav_duration_and_params(file_path)
    estimated_duration = (
        duration_params[0] if duration_params is not None else _probe_audio_duration(file_path)
    )
    if estimated_duration is None or estimated_duration <= max_seconds:
        return _transcribe_local_mlx(
            file_path, model_name, mlx_cfg, language=language, prompt=prompt,
            cancel_event=cancel_event, mlx_worker=mlx_worker,
        )
    if Path(file_path).suffix.lower() != ".wav":
        return {
            "success": False,
            "transcript": "",
            "provider": "local_mlx",
            "error_type": "configuration",
            "error": (
                f"MLX audio is {estimated_duration:.1f}s; long-input segmentation currently "
                "requires a WAV file"
            ),
        }

    work_dir = ""
    try:
        work_dir, chunks = _split_wav_for_local_mlx(file_path, max_seconds=max_seconds)
        if not chunks:
            return _mlx_configuration_error("MLX could not split the long WAV input")
        chunk_results: list[dict[str, Any]] = []
        transcripts: list[str] = []
        for index, (chunk_path, offset, chunk_duration) in enumerate(chunks):
            if cancel_event is not None and cancel_event.is_set():
                return {
                    "success": False,
                    "transcript": "",
                    "provider": "local_mlx",
                    "error_type": "cancelled",
                    "error": "MLX Whisper transcription was cancelled",
                    "completed_chunks": index,
                }
            transcribe_kwargs = {
                "language": language,
                "prompt": prompt,
                "cancel_event": cancel_event,
            }
            if mlx_worker is not None:
                transcribe_kwargs["mlx_worker"] = mlx_worker
            result = _transcribe_local_mlx(chunk_path, model_name, mlx_cfg, **transcribe_kwargs)
            if not result.get("success"):
                return {
                    "success": False,
                    "transcript": "",
                    "provider": "local_mlx",
                    "error_type": result.get("error_type", "runtime"),
                    "error": (
                        f"MLX chunk {index + 1}/{len(chunks)} failed: "
                        f"{result.get('error', 'unknown error')}"
                    ),
                    "completed_chunks": index,
                }
            text = str(result.get("transcript", result.get("text", ""))).strip()
            if text:
                transcripts.append(text)
            chunk_results.append({
                "index": index,
                "offset_seconds": round(offset, 3),
                "duration_seconds": round(chunk_duration, 3),
                "text": text,
                "no_speech": bool(result.get("no_speech", not text)),
            })
        transcript = " ".join(transcripts).strip()
        return {
            "success": True,
            "text": transcript,
            "transcript": transcript,
            "provider": "local_mlx",
            "language": "zh",
            "task": "transcribe",
            "no_speech": not bool(transcript),
            "chunked": True,
            "chunks": chunk_results,
        }
    finally:
        if work_dir:
            shutil.rmtree(work_dir, ignore_errors=True)
