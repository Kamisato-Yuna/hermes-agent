"""Adapter for speech-to-text in a separate Apple MLX Whisper environment."""
from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Optional

from hermes_cli._subprocess_compat import windows_hide_flags

logger = logging.getLogger(__name__)


def _transcribe_local_mlx(
    file_path: str,
    model_name: str,
    mlx_cfg: dict[str, Any],
    *,
    language: Optional[str] = None,
    prompt: Optional[str] = None,
) -> dict[str, Any]:
    """Transcribe through the user's separate MLX Whisper Python environment."""
    python_path = str(mlx_cfg.get("python") or sys.executable).strip()
    runner = Path(__file__).with_name("mlx_whisper_runner.py")
    language = str(language or mlx_cfg.get("language") or "zh")
    configured_task = str(mlx_cfg.get("task") or "transcribe").strip().lower()
    if configured_task not in {"", "transcribe"}:
        logger.warning("MLX Whisper task '%s' is unsupported; forcing transcribe", configured_task)
    timeout = mlx_cfg.get("timeout_seconds", 300)
    try:
        timeout = max(1, float(timeout))
    except (TypeError, ValueError):
        timeout = 300.0

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
    ]
    initial_prompt = (
        prompt if isinstance(prompt, str) and prompt.strip()
        else mlx_cfg.get("initial_prompt")
    )
    if isinstance(initial_prompt, str) and initial_prompt.strip():
        command.extend(["--initial-prompt", initial_prompt])

    try:
        from tools.environments.local import hermes_subprocess_env
        child_env = hermes_subprocess_env(inherit_credentials=False)
    except Exception:
        child_env = os.environ.copy()
    for key in ("HF_HOME", "HF_ENDPOINT", "HF_API_ENDPOINT", "HF_XET_HIGH_PERFORMANCE"):
        value = mlx_cfg.get(key.lower()) or mlx_cfg.get(key)
        if value:
            child_env[key] = str(value)

    try:
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
        if completed.returncode != 0:
            details = completed.stderr.strip() or completed.stdout.strip()
            return {
                "success": False,
                "transcript": "",
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
        return payload
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "transcript": "",
            "error": f"MLX Whisper timed out after {timeout:.0f}s",
            "provider": "local_mlx",
        }
    except (OSError, ValueError) as exc:
        return {
            "success": False,
            "transcript": "",
            "error": f"MLX Whisper unavailable: {exc}",
            "provider": "local_mlx",
        }
