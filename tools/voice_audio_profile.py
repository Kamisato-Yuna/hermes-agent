"""Optional local audio analysis for microphone voice input.

The analyzer is deliberately command-based so Hermes does not need to depend on
MLX, librosa, or an emotion model. A configured command receives two
placeholders: ``{audio}`` and ``{output}``; it must write a JSON profile to the
output path. Analysis is fail-open and never blocks normal STT delivery beyond
the configured timeout.
"""
from __future__ import annotations

import json
import logging
import math
import shlex
import subprocess
import tempfile
from pathlib import Path
from statistics import median
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT_SECONDS = 15.0


def _analysis_config(stt_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    if stt_config is None:
        try:
            from hermes_cli.config import load_config
            stt_config = load_config()
        except Exception:
            stt_config = {}
    voice = stt_config.get("voice") if isinstance(stt_config, dict) else {}
    analysis = voice.get("analysis") if isinstance(voice, dict) else {}
    return analysis if isinstance(analysis, dict) else {}


def _enabled(config: Dict[str, Any]) -> bool:
    return config.get("enabled") is True and bool(config.get("command"))


def _build_command(command: Any, audio: Path, output: Path) -> list[str]:
    if isinstance(command, list):
        parts = [str(part) for part in command]
    elif isinstance(command, str):
        parts = shlex.split(command)
    else:
        raise ValueError("voice.analysis.command must be a string or list")
    if not parts:
        raise ValueError("voice.analysis.command is empty")
    return [part.replace("{audio}", str(audio)).replace("{output}", str(output)) for part in parts]


def _finite_numbers(values: Any) -> list[float]:
    if not isinstance(values, list):
        return []
    result = []
    for value in values:
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            result.append(number)
    return result


def summarize_profile(profile: Dict[str, Any]) -> Dict[str, Any]:
    """Keep only compact, non-sensitive audio observations for prompt context."""
    analysis = profile.get("analysis") if isinstance(profile, dict) else {}
    if not isinstance(analysis, dict):
        analysis = {}
    emotions = analysis.get("emotion_result")
    top_emotions = []
    if isinstance(emotions, list):
        for item in emotions[:3]:
            if isinstance(item, dict) and item.get("label") is not None:
                top_emotions.append({"label": item["label"], "score": item.get("score")})

    windows = analysis.get("acoustic_windows")
    pitches = []
    rms = []
    zcr = []
    if isinstance(windows, list):
        for item in windows:
            if not isinstance(item, dict):
                continue
            pitches.extend(_finite_numbers([item.get("pitch_hz")]))
            rms.extend(_finite_numbers([item.get("rms")]))
            zcr.extend(_finite_numbers([item.get("zero_crossing_rate")]))

    acoustic: Dict[str, Any] = {"window_seconds": analysis.get("window_seconds")}
    if pitches:
        acoustic["pitch_hz_median"] = round(median(pitches), 2)
    if rms:
        acoustic["rms_mean"] = round(sum(rms) / len(rms), 6)
    if zcr:
        acoustic["zero_crossing_rate_mean"] = round(sum(zcr) / len(zcr), 6)

    result: Dict[str, Any] = {
        "emotion_tendency": top_emotions,
        "acoustic": acoustic,
    }
    notes = analysis.get("notes")
    if isinstance(notes, list):
        result["notes"] = [str(note) for note in notes[:3]]
    return result


def analyze_recording(
    audio_path: str | Path,
    *,
    stt_config: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """Run the configured analyzer, returning a compact profile or ``None``."""
    config = _analysis_config(stt_config)
    if not _enabled(config):
        return None

    audio = Path(audio_path)
    if not audio.is_file():
        logger.warning("Voice analysis skipped; audio file does not exist: %s", audio)
        return None

    try:
        timeout = float(config.get("timeout_seconds", _DEFAULT_TIMEOUT_SECONDS))
        if not math.isfinite(timeout) or timeout <= 0:
            timeout = _DEFAULT_TIMEOUT_SECONDS
    except (TypeError, ValueError):
        timeout = _DEFAULT_TIMEOUT_SECONDS

    try:
        with tempfile.NamedTemporaryFile(prefix="hermes_voice_profile_", suffix=".json", delete=False) as tmp:
            output = Path(tmp.name)
        try:
            command = _build_command(config["command"], audio, output)
            completed = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            if completed.returncode != 0:
                logger.warning("Voice analysis failed with exit code %s", completed.returncode)
                return None
            if not output.is_file():
                logger.warning("Voice analysis produced no profile: %s", output)
                return None
            profile = json.loads(output.read_text(encoding="utf-8"))
            if not isinstance(profile, dict):
                logger.warning("Voice analysis profile is not a JSON object")
                return None
            return summarize_profile(profile)
        finally:
            output.unlink(missing_ok=True)
    except subprocess.TimeoutExpired:
        logger.warning("Voice analysis timed out after %.1fs", timeout)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        logger.warning("Voice analysis unavailable: %s", exc)
    except Exception:
        logger.exception("Unexpected voice analysis failure")
    return None
