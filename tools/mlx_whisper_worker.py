from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict

try:
    from .mlx_whisper_runner import (
        DEFAULT_COMPRESSION_RATIO_THRESHOLD,
        DEFAULT_LOGPROB_THRESHOLD,
        DEFAULT_NO_SPEECH_THRESHOLD,
        DEFAULT_SILENCE_PEAK_THRESHOLD,
        DEFAULT_SILENCE_RMS_THRESHOLD,
        _filter_segments,
        _silence_payload,
        _wav_energy,
    )
except ImportError:  # pragma: no cover - used when executed as an external script
    from mlx_whisper_runner import (  # type: ignore[no-redef]
        DEFAULT_COMPRESSION_RATIO_THRESHOLD,
        DEFAULT_LOGPROB_THRESHOLD,
        DEFAULT_NO_SPEECH_THRESHOLD,
        DEFAULT_SILENCE_PEAK_THRESHOLD,
        DEFAULT_SILENCE_RMS_THRESHOLD,
        _filter_segments,
        _silence_payload,
        _wav_energy,
    )


def _json_line(payload: Dict[str, Any]) -> None:
    print(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), flush=True)


def _number(request: Dict[str, Any], key: str, default: float) -> float:
    value = request.get(key, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _transcribe_request(mlx: Any, request: Dict[str, Any], model: str, *, prewarm: bool = False) -> Dict[str, Any]:
    audio = request.get("audio")
    if not isinstance(audio, str) or not audio:
        return {"success": False, "error_type": "configuration", "error": "audio is required"}
    language = str(request.get("language") or "zh").strip().lower()
    task = str(request.get("task") or "transcribe").strip().lower()
    if language != "zh":
        return {"success": False, "error_type": "configuration", "error": "language must be zh"}
    if task != "transcribe":
        return {"success": False, "error_type": "configuration", "error": "task must be transcribe"}

    no_speech_threshold = _number(request, "no_speech_threshold", DEFAULT_NO_SPEECH_THRESHOLD)
    logprob_threshold = _number(request, "logprob_threshold", DEFAULT_LOGPROB_THRESHOLD)
    compression_ratio_threshold = _number(
        request,
        "compression_ratio_threshold",
        DEFAULT_COMPRESSION_RATIO_THRESHOLD,
    )
    silence_rms_threshold = _number(
        request,
        "silence_rms_threshold",
        DEFAULT_SILENCE_RMS_THRESHOLD,
    )
    silence_peak_threshold = _number(
        request,
        "silence_peak_threshold",
        DEFAULT_SILENCE_PEAK_THRESHOLD,
    )
    energy = _wav_energy(audio)
    if (
        energy is not None
        and energy[0] <= silence_rms_threshold
        and energy[1] <= silence_peak_threshold
    ):
        result = _silence_payload("zh", rms=energy[0], peak=energy[1])
        if prewarm:
            result["prewarm"] = True
        return result

    kwargs: Dict[str, Any] = {
        "path_or_hf_repo": model,
        "language": "zh",
        "task": "transcribe",
        "word_timestamps": False,
        "condition_on_previous_text": False,
        "no_speech_threshold": no_speech_threshold,
        "logprob_threshold": logprob_threshold,
        "compression_ratio_threshold": compression_ratio_threshold,
        "verbose": False,
    }
    initial_prompt = request.get("initial_prompt")
    if isinstance(initial_prompt, str) and initial_prompt.strip():
        kwargs["initial_prompt"] = initial_prompt

    result = mlx.transcribe(audio, **kwargs)
    if not isinstance(result, dict):
        raise ValueError("MLX Whisper returned a non-object result")
    raw_segments = result.get("segments", [])
    segments = raw_segments if isinstance(raw_segments, list) else []
    kept, dropped = _filter_segments(
        segments,
        no_speech_threshold=no_speech_threshold,
        logprob_threshold=logprob_threshold,
        compression_ratio_threshold=compression_ratio_threshold,
    )
    transcript = " ".join(kept).strip()
    if not segments and isinstance(result.get("text"), str):
        transcript = result["text"].strip()
    payload: Dict[str, Any] = {
        "success": True,
        "text": transcript,
        "transcript": transcript,
        "provider": "local_mlx",
        "language": "zh",
        "task": "transcribe",
        "no_speech": not bool(transcript),
        "segments_total": len(segments),
        "segments_dropped": dropped,
    }
    if prewarm:
        payload["prewarm"] = True
        payload["text"] = ""
        payload["transcript"] = ""
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Persistent MLX Whisper JSONL worker")
    parser.add_argument("--model", required=True)
    args = parser.parse_args()

    try:
        import mlx_whisper
    except Exception as exc:  # noqa: BLE001 - protocol must report startup failure
        print(f"MLX worker import failed: {exc}", file=sys.stderr, flush=True)
        _json_line({"event": "ready", "success": False, "error_type": "runtime", "error": str(exc)})
        return 1

    _json_line({"event": "ready", "success": True})
    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue
        request_id = None
        try:
            request = json.loads(line)
            if not isinstance(request, dict):
                raise ValueError("request must be a JSON object")
            request_id = request.get("id")
            operation = request.get("op", "transcribe")
            if operation == "shutdown":
                _json_line({"id": request_id, "success": True, "shutdown": True})
                return 0
            if operation not in {"transcribe", "warmup"}:
                raise ValueError(f"unknown operation: {operation}")
            response = _transcribe_request(
                mlx_whisper,
                request,
                args.model,
                prewarm=operation == "warmup",
            )
            response["id"] = request_id
            _json_line(response)
        except Exception as exc:  # noqa: BLE001 - keep the worker alive for later requests
            print(f"MLX worker request failed: {exc}", file=sys.stderr, flush=True)
            _json_line({
                "id": request_id,
                "success": False,
                "error_type": "runtime",
                "error": f"MLX worker request failed: {exc}",
            })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
