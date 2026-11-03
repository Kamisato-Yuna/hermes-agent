#!/usr/bin/env python3
"""Small JSON stdout adapter for the MLX Whisper environment."""
from __future__ import annotations

import argparse
from array import array
import json
import math
import sys
import wave


DEFAULT_NO_SPEECH_THRESHOLD = 0.6
DEFAULT_LOGPROB_THRESHOLD = -1.0
DEFAULT_COMPRESSION_RATIO_THRESHOLD = 2.4
DEFAULT_SILENCE_RMS_THRESHOLD = 32.0
DEFAULT_SILENCE_PEAK_THRESHOLD = 256.0


def _wav_energy(path: str) -> tuple[float, float] | None:
    """Return ``(rms, peak)`` for a PCM16 WAV, or None for other inputs."""
    if not path.lower().endswith(".wav"):
        return None
    try:
        with wave.open(path, "rb") as wav_file:
            if wav_file.getsampwidth() != 2:
                return None
            total = 0
            count = 0
            peak = 0
            while True:
                raw = wav_file.readframes(8192)
                if not raw:
                    break
                samples = array("h")
                usable = len(raw) - (len(raw) % 2)
                samples.frombytes(raw[:usable])
                if not samples:
                    continue
                if sys.byteorder != "little":
                    samples.byteswap()
                total += sum(int(sample) * int(sample) for sample in samples)
                count += len(samples)
                peak = max(peak, max(abs(int(sample)) for sample in samples))
            if not count:
                return 0.0, 0.0
            return math.sqrt(total / count), float(peak)
    except (OSError, EOFError, wave.Error, ValueError):
        return None


def _silence_payload(language: str, *, rms: float, peak: float) -> dict:
    return {
        "success": True,
        "text": "",
        "transcript": "",
        "provider": "local_mlx",
        "language": language,
        "task": "transcribe",
        "no_speech": True,
        "silence_gate": True,
        "energy_rms": round(rms, 4),
        "energy_peak": round(peak, 4),
        "segments_dropped": 0,
    }


def _segment_number(segment: dict, key: str) -> float | None:
    value = segment.get(key)
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _filter_segments(
    segments: list,
    *,
    no_speech_threshold: float,
    logprob_threshold: float,
    compression_ratio_threshold: float,
) -> tuple[list[str], int]:
    """Drop low-confidence Whisper segments before building the transcript."""
    kept: list[str] = []
    dropped = 0
    for segment in segments:
        if not isinstance(segment, dict):
            continue
        no_speech_prob = _segment_number(segment, "no_speech_prob")
        avg_logprob = _segment_number(segment, "avg_logprob")
        compression_ratio = _segment_number(segment, "compression_ratio")
        low_confidence = (
            (no_speech_prob is not None and no_speech_prob >= no_speech_threshold)
            or (avg_logprob is not None and avg_logprob < logprob_threshold)
            or (
                compression_ratio is not None
                and compression_ratio > compression_ratio_threshold
            )
        )
        text = str(segment.get("text", "")).strip()
        if low_confidence:
            if text:
                dropped += 1
            continue
        if text:
            kept.append(text)
    return kept, dropped


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("audio")
    parser.add_argument("--model", required=True)
    parser.add_argument("--language", default="zh")
    parser.add_argument("--task", default="transcribe")
    parser.add_argument("--initial-prompt", default="")
    parser.add_argument("--no-speech-threshold", type=float, default=DEFAULT_NO_SPEECH_THRESHOLD)
    parser.add_argument("--logprob-threshold", type=float, default=DEFAULT_LOGPROB_THRESHOLD)
    parser.add_argument(
        "--compression-ratio-threshold",
        type=float,
        default=DEFAULT_COMPRESSION_RATIO_THRESHOLD,
    )
    parser.add_argument(
        "--silence-rms-threshold",
        type=float,
        default=DEFAULT_SILENCE_RMS_THRESHOLD,
    )
    parser.add_argument(
        "--silence-peak-threshold",
        type=float,
        default=DEFAULT_SILENCE_PEAK_THRESHOLD,
    )
    args = parser.parse_args()

    task = args.task.strip().lower()
    if task != "transcribe":
        raise ValueError("MLX Whisper task must be transcribe")
    language = args.language.strip().lower()
    if language != "zh":
        raise ValueError("MLX Whisper language must be zh")

    energy = _wav_energy(args.audio)
    if (
        energy is not None
        and energy[0] <= args.silence_rms_threshold
        and energy[1] <= args.silence_peak_threshold
    ):
        print(json.dumps(_silence_payload(language, rms=energy[0], peak=energy[1]), ensure_ascii=False))
        return 0

    import mlx_whisper

    kwargs = {
        "path_or_hf_repo": args.model,
        "language": language,
        "task": task,
        "word_timestamps": False,
        "condition_on_previous_text": False,
        "no_speech_threshold": args.no_speech_threshold,
        "logprob_threshold": args.logprob_threshold,
        "compression_ratio_threshold": args.compression_ratio_threshold,
        "verbose": False,
    }
    if args.initial_prompt.strip():
        kwargs["initial_prompt"] = args.initial_prompt

    result = mlx_whisper.transcribe(args.audio, **kwargs)
    if not isinstance(result, dict):
        raise ValueError("MLX Whisper returned a non-object result")
    raw_segments = result.get("segments", [])
    segments = raw_segments if isinstance(raw_segments, list) else []
    kept, dropped = _filter_segments(
        segments,
        no_speech_threshold=args.no_speech_threshold,
        logprob_threshold=args.logprob_threshold,
        compression_ratio_threshold=args.compression_ratio_threshold,
    )
    transcript = " ".join(kept).strip()
    if not segments and isinstance(result.get("text"), str):
        transcript = result["text"].strip()
    payload = {
        "success": True,
        "text": transcript,
        "transcript": transcript,
        "provider": "local_mlx",
        "language": language,
        "task": task,
        "no_speech": not bool(transcript),
        "segments_total": len(segments),
        "segments_dropped": dropped,
    }
    print(json.dumps(payload, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(json.dumps({"success": False, "transcript": "", "error": str(exc)}), file=sys.stderr)
        raise
