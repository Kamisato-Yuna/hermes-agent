#!/usr/bin/env python3
"""Small JSON stdout adapter for the MLX Whisper environment."""
from __future__ import annotations

import argparse
import json
import os
import sys


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("audio")
    parser.add_argument("--model", required=True)
    parser.add_argument("--language", default="zh")
    parser.add_argument("--task", default="transcribe")
    parser.add_argument("--initial-prompt", default="")
    args = parser.parse_args()

    task = args.task.strip().lower()
    if task != "transcribe":
        raise ValueError("MLX Whisper task must be transcribe")

    import mlx_whisper

    kwargs = {
        "path_or_hf_repo": args.model,
        "language": args.language,
        "task": task,
        "word_timestamps": False,
        "condition_on_previous_text": False,
        "no_speech_threshold": 0.6,
        "logprob_threshold": -1.0,
        "compression_ratio_threshold": 2.4,
        "verbose": False,
    }
    if args.initial_prompt.strip():
        kwargs["initial_prompt"] = args.initial_prompt

    result = mlx_whisper.transcribe(args.audio, **kwargs)
    segments = result.get("segments", []) if isinstance(result, dict) else []
    transcript = " ".join(
        str(segment.get("text", "")).strip()
        for segment in segments
        if isinstance(segment, dict) and str(segment.get("text", "")).strip()
    ).strip()
    payload = {
        "success": True,
        "text": transcript,
        "transcript": transcript,
        "provider": "local_mlx",
        "language": args.language,
        "no_speech": not bool(transcript),
    }
    print(json.dumps(payload, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(json.dumps({"success": False, "transcript": "", "error": str(exc)}), file=sys.stderr)
        raise
