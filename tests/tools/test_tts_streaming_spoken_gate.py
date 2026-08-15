"""Tests for TTS entry-point safety gates."""

from unittest.mock import patch

from tools import tts_tool


def test_stream_tts_returns_without_speaking_when_semantic_rewrite_enabled():
    import queue
    import threading

    done = threading.Event()
    q = queue.Queue()
    q.put(None)

    with patch.object(
        tts_tool,
        "_load_tts_config",
        return_value={"spoken_rewrite": {"enabled": True}},
    ), patch("tools.tts_streaming.resolve_streaming_provider") as resolve:
        tts_tool.stream_tts_to_speaker(q, threading.Event(), done)

    resolve.assert_not_called()
    assert done.is_set()


def test_stream_tts_keeps_legacy_path_when_semantic_rewrite_disabled():
    import queue
    import threading

    done = threading.Event()
    q = queue.Queue()
    q.put(None)

    with patch.object(
        tts_tool,
        "_load_tts_config",
        return_value={"spoken_rewrite": {"enabled": False}},
    ), patch("tools.tts_streaming.resolve_streaming_provider", return_value=None), patch.object(
        tts_tool, "_SyncSentencePipeline"
    ) as pipeline:
        tts_tool.stream_tts_to_speaker(q, threading.Event(), done)

    pipeline.assert_called_once()
    assert done.is_set()
