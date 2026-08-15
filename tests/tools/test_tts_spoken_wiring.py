"""Tests for wiring the semantic spoken script into whole-file TTS."""

import json
from pathlib import Path
from types import SimpleNamespace

from tools import tts_tool
from tools.tts_spoken_script import SPOKEN_FAILURE_NOTICE


def _success_result(text: str):
    return SimpleNamespace(
        request_id="tts-test",
        status="success",
        spoken_text=text,
        rewrite_used=True,
        fallback_used=False,
    )


def _run_with_fake_audio(monkeypatch, tmp_path, rewrite_result):
    raw = "原始回复 **不要照读**，路径是 /tmp/secret.txt，内部编号 ABC_123。"
    received = {}

    monkeypatch.setattr(
        tts_tool,
        "_load_tts_config",
        lambda: {
            "provider": "fake",
            "spoken_rewrite": {"enabled": True},
        },
    )

    def fake_prepare(text, *, config):
        received["rewrite_input"] = text
        return rewrite_result

    monkeypatch.setattr("tools.tts_spoken_script.prepare_spoken_script", fake_prepare)
    monkeypatch.setattr(tts_tool, "_get_provider", lambda _cfg: "fake")
    monkeypatch.setattr(tts_tool, "_resolve_command_provider_config", lambda *_args: None)
    monkeypatch.setattr(tts_tool, "_resolve_max_text_length", lambda *_args: 4096)

    def fake_single(*, text, output_path, **_kwargs):
        received["tts_input"] = text
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_bytes(b"fake-audio")
        return json.dumps({"success": True, "file_path": output_path})

    monkeypatch.setattr(tts_tool, "_text_to_speech_single", fake_single)
    result = json.loads(
        tts_tool.text_to_speech_tool(
            text=raw,
            output_path=str(tmp_path / "spoken.mp3"),
        )
    )
    return raw, received, result


def test_enabled_mode_sends_rewritten_script_to_audio_provider(monkeypatch, tmp_path):
    raw, received, result = _run_with_fake_audio(
        monkeypatch,
        tmp_path,
        _success_result("八月七日之后，先检查总计。"),
    )

    assert result["success"] is True
    assert received["rewrite_input"] == raw
    assert received["tts_input"] == "八月七日之后，先检查总计。"
    assert "/tmp/secret.txt" not in received["tts_input"]
    assert "ABC_123" not in received["tts_input"]


def test_rewrite_failure_sends_only_fixed_notice(monkeypatch, tmp_path):
    raw, received, result = _run_with_fake_audio(
        monkeypatch,
        tmp_path,
        SimpleNamespace(
            request_id="tts-test",
            status="fallback_notice",
            spoken_text=SPOKEN_FAILURE_NOTICE,
            rewrite_used=False,
            fallback_used=True,
        ),
    )

    assert result["success"] is True
    assert received["rewrite_input"] == raw
    assert received["tts_input"] == SPOKEN_FAILURE_NOTICE
    assert raw not in received["tts_input"]


def test_platform_prepare_tts_text_preserves_full_reply_when_enabled(monkeypatch):
    from gateway.config import Platform, PlatformConfig
    from gateway.platforms.base import BasePlatformAdapter

    class DummyAdapter(BasePlatformAdapter):
        def __init__(self):
            super().__init__(PlatformConfig(enabled=True, token="test"), Platform.TELEGRAM)

        async def connect(self):
            return True

        async def disconnect(self):
            pass

        async def send(self, chat_id, content, **kwargs):
            raise AssertionError("not used")

        async def get_chat_info(self, chat_id):
            return {"id": chat_id, "type": "dm"}

    monkeypatch.setattr(
        "tools.tts_tool._load_tts_config",
        lambda: {"spoken_rewrite": {"enabled": True}},
    )
    raw = "完整回复 **含 Markdown** 和 /tmp/path。"
    assert DummyAdapter().prepare_tts_text(raw) == raw
