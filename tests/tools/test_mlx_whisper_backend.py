import json
import sys
from types import SimpleNamespace

import pytest

from tools import transcription_tools as tt
from tools import transcription_mlx


def test_local_mlx_backend_uses_transcribe_and_forwards_cache(monkeypatch, tmp_path):
    calls = {}

    def fake_run(command, **kwargs):
        calls["command"] = command
        calls["env"] = kwargs["env"]
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps(
                {
                    "success": True,
                    "text": "中文测试",
                    "language": "zh",
                }
            ),
            stderr="",
        )

    monkeypatch.setattr(transcription_mlx.subprocess, "run", fake_run)
    monkeypatch.setattr(transcription_mlx, "windows_hide_flags", lambda: 0)
    audio = tmp_path / "input.wav"
    audio.write_bytes(b"RIFF")

    result = tt._transcribe_local_mlx(
        str(audio),
        "mlx-community/whisper-large-v3-mlx",
        {
            "python": "/path/to/mlx-python",
            "language": "zh",
            "task": "translate",
            "hf_home": "/tmp/mlx-cache",
        },
    )

    assert result["success"] is True
    assert result["provider"] == "local_mlx"
    assert result["transcript"] == "中文测试"
    assert "--task" in calls["command"]
    assert calls["command"][calls["command"].index("--task") + 1] == "transcribe"
    assert calls["env"]["HF_HOME"] == "/tmp/mlx-cache"


def test_runner_rejects_translation_task(monkeypatch):
    import tools.mlx_whisper_runner as runner

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mlx_whisper_runner.py",
            "/tmp/input.wav",
            "--model",
            "mlx-community/whisper-large-v3-mlx",
            "--task",
            "translate",
        ],
    )

    with pytest.raises(ValueError, match="task must be transcribe"):
        runner.main()


def test_local_mlx_provider_is_exempt_from_remote_size_cap():
    assert tt._is_local_stt_provider("local_mlx", {}) is True
