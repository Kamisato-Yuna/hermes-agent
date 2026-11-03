import json
import os
import sys
import wave
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
            "python": sys.executable,
            "language": "zh",
            "task": "transcribe",
            "hf_home": "/tmp/mlx-cache",
        },
    )

    assert result["success"] is True
    assert result["provider"] == "local_mlx"
    assert result["transcript"] == "中文测试"
    assert "--task" in calls["command"]
    assert calls["command"][calls["command"].index("--task") + 1] == "transcribe"
    assert calls["env"]["HF_HOME"] == "/tmp/mlx-cache"


def test_local_mlx_rejects_translation_configuration(monkeypatch, tmp_path):
    calls = []

    def fake_run(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("translation configuration must not launch MLX")

    monkeypatch.setattr(transcription_mlx.subprocess, "run", fake_run)
    audio = tmp_path / "input.wav"
    audio.write_bytes(b"RIFF")

    result = tt._transcribe_local_mlx(
        str(audio),
        "mlx-community/whisper-large-v3-mlx",
        {"python": "/path/to/mlx-python", "language": "zh", "task": "translate"},
    )

    assert result["success"] is False
    assert result["error_type"] == "configuration"
    assert "transcribe" in result["error"]
    assert calls == []


def test_local_mlx_rejects_non_chinese_language_configuration(monkeypatch, tmp_path):
    monkeypatch.setattr(
        transcription_mlx.subprocess,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("invalid language must not launch MLX")
        ),
    )
    audio = tmp_path / "input.wav"
    audio.write_bytes(b"RIFF")

    result = tt._transcribe_local_mlx(
        str(audio),
        "mlx-community/whisper-large-v3-mlx",
        {"python": "/path/to/mlx-python", "language": "en", "task": "transcribe"},
    )

    assert result["success"] is False
    assert result["error_type"] == "configuration"
    assert "language" in result["error"]


def test_local_mlx_rejects_invalid_numeric_configuration(monkeypatch, tmp_path):
    monkeypatch.setattr(
        transcription_mlx.subprocess,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("invalid numeric config must not launch MLX")
        ),
    )
    result = tt._transcribe_local_mlx(
        str(tmp_path / "input.wav"),
        "test-model",
        {
            "python": sys.executable,
            "language": "zh",
            "task": "transcribe",
            "timeout_seconds": "not-a-number",
        },
    )
    assert result["success"] is False
    assert result["error_type"] == "configuration"
    assert "timeout_seconds" in result["error"]


def test_runner_filters_high_no_speech_segments(monkeypatch, tmp_path, capsys):
    import tools.mlx_whisper_runner as runner

    class FakeWhisper:
        @staticmethod
        def transcribe(*args, **kwargs):
            return {
                "segments": [
                    {
                        "text": "请不吝点赞 订阅 转发",
                        "no_speech_prob": 0.91,
                        "avg_logprob": -0.1,
                        "compression_ratio": 0.8,
                    }
                ]
            }

    monkeypatch.setitem(sys.modules, "mlx_whisper", FakeWhisper)
    audio = tmp_path / "input.wav"
    audio.write_bytes(b"RIFF")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mlx_whisper_runner.py",
            str(audio),
            "--model",
            "mlx-community/whisper-large-v3-mlx",
            "--language",
            "zh",
            "--task",
            "transcribe",
        ],
    )

    assert runner.main() == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["transcript"] == ""
    assert payload["no_speech"] is True
    assert payload["segments_dropped"] == 1


def test_runner_short_circuits_low_energy_wav_before_model_import(monkeypatch, tmp_path, capsys):
    import tools.mlx_whisper_runner as runner

    class ExplodingWhisper:
        @staticmethod
        def transcribe(*args, **kwargs):
            raise AssertionError("low-energy WAV should not load Whisper")

    monkeypatch.setitem(sys.modules, "mlx_whisper", ExplodingWhisper)
    audio = tmp_path / "quiet.wav"
    with wave.open(str(audio), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16000)
        wav_file.writeframes(b"\x01\x00" * 16000)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mlx_whisper_runner.py",
            str(audio),
            "--model",
            "mlx-community/whisper-large-v3-mlx",
            "--language",
            "zh",
            "--task",
            "transcribe",
        ],
    )

    assert runner.main() == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["transcript"] == ""
    assert payload["no_speech"] is True
    assert payload["silence_gate"] is True


def test_local_mlx_readiness_checks_import_and_model_cache(monkeypatch, tmp_path):
    python_path = tmp_path / "python"
    python_path.write_text("", encoding="utf-8")
    python_path.chmod(0o755)
    hf_home = tmp_path / "hf"
    model_cache = hf_home / "hub" / "models--mlx-community--whisper-large-v3-mlx" / "snapshots" / "abc"
    model_cache.mkdir(parents=True)
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(transcription_mlx.subprocess, "run", fake_run)
    result = transcription_mlx.check_local_mlx_requirements(
        {
            "enabled": True,
            "provider": "local_mlx",
            "mlx": {
                "python": str(python_path),
                "model": "mlx-community/whisper-large-v3-mlx",
                "language": "zh",
                "task": "transcribe",
                "hf_home": str(hf_home),
            },
        }
    )

    assert result["ready"] is True
    assert result["model_cached"] is True
    assert calls and calls[0][0] == str(python_path)
    assert "mlx_whisper" in calls[0][-1]


def test_local_mlx_prewarm_uses_a_temporary_non_silent_wav(monkeypatch):
    captured = {}

    monkeypatch.setattr(
        transcription_mlx,
        "check_local_mlx_requirements",
        lambda cfg: {"ready": True, "model": "test-model"},
    )

    def fake_transcribe(path, model, cfg):
        captured["path"] = path
        captured["model"] = model
        captured["cfg"] = cfg
        with wave.open(path, "rb") as wav_file:
            captured["peak"] = max(abs(int.from_bytes(wav_file.readframes(1), "little", signed=True)), 0)
        return {"success": True, "transcript": "", "provider": "local_mlx"}

    monkeypatch.setattr(transcription_mlx, "_transcribe_local_mlx", fake_transcribe)
    result = transcription_mlx.prewarm_local_mlx(
        {
            "provider": "local_mlx",
            "mlx": {
                "python": sys.executable,
                "model": "test-model",
                "language": "zh",
                "task": "transcribe",
            },
        }
    )

    assert result["success"] is True
    assert result["prewarm"] is True
    assert captured["path"].endswith(".wav")
    assert captured["peak"] > 0


def test_local_mlx_worker_start_prewarm_and_cleans_temporary_audio(monkeypatch, tmp_path):
    from tools import mlx_whisper_client

    monkeypatch.setattr(
        transcription_mlx,
        "check_local_mlx_requirements",
        lambda cfg: {"ready": True, "python": sys.executable, "model": "test-model"},
    )
    captured = {}

    class FakeWorker:
        def __init__(self, **kwargs):
            captured["kwargs"] = kwargs
            self.pid = 4321
            self.is_running = False

        def start(self):
            self.is_running = True
            return self

        def warmup(self, path):
            captured["warmup_path"] = path
            return {"success": True, "prewarm": True}

        def stop(self):
            self.is_running = False

    monkeypatch.setattr(mlx_whisper_client, "MLXWhisperWorker", FakeWorker)
    worker, status = tt.start_local_mlx_worker(
        {
            "provider": "local_mlx",
            "mlx": {
                "python": sys.executable,
                "model": "test-model",
                "language": "zh",
                "task": "transcribe",
            },
        },
        prewarm=True,
    )

    assert worker is not None
    assert status["success"] is True
    assert status["worker_pid"] == 4321
    assert captured["kwargs"]["model"] == "test-model"
    assert not os.path.exists(captured["warmup_path"])


def test_local_mlx_splits_long_wav_and_preserves_timeline(monkeypatch, tmp_path):
    audio = tmp_path / "long.wav"
    with wave.open(str(audio), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16000)
        wav_file.writeframes(b"\x00\x00" * int(2.2 * 16000))
    calls = []

    def fake_transcribe(
        path, model, cfg, *, language=None, prompt=None, cancel_event=None,
    ):
        calls.append(path)
        return {"success": True, "transcript": f"part-{len(calls)}", "no_speech": False}

    monkeypatch.setattr(transcription_mlx, "_transcribe_local_mlx", fake_transcribe)
    result = transcription_mlx._transcribe_local_mlx_with_segments(
        str(audio),
        "test-model",
        {"max_segment_seconds": 1, "language": "zh", "task": "transcribe"},
    )

    assert result["success"] is True
    assert result["chunked"] is True
    assert result["transcript"] == "part-1 part-2 part-3"
    assert [chunk["offset_seconds"] for chunk in result["chunks"]] == [0.0, 1.0, 2.0]
    assert result["chunks"][-1]["duration_seconds"] == pytest.approx(0.2, abs=0.001)
    assert all(not __import__("pathlib").Path(path).exists() for path in calls)


def test_local_mlx_cancel_event_terminates_child(monkeypatch, tmp_path):
    class FakeProcess:
        pid = 123
        returncode = None

        def __init__(self):
            self.terminated = False

        def communicate(self, timeout=None):
            if self.terminated:
                self.returncode = -15
                return "", "cancelled"
            raise transcription_mlx.subprocess.TimeoutExpired("mlx", timeout)

        def terminate(self):
            self.terminated = True

        def kill(self):
            self.terminated = True

    process = FakeProcess()
    monkeypatch.setattr(transcription_mlx, "windows_hide_flags", lambda: 0)
    monkeypatch.setattr(transcription_mlx.os, "getpgid", lambda pid: pid, raising=False)
    monkeypatch.setattr(transcription_mlx.os, "killpg", lambda *args: None, raising=False)
    cancel = __import__("threading").Event()
    def fake_popen(*args, **kwargs):
        cancel.set()
        return process

    monkeypatch.setattr(transcription_mlx.subprocess, "Popen", fake_popen)
    audio = tmp_path / "input.wav"
    audio.write_bytes(b"RIFF")

    result = tt._transcribe_local_mlx(
        str(audio),
        "mlx-community/whisper-large-v3-mlx",
        {"python": sys.executable, "language": "zh", "task": "transcribe"},
        cancel_event=cancel,
    )

    assert result["success"] is False
    assert result["error_type"] == "cancelled"
    assert process.terminated is True


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
