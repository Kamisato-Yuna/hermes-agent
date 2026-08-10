import io
import json
import os
import sys
import threading
import time
import types
from pathlib import Path


def test_worker_jsonl_protocol_handles_warmup_and_multiple_transcriptions(monkeypatch, tmp_path):
    import tools.mlx_whisper_worker as worker

    calls = []
    fake_mlx = types.ModuleType("mlx_whisper")

    def fake_transcribe(audio, **kwargs):
        calls.append((audio, kwargs))
        return {
            "segments": [
                {
                    "text": f"recognized-{Path(audio).stem}",
                    "no_speech_prob": 0.01,
                    "avg_logprob": -0.1,
                    "compression_ratio": 1.1,
                }
            ]
        }

    fake_mlx.transcribe = fake_transcribe
    monkeypatch.setitem(sys.modules, "mlx_whisper", fake_mlx)

    audio = str(tmp_path / "voice.wav")
    requests = [
        {"id": "ready-warmup", "op": "warmup", "audio": audio},
        {"id": "one", "op": "transcribe", "audio": audio},
        {"id": "two", "op": "transcribe", "audio": audio},
        {"id": "shutdown", "op": "shutdown"},
    ]
    monkeypatch.setattr(sys, "argv", [
        "mlx_whisper_worker.py",
        "--model",
        "fake-model",
    ])
    monkeypatch.setattr(sys, "stdin", io.StringIO("\n".join(json.dumps(r) for r in requests) + "\n"))
    output = io.StringIO()
    monkeypatch.setattr(sys, "stdout", output)

    assert worker.main() == 0

    messages = [json.loads(line) for line in output.getvalue().splitlines()]
    assert messages[0] == {"event": "ready", "success": True}
    assert [message["id"] for message in messages[1:]] == [
        "ready-warmup",
        "one",
        "two",
        "shutdown",
    ]
    assert all(message["success"] for message in messages[1:])
    assert messages[1]["prewarm"] is True
    assert messages[2]["transcript"] == "recognized-voice"
    assert messages[3]["transcript"] == "recognized-voice"
    assert len(calls) == 3
    assert all(call[1]["language"] == "zh" for call in calls)
    assert all(call[1]["task"] == "transcribe" for call in calls)


def test_worker_client_reports_spawn_failure(monkeypatch):
    from tools.mlx_whisper_client import MLXWhisperWorker, MLXWhisperWorkerError

    def fail_spawn(*args, **kwargs):
        raise OSError("python missing")

    monkeypatch.setattr("tools.mlx_whisper_client.subprocess.Popen", fail_spawn)
    worker = MLXWhisperWorker(
        python_path=sys.executable,
        model="fake-model",
        runner_path="/tmp/fake-worker.py",
        timeout_seconds=2,
    )
    try:
        worker.start()
    except MLXWhisperWorkerError as exc:
        assert "worker" in str(exc).lower()
    else:
        raise AssertionError("worker.start() should report spawn failure")


def test_worker_client_reuses_one_process_for_multiple_requests(monkeypatch, tmp_path):
    from tools.mlx_whisper_client import MLXWhisperWorker

    fake_module = tmp_path / "mlx_whisper.py"
    fake_module.write_text(
        "def transcribe(audio, **kwargs):\n"
        "    return {'segments': [{'text': 'fake-' + audio.split('/')[-1], "
        "'no_speech_prob': 0.0, 'avg_logprob': -0.1, 'compression_ratio': 1.0}]}\n",
        encoding="utf-8",
    )
    worker_script = Path(__file__).resolve().parents[2] / "tools" / "mlx_whisper_worker.py"
    worker = MLXWhisperWorker(
        python_path=sys.executable,
        model="fake-model",
        runner_path=str(worker_script),
        timeout_seconds=5,
    )
    monkeypatch.setattr(
        worker,
        "_child_env",
        lambda: {
            **os.environ,
            "PYTHONPATH": str(tmp_path) + os.pathsep + str(worker_script.parent),
        },
    )

    worker.start()
    first_pid = worker.pid
    first = worker.transcribe("first.wav")
    second = worker.transcribe("second.wav")
    assert first["success"] is True
    assert second["success"] is True
    assert first["transcript"] == "fake-first.wav"
    assert second["transcript"] == "fake-second.wav"
    assert first_pid == worker.pid
    worker.stop()
    assert worker.is_running is False


def test_worker_client_cancellation_terminates_current_worker(monkeypatch, tmp_path):
    from tools.mlx_whisper_client import MLXWhisperWorker

    fake_module = tmp_path / "mlx_whisper.py"
    fake_module.write_text(
        "import time\n"
        "def transcribe(audio, **kwargs):\n"
        "    time.sleep(10)\n"
        "    return {'segments': []}\n",
        encoding="utf-8",
    )
    worker_script = Path(__file__).resolve().parents[2] / "tools" / "mlx_whisper_worker.py"
    worker = MLXWhisperWorker(
        python_path=sys.executable,
        model="fake-model",
        runner_path=str(worker_script),
        timeout_seconds=20,
    )
    monkeypatch.setattr(
        worker,
        "_child_env",
        lambda: {
            **os.environ,
            "PYTHONPATH": str(tmp_path) + os.pathsep + str(worker_script.parent),
        },
    )
    worker.start()
    cancelled = threading.Event()
    result_box = {}

    def run_request():
        result_box["result"] = worker.transcribe("slow.wav", cancel_event=cancelled)

    thread = threading.Thread(target=run_request)
    thread.start()
    time.sleep(0.15)
    cancelled.set()
    thread.join(timeout=3)
    assert not thread.is_alive()
    assert result_box["result"]["error_type"] == "cancelled"
    assert worker.is_running is False
