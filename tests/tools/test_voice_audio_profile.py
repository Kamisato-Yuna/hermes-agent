import json
import sys

from tools.voice_audio_profile import analyze_recording, summarize_profile


def test_disabled_analysis_is_noop(tmp_path):
    audio = tmp_path / "input.wav"
    audio.write_bytes(b"audio")
    assert analyze_recording(audio, stt_config={"voice": {"analysis": {"enabled": False}}}) is None


def test_summarize_profile_keeps_compact_acoustic_evidence():
    profile = {
        "analysis": {
            "emotion_result": [
                {"label": "happy", "score": 0.9},
                {"label": "calm", "score": 0.1},
            ],
            "window_seconds": 5,
            "acoustic_windows": [
                {"pitch_hz": 100, "rms": 0.1, "zero_crossing_rate": 0.2},
                {"pitch_hz": 200, "rms": 0.3, "zero_crossing_rate": 0.4},
            ],
            "notes": ["uncertain"],
        }
    }
    result = summarize_profile(profile)
    assert result["emotion_tendency"][0]["label"] == "happy"
    assert result["acoustic"]["pitch_hz_median"] == 150
    assert result["acoustic"]["rms_mean"] == 0.2
    assert len(result["notes"]) == 1
    assert "acoustic_windows" not in json.dumps(result)


def test_command_analysis_returns_profile_and_cleans_temp_output(tmp_path):
    audio = tmp_path / "input.wav"
    audio.write_bytes(b"audio")
    command = [
        sys.executable,
        "-c",
        (
            "import json,sys; "
            "json.dump({'analysis': {'emotion_result': [{'label':'calm','score':0.8}], "
            "'acoustic_windows': []}}, open(sys.argv[2], 'w'))"
        ),
        "{audio}",
        "{output}",
    ]
    result = analyze_recording(
        audio,
        stt_config={
            "voice": {
                "analysis": {
                    "enabled": True,
                    "command": command,
                    "timeout_seconds": 5,
                }
            }
        },
    )
    assert result is not None
    assert result["emotion_tendency"][0]["label"] == "calm"


def test_command_analysis_fails_open_on_timeout(tmp_path):
    audio = tmp_path / "input.wav"
    audio.write_bytes(b"audio")
    result = analyze_recording(
        audio,
        stt_config={
            "voice": {
                "analysis": {
                    "enabled": True,
                    "command": [sys.executable, "-c", "import time; time.sleep(2)"],
                    "timeout_seconds": 0.05,
                }
            }
        },
    )
    assert result is None
