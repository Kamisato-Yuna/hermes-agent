# Voice audio and transcription

## MLX Whisper STT backend

On Apple Silicon, Hermes can delegate local STT to a separate MLX Python
environment instead of loading `faster-whisper` inside the Hermes runtime:

```yaml
stt:
  enabled: true
  provider: local_mlx
  mlx:
    python: /Users/yuna/LLM-Local/qwen36-mlx-vlm/.venv/bin/python
    model: mlx-community/whisper-large-v3-mlx
    language: zh
    task: transcribe
    timeout_seconds: 300
    prewarm_on_voice_start: true
    no_speech_threshold: 0.6
    logprob_threshold: -1.0
    compression_ratio_threshold: 2.4
    silence_rms_threshold: 32
    silence_peak_threshold: 256
    max_segment_seconds: 60
    hf_home: /Users/yuna/LLM-Local/qwen36-mlx-vlm/.cache/huggingface
    hf_endpoint: https://hf-mirror.com
    hf_api_endpoint: https://huggingface.co
```

`local_mlx` normally launches `tools/mlx_whisper_worker.py` for Voice Mode and
keeps MLX in a child process, separate from the Hermes runtime. The worker uses
one JSONL request/response per utterance and `mlx_whisper`'s in-process
`ModelHolder` cache, so the model is initialized once per Voice Mode session.
The existing `tools/mlx_whisper_runner.py` remains available as a one-shot
fallback for worker transport failures. Both paths reject `task` values other
than `transcribe` and languages other than `zh` before decoding. Set
`initial_prompt` under `stt.mlx` for domain vocabulary. The `hf_*` values are
important when the MLX model is already in a project-specific cache; otherwise
the worker may start a second download from the default Hugging Face cache.

Before enabling Voice Mode, the requirements check verifies the configured
Python executable, the one-shot runner, and `import mlx_whisper`. It reports a
missing model snapshot as a warning rather than a hard failure, because the
first worker prewarm may download it. With `prewarm_on_voice_start: true` (the
default), `/voice on` starts the persistent worker and sends a short temporary
non-silent WAV through it once; the WAV is deleted after the ready response. Set
it to `false` to start the worker without loading the model until the first
utterance. If the worker cannot start, Voice Mode remains usable through the
existing one-shot MLX runner and prints a visible fallback warning.

The worker's request protocol is private and machine-readable: diagnostics go
to stderr, stdout contains only JSONL, and every request has a unique ID. A
single request timeout or cancellation terminates that worker process group so
its late response cannot enter a later turn; the next utterance can start a
fresh worker. `/voice off` cancels the current request, stops the Voice-Mode
worker, and releases its MLX memory. The worker is not used by ordinary
non-Voice STT calls.

The worker and one-shot runner have two independent empty-speech guards. PCM16
WAVs below the RMS and peak thresholds return an empty `no_speech` result
without loading Whisper; all formats additionally filter segments with high
`no_speech_prob`, poor `avg_logprob`, or excessive compression ratio. Thresholds
are configurable under `stt.mlx`; the defaults are conservative and can be
tuned for a quiet microphone. Long WAV inputs are split into
`max_segment_seconds` chunks (60 seconds by default), then merged with
per-chunk timeline offsets in the result. Non-WAV inputs longer than the
configured limit fail clearly instead of waiting for an unbounded Whisper call.

For machines with more than one microphone, run `/voice devices` in the CLI
and set the displayed numeric index (or the exact device name) under the voice
section:

```yaml
voice:
  input_device: 2
```

Leaving `voice.input_device` unset keeps the PortAudio default input device.


Hermes CLI voice input can run an external analyzer after STT succeeds. The
feature is disabled by default and is fail-open: a missing analyzer, timeout,
non-zero exit, invalid JSON, or analysis exception does not block the
transcribed message.

Configure it in `~/.hermes/config.yaml`:

```yaml
voice:
  analysis:
    enabled: true
    command: "/path/to/analyze-audio.sh {audio} {output}"
    timeout_seconds: 15
```

The command is executed without a shell. It must accept the `{audio}` and
`{output}` placeholders and write a JSON object to `{output}`. Hermes passes a
compact summary to the model as temporary, API-local context; the original
transcript remains the only durable user message. The generated profile file
is deleted after reading unless the external analyzer persists its own copy.

Recommended analyzer contract:

- keep raw audio handling local;
- label emotion output as an uncertain tendency, not a fact;
- return aggregate acoustic values rather than long per-window arrays;
- avoid writing secrets or raw recordings to logs;
- keep the command within the configured timeout.

Example using the local audio-analysis helper:

```yaml
voice:
  analysis:
    enabled: true
    command: "/path/to/audio-analysis/analyze-audio.sh {audio} {output}"
    timeout_seconds: 15
```

The helper writes the JSON profile to the supplied output path; no stdout
parsing is required.
