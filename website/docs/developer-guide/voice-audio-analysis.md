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
    hf_home: /Users/yuna/LLM-Local/qwen36-mlx-vlm/.cache/huggingface
    hf_endpoint: https://hf-mirror.com
    hf_api_endpoint: https://huggingface.co
```

`local_mlx` launches `tools/mlx_whisper_runner.py` in a child process. This
keeps MLX and Hermes dependencies isolated and ensures `task: transcribe` is
used rather than translation. Set `initial_prompt` under `stt.mlx` for domain
vocabulary. The `hf_*` values are important when the MLX model is already in a
project-specific cache; otherwise the runner may start a second download from
the default Hugging Face cache.


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
