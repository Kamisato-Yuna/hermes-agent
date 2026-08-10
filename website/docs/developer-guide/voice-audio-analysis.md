# Optional voice audio analysis

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
