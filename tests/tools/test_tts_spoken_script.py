"""Tests for the semantic spoken-script layer."""

from types import SimpleNamespace

from tools.tts_spoken_script import (
    SPOKEN_REWRITE_TASK,
    SPOKEN_FAILURE_NOTICE,
    normalize_chinese_semantics,
    prepare_spoken_script,
    validate_spoken_text,
)


def _response(content):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
        model="test-model",
    )


def test_chinese_semantics_expand_date_percentage_and_duration():
    text = "八月七日之后，完成率百分之三十五，预计八小时。"
    assert normalize_chinese_semantics("8/07之后，完成率35%，预计8h。") == text


def test_chinese_semantics_expand_iso_date_with_digit_year():
    assert normalize_chinese_semantics("截止日期是2026-08-07。") == "截止日期是二零二六年八月七日。"


def test_validator_rejects_urls_paths_english_and_raw_symbols():
    assert validate_spoken_text("请查看 https://example.com")
    assert validate_spoken_text("文件路径是 /tmp/demo.txt")
    assert validate_spoken_text("内部 ID 是 ABC_123")
    assert validate_spoken_text("完成率是35%")


def test_prepare_spoken_script_uses_configured_auxiliary_rewriter():
    calls = []

    def fake_call(*, task, provider, model, messages, **kwargs):
        calls.append({"task": task, "provider": provider, "model": model, "messages": messages})
        return _response('{"spoken_text":"八月七日之后，先检查总计。"}')

    result = prepare_spoken_script(
        "8/07之后的完整回复，包含 Markdown、URL 和内部细节。",
        config={
            "enabled": True,
            "provider": "custom",
            "model": "deepseek-v4-flash",
            "task": SPOKEN_REWRITE_TASK,
        },
        call_llm_fn=fake_call,
    )

    assert result.status == "success"
    assert result.spoken_text == "八月七日之后，先检查总计。"
    assert result.rewrite_used is True
    assert result.fallback_used is False
    assert calls[0]["task"] == SPOKEN_REWRITE_TASK
    assert calls[0]["provider"] == "custom"
    assert calls[0]["model"] == "deepseek-v4-flash"
    assert result.events[0]["stage"] == "received"
    assert any(event["stage"] == "validation_succeeded" for event in result.events)


def test_primary_internal_route_reuses_active_custom_endpoint():
    calls = []

    def fake_call(**kwargs):
        calls.append(kwargs)
        return _response('{"spoken_text":"播报稿已经整理完成。"}')

    result = prepare_spoken_script(
        "这是完整回复。",
        config={
            "enabled": True,
            "provider": "custom",
            "model": "deepseek-v4-flash",
        },
        main_route={
            "provider": "custom",
            "model": "DeepSeek-V4-Flash",
            "base_url": "https://internal.example/v1",
            "api_key": "[REDACTED]",
        },
        call_llm_fn=fake_call,
    )

    assert result.status == "success"
    assert calls[0]["provider"] == "custom"
    assert calls[0]["model"] == "deepseek-v4-flash"
    assert calls[0]["base_url"] == "https://internal.example/v1"
    assert calls[0]["api_key"] == "[REDACTED]"


def test_prepare_spoken_script_falls_back_to_main_model_after_aux_failure():
    calls = []

    def fake_call(*, task, provider, model, messages, **kwargs):
        calls.append({"task": task, "provider": provider, "model": model})
        if task == SPOKEN_REWRITE_TASK:
            raise RuntimeError("aux unavailable")
        return _response('{"spoken_text":"辅助模型不可用时，主模型仍然可以整理播报。"}')

    result = prepare_spoken_script(
        "原始回复不应该被直接朗读。",
        config={
            "enabled": True,
            "provider": "custom",
            "model": "deepseek-v4-flash",
            "task": SPOKEN_REWRITE_TASK,
        },
        main_route={
            "provider": "custom",
            "model": "DeepSeek-V4-Flash",
        },
        call_llm_fn=fake_call,
    )

    assert result.status == "success"
    assert result.fallback_used is True
    assert result.spoken_text.startswith("辅助模型不可用")
    assert calls == [
        {"task": SPOKEN_REWRITE_TASK, "provider": "custom", "model": "deepseek-v4-flash"},
        {"task": None, "provider": "custom", "model": "DeepSeek-V4-Flash"},
    ]
    assert any(event["stage"] == "rewrite_failed" for event in result.events)
    assert any(event["stage"] == "fallback_main_started" for event in result.events)


def test_prepare_spoken_script_fail_closed_without_raw_reply_fallback():
    raw = "原始回复包含 /tmp/secret.txt 和 ABC_123，不可以直接播报。"

    def fake_call(**kwargs):
        return _response("not json and not a safe script")

    result = prepare_spoken_script(
        raw,
        config={"enabled": True, "provider": "custom", "model": "deepseek-v4-flash"},
        call_llm_fn=fake_call,
    )

    assert result.status == "fallback_notice"
    assert result.spoken_text == SPOKEN_FAILURE_NOTICE
    assert "/tmp/secret.txt" not in result.spoken_text
    assert "ABC_123" not in result.spoken_text
    assert any(event["stage"] == "validation_rejected" for event in result.events)


def test_disabled_semantic_layer_preserves_existing_deterministic_cleaner():
    result = prepare_spoken_script(
        "<think>不要播报</think>**正常回复**",
        config={"enabled": False},
    )

    assert result.status == "disabled"
    assert result.rewrite_used is False
    assert "不要播报" not in result.spoken_text
    assert "正常回复" in result.spoken_text


def test_spoken_rewrite_enabled_parses_false_string_as_disabled():
    from tools.tts_spoken_script import spoken_rewrite_enabled

    assert spoken_rewrite_enabled({"enabled": "false"}) is False
    assert spoken_rewrite_enabled({"enabled": "true"}) is True


def test_ascii_punctuation_is_normalized_before_validation():
    assert normalize_chinese_semantics("检查结果: 完成率35%.") == "检查结果：完成率百分之三十五。"


def test_default_auxiliary_task_prefers_internal_deepseek_model():
    from hermes_cli.config_defaults import DEFAULT_CONFIG

    task = DEFAULT_CONFIG["auxiliary"][SPOKEN_REWRITE_TASK]
    assert task["provider"] == "custom"
    assert task["model"] == "deepseek-v4-flash"
    assert DEFAULT_CONFIG["tts"]["spoken_rewrite"]["enabled"] is False
    assert DEFAULT_CONFIG["tts"]["spoken_rewrite"]["number_style"] == "chinese_semantic"
