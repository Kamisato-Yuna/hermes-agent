"""Real AIAgent reset routing regression (no network or real credentials)."""
import json
from unittest.mock import MagicMock
import pytest

from run_agent import AIAgent
from tui_gateway import server


def test_reset_same_identity_restores_endpoint_and_auth(monkeypatch, tmp_path):
    (tmp_path / "config.yaml").write_text(json.dumps({"model": {"default": "same-model", "provider": "custom"}}))
    monkeypatch.setattr(server, "_hermes_home", tmp_path)
    monkeypatch.setattr("hermes_cli.runtime_provider.resolve_runtime_provider", lambda **kw: {
        "provider": "custom", "base_url": "https://configured.example/v1",
        "api_key": "", "api_mode": "chat_completions"})
    agent = AIAgent.__new__(AIAgent)
    agent.model = "same-model"
    agent.provider = agent.requested_provider = "custom"
    agent.base_url = "https://override.example/v1"
    agent.api_key = "old-test-key"
    agent.api_mode = "chat_completions"
    agent.client = MagicMock()
    agent._client_kwargs = {"api_key": agent.api_key, "base_url": agent.base_url}
    agent._credential_pool = None
    agent.context_compressor = None
    agent._cached_system_prompt = None
    agent._primary_runtime = {}
    agent._fallback_chain = []
    agent._fallback_model = None
    agent.runtime_capabilities = {}
    agent._create_openai_client = lambda kwargs, **kw: MagicMock()
    session = {"agent": agent, "model_override": {"model": "same-model", "provider": "custom"}}
    server._apply_model_switch("sid", session, "reset")
    assert agent.base_url == "https://configured.example/v1"
    assert not agent.api_key
    assert not agent._client_kwargs["api_key"]
    assert "model_override" not in session


def test_reset_same_route_rebinds_configured_pool(monkeypatch, tmp_path):
    (tmp_path / "config.yaml").write_text(json.dumps({"model": {"default": "same-model", "provider": "custom"}}))
    monkeypatch.setattr(server, "_hermes_home", tmp_path)
    old_pool, configured_pool = object(), object()
    monkeypatch.setattr("hermes_cli.runtime_provider.resolve_runtime_provider", lambda **kw: {
        "provider": "custom", "base_url": "https://configured.example/v1", "api_key": "test-key",
        "api_mode": "chat_completions", "credential_pool": configured_pool})
    agent = AIAgent.__new__(AIAgent)
    agent.model, agent.provider = "same-model", "custom"
    agent.base_url, agent.api_key, agent.api_mode = "https://configured.example/v1", "test-key", "chat_completions"
    agent._credential_pool = old_pool
    session = {"agent": agent, "model_override": {"model": "same-model", "provider": "custom"}}
    server._apply_model_switch("sid", session, "reset")
    assert agent._credential_pool is configured_pool


def test_reset_rejects_missing_endpoint_without_reusing_old_route(monkeypatch, tmp_path):
    (tmp_path / "config.yaml").write_text(json.dumps({"model": {"default": "same-model", "provider": "custom"}}))
    monkeypatch.setattr(server, "_hermes_home", tmp_path)
    monkeypatch.setattr("hermes_cli.runtime_provider.resolve_runtime_provider", lambda **kw: {
        "provider": "custom", "base_url": "", "api_key": "new-key", "api_mode": "chat_completions"})
    agent = AIAgent.__new__(AIAgent)
    agent.model = "same-model"
    agent.provider = agent.requested_provider = "custom"
    agent.base_url, agent.api_key, agent.api_mode = "https://override.example/v1", "old-key", "chat_completions"
    agent._credential_pool = object()
    before_pool = agent._credential_pool
    session = {"agent": agent, "model_override": {"model": "same-model"}}
    with pytest.raises(ValueError):
        server._apply_model_switch("sid", session, "reset")
    assert (agent.base_url, agent.api_key, agent._credential_pool) == ("https://override.example/v1", "old-key", before_pool)
    assert "model_override" in session


def test_reset_post_rebuild_failure_restores_agent_and_session(monkeypatch, tmp_path):
    (tmp_path / "config.yaml").write_text(json.dumps({"model": {"default": "new-model", "provider": "custom"}}))
    monkeypatch.setattr(server, "_hermes_home", tmp_path)
    monkeypatch.setattr("hermes_cli.runtime_provider.resolve_runtime_provider", lambda **kw: {
        "provider": "custom", "base_url": "https://configured.example/v1", "api_key": "new-key",
        "api_mode": "chat_completions", "credential_pool": object()})
    agent = AIAgent.__new__(AIAgent)
    agent.model, agent.provider, agent.requested_provider = "old-model", "custom", "custom"
    agent.base_url, agent.api_key, agent.api_mode = "https://override.example/v1", "old-key", "chat_completions"
    agent._credential_pool = object()
    agent.client = MagicMock()
    agent._client_kwargs = {"api_key": "old-key", "base_url": agent.base_url}
    agent._create_openai_client = lambda *args, **kw: MagicMock()
    agent._anthropic_prompt_cache_policy = MagicMock(side_effect=RuntimeError("cache setup failed"))
    before = (agent.model, agent.provider, agent.base_url, agent.api_key, agent.client, agent._credential_pool, dict(agent._client_kwargs))
    session = {"agent": agent, "model_override": {"model": "old-model"}, "one_turn_model_restore": {"model": "older"}}
    with pytest.raises(ValueError):
        server._apply_model_switch("sid", session, "reset")
    assert (agent.model, agent.provider, agent.base_url, agent.api_key, agent.client, agent._credential_pool, agent._client_kwargs) == before
    assert session["model_override"] == {"model": "old-model"}
    assert session["one_turn_model_restore"] == {"model": "older"}


def test_reset_does_not_surface_client_exception_containing_key(monkeypatch, tmp_path, caplog):
    (tmp_path / "config.yaml").write_text(json.dumps({"model": {"default": "new-model", "provider": "custom"}}))
    monkeypatch.setattr(server, "_hermes_home", tmp_path)
    secret = "sensitive-test-credential"
    monkeypatch.setattr("hermes_cli.runtime_provider.resolve_runtime_provider", lambda **kw: {
        "provider": "custom", "base_url": "https://configured.example/v1", "api_key": secret,
        "api_mode": "chat_completions"})
    agent = AIAgent.__new__(AIAgent)
    agent.model, agent.provider = "old-model", "custom"
    agent.base_url, agent.api_key, agent.api_mode = "https://old.example/v1", "old-key", "chat_completions"
    agent._create_openai_client = MagicMock(side_effect=ValueError("failed with " + secret))
    with pytest.raises(ValueError) as error:
        server._apply_model_switch("sid", {"agent": agent, "model_override": {"model": "old-model"}}, "reset")
    assert secret not in str(error.value)
    assert secret not in caplog.text
