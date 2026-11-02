"""TUI gateway ``/model reset`` (``config.set model`` and Desktop's slash mirror both land here)."""

import json
import pytest

from tui_gateway import server


class _Agent:
    def __init__(self):
        self.model, self.provider, self.base_url, self.api_key, self.api_mode = "pinned", "pin-prov", "", "", ""
        self.switched_to = []

    def switch_model(self, **kw):
        self.switched_to.append((kw["new_model"], kw["new_provider"]))
        self.model, self.provider = kw["new_model"], kw["new_provider"]
        self.base_url, self.api_key, self.api_mode = kw["base_url"], kw["api_key"], kw["api_mode"]


def test_reset_drops_the_pin_and_adopts_config_without_repinning(monkeypatch, tmp_path):
    (tmp_path / "config.yaml").write_text(json.dumps({"model": {"default": "cfg-model", "provider": "cfg-prov"}}))
    monkeypatch.setattr(server, "_hermes_home", tmp_path)
    resolved = []

    def resolve(**kw):
        resolved.append((kw["target_model"], kw["requested"]))
        return {"provider": kw["requested"], "base_url": "", "api_key": "k", "api_mode": "chat_completions"}

    monkeypatch.setattr("hermes_cli.runtime_provider.resolve_runtime_provider", resolve)
    agent = _Agent()
    session = {"agent": agent, "model_override": {"model": "pinned", "provider": "pin-prov"},
               "one_turn_model_restore": {"model": "pinned"}, "config_model_seen": ("cfg-model", "cfg-prov")}

    out = server._apply_model_switch("sid", session, "reset")

    assert resolved == [("cfg-model", "cfg-prov")]
    assert agent.switched_to == [("cfg-model", "cfg-prov")]
    assert "model_override" not in session  # no fresh pin: later config edits reach this chat
    assert "one_turn_model_restore" not in session
    assert out["value"] == "cfg-model" and out["scope"] == "session"

    again = server._apply_model_switch("sid", session, "reset")
    assert again["warning"] != out["warning"] and agent.switched_to == [("cfg-model", "cfg-prov")]


@pytest.mark.parametrize("failure", ["resolve", "switch"])
def test_reset_failure_preserves_session_and_one_turn_state(monkeypatch, tmp_path, failure):
    (tmp_path / "config.yaml").write_text(json.dumps({"model": {"default": "cfg-model", "provider": "cfg-prov"}}))
    monkeypatch.setattr(server, "_hermes_home", tmp_path)

    def resolve(**kw):
        if failure == "resolve":
            raise ValueError("unavailable")
        return {"provider": kw["requested"], "base_url": "", "api_key": "k", "api_mode": "chat_completions"}

    monkeypatch.setattr("hermes_cli.runtime_provider.resolve_runtime_provider", resolve)
    agent = _Agent()
    agent.model = "one-turn"
    if failure == "switch":
        def fail_switch(**kw):
            raise RuntimeError("unavailable")
        monkeypatch.setattr(agent, "switch_model", fail_switch)
    session = {"agent": agent, "model_override": {"model": "pinned", "provider": "pin-prov"},
               "one_turn_model_restore": {"model": "pinned"}, "config_model_seen": ("cfg-model", "cfg-prov"),
               "composer_override_profile": {"model": "old-default", "provider": "cfg-prov"}}
    before = dict(session)

    with pytest.raises(ValueError, match="unavailable"):
        server._apply_model_switch("sid", session, "reset")

    assert session == before
    assert (agent.model, agent.provider) == ("one-turn", "pin-prov")


def test_reset_restores_configured_endpoint_and_credentials(monkeypatch, tmp_path):
    config = {"model": {"default": "same-model", "provider": "custom",
                        "base_url": "https://configured.example/v1", "api_key": "configured-key"}}
    (tmp_path / "config.yaml").write_text(json.dumps(config))
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(server, "_hermes_home", tmp_path)
    agent = _Agent()
    agent.model, agent.provider = "same-model", "custom"
    agent.base_url, agent.api_key, agent.api_mode = "https://override.example/v1", "override-key", "chat_completions"

    def switch(**kw):
        agent.model, agent.provider = kw["new_model"], kw["new_provider"]
        agent.base_url, agent.api_key, agent.api_mode = kw["base_url"], kw["api_key"], kw["api_mode"]

    monkeypatch.setattr(agent, "switch_model", switch)
    session = {"agent": agent, "model_override": {"model": "same-model", "provider": "custom",
                                                   "base_url": agent.base_url, "api_key": agent.api_key}}
    server._apply_model_switch("sid", session, "reset")
    assert (agent.base_url, agent.api_key) == ("https://configured.example/v1", "configured-key")
    assert "model_override" not in session
