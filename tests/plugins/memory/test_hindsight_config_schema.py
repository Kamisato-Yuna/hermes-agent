"""Tests for Hindsight's declared config surface."""

from plugins.memory.config_schema import (
    KIND_BOOL,
    KIND_JSON,
    KIND_NUMBER,
    KIND_SECRET,
    KIND_SELECT,
    get_provider_config_schema,
)


def test_hindsight_is_declared():
    provider = get_provider_config_schema("hindsight")

    assert provider is not None
    assert provider.label == "Hindsight"
    assert {field.key for field in provider.fields} == {
        "mode",
        "api_key",
        "api_url",
        "bank_id",
        "recall_budget",
        "recall_auto_route",
        "recall_auto_route_fail_open",
        "recall_routes",
        "recall_max_results",
    }


def test_routing_fields_are_declared_for_full_config():
    provider = get_provider_config_schema("hindsight")
    assert provider is not None

    fields = {field.key: field for field in provider.fields}
    assert fields["recall_auto_route"].kind == KIND_BOOL
    assert fields["recall_auto_route_fail_open"].kind == KIND_BOOL
    assert fields["recall_routes"].kind == KIND_JSON
    assert fields["recall_max_results"].kind == KIND_NUMBER
    assert fields["recall_auto_route"].default == "false"
    assert fields["recall_auto_route_fail_open"].default == "false"
    assert fields["recall_routes"].inline is False


def test_mode_gating_is_expressed_as_select_options():
    provider = get_provider_config_schema("hindsight")
    assert provider is not None

    mode = next(field for field in provider.fields if field.key == "mode")
    assert mode.kind == KIND_SELECT
    assert mode.allowed_values() == {"cloud", "local_external"}
    # local_embedded is intentionally unsupported on desktop.
    assert "local_embedded" not in mode.allowed_values()


def test_api_key_is_a_secret_bound_to_env():
    provider = get_provider_config_schema("hindsight")
    assert provider is not None

    api_key = next(field for field in provider.fields if field.key == "api_key")
    assert api_key.kind == KIND_SECRET
    assert api_key.is_secret is True
    assert api_key.env_key == "HINDSIGHT_API_KEY"
