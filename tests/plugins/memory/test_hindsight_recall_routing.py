"""Regression tests for configuration-driven Hindsight recall routing.

These tests deliberately use an async fake client.  A synchronous fake would
allow production code that creates, but never awaits, ``arecall`` coroutines to
appear correct.
"""

from __future__ import annotations

import inspect
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from plugins.memory.hindsight import (
    HindsightMemoryProvider,
    _arecall_with_fail_open,
)


class _AsyncRecallClient:
    """Small async fake that records every request in order."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []
        self.arecall = AsyncMock(side_effect=self._arecall)

    async def _arecall(self, **kwargs):
        self.calls.append(dict(kwargs))
        if not self.responses:
            raise AssertionError("unexpected extra arecall")
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response


def _response(*results):
    return SimpleNamespace(results=list(results))


def _result(identifier, text, tags=()):
    return SimpleNamespace(id=identifier, text=text, tags=list(tags))


def _provider(tmp_path, monkeypatch, client, **overrides):
    config = {
        "mode": "cloud",
        "apiKey": "test-key",
        "api_url": "http://localhost:9999",
        "bank_id": "test-bank",
        "budget": "mid",
        "memory_mode": "hybrid",
    }
    config.update(overrides)
    config_path = tmp_path / "hindsight" / "config.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(config), encoding="utf-8")
    monkeypatch.setattr(
        "plugins.memory.hindsight.get_hermes_home", lambda: tmp_path
    )

    provider = HindsightMemoryProvider()
    provider.initialize(session_id="test-session", platform="cli")
    provider._client = client
    return provider


def test_fail_open_helper_is_async():
    assert inspect.iscoroutinefunction(_arecall_with_fail_open)


def test_auto_route_is_opt_in_and_default_recall_is_unchanged(
    tmp_path, monkeypatch
):
    client = _AsyncRecallClient(_response(_result("one", "memory")))
    provider = _provider(
        tmp_path,
        monkeypatch,
        client,
        recall_routes={
            "alpha": {
                "keywords": ["marker"],
                "tags": ["scope:alpha"],
                "types": ["world"],
            }
        },
    )

    recalled = provider._do_recall("marker query")

    assert recalled.text == "- memory"
    assert len(client.calls) == 1
    assert client.calls[0]["types"] == ["observation"]
    assert "tags" not in client.calls[0]
    assert "tag_groups" not in client.calls[0]


def test_route_config_accepts_json_and_passes_filters_and_cap(
    tmp_path, monkeypatch
):
    client = _AsyncRecallClient(
        _response(
            _result("private", "private", ["scope:alpha", "sensitive"]),
            _result("one", "one", ["scope:alpha"]),
            _result("two", "two", ["scope:alpha"]),
        )
    )
    routes = json.dumps(
        {
            "alpha": {
                "keywords": ["marker"],
                "tags": ["scope:alpha", "kind:plan"],
                "tags_match": "all_strict",
                "types": ["world", "observation"],
                "max_results": 1,
                "exclude_tags": ["sensitive"],
            }
        }
    )
    provider = _provider(
        tmp_path,
        monkeypatch,
        client,
        recall_auto_route=True,
        recall_routes=routes,
    )

    assert list(provider._recall_routes) == ["alpha"]
    assert provider._recall_domain_routing is True

    recalled = provider._do_recall("marker query")

    assert recalled.text == "- one"
    assert recalled.count == 1
    assert len(client.calls) == 1
    request = client.calls[0]
    assert request["types"] == ["world", "observation"]
    assert "tags" not in request
    assert request["tag_groups"] == [
        {
            "and": [
                {"tags": ["scope:alpha", "kind:plan"], "match": "all_strict"},
                {"not": {"tags": ["sensitive"], "match": "any_strict"}},
            ]
        }
    ]


def test_route_selection_prefers_chat_id_then_name_then_keyword(
    tmp_path, monkeypatch
):
    client = _AsyncRecallClient(
        _response(_result("one", "memory")),
        _response(_result("two", "memory")),
        _response(_result("three", "memory")),
    )
    provider = _provider(
        tmp_path,
        monkeypatch,
        client,
        recall_auto_route=True,
        recall_routes={
            "keyword": {"keywords": ["shared"], "tags": ["route:keyword"]},
            "name": {"chat_names": ["Team Alpha"], "tags": ["route:name"]},
            "id": {"chat_ids": ["chat-alpha"], "tags": ["route:id"]},
        },
    )

    provider._chat_id = "chat-alpha"
    provider._chat_name = "Team Beta"
    provider._do_recall("shared query")
    assert client.calls[-1]["tags"] == ["route:id"]

    provider._chat_id = "chat-unknown"
    provider._chat_name = "TEAM ALPHA"
    provider._do_recall("shared query")
    assert client.calls[-1]["tags"] == ["route:name"]

    provider._chat_name = "unknown"
    provider._do_recall("shared query")
    assert client.calls[-1]["tags"] == ["route:keyword"]


def test_same_signal_uses_configuration_order(tmp_path, monkeypatch):
    client = _AsyncRecallClient(_response(_result("one", "memory")))
    provider = _provider(
        tmp_path,
        monkeypatch,
        client,
        recall_auto_route=True,
        recall_routes={
            "first": {"keywords": ["shared"], "tags": ["route:first"]},
            "second": {"keywords": ["shared"], "tags": ["route:second"]},
        },
    )

    provider._do_recall("shared query")

    assert client.calls[0]["tags"] == ["route:first"]


@pytest.mark.asyncio
async def test_fallback_awaits_each_stage_in_order():
    client = _AsyncRecallClient(
        _response(),
        _response(),
        _response(_result("one", "memory")),
    )
    strict = {
        "bank_id": "test-bank",
        "query": "marker query",
        "budget": "mid",
        "max_tokens": 4096,
        "tags": ["scope:alpha", "kind:plan"],
        "tags_match": "all_strict",
        "types": ["observation"],
    }
    fallback = {
        **strict,
        "tags": ["scope:alpha"],
        "tags_match": "all_strict",
    }
    unfiltered = {key: value for key, value in strict.items() if key not in {"tags", "tags_match"}}

    response = await _arecall_with_fail_open(
        client,
        strict,
        auto_routed=True,
        fail_open=True,
        fallback_kwargs=[fallback],
        unfiltered_kwargs=unfiltered,
    )

    assert response.results[0].text == "memory"
    assert client.arecall.await_count == 3
    assert [call["tags"] for call in client.calls[:2]] == [
        ["scope:alpha", "kind:plan"],
        ["scope:alpha"],
    ]
    assert "tags" not in client.calls[2]
    assert "tags_match" not in client.calls[2]


def test_route_fallback_stays_tag_scoped_when_fail_open_is_off(
    tmp_path, monkeypatch
):
    client = _AsyncRecallClient(_response(), _response(_result("one", "memory")))
    provider = _provider(
        tmp_path,
        monkeypatch,
        client,
        recall_auto_route=True,
        recall_routes={
            "alpha": {
                "keywords": ["marker"],
                "tags": ["scope:alpha", "kind:plan"],
            }
        },
    )

    recalled = provider._do_recall("marker query")

    assert recalled.text == "- memory"
    assert client.arecall.await_count == 2
    assert client.calls[-1]["tags"] == ["scope:alpha"]
    assert "tag_groups" not in client.calls[-1]
    assert "tags" in client.calls[-1]


def test_route_config_without_tags_does_not_create_unfiltered_fallback(
    tmp_path, monkeypatch
):
    client = _AsyncRecallClient(_response())
    provider = _provider(
        tmp_path,
        monkeypatch,
        client,
        recall_auto_route=True,
        recall_auto_route_fail_open=True,
        recall_routes={"types-only": {"keywords": ["marker"], "types": ["world"]}},
    )

    recalled = provider._do_recall("marker query")

    assert recalled.text == ""
    assert client.arecall.await_count == 1
    assert client.calls[0]["types"] == ["world"]
    assert "tags" not in client.calls[0]
    assert "tag_groups" not in client.calls[0]


def test_route_config_accepts_wrapper_mapping(tmp_path, monkeypatch):
    client = _AsyncRecallClient(_response(_result("one", "memory")))
    provider = _provider(
        tmp_path,
        monkeypatch,
        client,
        recall_auto_route=True,
        recall_routes={
            "routes": {
                "wrapped": {
                    "keywords": ["marker"],
                    "tags": ["route:wrapped"],
                }
            }
        },
    )

    provider._do_recall("marker query")

    assert list(provider._recall_routes) == ["wrapped"]
    assert provider._recall_routes["wrapped"]["name"] == "wrapped"
    assert client.calls[0]["tags"] == ["route:wrapped"]


def test_invalid_route_entries_are_ignored(tmp_path, monkeypatch):
    client = _AsyncRecallClient(_response(_result("one", "memory")))
    provider = _provider(
        tmp_path,
        monkeypatch,
        client,
        recall_auto_route=True,
        recall_routes=["not-a-route", {"keywords": ["marker"], "tags": ["route:valid"]}],
    )

    provider._do_recall("marker query")

    assert client.calls[0]["tags"] == ["route:valid"]


def test_tag_group_sdk_failure_retries_the_same_stage_with_positive_tags(
    tmp_path, monkeypatch
):
    client = MagicMock()
    client.arecall = AsyncMock(
        side_effect=[
            ModuleNotFoundError("hindsight_client_api.models.recall_request_tag_groups_inner"),
            _response(_result("one", "memory", ["scope:alpha"])),
        ]
    )
    provider = _provider(
        tmp_path,
        monkeypatch,
        client,
        recall_auto_route=True,
        recall_routes={
            "alpha": {
                "keywords": ["marker"],
                "tags": ["scope:alpha"],
                "exclude_tags": ["sensitive"],
            }
        },
    )

    recalled = provider._do_recall("marker query")

    assert recalled.text == "- memory"
    assert client.arecall.await_count == 2
    first, second = client.arecall.await_args_list
    assert "tag_groups" in first.kwargs
    assert second.kwargs["tags"] == ["scope:alpha"]
    assert second.kwargs["tags_match"] == "all_strict"


def test_nonempty_route_result_stops_fallback(tmp_path, monkeypatch):
    client = _AsyncRecallClient(_response(_result("one", "memory")))
    provider = _provider(
        tmp_path,
        monkeypatch,
        client,
        recall_auto_route=True,
        recall_auto_route_fail_open=True,
        recall_routes={
            "alpha": {
                "keywords": ["marker"],
                "tags": ["scope:alpha", "kind:plan"],
            }
        },
    )

    provider._do_recall("marker query")

    assert client.arecall.await_count == 1


def test_explicit_tags_keep_precedence_and_never_fail_open(tmp_path, monkeypatch):
    client = _AsyncRecallClient(_response())
    provider = _provider(
        tmp_path,
        monkeypatch,
        client,
        recall_auto_route=True,
        recall_auto_route_fail_open=True,
        recall_tags=["explicit:scope"],
        recall_tags_match="all_strict",
        recall_types=["experience"],
        recall_routes={
            "alpha": {
                "keywords": ["marker"],
                "tags": ["scope:alpha", "kind:plan"],
                "types": ["world"],
            }
        },
    )

    recalled = provider._do_recall("marker query")

    assert recalled.text == ""
    assert client.arecall.await_count == 1
    request = client.calls[0]
    assert request["tags"] == ["explicit:scope"]
    assert request["tags_match"] == "all_strict"
    assert request["types"] == ["experience"]
    assert "tag_groups" not in request


def test_explicit_types_override_route_types(tmp_path, monkeypatch):
    client = _AsyncRecallClient(_response(_result("one", "memory")))
    provider = _provider(
        tmp_path,
        monkeypatch,
        client,
        recall_auto_route=True,
        recall_types=["experience"],
        recall_routes={
            "alpha": {
                "keywords": ["marker"],
                "tags": ["scope:alpha"],
                "types": ["world"],
            }
        },
    )

    provider._do_recall("marker query")

    assert client.calls[0]["types"] == ["experience"]
    assert client.calls[0]["tags"] == ["scope:alpha"]


def test_fail_open_keeps_exclusions_on_unfiltered_fallback(tmp_path, monkeypatch):
    client = _AsyncRecallClient(
        _response(),
        _response(),
        _response(
            _result("private", "private", ["sensitive"]),
            _result("public", "public", []),
        ),
    )
    provider = _provider(
        tmp_path,
        monkeypatch,
        client,
        recall_auto_route=True,
        recall_auto_route_fail_open=True,
        recall_routes={
            "alpha": {
                "keywords": ["marker"],
                "tags": ["scope:alpha", "kind:plan"],
                "exclude_tags": ["sensitive"],
            }
        },
    )

    recalled = provider._do_recall("marker query")

    assert recalled.text == "- public"
    assert client.arecall.await_count == 3
    final = client.calls[-1]
    assert "tags" not in final
    assert "tags_match" not in final
    assert final["tag_groups"] == [
        {"not": {"tags": ["sensitive"], "match": "any_strict"}}
    ]


def test_sdk_error_does_not_trigger_unfiltered_fallback(tmp_path, monkeypatch):
    client = _AsyncRecallClient(RuntimeError("service unavailable"))
    provider = _provider(
        tmp_path,
        monkeypatch,
        client,
        recall_auto_route=True,
        recall_auto_route_fail_open=True,
        recall_routes={
            "alpha": {
                "keywords": ["marker"],
                "tags": ["scope:alpha"],
            }
        },
    )

    recalled = provider._do_recall("marker query")

    assert recalled.text == ""
    assert client.arecall.await_count == 1


def test_route_retain_tags_are_opt_in_and_merged(tmp_path, monkeypatch):
    client = MagicMock()
    client.aretain_batch = AsyncMock(return_value=SimpleNamespace(ok=True))
    provider = _provider(
        tmp_path,
        monkeypatch,
        client,
        recall_auto_route=True,
        retain_tags=["global"],
        recall_routes={
            "alpha": {
                "keywords": ["marker"],
                "tags": ["scope:alpha"],
                "retain_tags": ["route:alpha", "global"],
            }
        },
    )

    result = json.loads(
        provider.handle_tool_call(
            "hindsight_retain", {"content": "marker fact", "tags": ["manual"]}
        )
    )

    assert result["result"] == "Memory stored successfully."
    item = client.aretain_batch.call_args.kwargs["items"][0]
    assert item["tags"] == ["global", "route:alpha", "manual"]


def test_reflect_uses_route_filters(tmp_path, monkeypatch):
    client = MagicMock()
    client.areflect = AsyncMock(
        return_value=SimpleNamespace(text="Synthesized answer")
    )
    provider = _provider(
        tmp_path,
        monkeypatch,
        client,
        recall_auto_route=True,
        recall_routes={
            "alpha": {
                "keywords": ["marker"],
                "tags": ["scope:alpha"],
                "exclude_tags": ["sensitive"],
            }
        },
    )

    result = json.loads(
        provider.handle_tool_call(
            "hindsight_reflect", {"query": "marker query"}
        )
    )

    assert result["result"] == "Synthesized answer"
    request = client.areflect.await_args.kwargs
    assert request["fact_types"] == ["observation"]
    assert request["tag_groups"] == [
        {
            "and": [
                {"tags": ["scope:alpha"], "match": "all_strict"},
                {"not": {"tags": ["sensitive"], "match": "any_strict"}},
            ]
        }
    ]


def test_recall_tool_uses_same_route_plan(tmp_path, monkeypatch):
    client = _AsyncRecallClient(
        _response(_result("one", "memory", ["scope:alpha"]))
    )
    provider = _provider(
        tmp_path,
        monkeypatch,
        client,
        recall_auto_route=True,
        recall_routes={
            "alpha": {
                "keywords": ["marker"],
                "tags": ["scope:alpha"],
                "types": ["world"],
                "max_results": 1,
            }
        },
    )

    result = json.loads(
        provider.handle_tool_call(
            "hindsight_recall", {"query": "marker query"}
        )
    )

    assert result["result"] == "1. memory"
    assert client.calls[0]["types"] == ["world"]
    assert client.calls[0]["tags"] == ["scope:alpha"]


def test_route_is_selected_before_query_truncation(tmp_path, monkeypatch):
    client = _AsyncRecallClient(_response(_result("one", "memory")))
    provider = _provider(
        tmp_path,
        monkeypatch,
        client,
        recall_auto_route=True,
        recall_max_input_chars=10,
        recall_routes={
            "marker": {
                "keywords": ["route-marker"],
                "tags": ["scope:marker"],
            }
        },
    )

    provider._do_recall("prefix text route-marker")

    assert client.calls[0]["query"] == "prefix tex"
    assert client.calls[0]["tags"] == ["scope:marker"]
