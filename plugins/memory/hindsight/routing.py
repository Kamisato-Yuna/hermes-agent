"""Configuration-driven recall routing helpers for the Hindsight provider."""

from __future__ import annotations

import json
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Dict, Iterable


_ROUTE_KEYS = {
    "name", "chat_ids", "chat_names", "keywords", "tags", "tags_match",
    "exclude_tags", "fallback_tags", "fallback_tags_match", "query_prefix",
    "types", "max_results", "retain_tags",
}
_VALID_MATCHES = {"any", "all", "any_strict", "all_strict"}


@dataclass(frozen=True)
class RecallPlan:
    kwargs: Dict[str, Any]
    route: Dict[str, Any] | None
    auto_routed: bool
    fail_open: bool
    fallback_kwargs: tuple[Dict[str, Any], ...]
    unfiltered_kwargs: Dict[str, Any] | None
    exclude_tags: tuple[str, ...]
    max_results: int


def parse_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None or value == "":
        return default
    if isinstance(value, (int, float)):
        return bool(value)
    value = str(value).strip().casefold()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    return default


def normalize_values(value: Any) -> list[str]:
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, (list, tuple, set)):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def normalize_tags(value: Any) -> list[str]:
    if isinstance(value, str):
        value = value.replace("\n", ",").split(",")
    return normalize_values(value)


def normalize_routes(value: Any) -> dict[str, dict[str, Any]]:
    if value in (None, ""):
        return {}
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (TypeError, ValueError):
            return {}
    if isinstance(value, dict) and isinstance(value.get("routes"), (dict, list)):
        value = value["routes"]
    if isinstance(value, dict):
        entries = list(value.items()) if all(isinstance(v, dict) for v in value.values()) else [(str(value.get("name", "route")), value)]
    elif isinstance(value, list):
        entries = [(str(v.get("name", i)) if isinstance(v, dict) else str(i), v) for i, v in enumerate(value)]
    else:
        return {}
    routes: dict[str, dict[str, Any]] = {}
    for index, (name, route) in enumerate(entries):
        if not isinstance(route, dict):
            continue
        route = dict(route)
        route_name = str(route.get("name") or name or index)
        if route_name in routes:
            route_name = f"{route_name}-{index}"
        route.setdefault("name", route_name)
        routes[route_name] = route
    return routes


def result_tags(result: Any) -> set[str]:
    tags = getattr(result, "tags", None)
    if tags is None and isinstance(result, dict):
        tags = result.get("tags")
    return set(normalize_tags(tags))


def filter_results(results: Iterable[Any] | None, exclude_tags: Any = (), max_results: int = 0) -> list[Any]:
    excluded = set(normalize_tags(exclude_tags))
    filtered = [result for result in list(results or []) if not (result_tags(result) & excluded)]
    return filtered[:max_results] if max_results else filtered


def strip_internal(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    return {key: value for key, value in kwargs.items() if not key.startswith("_hindsight_")}


def response_with_results(response: Any, results: list[Any]) -> Any:
    try:
        response.results = results
        return response
    except Exception:
        return SimpleNamespace(results=results)


async def arecall_with_fail_open(
    client: Any, recall_kwargs: Dict[str, Any], *, auto_routed: bool,
    fail_open: bool = False, fallback_kwargs: Iterable[Dict[str, Any]] = (),
    unfiltered_kwargs: Dict[str, Any] | None = None, exclude_tags: Any = (),
) -> Any:
    async def call(request: Dict[str, Any]) -> Any:
        api_kwargs = strip_internal(request)
        try:
            response = await client.arecall(**api_kwargs)
        except (TypeError, ModuleNotFoundError):
            if "tag_groups" not in api_kwargs:
                raise
            fallback = dict(api_kwargs)
            fallback.pop("tag_groups", None)
            positive = normalize_tags(request.get("_hindsight_positive_tags"))
            if positive:
                fallback["tags"] = positive
                fallback["tags_match"] = request.get("_hindsight_tags_match", "all_strict")
            response = await client.arecall(**fallback)
            return response_with_results(response, filter_results(response.results, exclude_tags))
        return response

    response = await call(recall_kwargs)
    if not auto_routed or filter_results(getattr(response, "results", None), exclude_tags):
        return response
    for fallback in fallback_kwargs:
        response = await call(fallback)
        if filter_results(getattr(response, "results", None), exclude_tags):
            return response
    if fail_open and unfiltered_kwargs is not None:
        return await call(unfiltered_kwargs)
    return response


def valid_match(value: Any, default: str = "all_strict") -> str:
    value = str(value or default).strip()
    return value if value in _VALID_MATCHES else default
