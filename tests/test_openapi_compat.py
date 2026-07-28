"""OpenAPI compatibility tests for the pinned Google ADK release."""

from __future__ import annotations

import copy
import os
import warnings
from pathlib import Path
from typing import Any, cast
from unittest.mock import MagicMock, create_autospec, patch

import pytest
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.routing import APIRoute
from google.adk.cli.fast_api import get_fast_api_app
from openinference.instrumentation.google_adk import GoogleADKInstrumentor

from agent import server
from agent.health import live
from agent.openapi_compat import (
    _derived_operation_id,
    ensure_unique_operation_ids,
)

HTTP_METHODS = frozenset(
    {"delete", "get", "head", "options", "patch", "post", "put", "trace"}
)
CANONICAL_ADD_SESSION_PATH = "/apps/{app_name}/eval-sets/{eval_set_id}/add-session"
LEGACY_ADD_SESSION_PATH = "/apps/{app_name}/eval_sets/{eval_set_id}/add_session"
ORIGINAL_ADD_SESSION_ID = (
    "add_session_to_eval_set_apps__app_name__eval_sets__eval_set_id__add_session_post"
)
LEGACY_ADD_SESSION_ID = (
    f"{ORIGINAL_ADD_SESSION_ID}__post_9d26a7c1b0b384aa6c1e01bda285c413"
)


def _documented_routes(app: FastAPI) -> list[APIRoute]:
    """Return the routes that participate in OpenAPI generation."""
    return [
        route
        for route in app.routes
        if isinstance(route, APIRoute) and route.include_in_schema
    ]


def _route_metadata(app: FastAPI) -> list[tuple[Any, ...]]:
    """Capture route metadata whose mutation could affect API behavior."""
    return [
        (
            id(route),
            route.path,
            route.path_format,
            tuple(sorted(route.methods)),
            route.name,
            route.operation_id,
            route.unique_id,
            id(route.endpoint),
            tuple(id(dependency) for dependency in route.dependencies),
            route.response_model,
        )
        for route in _documented_routes(app)
    ]


def _operation_ids(document: dict[str, Any]) -> list[str]:
    """Extract every operation ID in path and method order."""
    return [
        operation["operationId"]
        for path_item in document["paths"].values()
        for method, operation in path_item.items()
        if method in HTTP_METHODS
    ]


def _without_operation_ids(document: dict[str, Any]) -> dict[str, Any]:
    """Copy an OpenAPI document while removing operation IDs."""
    stripped = copy.deepcopy(document)
    for path_item in stripped["paths"].values():
        for method, operation in path_item.items():
            if method in HTTP_METHODS:
                operation.pop("operationId")
    return stripped


def _raw_adk_app(agent_dir: Path) -> FastAPI:
    """Construct the real ADK app and add repository-owned health routes."""
    app = get_fast_api_app(agents_dir=str(agent_dir), web=False)

    async def ready() -> JSONResponse:
        return JSONResponse({"status": "ready"})

    app.get("/live")(live)
    app.get("/ready")(ready)
    return app


def _duplicate_route_app(paths: tuple[str, str]) -> FastAPI:
    """Create a real FastAPI app whose normalized route IDs collide."""
    app = FastAPI()

    async def do_thing() -> dict[str, bool]:
        return {"ok": True}

    for path in paths:
        app.add_api_route(
            path,
            do_thing,
            methods=["POST"],
            name="do_thing",
        )
    return app


def _path_to_emitted_id(app: FastAPI) -> dict[str, str]:
    """Map documented paths to their effective operation IDs."""
    return {
        route.path: route.operation_id or route.unique_id
        for route in _documented_routes(app)
    }


def test_real_adk_schema_changes_only_colliding_operation_id(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Prove the shim repairs ADK's live schema without changing its contract."""
    monkeypatch.chdir(tmp_path)
    agent_dir = tmp_path / "agents"
    agent_dir.mkdir()
    app = _raw_adk_app(agent_dir)
    metadata_before = _route_metadata(app)

    with warnings.catch_warnings(record=True) as warnings_before:
        warnings.simplefilter("always")
        openapi_before = app.openapi()

    duplicate_warnings = [
        warning
        for warning in warnings_before
        if "Duplicate Operation ID" in str(warning.message)
    ]
    assert len(duplicate_warnings) == 1
    assert ensure_unique_operation_ids(app) == 1
    metadata_after = _route_metadata(app)

    changed_indexes = [
        index
        for index, (before, after) in enumerate(
            zip(metadata_before, metadata_after, strict=True)
        )
        if before != after
    ]
    assert len(changed_indexes) == 1
    changed_index = changed_indexes[0]
    changed_before = metadata_before[changed_index]
    changed_after = metadata_after[changed_index]
    assert changed_before[:5] == changed_after[:5]
    assert changed_before[5] is None
    assert changed_after[5] == LEGACY_ADD_SESSION_ID
    assert changed_before[6:] == changed_after[6:]
    assert changed_after[1] == LEGACY_ADD_SESSION_PATH

    with warnings.catch_warnings(record=True) as warnings_after:
        warnings.simplefilter("always")
        openapi_after = app.openapi()

    assert not [
        warning
        for warning in warnings_after
        if "Duplicate Operation ID" in str(warning.message)
    ]
    operation_ids = _operation_ids(openapi_after)
    assert len(operation_ids) == 47
    assert all(operation_ids)
    assert len(operation_ids) == len(set(operation_ids))
    assert (
        openapi_after["paths"][CANONICAL_ADD_SESSION_PATH]["post"]["operationId"]
        == ORIGINAL_ADD_SESSION_ID
    )
    assert (
        openapi_after["paths"][LEGACY_ADD_SESSION_PATH]["post"]["operationId"]
        == LEGACY_ADD_SESSION_ID
    )
    assert _without_operation_ids(openapi_before) == _without_operation_ids(
        openapi_after
    )

    cached_schema = app.openapi_schema
    assert ensure_unique_operation_ids(app) == 0
    assert app.openapi_schema is cached_schema


def test_ids_are_stable_when_duplicate_registration_order_reverses() -> None:
    """Keep path-to-ID mapping independent from route registration order."""
    canonical_path = "/items/{item_id}/do-thing"
    legacy_path = "/items/{item_id}/do_thing"
    first_app = _duplicate_route_app((canonical_path, legacy_path))
    second_app = _duplicate_route_app((legacy_path, canonical_path))
    original_id = _path_to_emitted_id(first_app)[canonical_path]

    assert len(set(_path_to_emitted_id(first_app).values())) == 1
    assert len(set(_path_to_emitted_id(second_app).values())) == 1
    assert ensure_unique_operation_ids(first_app) == 1
    assert ensure_unique_operation_ids(second_app) == 1

    first_ids = _path_to_emitted_id(first_app)
    second_ids = _path_to_emitted_id(second_app)
    assert first_ids == second_ids
    assert first_ids[canonical_path] == original_id
    assert first_ids[legacy_path] != original_id


def test_candidate_collision_uses_deterministic_suffix() -> None:
    """Avoid overwriting an unrelated operation ID matching the candidate."""
    canonical_path = "/items/{item_id}/do-thing"
    legacy_path = "/items/{item_id}/do_thing"
    app = _duplicate_route_app((canonical_path, legacy_path))
    routes_by_path = {route.path: route for route in _documented_routes(app)}
    existing_id = routes_by_path[canonical_path].unique_id
    base_candidate = _derived_operation_id(
        existing_id,
        routes_by_path[legacy_path],
    )

    async def reserved() -> dict[str, bool]:
        return {"reserved": True}

    app.add_api_route(
        "/reserved",
        reserved,
        methods=["GET"],
        operation_id=base_candidate,
    )

    assert ensure_unique_operation_ids(app) == 1
    assert routes_by_path[legacy_path].operation_id == f"{base_candidate}_2"
    assert _path_to_emitted_id(app)["/reserved"] == base_candidate


def test_unique_app_is_complete_no_op() -> None:
    """Leave unique explicit and generated IDs, including the schema cache, alone."""
    app = FastAPI()

    @app.get("/explicit", operation_id="explicit_operation")
    async def explicit() -> dict[str, bool]:
        return {"explicit": True}

    @app.get("/generated")
    async def generated() -> dict[str, bool]:
        return {"generated": True}

    @app.get("/hidden", include_in_schema=False)
    async def hidden() -> dict[str, bool]:
        return {"hidden": True}

    metadata_before = _route_metadata(app)
    cached_schema = app.openapi()

    assert ensure_unique_operation_ids(app) == 0
    assert _route_metadata(app) == metadata_before
    assert app.openapi_schema is cached_schema
    assert _operation_ids(cached_schema) == [
        "explicit_operation",
        "generated_generated_get",
    ]


def test_server_factory_applies_compatibility_workaround(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Verify the supported factory emits a warning-free, unique schema."""
    monkeypatch.chdir(tmp_path)
    agent_dir = tmp_path / "agents"
    agent_dir.mkdir()
    instrumentor_class = cast(
        MagicMock,
        create_autospec(GoogleADKInstrumentor, spec_set=True),
    )
    instrumentor_class.return_value.is_instrumented_by_opentelemetry = False

    with (
        patch.dict(
            os.environ,
            {
                "ADK_DISABLE_LOAD_DOTENV": "true",
                "AGENT_DIR": str(agent_dir),
                "AGENT_NAME": "openapi-test-agent",
                "ALLOW_ORIGINS": "[]",
                "OTEL_SDK_DISABLED": "true",
            },
            clear=True,
        ),
        patch.object(
            server,
            "GoogleADKInstrumentor",
            new=instrumentor_class,
        ),
    ):
        app = server.create_app()

    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        document = app.openapi()

    assert not [
        warning
        for warning in caught_warnings
        if "Duplicate Operation ID" in str(warning.message)
    ]
    operation_ids = _operation_ids(document)
    assert len(operation_ids) == len(set(operation_ids))
    assert (
        document["paths"][CANONICAL_ADD_SESSION_PATH]["post"]["operationId"]
        == ORIGINAL_ADD_SESSION_ID
    )
    assert (
        document["paths"][LEGACY_ADD_SESSION_PATH]["post"]["operationId"]
        == LEGACY_ADD_SESSION_ID
    )
