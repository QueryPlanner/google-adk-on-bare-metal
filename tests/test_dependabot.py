"""Dependabot configuration syntax and policy contract tests."""

import re
import tomllib
from pathlib import Path
from typing import Any, cast

import pytest
import yaml  # type: ignore[import-untyped]

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPOSITORY_ROOT / ".github" / "dependabot.yml"
COMPOSE_PATH = REPOSITORY_ROOT / "compose.yaml"
TRACE_GATEWAY_COMPOSE_PATH = REPOSITORY_ROOT / "compose.trace-gateway.yaml"
DOCKERFILE_PATH = REPOSITORY_ROOT / "Dockerfile"
PYPROJECT_PATH = REPOSITORY_ROOT / "pyproject.toml"
LOCKFILE_PATH = REPOSITORY_ROOT / "uv.lock"
WORKFLOW_DIRECTORY = REPOSITORY_ROOT / ".github" / "workflows"

EXPECTED_POLICIES = {
    ("uv", "/"): ("weekly", 4, "chore(deps)"),
    ("github-actions", "/"): ("daily", 3, "ci(deps)"),
    ("docker", "/"): ("weekly", 2, "build(deps)"),
    ("docker-compose", "/"): ("weekly", 1, "build(deps)"),
}
EXPECTED_IGNORES = {
    ("uv", "/"): [
        {
            "dependency-name": "*",
            "update-types": ["version-update:semver-major"],
        },
        {
            "dependency-name": "pydantic",
            "update-types": [
                "version-update:semver-patch",
                "version-update:semver-minor",
            ],
        },
        {
            "dependency-name": "opentelemetry-exporter-gcp-logging",
            "update-types": ["version-update:semver-minor"],
        },
        {
            "dependency-name": "opentelemetry-exporter-otlp-proto-http",
            "update-types": ["version-update:semver-minor"],
        },
        {
            "dependency-name": "opentelemetry-instrumentation-logging",
            "update-types": ["version-update:semver-minor"],
        },
    ],
    ("github-actions", "/"): [
        {
            "dependency-name": "*",
            "update-types": ["version-update:semver-major"],
        },
    ],
    ("docker", "/"): [
        {
            "dependency-name": "python",
            "update-types": [
                "version-update:semver-minor",
                "version-update:semver-major",
            ],
        },
    ],
}
SUPPORTED_UPDATE_TYPES = {
    "version-update:semver-patch",
    "version-update:semver-minor",
    "version-update:semver-major",
}
BASE_UPDATE_KEYS = {
    "package-ecosystem",
    "directory",
    "schedule",
    "open-pull-requests-limit",
    "commit-message",
}
FORBIDDEN_UPDATE_KEYS = {
    "allow",
    "groups",
    "insecure-external-code-execution",
    "multi-ecosystem-group",
    "registries",
    "target-branch",
}


class UniqueKeySafeLoader(yaml.SafeLoader):
    """Safe YAML loader that fails instead of overwriting duplicate keys."""


def _construct_unique_mapping(
    loader: UniqueKeySafeLoader,
    node: Any,
    deep: bool = False,
) -> dict[object, object]:
    """Construct a mapping after proving every key is unique."""
    loader.flatten_mapping(node)
    mapping: dict[object, object] = {}

    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise ValueError(f"Duplicate YAML key: {key!r}")
        mapping[key] = loader.construct_object(value_node, deep=deep)

    return mapping


UniqueKeySafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def _parse_yaml(document: str) -> object:
    """Parse one YAML document without allowing unsafe constructors."""
    loader = UniqueKeySafeLoader(document)
    try:
        return loader.get_single_data()
    finally:
        loader.dispose()


def _load_configuration() -> dict[str, Any]:
    """Return the tracked Dependabot document as a mapping."""
    configuration = _parse_yaml(CONFIG_PATH.read_text(encoding="utf-8"))
    assert isinstance(configuration, dict)
    return cast(dict[str, Any], configuration)


def _updates_by_ecosystem(
    configuration: dict[str, Any],
) -> dict[tuple[str, str], dict[str, Any]]:
    """Index update entries while rejecting duplicate ecosystem coverage."""
    raw_updates = configuration.get("updates")
    assert isinstance(raw_updates, list)
    updates: dict[tuple[str, str], dict[str, Any]] = {}

    for raw_update in raw_updates:
        assert isinstance(raw_update, dict)
        update = cast(dict[str, Any], raw_update)
        ecosystem = update.get("package-ecosystem")
        directory = update.get("directory")
        assert isinstance(ecosystem, str)
        assert isinstance(directory, str)
        key = (ecosystem, directory)
        assert key not in updates
        updates[key] = update

    return updates


@pytest.mark.parametrize(
    "document",
    (
        "version: 2\nversion: 3\nupdates: []\n",
        (
            "version: 2\n"
            "updates:\n"
            "  - package-ecosystem: uv\n"
            "    directory: /\n"
            "    schedule:\n"
            "      interval: weekly\n"
            "      interval: daily\n"
        ),
    ),
)
def test_yaml_parser_rejects_duplicate_mapping_keys(document: str) -> None:
    """Prevent valid-looking duplicate keys from silently replacing policy."""
    with pytest.raises(ValueError, match="Duplicate YAML key"):
        _parse_yaml(document)


def test_configuration_has_exact_individual_update_policy() -> None:
    """Keep each supported dependency update independently reviewable."""
    configuration = _load_configuration()

    assert set(configuration) == {"version", "updates"}
    assert type(configuration["version"]) is int
    assert configuration["version"] == 2

    updates = _updates_by_ecosystem(configuration)
    assert set(updates) == set(EXPECTED_POLICIES)

    for key, (interval, pull_request_limit, commit_prefix) in EXPECTED_POLICIES.items():
        update = updates[key]
        expected_keys = BASE_UPDATE_KEYS | (
            {"exclude-paths"} if key == ("docker", "/") else set()
        )
        if key in EXPECTED_IGNORES:
            expected_keys.add("ignore")

        assert set(update) == expected_keys
        assert FORBIDDEN_UPDATE_KEYS.isdisjoint(update)
        assert update["schedule"] == {"interval": interval}
        assert type(update["open-pull-requests-limit"]) is int
        assert update["open-pull-requests-limit"] == pull_request_limit
        assert update["commit-message"] == {"prefix": commit_prefix}
        if key in EXPECTED_IGNORES:
            assert update["ignore"] == EXPECTED_IGNORES[key]

    assert updates[("docker", "/")]["exclude-paths"] == ["compose.yaml"]


def test_ignore_policy_is_narrow_and_version_only() -> None:
    """Keep resolver safeguards explicit without hiding security updates."""
    updates = _updates_by_ecosystem(_load_configuration())

    for key, expected_conditions in EXPECTED_IGNORES.items():
        conditions = updates[key]["ignore"]
        assert isinstance(conditions, list)
        assert conditions == expected_conditions

        dependency_names: set[str] = set()
        for condition in conditions:
            assert isinstance(condition, dict)
            assert set(condition) == {"dependency-name", "update-types"}

            dependency_name = condition["dependency-name"]
            update_types = condition["update-types"]
            assert isinstance(dependency_name, str)
            assert dependency_name not in dependency_names
            dependency_names.add(dependency_name)
            assert isinstance(update_types, list)
            assert update_types
            assert len(update_types) == len(set(update_types))
            assert set(update_types) <= SUPPORTED_UPDATE_TYPES
            assert all(isinstance(update_type, str) for update_type in update_types)

    uv_dependency_names = {
        condition["dependency-name"] for condition in EXPECTED_IGNORES[("uv", "/")]
    }
    assert "google-adk" not in uv_dependency_names
    assert "opentelemetry-*" not in uv_dependency_names
    uv_ignores_by_dependency = {
        condition["dependency-name"]: set(condition["update-types"])
        for condition in EXPECTED_IGNORES[("uv", "/")]
    }
    assert (
        uv_ignores_by_dependency["*"] | uv_ignores_by_dependency["pydantic"]
        == SUPPORTED_UPDATE_TYPES
    )
    assert "versions" not in {
        key
        for conditions in EXPECTED_IGNORES.values()
        for condition in conditions
        for key in condition
    }
    assert "ignore" not in updates[("docker-compose", "/")]


def test_configuration_targets_real_supported_manifests() -> None:
    """Tie every configured ecosystem to its repository manifest."""
    assert PYPROJECT_PATH.is_file()
    assert LOCKFILE_PATH.is_file()

    project = tomllib.loads(PYPROJECT_PATH.read_text(encoding="utf-8"))
    assert project["project"]["requires-python"] == ">=3.13,<3.14"
    project_dependencies = project["project"]["dependencies"]
    direct_dependency_names = {
        re.split(r"[<>=!~ ;\[]", dependency, maxsplit=1)[0]
        .strip()
        .lower()
        .replace("_", "-")
        for dependency in project_dependencies
    }
    uv_ignore_names = {
        condition["dependency-name"]
        for condition in EXPECTED_IGNORES[("uv", "/")]
        if condition["dependency-name"] != "*"
    }
    assert uv_ignore_names <= direct_dependency_names

    development_dependencies = project["dependency-groups"]["dev"]
    assert "pyyaml>=6.0.3,<7.0.0" in development_dependencies

    workflow_paths = sorted(
        path
        for pattern in ("*.yml", "*.yaml")
        for path in WORKFLOW_DIRECTORY.glob(pattern)
    )
    workflow_documents = [path.read_text(encoding="utf-8") for path in workflow_paths]
    assert workflow_documents
    assert any(
        re.search(r"(?m)^\s*uses:\s*(?!\./)[^@\s]+@", document)
        for document in workflow_documents
    )

    dockerfile = DOCKERFILE_PATH.read_text(encoding="utf-8")
    assert re.search(
        r"(?mi)^FROM\s+python:3\.13(?:\.\d+)?-slim\s+AS\s+python-base\s*$",
        dockerfile,
    )

    compose = COMPOSE_PATH.read_text(encoding="utf-8")
    assert re.findall(r"(?m)^\s*image:\s*(\S+)\s*$", compose) == ["${IMAGE:-agent}"]
    assert re.search(r"(?m)^\s*build:\s*\.\s*$", compose)

    trace_gateway_compose = TRACE_GATEWAY_COMPOSE_PATH.read_text(encoding="utf-8")
    assert (
        "opentelemetry-collector-contrib:0.157.0@"
        "sha256:f2f01157055a9b2aab9df7118e1f1c9abf345e99b23bc7a2bc791db374a7d0f6"
    ) in trace_gateway_compose

    dependabot_document = CONFIG_PATH.read_text(encoding="utf-8")
    assert (
        "Track the optional pinned Collector overlay on the same weekly cadence."
        in dependabot_document
    )
    assert (
        "Remove when the supported LiteLLM range no longer pins "
        "Pydantic to 2.12.5." in dependabot_document
    )


def test_pydantic_constraint_matches_locked_litellm_compatibility() -> None:
    """Keep the direct constraint truthful without changing resolved runtime code."""
    project = tomllib.loads(PYPROJECT_PATH.read_text(encoding="utf-8"))
    assert "pydantic==2.12.5" in project["project"]["dependencies"]

    lockfile = tomllib.loads(LOCKFILE_PATH.read_text(encoding="utf-8"))
    packages = lockfile["package"]
    root_package = next(
        package for package in packages if package["name"] == "google-adk-on-bare-metal"
    )
    pydantic_requirement = next(
        requirement
        for requirement in root_package["metadata"]["requires-dist"]
        if requirement["name"] == "pydantic"
    )
    assert pydantic_requirement == {
        "name": "pydantic",
        "specifier": "==2.12.5",
    }

    resolved_versions = {
        package["name"]: package["version"]
        for package in packages
        if package["name"] in {"google-adk", "litellm", "pydantic"}
    }
    assert resolved_versions == {
        "google-adk": "1.36.2",
        "litellm": "1.83.14",
        "pydantic": "2.12.5",
    }
