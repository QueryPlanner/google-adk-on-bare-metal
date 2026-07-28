"""Hosted Docker workflow contracts for VM deployment-state adoption."""

from __future__ import annotations

import ast
import re
from pathlib import Path

import yaml  # type: ignore[import-untyped]

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_PATH = REPOSITORY_ROOT / ".github" / "workflows" / "test-docker-compose.yml"
RUNTIME_PATH = REPOSITORY_ROOT / "tests" / "test_deployment_state_runtime.py"
CHECKOUT_PIN = "3d3c42e5aac5ba805825da76410c181273ba90b1"
SETUP_UV_PIN = "d4b2f3b6ecc6e67c4457f6d3e41ec42d3d0fcb86"


def _workflow() -> dict[str, object]:
    document = yaml.safe_load(WORKFLOW_PATH.read_text(encoding="utf-8"))
    assert isinstance(document, dict)
    return document


def _job() -> dict[str, object]:
    jobs = _workflow()["jobs"]
    assert isinstance(jobs, dict)
    selected = jobs["deployment-state"]
    assert isinstance(selected, dict)
    return selected


def _function_source(document: str, name: str) -> str:
    tree = ast.parse(document)
    function = next(
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == name
    )
    return ast.get_source_segment(document, function) or ""


def test_deployment_state_job_is_read_only_and_secret_free() -> None:
    """Run Docker-root-equivalent proof only on an ephemeral least-privilege host."""
    workflow = _workflow()
    job = _job()
    job_source = yaml.safe_dump(job, sort_keys=False)

    assert workflow["permissions"] == {"contents": "read"}
    assert job["name"] == "Validate VM deployment-state adoption"
    assert job["runs-on"] == "ubuntu-latest"
    assert job["timeout-minutes"] == 20
    assert "permissions" not in job
    assert ": write" not in job_source
    assert "secrets." not in job_source
    assert "github.token" not in job_source
    assert "GITHUB_TOKEN" not in job_source
    assert "docker/login-action" not in job_source
    assert "push: true" not in job_source


def test_deployment_state_job_uses_pinned_locked_focused_tools() -> None:
    """Keep the hosted proof reproducible and limited to one runtime test."""
    steps = _job()["steps"]
    assert isinstance(steps, list)
    rendered = yaml.safe_dump(steps, sort_keys=False)

    assert rendered.count("actions/checkout@") == 1
    assert f"actions/checkout@{CHECKOUT_PIN}" in rendered
    assert rendered.count("persist-credentials: false") == 1
    assert rendered.count("astral-sh/setup-uv@") == 1
    assert f"astral-sh/setup-uv@{SETUP_UV_PIN}" in rendered
    assert "run: uv python install 3.13" in rendered
    assert "run: uv sync --locked" in rendered
    assert rendered.count("uv run pytest") == 1
    assert "uv run pytest tests/test_deployment_state_runtime.py -q" in rendered


def test_deployment_state_job_enables_unique_opt_in_boundary() -> None:
    """Require explicit real-Docker execution and a per-run owned namespace."""
    environment = _job()["env"]
    assert isinstance(environment, dict)
    assert environment == {
        "RUN_DEPLOYMENT_STATE_INTEGRATION": "1",
        "DEPLOYMENT_STATE_TEST_PREFIX": "adk-state-${{ github.run_id }}",
    }
    assert "COMPOSE_PROJECT_NAME" not in environment


def test_runtime_uses_candidate_not_production_compose() -> None:
    """Prove adoption without publishing a production port or touching its project."""
    source = RUNTIME_PATH.read_text(encoding="utf-8")

    assert (
        'CANDIDATE_COMPOSE_PATH = REPOSITORY_ROOT / "compose.candidate.yaml"' in source
    )
    assert 'REPOSITORY_ROOT / "compose.yaml"' not in source
    assert '"127.0.0.1::5000"' in source
    assert '"up",' in source
    assert '"--no-build",' in source
    assert '"--pull",' in source
    assert '"never",' in source
    assert '"--wait",' in source
    assert '"deployment-state-runtime"' in source


def test_runtime_cleanup_is_exact_and_never_prunes() -> None:
    """Preserve unrelated VM resources and persistent volumes."""
    source = RUNTIME_PATH.read_text(encoding="utf-8")
    compose_source = _function_source(source, "_compose")
    runtime_source = _function_source(
        source,
        "test_real_docker_adopts_exact_healthy_compose_state",
    )
    for forbidden in (
        "docker system prune",
        "docker image prune",
        "docker builder prune",
        '"volume", "prune"',
        '"network", "prune"',
        '"container", "prune"',
    ):
        assert forbidden not in source
    assert 'project.startswith("adk-state-")' in compose_source
    assert '"down", "--remove-orphans", "--timeout", "30"' in runtime_source
    assert runtime_source.count('"--volumes"') == 1
    assert re.search(
        r'"container",\s*"rm",\s*"--force",\s*"--volumes",\s*registry_name',
        runtime_source,
    )
    assert "baseline_volumes" in runtime_source
    assert "remaining_volumes != baseline_volumes" in runtime_source
    assert "sentinel_container_identity" in runtime_source
    assert "sentinel_volume_identity" in runtime_source
    assert "sentinel_network_id" in runtime_source
    assert "sentinel_image_identity" in runtime_source


def test_runtime_pins_supported_registry_and_prearms_cleanup() -> None:
    """Use an immutable Registry v3 image and clean partial startup failures."""
    source = RUNTIME_PATH.read_text(encoding="utf-8")
    runtime_source = _function_source(
        source,
        "test_real_docker_adopts_exact_healthy_compose_state",
    )
    owned_source = _function_source(source, "_owned_mutation")
    registry_source = _function_source(source, "_ensure_registry_image")
    digest_match = re.search(
        r"REGISTRY_IMAGE_REFERENCE = \(\n"
        r'    "registry@sha256:([0-9a-f]{64})"\n'
        r"\)",
        source,
    )

    assert digest_match is not None
    assert digest_match.group(1) != "0" * 64
    assert "registry:2" not in source
    assert "registry:3" not in source
    assert owned_source.index("cleanup_commands.append") < owned_source.index(
        "return operation()"
    )
    assert runtime_source.count("_owned_mutation(") >= 7
    assert "lambda: _build_image(" in runtime_source
    assert "lambda: _create_registry(" in runtime_source
    assert "_owned_mutation(" in registry_source
    assert "REGISTRY_IMAGE_REFERENCE" in registry_source
    assert runtime_source.index(
        "compose_cleanup_required = True"
    ) < runtime_source.index("_compose(")


def test_runtime_proves_real_cli_lock_and_durable_private_state() -> None:
    """Keep the opt-in proof tied to the operator-facing state boundary."""
    source = RUNTIME_PATH.read_text(encoding="utf-8")
    runtime_source = _function_source(
        source,
        "test_real_docker_adopts_exact_healthy_compose_state",
    )

    assert 'os.environ.get(RUN_ENVIRONMENT_NAME) != "1"' in source
    assert "_require_docker(environment)" in runtime_source
    assert '"-m", "agent.deployment_state_cli"' in source
    assert '"adopt",' in runtime_source
    assert '"inspect",' in runtime_source
    assert "with store.transaction():" in runtime_source
    assert "busy.returncode == 75" in runtime_source
    assert "snapshot.read_bytes() == environment_before" in runtime_source
    assert "env_file.read_bytes() == environment_before" in runtime_source
    assert "environment_sha256" in runtime_source
    assert "0o700" in runtime_source
    assert "0o600" in runtime_source
    assert "PRIVATE_CANARY not in" in runtime_source
