"""Hosted workflow and safety contracts for the real VM promotion proof."""

from __future__ import annotations

import ast
import re
from pathlib import Path

import yaml  # type: ignore[import-untyped]

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_PATH = REPOSITORY_ROOT / ".github" / "workflows" / "test-docker-compose.yml"
RUNTIME_PATH = REPOSITORY_ROOT / "tests" / "test_deployment_promotion_runtime.py"
CHECKOUT_PIN = "3d3c42e5aac5ba805825da76410c181273ba90b1"
SETUP_UV_PIN = "d4b2f3b6ecc6e67c4457f6d3e41ec42d3d0fcb86"
REGISTRY_DIGEST = "1be55279f18a2fe1a74edf2664cac61c1bea305b7b4642dab412e7affdcb3e33"


def _workflow() -> dict[str, object]:
    document = yaml.safe_load(WORKFLOW_PATH.read_text(encoding="utf-8"))
    assert isinstance(document, dict)
    return document


def _job() -> dict[str, object]:
    jobs = _workflow()["jobs"]
    assert isinstance(jobs, dict)
    selected = jobs["deployment-promotion"]
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


def test_promotion_job_is_ephemeral_read_only_and_secret_free() -> None:
    """Run Docker-root-equivalent behavior only with least-privilege CI access."""
    workflow = _workflow()
    job = _job()
    rendered = yaml.safe_dump(job, sort_keys=False)

    assert workflow["permissions"] == {"contents": "read"}
    assert job["name"] == "Validate atomic VM promotion and rollback"
    assert job["runs-on"] == "ubuntu-latest"
    assert job["timeout-minutes"] == 35
    assert "permissions" not in job
    assert ": write" not in rendered
    assert "secrets." not in rendered
    assert "github.token" not in rendered
    assert "GITHUB_TOKEN" not in rendered
    assert "docker/login-action" not in rendered
    assert "push: true" not in rendered


def test_promotion_job_uses_pinned_locked_focused_boundaries() -> None:
    """Keep hosted execution reproducible and limited to this exact proof."""
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
    assert "uv run pytest tests/test_deployment_promotion_runtime.py -q" in rendered


def test_promotion_job_has_one_unique_explicit_opt_in_namespace() -> None:
    """Prevent accidental local execution or a shared Compose project."""
    environment = _job()["env"]
    assert isinstance(environment, dict)
    assert environment == {
        "RUN_DEPLOYMENT_PROMOTION_INTEGRATION": "1",
        "DEPLOYMENT_PROMOTION_TEST_PREFIX": ("adk-promotion-${{ github.run_id }}"),
    }
    assert "COMPOSE_PROJECT_NAME" not in environment
    assert "github.run_attempt" not in environment["DEPLOYMENT_PROMOTION_TEST_PREFIX"]


def test_runtime_uses_one_real_base_and_three_thin_immutable_releases() -> None:
    """Build the real app once, then vary only wrapper and OCI revision layers."""
    source = RUNTIME_PATH.read_text(encoding="utf-8")
    runtime = _function_source(
        source,
        "test_real_docker_promotes_then_restores_exact_verified_baseline",
    )

    assert 'os.environ.get(RUN_ENVIRONMENT_NAME) != "1"' in source
    assert f'"registry@sha256:{REGISTRY_DIGEST}"' in source
    assert "registry:2" not in source
    assert "registry:3" not in source
    assert runtime.count("_build_base_image(") == 1
    assert runtime.count("_build_release_image,") == 1
    assert '("old", old_tag, old_revision)' in runtime
    assert '("good", good_tag, good_revision)' in runtime
    assert '("failing", failing_tag, failing_revision)' in runtime
    assert "FROM ${BASE_IMAGE}" in source
    assert 'LABEL org.opencontainers.image.revision="${SOURCE_REVISION}"' in source
    assert "IMAGE_REFERENCE_PATTERN.fullmatch(exact_reference)" in source
    assert '"image", "pull", config.image_reference' not in runtime


def test_runtime_prearms_exact_cleanup_and_never_prunes() -> None:
    """Leave unrelated VM resources untouched, including on partial failures."""
    source = RUNTIME_PATH.read_text(encoding="utf-8")
    owned = _function_source(source, "_owned_mutation")
    cleanup = _function_source(source, "_execute_exact_cleanup")
    production_cleanup = _function_source(source, "_production_cleanup")
    runtime = _function_source(
        source,
        "test_real_docker_promotes_then_restores_exact_verified_baseline",
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
    assert owned.index("cleanup_commands.append") < owned.index("return operation()")
    assert '"down",' in production_cleanup
    assert '"--volumes",' in production_cleanup
    finally_index = runtime.index("finally:")
    assert "_production_cleanup(" not in runtime[:finally_index]
    assert "_production_cleanup(" in runtime[finally_index:]
    assert runtime.index("production_cleanup_armed = True") < runtime.index(
        "_compose(\n"
        "            docker,\n"
        "            base_environment,\n"
        "            checkout=checkout"
    )
    assert cleanup.count("_run(") == 1
    assert '"system"' not in cleanup
    assert '"prune"' not in cleanup
    assert runtime.count("_owned_mutation(") >= 7
    assert "sentinel_container_identity" in runtime
    assert "sentinel_volume_identity" in runtime
    assert "sentinel_network_identity" in runtime
    assert "sentinel_image_identity" in runtime


def test_runtime_proves_success_then_production_only_failure_and_rollback() -> None:
    """Exercise the actual controller and inspect every restored identity."""
    source = RUNTIME_PATH.read_text(encoding="utf-8")
    wrapper = re.search(
        r'PROMOTION_WRAPPER = """\\\n(?P<body>.*?)\n"""',
        source,
        flags=re.DOTALL,
    )
    runtime = _function_source(
        source,
        "test_real_docker_promotes_then_restores_exact_verified_baseline",
    )

    assert wrapper is not None
    assert "PROMOTION_TEST_FAIL" in wrapper.group("body")
    assert "TELEMETRY_NAMESPACE" in wrapper.group("body")
    assert '"candidate"' in wrapper.group("body")
    assert "promotion-failure-sentinel" in wrapper.group("body")
    assert runtime.count("_promotion_cli(") == 2
    assert "adopt_existing=True" in runtime
    assert "adopt_existing=False" in runtime
    assert "failing_result.returncode == 1" in runtime
    assert "the recorded baseline was restored" in runtime
    assert "_read_production_failure_sentinel(" in runtime
    assert 'restored_state["Health"]["Status"] == "healthy"' in runtime
    assert 'restored_config["Image"] == references["good"]' in runtime
    assert 'restored_image_id == release_images["good"]' in runtime
    assert "restored_labels" in runtime
    assert "env_file.read_bytes() == good_environment_bytes" in runtime
    assert "== good_revision" in runtime


def test_runtime_proves_candidate_receipt_journal_volume_and_secret_contracts() -> None:
    """Bind rollback evidence to candidate, state, volume, and private env facts."""
    source = RUNTIME_PATH.read_text(encoding="utf-8")
    runtime = _function_source(
        source,
        "test_real_docker_promotes_then_restores_exact_verified_baseline",
    )

    assert "good_volume_identity == old_volume_identity" in runtime
    assert "VOLUME_SENTINEL" in runtime
    assert "rollback.persistent_volumes[0].as_document()" in runtime
    assert 'intent["candidate"]["image_reference"]' in runtime
    assert 'intent["candidate"]["container_id"]' in runtime
    assert 'intent["baseline_journal_sequence"] == 2' in runtime
    assert '"adopted",\n            "promoted",\n            "rolled_back"' in runtime
    assert "not store.pending_path.exists()" in runtime
    assert "canary.encode() not in intent_bytes" in runtime
    assert "canary.encode() not in store.current_path.read_bytes()" in runtime
    assert "_assert_safe_output(good_result)" in runtime
    assert "_assert_safe_output(failing_result)" in runtime
