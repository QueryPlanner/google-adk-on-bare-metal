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


def _step(name: str) -> dict[str, object]:
    steps = _job()["steps"]
    assert isinstance(steps, list)
    matches = [
        step for step in steps if isinstance(step, dict) and step.get("name") == name
    ]
    assert len(matches) == 1
    return matches[0]


def _run_text(step: dict[str, object]) -> str:
    run = step["run"]
    assert isinstance(run, str)
    return run


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
    job = _job()
    steps = job["steps"]
    assert isinstance(steps, list)
    rendered = yaml.safe_dump(steps, sort_keys=False)
    execution = _step("Run candidate isolation, promotion, and rollback proof")
    normalized_command = " ".join(_run_text(execution).split())

    assert set(job) == {"name", "runs-on", "timeout-minutes", "env", "steps"}
    assert set(execution) == {"name", "env", "run"}
    assert rendered.count("actions/checkout@") == 1
    assert f"actions/checkout@{CHECKOUT_PIN}" in rendered
    assert rendered.count("persist-credentials: false") == 1
    assert rendered.count("astral-sh/setup-uv@") == 1
    assert f"astral-sh/setup-uv@{SETUP_UV_PIN}" in rendered
    assert "run: uv python install 3.13" in rendered
    assert "run: uv sync --locked" in rendered
    assert execution["env"] == {
        "PYTEST_ADDOPTS": "",
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
        "PYTEST_PLUGINS": "",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    assert normalized_command == (
        "uv run --locked --no-sync pytest "
        "--noconftest --confcutdir=tests "
        "-o addopts= -p no:cacheprovider "
        "tests/test_deployment_promotion_runtime.py::"
        "test_real_docker_promotes_then_restores_exact_verified_baseline "
        "-q --tb=line --disable-warnings --show-capture=no"
    )
    for bypass in ("--collect-only", " -k ", "|| true", "; true"):
        assert bypass not in f" {normalized_command} "


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


def test_runtime_uses_one_real_base_and_four_thin_immutable_releases() -> None:
    """Build the real app once, then vary only wrapper and OCI revision layers."""
    source = RUNTIME_PATH.read_text(encoding="utf-8")
    release_builder = _function_source(source, "_build_release_image")
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
    assert "for phase, tag, revision, context, expected_command in (" in runtime
    for tag in ("old_tag", "good_tag", "unhealthy_tag", "failing_tag"):
        assert runtime.count(tag) >= 2
    assert "FROM ${BASE_IMAGE}" in source
    assert 'LABEL org.opencontainers.image.revision="${SOURCE_REVISION}"' in source
    assert 'HEALTHY_RELEASE_COMMAND = ("python", "-m", "agent.server")' in source
    assert '"import time; time.sleep(300)",' in source
    assert "__RELEASE_COMMAND__" in source
    assert runtime.count("_write_derivative_context(") == 2
    assert '"{{json .Config.Cmd}}"' in release_builder
    assert "configured_command == list(expected_command)" in release_builder
    assert "IMAGE_REFERENCE_PATTERN.fullmatch(exact_reference)" in source
    assert '"image", "pull", config.image_reference' not in runtime
    assert "assert len(set(revisions.values())) == 4" in runtime
    assert "assert len(set(release_images.values())) == 4" in runtime
    assert "assert len(set(references.values())) == 4" in runtime
    assert "assert oci_revision == revision" in runtime
    assert 'old_container_config["Cmd"] == list(HEALTHY_RELEASE_COMMAND)' in runtime


def test_runtime_prearms_exact_cleanup_and_never_prunes() -> None:
    """Leave unrelated VM resources untouched, including on partial failures."""
    source = RUNTIME_PATH.read_text(encoding="utf-8")
    owned = _function_source(source, "_owned_mutation")
    cleanup = _function_source(source, "_execute_exact_cleanup")
    cleanup_target = _function_source(source, "_cleanup_owned_target")
    ownership = _function_source(source, "_owned_resource_document")
    production_cleanup = _function_source(source, "_production_cleanup")
    project_ownership = _function_source(source, "_assert_compose_project_owned")
    project_verification = _function_source(
        source,
        "_verified_compose_project_targets",
    )
    registry_image = _function_source(source, "_ensure_registry_image")
    candidate_cleanup = _function_source(source, "_cleanup_candidate_containers")
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
    assert owned.index("cleanup_targets.append") < owned.index("return operation()")
    assert '"down",' in production_cleanup
    assert '"--volumes",' in production_cleanup
    assert production_cleanup.index("_verified_compose_project_targets(") < (
        production_cleanup.index("_compose(")
    )
    assert production_cleanup.index("_compose(") < production_cleanup.index(
        "_assert_compose_project_absent("
    )
    assert "_verified_compose_project_targets(" in project_ownership
    assert "_owned_resource_document(" in project_verification
    assert "if not targets:" in production_cleanup
    assert "_cleanup_owned_target(docker, environment, target)" in production_cleanup
    assert "cleanup target owner did not match" in ownership
    assert "cleanup target identity did not match" in ownership
    assert "expected_owner" in ownership
    assert cleanup_target.index("_owned_resource_document(") < cleanup_target.index(
        'command = [docker, "image", "rm", target.reference]'
    )
    assert "_listed_resource_ids(docker, environment, target)" in cleanup_target
    assert "owned Docker cleanup left its exact target" in cleanup_target
    assert "_cleanup_owned_target(docker, environment, target)" in cleanup
    assert "No such" not in cleanup
    assert "not found" not in cleanup
    assert "does not exist" not in cleanup
    assert "_owned_mutation(" not in registry_image
    assert "REGISTRY_IMAGE_REFERENCE" in registry_image
    finally_index = runtime.index("finally:")
    assert "_production_cleanup(" not in runtime[:finally_index]
    assert "_production_cleanup(" in runtime[finally_index:]
    assert "_cleanup_candidate_containers(" not in runtime[:finally_index]
    assert "_cleanup_candidate_containers(" in runtime[finally_index:]
    assert runtime.index("production_cleanup_armed = True") < runtime.index(
        "_compose(\n"
        "            docker,\n"
        "            base_environment,\n"
        "            checkout=checkout"
    )
    assert '"system"' not in cleanup
    assert '"prune"' not in cleanup
    assert runtime.count("_owned_mutation(") >= 7
    assert "sentinel_container_identity" in runtime
    assert "sentinel_volume_identity" in runtime
    assert "sentinel_network_identity" in runtime
    assert "sentinel_image_identity" in runtime
    assert source.count("io.queryplanner.adk.promotion-test.owner") >= 4
    assert 'document.get("Image") != image_id' in candidate_cleanup
    assert 'project.startswith("candidate-")' in candidate_cleanup
    assert '"com.docker.compose.project.working_dir"' in candidate_cleanup
    assert '"container",\n                    "rm",' in candidate_cleanup
    assert runtime.index("_production_cleanup(", finally_index) < runtime.index(
        "_cleanup_candidate_containers(",
        finally_index,
    )
    assert runtime.index(
        "_cleanup_candidate_containers(",
        finally_index,
    ) < runtime.index("_execute_exact_cleanup(", finally_index)


def test_runtime_reserves_generated_docker_resources_before_mutation() -> None:
    """Fail closed instead of adopting or overwriting a colliding resource."""
    source = RUNTIME_PATH.read_text(encoding="utf-8")
    runtime = _function_source(
        source,
        "test_real_docker_promotes_then_restores_exact_verified_baseline",
    )
    push = _function_source(source, "_push_exact_image")
    project_guard = _function_source(source, "_assert_compose_project_absent")

    assert runtime.index(
        "_assert_image_reference_absent(docker, base_environment, base_tag)"
    ) < runtime.index("_owned_mutation(")
    assert runtime.index(
        "_assert_image_reference_absent(docker, base_environment, tag)"
    ) < runtime.index("release_images[phase] = _owned_mutation(")
    assert runtime.index(
        "_assert_container_name_absent(\n"
        "            docker,\n"
        "            base_environment,\n"
        "            registry_name,"
    ) < runtime.index("lambda: _create_registry(")
    assert runtime.index(
        "_assert_image_reference_absent(\n"
        "            docker,\n"
        "            base_environment,\n"
        "            sentinel_image,"
    ) < runtime.index('[docker, "image", "tag", registry_image_id, sentinel_image]')
    assert runtime.index(
        "_assert_container_name_absent(\n"
        "            docker,\n"
        "            base_environment,\n"
        "            sentinel_container,"
    ) < runtime.index('"--name",\n                    sentinel_container,')
    assert runtime.index(
        "_assert_volume_name_absent(\n"
        "            docker,\n"
        "            base_environment,\n"
        "            sentinel_volume,"
    ) < runtime.index('kind="volume",\n                reference=sentinel_volume,')
    assert runtime.index(
        "_assert_network_name_absent(\n"
        "            docker,\n"
        "            base_environment,\n"
        "            sentinel_network,"
    ) < runtime.index('kind="network",\n                reference=sentinel_network,')
    assert push.index("_assert_image_reference_absent(") < push.index(
        "_owned_mutation("
    )
    assert runtime.index("_assert_compose_project_absent(") < runtime.index(
        "production_cleanup_armed = True"
    )
    assert project_guard.count("com.docker.compose.project=") == 1
    assert '"container", "ls"' in project_guard
    assert '"volume", "ls"' in project_guard
    assert '"network", "ls"' in project_guard
    assert "project}-agent-1" in project_guard
    assert "project}_agent_artifacts" in project_guard
    assert "project}_default" in project_guard
    assert runtime.count("expected_owner=prefix") >= 6
    assert "expected_id=registry_image_id" in runtime
    assert runtime.count('f"{OWNER_LABEL}={prefix}"') >= 3
    assert "_assert_compose_project_owned(" in runtime


def test_runtime_proves_candidate_isolation_then_promotion_rollback() -> None:
    """Exercise the controller across pre-cutover rejection and rollback."""
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
    lifecycle = _function_source(source, "_candidate_lifecycle_evidence")

    assert wrapper is not None
    assert "PROMOTION_TEST_FAIL" in wrapper.group("body")
    assert "TELEMETRY_NAMESPACE" in wrapper.group("body")
    assert '"candidate"' in wrapper.group("body")
    assert "promotion-failure-sentinel" in wrapper.group("body")
    assert runtime.count("_promotion_cli(") == 3
    assert "adopt_existing=True" in runtime
    assert runtime.count("adopt_existing=False") == 2
    assert "unhealthy_result.returncode == 1" in runtime
    assert "candidate Compose start (exit 1)" in runtime
    assert "CANDIDATE_START_PERIOD_SECONDS" in runtime
    assert "CANDIDATE_FAILURE_BOUND_SECONDS" in runtime
    assert "start_period: {CANDIDATE_START_PERIOD_SECONDS}s" in runtime
    assert 'unhealthy_release_checkout / "compose.candidate.yaml"' in runtime
    assert "_candidate_lifecycle_evidence(" in runtime
    assert '"events",' in lifecycle
    assert '"type=container"' in lifecycle
    assert "label=com.docker.compose.project.working_dir=" in lifecycle
    assert 'attributes.get("image") != image_reference' in lifecycle
    for action in (
        '"create"',
        '"start"',
        '"health_status: unhealthy"',
        '"destroy"',
    ):
        assert action in lifecycle
    assert "unhealthy_at - started_at" in lifecycle
    assert "CANDIDATE_START_PERIOD_SECONDS * 1_000_000_000" in lifecycle
    assert runtime.count("_production_process_identity(") == 2
    assert "good_container_id," in runtime
    assert "after_unhealthy_container_id," in runtime
    assert "_private_file_fingerprint(env_file) == good_environment_fingerprint" in (
        runtime
    )
    assert "_state_tree_fingerprint(state_directory) == good_state_tree" in runtime
    assert "== good_candidate_containers" in runtime
    assert "failing_result.returncode == 1" in runtime
    assert "the recorded baseline was restored" in runtime
    assert "_read_production_failure_sentinel(" in runtime
    assert 'restored_state["Health"]["Status"] == "healthy"' in runtime
    assert 'restored_config["Image"] == references["good"]' in runtime
    assert 'restored_image_id == release_images["good"]' in runtime
    assert "restored_labels" in runtime
    assert "restored_environment_fingerprint" in runtime
    assert "env_file.read_bytes() == good_environment_bytes" not in runtime
    assert "== good_revision" in runtime


def test_runtime_proves_candidate_receipt_journal_volume_and_secret_contracts() -> None:
    """Bind rollback evidence to candidate, state, volume, and private env facts."""
    source = RUNTIME_PATH.read_text(encoding="utf-8")
    process_identity = _function_source(source, "_production_process_identity")
    environment_fingerprint = _function_source(
        source,
        "_private_file_fingerprint",
    )
    byte_comparison = _function_source(source, "_assert_private_file_contents")
    state_fingerprint = _function_source(source, "_state_tree_fingerprint")
    candidate_ids = _function_source(source, "_candidate_service_container_ids")
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
    assert "durable_payloads" in runtime
    assert "durable deployment state exposed a canary" in runtime
    assert "_assert_safe_output(good_result)" in runtime
    assert "_assert_safe_output(unhealthy_result)" in runtime
    assert "_assert_safe_output(failing_result)" in runtime
    assert "promotion output exposed a private canary" in source
    for field in (
        '"container_id"',
        '"image_reference"',
        '"image_id"',
        '"pid"',
        '"started_at"',
        '"restart_count"',
        '"running"',
        '"status"',
        '"health_status"',
    ):
        assert field in process_identity
    assert "path.lstat()" in environment_fingerprint
    assert "mode != 0o600" in environment_fingerprint
    assert "hashlib.sha256(contents).hexdigest()" in environment_fingerprint
    assert "hmac.compare_digest" in byte_comparison
    assert "private environment bytes changed" in byte_comparison
    assert runtime.count("_assert_private_file_contents(") == 3
    assert "path.lstat()" in state_fingerprint
    assert "deployment-state tree contains a special path" in state_fingerprint
    assert "hashlib.sha256(contents).hexdigest()" in state_fingerprint
    assert '"label=com.docker.compose.service=agent"' in candidate_ids
    assert 'f"ancestor={image_id}"' in candidate_ids
    assert "label=com.docker.compose.project.working_dir=" in candidate_ids
