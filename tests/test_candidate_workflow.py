"""Static CI and safety contracts for the candidate runtime proof."""

from __future__ import annotations

import ast
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_PATH = REPOSITORY_ROOT / ".github" / "workflows" / "test-docker-compose.yml"
RUNTIME_TEST_PATH = REPOSITORY_ROOT / "tests" / "test_candidate_runtime.py"
CHECKOUT_PIN = "3d3c42e5aac5ba805825da76410c181273ba90b1"
SETUP_UV_PIN = "d4b2f3b6ecc6e67c4457f6d3e41ec42d3d0fcb86"
EXPECTED_ENV_ALLOWLIST = (
    "AGENT_NAME",
    "ROOT_AGENT_MODEL",
    "LOG_LEVEL",
    "TELEMETRY_NAMESPACE",
    "K_REVISION",
    "CANDIDATE_ENV_CANARY",
)


def _job_block(document: str, job_name: str) -> str:
    """Return one complete top-level workflow job."""
    lines = document.splitlines()
    start = lines.index(f"  {job_name}:")
    block = [lines[start]]
    for line in lines[start + 1 :]:
        if line.startswith("  ") and not line.startswith("    ") and line.endswith(":"):
            break
        block.append(line)
    return "\n".join(block)


def _assigned_literal(source: str, name: str) -> object:
    """Return one module-level literal assignment without importing the test."""
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign):
            targets = node.targets
            assigned_value: ast.expr | None = node.value
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
            assigned_value = node.value
        else:
            continue
        if any(
            isinstance(target, ast.Name) and target.id == name for target in targets
        ):
            assert assigned_value is not None
            return ast.literal_eval(assigned_value)
    raise AssertionError(f"{name} is not assigned")


def _function_source(source: str, name: str) -> str:
    """Return exact source for one top-level function using its AST range."""
    for node in ast.parse(source).body:
        if (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == name
        ):
            assert node.end_lineno is not None
            return "\n".join(source.splitlines()[node.lineno - 1 : node.end_lineno])
    raise AssertionError(f"{name} is not defined")


def _class_source(source: str, name: str) -> str:
    """Return exact source for one top-level class using its AST range."""
    for node in ast.parse(source).body:
        if isinstance(node, ast.ClassDef) and node.name == name:
            assert node.end_lineno is not None
            return "\n".join(source.splitlines()[node.lineno - 1 : node.end_lineno])
    raise AssertionError(f"{name} is not defined")


def test_candidate_job_inherits_read_only_least_privilege_context() -> None:
    """Run on normal main changes without secrets or write permissions."""
    document = WORKFLOW_PATH.read_text(encoding="utf-8")
    candidate_job = _job_block(document, "candidate")

    assert "pull_request_target" not in document
    assert '  push:\n    branches: ["main"]' in document
    assert '  pull_request:\n    branches: ["main"]' in document
    assert "\npermissions:\n  contents: read\n" in document
    assert "permissions:" not in candidate_job
    assert ": write" not in candidate_job
    assert "id-token:" not in candidate_job
    assert "secrets." not in candidate_job
    assert "github.token" not in candidate_job
    assert "GITHUB_TOKEN" not in candidate_job
    assert "runs-on: ubuntu-latest" in candidate_job
    assert "timeout-minutes: 20" in candidate_job


def test_candidate_job_uses_pinned_tools_and_one_focused_test() -> None:
    """Keep the daemon-capable job immutable, locked, and narrowly scoped."""
    document = WORKFLOW_PATH.read_text(encoding="utf-8")
    candidate_job = _job_block(document, "candidate")

    assert candidate_job.count("actions/checkout@") == 1
    assert f"uses: actions/checkout@{CHECKOUT_PIN} # v7.0.1" in candidate_job
    assert candidate_job.count("persist-credentials: false") == 1
    assert candidate_job.count("astral-sh/setup-uv@") == 1
    assert f"uses: astral-sh/setup-uv@{SETUP_UV_PIN} # v5" in candidate_job
    assert "run: uv python install 3.13" in candidate_job
    assert "run: uv sync --locked" in candidate_job
    assert candidate_job.count("uv run pytest") == 1
    assert "run: uv run pytest tests/test_candidate_runtime.py -q" in candidate_job
    assert "docker/login-action" not in candidate_job
    assert "push: true" not in candidate_job


def test_candidate_job_explicitly_enables_a_unique_project_prefix() -> None:
    """Turn on the real proof only in its isolated CI job."""
    candidate_job = _job_block(
        WORKFLOW_PATH.read_text(encoding="utf-8"),
        "candidate",
    )

    assert 'RUN_CANDIDATE_INTEGRATION: "1"' in candidate_job
    assert "adk-candidate-${{ github.run_id }}" in candidate_job
    assert "github.run_attempt" not in candidate_job
    assert "COMPOSE_PROJECT_NAME" not in candidate_job


def test_candidate_runtime_uses_the_exact_environment_allowlist() -> None:
    """Prevent unrelated host or production settings from entering the candidate."""
    source = RUNTIME_TEST_PATH.read_text(encoding="utf-8")
    assert (
        _assigned_literal(source, "CANDIDATE_ENV_ALLOWLIST") == EXPECTED_ENV_ALLOWLIST
    )

    serializer_source = _function_source(source, "_write_candidate_environment")
    assert '"agent.compose_env"' in serializer_source
    assert "*CANDIDATE_ENV_ALLOWLIST" in serializer_source
    assert "env_file.write" not in serializer_source
    assert "OPENROUTER_API_KEY" not in source
    assert "GOOGLE_API_KEY" not in source
    assert "DATABASE_URL" not in source


def test_candidate_runtime_cleanup_is_exact_and_non_destructive() -> None:
    """Forbid broad Compose, image, volume, and daemon cleanup."""
    source = RUNTIME_TEST_PATH.read_text(encoding="utf-8")
    down_source = _function_source(source, "_down_candidate")

    for forbidden in ("--volumes", "--rmi", "prune"):
        assert forbidden not in source
    assert '"down"' in down_source
    assert '"--remove-orphans"' in down_source
    assert '"--timeout"' in down_source
    assert '"30"' in down_source
    assert "check=False" in down_source


def test_candidate_runtime_declares_both_project_ownership_guards() -> None:
    """Statically require prefix and membership checks at Compose cleanup boundaries."""
    source = RUNTIME_TEST_PATH.read_text(encoding="utf-8")
    harness_source = _class_source(source, "CandidateHarness")
    compose_source = _function_source(source, "_compose")
    down_source = _function_source(source, "_down_candidate")

    assert "resource_prefix: str" in harness_source
    assert "owned_projects: frozenset[str]" in harness_source
    assert 'project.startswith(f"{self.resource_prefix}-")' in harness_source
    assert "project in self.owned_projects" in harness_source
    assert "harness.assert_owned_project(project)" in compose_source
    assert "harness.assert_owned_project(project)" in down_source


def test_candidate_unhealthy_proof_checks_the_exact_running_contract() -> None:
    """Statically keep the opt-in rejection path tied to the intended container."""
    source = RUNTIME_TEST_PATH.read_text(encoding="utf-8")
    runtime_source = _function_source(
        source,
        "test_candidate_runtime_is_isolated_and_fail_closed",
    )
    isolation_source = _function_source(source, "_assert_runtime_isolation")

    assert _assigned_literal(source, "EXPECTED_ENTRYPOINT") == ("/app/entrypoint.sh",)
    assert _assigned_literal(source, "UNHEALTHY_COMMAND") == (
        "python",
        "-c",
        "import time; time.sleep(300)",
    )
    assert "assert unhealthy_up.returncode != 0" in runtime_source
    assert "unhealthy_container_id = _container_id(" in runtime_source
    assert "expected_command=UNHEALTHY_COMMAND" in runtime_source
    assert 'expected_health_status="unhealthy"' in runtime_source
    assert 'build_arguments=(f"BASE_IMAGE={healthy_tag}",)' in runtime_source
    assert 'build_arguments=(f"BASE_IMAGE={healthy_image_id}",)' not in runtime_source
    for inspected_field in (
        "{{.Image}}",
        "{{.Config.User}}",
        "{{json .Config.Entrypoint}}",
        "{{json .Config.Cmd}}",
        "{{.HostConfig.NetworkMode}}",
        "{{json .HostConfig.PortBindings}}",
        "{{json .Mounts}}",
        "{{.HostConfig.RestartPolicy.Name}}",
        "{{json .HostConfig.CapDrop}}",
        "{{json .HostConfig.SecurityOpt}}",
        "{{.State.Status}}",
        "{{.State.Health.Status}}",
    ):
        assert inspected_field in isolation_source
    assert '== "running"' in isolation_source
    assert "== expected_health_status" in isolation_source


def test_candidate_volume_sentinel_uses_replacement_sensitive_identity() -> None:
    """Keep the sentinel comparison stronger than volume-name existence."""
    source = RUNTIME_TEST_PATH.read_text(encoding="utf-8")
    identity_source = _function_source(source, "_volume_identity")
    runtime_source = _function_source(
        source,
        "test_candidate_runtime_is_isolated_and_fail_closed",
    )

    for field in ('payload["Name"]', 'payload["Mountpoint"]', 'payload["CreatedAt"]'):
        assert field in identity_source
    assert "sentinel_volume_identity = _volume_identity(" in runtime_source
    assert "== sentinel_volume_identity" in runtime_source


def test_candidate_runtime_fails_closed_only_at_the_real_docker_boundary() -> None:
    """Keep helper contracts runnable while opt-in governs only daemon work."""
    source = RUNTIME_TEST_PATH.read_text(encoding="utf-8")
    module = ast.parse(source)
    runtime_test = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "test_candidate_runtime_is_isolated_and_fail_closed"
    )

    assert "pytestmark" not in source
    assert any(
        isinstance(decorator, ast.Call)
        and isinstance(decorator.func, ast.Attribute)
        and decorator.func.attr == "skipif"
        for decorator in runtime_test.decorator_list
    )
    assert "def test_require_docker_rejects_missing_cli" in source
    assert "def test_require_docker_rejects_missing_compose_or_daemon" in source
    assert "_require_docker(environment)" in _function_source(
        source,
        runtime_test.name,
    )
