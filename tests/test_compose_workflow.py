"""Compose smoke workflow safety and behavior contract tests."""

import re
import subprocess
from pathlib import Path
from typing import NamedTuple

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_PATH = REPOSITORY_ROOT / ".github" / "workflows" / "test-docker-compose.yml"
CHECKOUT_USES_PATTERN = re.compile(
    r"^[ \t]*uses:[ \t]+actions/checkout@"
    r"(?P<reference>[^ \t#\r\n]+)"
    r"(?P<annotation>[ \t]+#[ \t]*v\d+\.\d+\.\d+)?[ \t]*$",
    re.MULTILINE,
)
SMOKE_STEP_NAME = "Build and smoke-test Compose project"


class SmokeHarness(NamedTuple):
    """Synthetic process boundary for exercising the tracked Bash program."""

    environment: dict[str, str]
    docker_log: Path
    python_log: Path
    runner_temp: Path


def _workflow_step_block(document: str, step_name: str) -> str:
    """Extract one complete workflow step by its display name."""
    lines = document.splitlines()
    step_start = lines.index(f"      - name: {step_name}")
    step_lines = [lines[step_start]]

    for line in lines[step_start + 1 :]:
        if line.startswith("      - name: "):
            break
        if line and not line.startswith("        "):
            break
        step_lines.append(line)

    return "\n".join(step_lines)


def _workflow_step_script(document: str, step_name: str) -> str:
    """Extract the literal Bash program from a named workflow step."""
    lines = document.splitlines()
    step_start = lines.index(f"      - name: {step_name}")
    run_start = lines.index("        run: |", step_start) + 1
    script_lines: list[str] = []

    for line in lines[run_start:]:
        if line and not line.startswith("          "):
            break
        script_lines.append(line[10:] if line else "")

    return "\n".join(script_lines)


@pytest.fixture
def smoke_harness(tmp_path: Path) -> SmokeHarness:
    """Provide deterministic fake Docker and Python process boundaries."""
    bin_dir = tmp_path / "bin"
    runner_temp = tmp_path / "runner-temp"
    docker_log = tmp_path / "docker.log"
    python_log = tmp_path / "python.log"
    bin_dir.mkdir()
    runner_temp.mkdir()
    docker_log.touch()
    python_log.touch()

    docker_path = bin_dir / "docker"
    docker_path.write_text(
        """#!/bin/bash
set -eu

printf '%s\\n' "$*" >> "$FAKE_DOCKER_LOG"

case " $* " in
  *" config --images "*)
    printf '%s\\n' "$IMAGE"
    exit 0
    ;;
  *" config --quiet "*|*" build "*)
    exit 0
    ;;
  *" up "*)
    exit "${FAKE_DOCKER_UP_EXIT:-0}"
    ;;
  *" ps --all -q agent "*)
    printf '%s\\n' "synthetic-container-id"
    exit 0
    ;;
  *" ps --all "*|*" logs "*)
    exit 0
    ;;
  *" down "*)
    exit "${FAKE_DOCKER_DOWN_EXIT:-0}"
    ;;
  *)
    printf 'Unexpected fake Docker invocation: %s\\n' "$*" >&2
    exit 99
    ;;
esac
""",
        encoding="utf-8",
    )
    docker_path.chmod(0o755)

    python_path = bin_dir / "python3"
    python_path.write_text(
        """#!/bin/bash
set -eu

printf '%s\\n' "called" >> "$FAKE_PYTHON_LOG"
exit "${FAKE_PYTHON_EXIT:-0}"
""",
        encoding="utf-8",
    )
    python_path.chmod(0o755)

    environment = {
        "COMPOSE_DISABLE_ENV_FILE": "1",
        "FAKE_DOCKER_LOG": str(docker_log),
        "FAKE_PYTHON_LOG": str(python_log),
        "HOME": str(tmp_path),
        "IMAGE": "synthetic-compose-image",
        "LANG": "C",
        "PATH": f"{bin_dir}:/usr/bin:/bin",
        "RUNNER_TEMP": str(runner_temp),
        "SMOKE_PROJECT": "synthetic-compose-project",
    }
    return SmokeHarness(environment, docker_log, python_log, runner_temp)


def _run_smoke_script(
    script: str,
    harness: SmokeHarness,
    **environment_overrides: str,
) -> subprocess.CompletedProcess[str]:
    """Execute the workflow program against the synthetic boundaries."""
    environment = harness.environment | environment_overrides
    return subprocess.run(  # noqa: S603 - the tracked workflow is trusted input
        ["/bin/bash"],
        input=script,
        cwd=REPOSITORY_ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )


def _docker_calls(harness: SmokeHarness) -> list[str]:
    """Return the fake Docker invocations in execution order."""
    return harness.docker_log.read_text(encoding="utf-8").splitlines()


def _assert_runner_temp_is_empty(harness: SmokeHarness) -> None:
    """Require the workflow to remove every temporary smoke file."""
    assert list(harness.runner_temp.iterdir()) == []


def test_workflow_has_unprivileged_complete_ci_triggers() -> None:
    """Run on every main change without exposing privileged PR contexts."""
    document = WORKFLOW_PATH.read_text(encoding="utf-8")

    assert "pull_request_target" not in document
    assert '  push:\n    branches: ["main"]' in document
    assert '  pull_request:\n    branches: ["main"]' in document
    assert "\npermissions:\n  contents: read\n" in document
    assert document.count("permissions:") == 1
    assert ": write" not in document
    assert "id-token:" not in document
    assert "group: compose-smoke-${{ github.workflow }}-${{ github.ref }}" in document
    assert "cancel-in-progress: true" in document
    assert "runs-on: ubuntu-latest" in document
    assert "timeout-minutes: 20" in document


def test_workflow_checkout_is_immutable_and_drops_credentials() -> None:
    """Keep PR-controlled Docker builds away from persistent Git credentials."""
    document = WORKFLOW_PATH.read_text(encoding="utf-8")
    checkout_step = _workflow_step_block(document, "Checkout repository")
    checkout_uses = list(CHECKOUT_USES_PATTERN.finditer(document))

    assert len(checkout_uses) == 1
    checkout_use = checkout_uses[0]
    checkout_reference = checkout_use.group("reference")
    assert checkout_use.group(0).strip() in checkout_step
    assert re.fullmatch(r"[0-9a-f]{40}", checkout_reference)
    assert checkout_reference != "0" * 40
    assert checkout_use.group("annotation") is not None
    assert "persist-credentials: false" in checkout_step
    assert "secrets." not in document
    assert "packages: write" not in document
    assert "docker/login-action" not in document
    assert "setup-qemu" not in document
    assert "push: true" not in document


def test_checkout_matcher_ignores_commented_pins() -> None:
    """Do not let documentation hide an active mutable checkout reference."""
    document = """
        # uses: actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1 # v7.0.1
        uses: actions/checkout@v7
    """

    checkout_references = [
        match.group("reference") for match in CHECKOUT_USES_PATTERN.finditer(document)
    ]

    assert checkout_references == ["v7"]


def test_smoke_script_is_secret_free_bounded_and_self_cleaning() -> None:
    """Lock the startup, health assertion, diagnostics, and cleanup contract."""
    document = WORKFLOW_PATH.read_text(encoding="utf-8")
    script = _workflow_step_script(document, SMOKE_STEP_NAME)

    for forbidden in (
        "OPENROUTER_API_KEY",
        "GOOGLE_API_KEY",
        "DATABASE_URL",
    ):
        assert forbidden not in document

    required_fragments = (
        'COMPOSE_DISABLE_ENV_FILE: "1"',
        "SMOKE_PROJECT: adk-smoke-${{ github.run_id }}-${{ github.run_attempt }}",
        "set -Eeuo pipefail",
        "trap cleanup EXIT",
        "trap 'exit 130' INT",
        "trap 'exit 143' TERM",
        "umask 077",
        "IMAGE: ghcr.io/example/agent:compose-${{ github.sha }}",
        'mktemp "${RUNNER_TEMP}/compose-smoke-env.XXXXXX"',
        'mktemp "${RUNNER_TEMP}/compose-smoke-override.XXXXXX.yaml"',
        "export ENV_FILE",
        "env_file: !override",
        "ports: !override",
        '"127.0.0.1:8080:8080"',
        'restart: "no"',
        'OTEL_SDK_DISABLED: "true"',
        '"${compose[@]}" config --images',
        'if [ "$resolved_images" != "$IMAGE" ]; then',
        "Unexpected Compose image:",
        '"${compose[@]}" config --quiet',
        '"${compose[@]}" build',
        "--detach --no-build --wait --wait-timeout 180",
        "python3 -c",
        'if payload != {"status": "ok"}:',
        'raise SystemExit(f"Unexpected health payload: {payload!r}")',
        '"${compose[@]}" ps --all || true',
        "--no-color --timestamps --tail=200 agent || true",
        "--volumes --remove-orphans --timeout 30",
        "|| down_exit_code=$?",
        'rm -f "$ENV_FILE"',
        'rm -f "$OVERRIDE_FILE"',
        'exit "$exit_code"',
    )
    for fragment in required_fragments:
        assert fragment in document or fragment in script

    assert "volumes: !reset" not in document
    assert "assert payload" not in script

    ordered_fragments = (
        "trap cleanup EXIT",
        "umask 077",
        '"${compose[@]}" config --images',
        '"${compose[@]}" config --quiet',
        '"${compose[@]}" build',
        '"${compose[@]}" up',
        "python3 -c",
    )
    positions = [script.index(fragment) for fragment in ordered_fragments]
    assert positions == sorted(positions)


def test_smoke_script_has_valid_bash_syntax() -> None:
    """Reject workflow edits that break the cleanup or smoke shell program."""
    document = WORKFLOW_PATH.read_text(encoding="utf-8")
    script = _workflow_step_script(document, SMOKE_STEP_NAME)

    result = subprocess.run(  # noqa: S603 - the tracked workflow is trusted input
        ["/bin/bash", "-n"],
        input=script,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_smoke_script_success_probes_and_cleans_up(
    smoke_harness: SmokeHarness,
) -> None:
    """Exercise the success path through health probing and teardown."""
    document = WORKFLOW_PATH.read_text(encoding="utf-8")
    script = _workflow_step_script(document, SMOKE_STEP_NAME)

    result = _run_smoke_script(script, smoke_harness)

    assert result.returncode == 0, result.stderr
    assert "Compose startup and health checks passed." in result.stdout
    calls = _docker_calls(smoke_harness)
    ordered_fragments = (
        " config --images",
        " config --quiet",
        " build",
        " up --detach",
        " ps --all -q agent",
        " down --volumes",
    )
    positions = [
        next(index for index, call in enumerate(calls) if fragment in call)
        for fragment in ordered_fragments
    ]
    assert positions == sorted(positions)
    assert not any(call.endswith(" ps --all") for call in calls)
    assert not any(" logs " in call for call in calls)
    assert smoke_harness.python_log.read_text(encoding="utf-8") == "called\n"
    _assert_runner_temp_is_empty(smoke_harness)


def test_smoke_script_preserves_startup_failure_and_diagnostics(
    smoke_harness: SmokeHarness,
) -> None:
    """Keep the original startup status while logging and tearing down."""
    document = WORKFLOW_PATH.read_text(encoding="utf-8")
    script = _workflow_step_script(document, SMOKE_STEP_NAME)

    result = _run_smoke_script(
        script,
        smoke_harness,
        FAKE_DOCKER_UP_EXIT="17",
    )

    assert result.returncode == 17
    calls = _docker_calls(smoke_harness)
    ordered_fragments = (
        " up --detach",
        " ps --all",
        " logs ",
        " down --volumes",
    )
    positions = [
        next(index for index, call in enumerate(calls) if fragment in call)
        for fragment in ordered_fragments
    ]
    assert positions == sorted(positions)
    assert smoke_harness.python_log.read_text(encoding="utf-8") == ""
    _assert_runner_temp_is_empty(smoke_harness)


def test_smoke_script_propagates_health_probe_failure(
    smoke_harness: SmokeHarness,
) -> None:
    """Fail, diagnose, and clean up when the exact health probe fails."""
    document = WORKFLOW_PATH.read_text(encoding="utf-8")
    script = _workflow_step_script(document, SMOKE_STEP_NAME)

    result = _run_smoke_script(
        script,
        smoke_harness,
        FAKE_PYTHON_EXIT="19",
    )

    assert result.returncode == 19
    calls = _docker_calls(smoke_harness)
    assert any(call.endswith(" ps --all") for call in calls)
    assert any(" logs " in call for call in calls)
    assert any(" down --volumes" in call for call in calls)
    assert smoke_harness.python_log.read_text(encoding="utf-8") == "called\n"
    _assert_runner_temp_is_empty(smoke_harness)


def test_smoke_script_surfaces_teardown_failure(
    smoke_harness: SmokeHarness,
) -> None:
    """Do not report success when Compose cannot remove its resources."""
    document = WORKFLOW_PATH.read_text(encoding="utf-8")
    script = _workflow_step_script(document, SMOKE_STEP_NAME)

    result = _run_smoke_script(
        script,
        smoke_harness,
        FAKE_DOCKER_DOWN_EXIT="23",
    )

    assert result.returncode == 23
    calls = _docker_calls(smoke_harness)
    assert any(" down --volumes" in call for call in calls)
    assert not any(call.endswith(" ps --all") for call in calls)
    assert not any(" logs " in call for call in calls)
    _assert_runner_temp_is_empty(smoke_harness)
