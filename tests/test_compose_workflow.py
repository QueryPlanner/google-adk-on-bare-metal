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
    event_log: Path
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
    docker_ps_counter = tmp_path / "docker-ps-counter"
    event_log = tmp_path / "events.log"
    python_log = tmp_path / "python.log"
    bin_dir.mkdir()
    runner_temp.mkdir()
    docker_log.touch()
    docker_ps_counter.write_text("0\n", encoding="utf-8")
    event_log.touch()
    python_log.touch()

    docker_path = bin_dir / "docker"
    docker_path.write_text(
        """#!/bin/bash
set -eu

printf '%s\\n' "$*" >> "$FAKE_DOCKER_LOG"
printf 'docker:%s\\n' "$*" >> "$FAKE_EVENT_LOG"

case " $* " in
  *" config --images "*)
    printf '%s\\n' "$IMAGE"
    exit 0
    ;;
  *" config --quiet "*|*" build "*)
    exit 0
    ;;
  *"compose-smoke-read-only"*" up "*)
    exit "${FAKE_READ_ONLY_UP_EXIT:-17}"
    ;;
  *" up "*)
    exit "${FAKE_DOCKER_UP_EXIT:-0}"
    ;;
  *" ps --all -q agent "*)
    ps_count="$(cat "$FAKE_DOCKER_PS_COUNTER")"
    ps_count="$((ps_count + 1))"
    printf '%s\\n' "$ps_count" > "$FAKE_DOCKER_PS_COUNTER"
    if [ "${FAKE_REUSE_CONTAINER_ID:-0}" = "1" ]; then
      printf '%s\\n' "synthetic-container-id"
    else
      printf 'synthetic-container-id-%s\\n' "$ps_count"
    fi
    exit 0
    ;;
  *" port agent 8080 "*)
    printf '%s\\n' "${FAKE_PUBLISHED_ADDRESS:-127.0.0.1:8080}"
    exit 0
    ;;
  *" inspect "*"State.Health.Status"*)
    printf '%s\\n' "${FAKE_READ_ONLY_HEALTH:-none}"
    exit 0
    ;;
  *" inspect "*"State.ExitCode"*)
    printf '%s\\n' "${FAKE_READ_ONLY_EXIT_CODE:-1}"
    exit 0
    ;;
  *" inspect "*"State.Status"*)
    printf '%s\\n' "${FAKE_READ_ONLY_STATUS:-exited}"
    exit 0
    ;;
  *"compose-smoke-read-only"*" logs "*)
    default_readonly_log='[ERROR] agent.server: Artifact storage is unavailable.'
    printf '%s\\n' \
      "${FAKE_READ_ONLY_LOG_MESSAGE-$default_readonly_log}"
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

phase="${SMOKE_PHASE:-unknown}"
printf '%s\\n' "$phase" >> "$FAKE_PYTHON_LOG"
printf 'python:%s\\n' "$phase" >> "$FAKE_EVENT_LOG"

if [ "${FAKE_PYTHON_FAIL_PHASE:-}" = "$phase" ]; then
  exit "${FAKE_PYTHON_EXIT:-19}"
fi
exit 0
""",
        encoding="utf-8",
    )
    python_path.chmod(0o755)

    environment = {
        "COMPOSE_DISABLE_ENV_FILE": "1",
        "FAKE_DOCKER_LOG": str(docker_log),
        "FAKE_DOCKER_PS_COUNTER": str(docker_ps_counter),
        "FAKE_EVENT_LOG": str(event_log),
        "FAKE_PYTHON_LOG": str(python_log),
        "IMAGE": "synthetic-compose-image",
        "LANG": "C",
        "PATH": f"{bin_dir}:/usr/bin:/bin",
        "RUNNER_TEMP": str(runner_temp),
        "SMOKE_PROJECT": "synthetic-compose-project",
    }
    return SmokeHarness(environment, docker_log, event_log, python_log, runner_temp)


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


def _events(harness: SmokeHarness) -> list[str]:
    """Return all fake boundary events in execution order."""
    return harness.event_log.read_text(encoding="utf-8").splitlines()


def _assert_ordered_fragments(events: list[str], fragments: tuple[str, ...]) -> None:
    """Require each fragment to occur after the preceding boundary event."""
    position = -1
    for fragment in fragments:
        position = next(
            index
            for index in range(position + 1, len(events))
            if fragment in events[index]
        )


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
    """Lock persistence, read-only failure, diagnostics, and cleanup contracts."""
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
        "SUPPRESS_CLEANUP_AGENT_LOGS=0",
        "SUPPRESS_CLEANUP_AGENT_LOGS=1",
        "Agent logs withheld after storage failure.",
        "IMAGE: ghcr.io/example/agent:compose-${{ github.sha }}",
        'mktemp "${RUNNER_TEMP}/compose-smoke-env.XXXXXX"',
        'mktemp "${RUNNER_TEMP}/compose-smoke-override.XXXXXX.yaml"',
        'mktemp "${RUNNER_TEMP}/compose-smoke-read-only.XXXXXX.yaml"',
        'mktemp -d "${RUNNER_TEMP}/compose-smoke-artifacts.XXXXXX"',
        "export ENV_FILE READ_ONLY_ARTIFACT_PARENT",
        "env_file: !override",
        'restart: "no"',
        'OTEL_SDK_DISABLED: "true"',
        "volumes: !override",
        "source: ${READ_ONLY_ARTIFACT_PARENT:?Set READ_ONLY_ARTIFACT_PARENT}",
        "target: /app/src/.adk",
        "read_only: true",
        "create_host_path: false",
        '"${compose[@]}" config --images',
        'if [ "$resolved_images" != "$IMAGE" ]; then',
        "Unexpected Compose image:",
        '"${compose[@]}" config --quiet',
        '"${readonly_compose[@]}" config --quiet',
        '"${compose[@]}" build',
        "--detach --no-build --wait --wait-timeout 180",
        '"${compose[@]}" port agent 8080',
        'if [ "$published_address" != "127.0.0.1:8080" ]; then',
        "Unexpected published address:",
        "SMOKE_PHASE=save python3 - <<'PY'",
        '"filename": "compose-smoke.txt"',
        '"artifact": {"text": artifact_text}',
        'if saved.get("version") != 0:',
        "--force-recreate --no-deps",
        "--wait --wait-timeout 180 agent",
        'recreated_container_id="$("${compose[@]}" ps --all -q agent)"',
        'if [ "$recreated_container_id" = "$container_id" ]; then',
        "Compose reused the original agent container.",
        "SMOKE_PHASE=load python3 - <<'PY'",
        "compose-smoke.txt/versions/0",
        'if loaded != {"text": artifact_text}:',
        "--wait --wait-timeout 30 agent",
        'if [ "$readonly_up_exit" -eq 0 ]; then',
        "Read-only artifact storage unexpectedly became healthy.",
        "docker inspect --format '{{.State.Status}}'",
        "docker inspect --format '{{.State.ExitCode}}'",
        "State.Health.Status",
        "Read-only agent did not exit:",
        "Artifact storage is unavailable.",
        "Traceback (most recent call last)",
        "PermissionError",
        "FileNotFoundError",
        "NotADirectoryError",
        "OSError",
        "ValueError",
        "[Errno",
        "/app/src",
        ".artifact-storage-probe-",
        ".probe-",
        "AGENT_DIR",
        'storage_message_count="$(',
        'storage_message_line="$(',
        "Read-only public storage line was not exact.",
        '"${compose[@]}" ps --all || true',
        "--no-color --timestamps --tail=200 agent || true",
        "--volumes --remove-orphans --timeout 30",
        "|| down_exit_code=$?",
        'rm -f "$ENV_FILE"',
        'rm -f "$OVERRIDE_FILE"',
        'rm -f "$READ_ONLY_OVERRIDE_FILE"',
        'rmdir "$READ_ONLY_ARTIFACT_PARENT"',
        'exit "$exit_code"',
    )
    for fragment in required_fragments:
        assert fragment in document or fragment in script

    assert "ports: !override" not in document
    assert "volumes: !reset" not in document
    assert "assert " not in script
    assert script.count("--volumes") == 1
    assert script.count("logs \\") == 2
    assert script.count("--tail=200 agent") == 2
    assert '--project-name "$SMOKE_PROJECT"' in script

    trap_position = script.index("trap cleanup EXIT")
    umask_position = script.index("umask 077")
    image_position = script.index('"${compose[@]}" config --images')
    config_position = script.index('"${compose[@]}" config --quiet')
    readonly_config_position = script.index('"${readonly_compose[@]}" config --quiet')
    build_position = script.index('"${compose[@]}" build')
    initial_up_position = script.index('"${compose[@]}" up', build_position)
    port_position = script.index('"${compose[@]}" port agent 8080')
    save_position = script.index("SMOKE_PHASE=save")
    recreate_position = script.index('"${compose[@]}" up', initial_up_position + 1)
    recreated_id_position = script.index(
        'recreated_container_id="$("${compose[@]}" ps --all -q agent)"'
    )
    load_position = script.index("SMOKE_PHASE=load")
    readonly_up_position = script.index('"${readonly_compose[@]}" up')
    readonly_logs_position = script.index(
        '"${readonly_compose[@]}" logs',
        readonly_up_position,
    )

    assert (
        trap_position
        < umask_position
        < image_position
        < config_position
        < readonly_config_position
        < build_position
        < initial_up_position
        < port_position
        < save_position
        < recreate_position
        < recreated_id_position
        < load_position
        < readonly_up_position
        < readonly_logs_position
    )


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
    """Exercise persistence, fail-closed storage, and scoped teardown."""
    document = WORKFLOW_PATH.read_text(encoding="utf-8")
    script = _workflow_step_script(document, SMOKE_STEP_NAME)

    result = _run_smoke_script(script, smoke_harness)

    assert result.returncode == 0, result.stderr
    assert "Artifact storage is unavailable." in result.stdout
    assert "Compose artifact persistence and failure checks passed." in result.stdout
    _assert_ordered_fragments(
        _events(smoke_harness),
        (
            " config --images",
            " config --quiet",
            "compose-smoke-read-only",
            " build",
            " up --detach --no-build --wait --wait-timeout 180",
            " ps --all -q agent",
            " port agent 8080",
            "python:save",
            (
                " up --detach --no-build --force-recreate --no-deps "
                "--wait --wait-timeout 180 agent"
            ),
            " ps --all -q agent",
            "python:load",
            "--wait --wait-timeout 30 agent",
            " ps --all -q agent",
            "docker:inspect --format {{.State.Status}}",
            "docker:inspect --format {{.State.ExitCode}}",
            "docker:inspect --format {{if .State.Health}}",
            " logs --no-color --timestamps --tail=200 agent",
            " down --volumes",
        ),
    )
    calls = _docker_calls(smoke_harness)
    assert not any(call.endswith(" ps --all") for call in calls)
    log_calls = [call for call in calls if " logs " in call]
    assert len(log_calls) == 1
    assert all("--tail=200 agent" in call for call in log_calls)
    down_calls = [call for call in calls if " down " in call]
    assert len(down_calls) == 1
    assert "--project-name synthetic-compose-project" in down_calls[0]
    assert " down --volumes --remove-orphans --timeout 30" in down_calls[0]
    assert smoke_harness.python_log.read_text(encoding="utf-8") == "save\nload\n"
    _assert_runner_temp_is_empty(smoke_harness)


def test_smoke_script_rejects_non_loopback_publication(
    smoke_harness: SmokeHarness,
) -> None:
    """Fail before probing health when Compose publishes on every interface."""
    document = WORKFLOW_PATH.read_text(encoding="utf-8")
    script = _workflow_step_script(document, SMOKE_STEP_NAME)

    result = _run_smoke_script(
        script,
        smoke_harness,
        FAKE_PUBLISHED_ADDRESS="0.0.0.0:8080",
    )

    assert result.returncode == 1
    assert "Unexpected published address: 0.0.0.0:8080" in result.stderr
    calls = _docker_calls(smoke_harness)
    assert any(" port agent 8080" in call for call in calls)
    assert any(call.endswith(" ps --all") for call in calls)
    assert any(
        " logs --no-color --timestamps --tail=200 agent" in call for call in calls
    )
    assert any(" down --volumes" in call for call in calls)
    assert smoke_harness.python_log.read_text(encoding="utf-8") == ""
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
    _assert_ordered_fragments(
        _events(smoke_harness),
        (
            " up --detach --no-build --wait --wait-timeout 180",
            " ps --all",
            " logs --no-color --timestamps --tail=200 agent",
            " down --volumes",
        ),
    )
    assert all("--tail=200 agent" in call for call in calls if " logs " in call)
    assert smoke_harness.python_log.read_text(encoding="utf-8") == ""
    _assert_runner_temp_is_empty(smoke_harness)


def test_smoke_script_proves_agent_container_was_recreated(
    smoke_harness: SmokeHarness,
) -> None:
    """Reject persistence evidence when Compose reuses the original container."""
    document = WORKFLOW_PATH.read_text(encoding="utf-8")
    script = _workflow_step_script(document, SMOKE_STEP_NAME)

    result = _run_smoke_script(
        script,
        smoke_harness,
        FAKE_REUSE_CONTAINER_ID="1",
    )

    assert result.returncode == 1
    assert "Compose reused the original agent container." in result.stderr
    calls = _docker_calls(smoke_harness)
    assert sum(" ps --all -q agent" in call for call in calls) == 2
    assert any(" --force-recreate --no-deps " in f" {call} " for call in calls)
    assert any(" down --volumes" in call for call in calls)
    assert smoke_harness.python_log.read_text(encoding="utf-8") == "save\n"
    _assert_runner_temp_is_empty(smoke_harness)


@pytest.mark.parametrize(
    ("failure_phase", "expected_python_log", "recreate_started"),
    (
        ("save", "save\n", False),
        ("load", "save\nload\n", True),
    ),
)
def test_smoke_script_propagates_artifact_probe_failure(
    smoke_harness: SmokeHarness,
    failure_phase: str,
    expected_python_log: str,
    recreate_started: bool,
) -> None:
    """Fail, diagnose, and clean up when either real HTTP probe fails."""
    document = WORKFLOW_PATH.read_text(encoding="utf-8")
    script = _workflow_step_script(document, SMOKE_STEP_NAME)

    result = _run_smoke_script(
        script,
        smoke_harness,
        FAKE_PYTHON_FAIL_PHASE=failure_phase,
        FAKE_PYTHON_EXIT="19",
    )

    assert result.returncode == 19
    calls = _docker_calls(smoke_harness)
    assert any(call.endswith(" ps --all") for call in calls)
    assert any(
        " logs --no-color --timestamps --tail=200 agent" in call for call in calls
    )
    assert any(" down --volumes" in call for call in calls)
    recreate_calls = [
        call
        for call in calls
        if " --force-recreate --no-deps " in f" {call} "
        and "compose-smoke-read-only" not in call
    ]
    assert bool(recreate_calls) is recreate_started
    assert smoke_harness.python_log.read_text(encoding="utf-8") == expected_python_log
    _assert_runner_temp_is_empty(smoke_harness)


def test_smoke_script_rejects_healthy_read_only_case(
    smoke_harness: SmokeHarness,
) -> None:
    """Never accept an in-memory fallback when the bind mount is read-only."""
    document = WORKFLOW_PATH.read_text(encoding="utf-8")
    script = _workflow_step_script(document, SMOKE_STEP_NAME)

    result = _run_smoke_script(
        script,
        smoke_harness,
        FAKE_READ_ONLY_UP_EXIT="0",
    )

    assert result.returncode == 1
    assert "Read-only artifact storage unexpectedly became healthy." in result.stderr
    calls = _docker_calls(smoke_harness)
    assert any(
        "compose-smoke-read-only" in call and "--wait-timeout 30 agent" in call
        for call in calls
    )
    assert any(call.endswith(" ps --all") for call in calls)
    assert any(" down --volumes" in call for call in calls)
    assert smoke_harness.python_log.read_text(encoding="utf-8") == "save\nload\n"
    _assert_runner_temp_is_empty(smoke_harness)


@pytest.mark.parametrize(
    ("status", "exit_code", "health", "expected_error"),
    (
        ("running", "0", "unhealthy", "Read-only agent did not exit: running."),
        ("created", "0", "none", "Read-only agent did not exit: created."),
        ("dead", "1", "none", "Read-only agent did not exit: dead."),
        ("exited", "0", "none", "Read-only agent exited successfully."),
        ("exited", "1", "healthy", "Read-only agent reported healthy."),
    ),
)
def test_smoke_script_requires_fail_closed_read_only_state(
    smoke_harness: SmokeHarness,
    status: str,
    exit_code: str,
    health: str,
    expected_error: str,
) -> None:
    """Accept only an exited, nonzero, never-healthy read-only container."""
    document = WORKFLOW_PATH.read_text(encoding="utf-8")
    script = _workflow_step_script(document, SMOKE_STEP_NAME)

    result = _run_smoke_script(
        script,
        smoke_harness,
        FAKE_READ_ONLY_STATUS=status,
        FAKE_READ_ONLY_EXIT_CODE=exit_code,
        FAKE_READ_ONLY_HEALTH=health,
    )

    assert result.returncode == 1
    assert expected_error in result.stderr
    calls = _docker_calls(smoke_harness)
    assert any(call.startswith("inspect ") for call in calls)
    assert any(" down --volumes" in call for call in calls)
    _assert_runner_temp_is_empty(smoke_harness)


@pytest.mark.parametrize(
    ("log_message", "expected_error"),
    (
        ("", "Read-only failure omitted the public storage message."),
        (
            "[ERROR] agent.server: Artifact storage is unavailable. PermissionError",
            "Read-only diagnostics failed sanitization.",
        ),
        (
            (
                "[ERROR] agent.server: /private/secret-path "
                "Artifact storage is unavailable."
            ),
            "Read-only public storage line was not exact.",
        ),
        (
            (
                "[ERROR] agent.server: Artifact storage is unavailable. "
                "Artifact storage is unavailable."
            ),
            "Read-only public storage line was not exact.",
        ),
    ),
)
def test_smoke_script_requires_sanitized_read_only_diagnostics(
    smoke_harness: SmokeHarness,
    log_message: str,
    expected_error: str,
) -> None:
    """Require the stable public error without exposing filesystem causes."""
    document = WORKFLOW_PATH.read_text(encoding="utf-8")
    script = _workflow_step_script(document, SMOKE_STEP_NAME)

    result = _run_smoke_script(
        script,
        smoke_harness,
        FAKE_READ_ONLY_LOG_MESSAGE=log_message,
    )

    assert result.returncode == 1
    assert expected_error in result.stderr
    assert "/private/secret-path" not in result.stdout + result.stderr
    calls = _docker_calls(smoke_harness)
    assert all("--tail=200 agent" in call for call in calls if " logs " in call)
    assert any(" down --volumes" in call for call in calls)
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
    log_calls = [call for call in calls if " logs " in call]
    assert len(log_calls) == 1
    assert "compose-smoke-read-only" in log_calls[0]
    assert "--tail=200 agent" in log_calls[0]
    _assert_runner_temp_is_empty(smoke_harness)
