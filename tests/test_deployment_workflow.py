"""Deployment workflow safety contract tests."""

import subprocess
from pathlib import Path
from typing import NamedTuple

import pytest

WORKFLOW_PATH = (
    Path(__file__).resolve().parents[1] / ".github" / "workflows" / "docker-publish.yml"
)
EXPECTED_DEPLOY_GUARD = (
    "github.event_name == 'workflow_dispatch' && "
    "github.ref == 'refs/heads/main' && "
    "inputs.deploy"
)
EXPECTED_DEPLOY_CLAUSES = {
    "github.event_name == 'workflow_dispatch'",
    "github.ref == 'refs/heads/main'",
    "inputs.deploy",
}
SECRET_CANARIES = {
    "DATABASE_URL": "postgresql://user:p$UNSET@database/agent",
    "OPENROUTER_API_KEY": "$(printf command-substitution)",
    "GOOGLE_API_KEY": "`printf backtick-substitution`",
    "ROOT_AGENT_MODEL": "openrouter/provider/model",
    "LANGFUSE_PUBLIC_KEY": "public-key",
    "LANGFUSE_SECRET_KEY": "secret-key",
    "LANGFUSE_HOST": "https://observability.example",
}


class DeployHarness(NamedTuple):
    """Synthetic remote host boundaries for the tracked deployment script."""

    environment: dict[str, str]
    docker_log: Path
    git_log: Path
    home: Path


def _indented_block(document: str, heading: str, indentation: int) -> str:
    """Return the lines nested beneath an exact YAML heading."""
    lines = document.splitlines()
    start = lines.index(f"{' ' * indentation}{heading}") + 1
    block: list[str] = []

    for line in lines[start:]:
        if line and not line.startswith(" " * (indentation + 1)):
            break
        block.append(line)

    return "\n".join(block)


def _deploy_guard(document: str) -> str:
    """Extract and normalize the deploy job condition."""
    deploy_block = _indented_block(document, "deploy:", 2)
    lines = deploy_block.splitlines()
    condition_start = lines.index("    if: >-") + 1
    condition_lines: list[str] = []

    for line in lines[condition_start:]:
        if not line.startswith("      "):
            break
        condition_lines.append(line.strip())

    return " ".join(condition_lines)


def _deploy_script(document: str) -> str:
    """Extract the remote shell program from the deployment action."""
    deploy_block = _indented_block(document, "deploy:", 2)
    lines = deploy_block.splitlines()
    script_start = lines.index("          script: |") + 1
    script_lines: list[str] = []

    for line in lines[script_start:]:
        if line and not line.startswith("            "):
            break
        script_lines.append(line[12:] if line else "")

    return "\n".join(script_lines)


def _materialize_deploy_script(script: str) -> str:
    """Replace GitHub expressions with deterministic non-secret values."""
    replacements = {
        "${{ github.event.repository.name }}": "Mixed-Repository",
        "${{ github.repository }}": "MixedOwner/Mixed-Repository",
    }
    replacements.update(
        {
            "${{ secrets." + secret_name + " }}": secret_value
            for secret_name, secret_value in SECRET_CANARIES.items()
        }
    )

    for expression, value in replacements.items():
        script = script.replace(expression, value)

    assert "${{" not in script
    return script


@pytest.fixture
def deploy_harness(tmp_path: Path) -> DeployHarness:
    """Provide fake Git and Docker executables on an isolated remote home."""
    bin_directory = tmp_path / "bin"
    home = tmp_path / "home"
    docker_log = tmp_path / "docker.log"
    git_log = tmp_path / "git.log"
    bin_directory.mkdir()
    home.mkdir()
    docker_log.touch()
    git_log.touch()

    git_path = bin_directory / "git"
    git_path.write_text(
        """#!/bin/sh
set -eu

printf '%s\\n' "$*" >> "$FAKE_GIT_LOG"

case "$1" in
  clone)
    mkdir -p "$3"
    ;;
  pull)
    ;;
  *)
    exit 99
    ;;
esac
""",
        encoding="utf-8",
    )
    git_path.chmod(0o755)

    docker_path = bin_directory / "docker"
    docker_path.write_text(
        """#!/bin/sh
set -eu

printf 'IMAGE=%s|%s\\n' "${IMAGE:-}" "$*" >> "$FAKE_DOCKER_LOG"

if [ "$1" = "compose" ] && [ "$2" = "pull" ]; then
  exit "${FAKE_DOCKER_PULL_EXIT:-0}"
fi
""",
        encoding="utf-8",
    )
    docker_path.chmod(0o755)

    environment = {
        "FAKE_DOCKER_LOG": str(docker_log),
        "FAKE_DOCKER_PULL_EXIT": "0",
        "FAKE_GIT_LOG": str(git_log),
        "HOME": str(home),
        "LANG": "C",
        "PATH": f"{bin_directory}:/usr/bin:/bin",
    }
    return DeployHarness(environment, docker_log, git_log, home)


def _run_deploy_script(
    script: str,
    harness: DeployHarness,
    **environment_overrides: str,
) -> subprocess.CompletedProcess[str]:
    """Execute the materialized remote program against synthetic boundaries."""
    environment = harness.environment | environment_overrides
    return subprocess.run(  # noqa: S603 - execute the tracked trusted script
        ["/bin/sh"],
        input=_materialize_deploy_script(script),
        cwd=WORKFLOW_PATH.parents[2],
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )


def _evaluate_guard(
    expression: str,
    *,
    event_name: str,
    git_ref: str,
    deploy: bool | None,
) -> bool:
    """Evaluate the supported clauses extracted from the workflow guard."""
    clauses = tuple(clause.strip() for clause in expression.split("&&"))
    outcomes = {
        "github.event_name == 'workflow_dispatch'": (event_name == "workflow_dispatch"),
        "github.ref == 'refs/heads/main'": git_ref == "refs/heads/main",
        "inputs.deploy": deploy is True,
    }

    assert len(clauses) == len(EXPECTED_DEPLOY_CLAUSES)
    assert set(clauses) == EXPECTED_DEPLOY_CLAUSES
    return all(outcomes[clause] for clause in clauses)


def test_workflow_exposes_explicit_deploy_confirmation() -> None:
    document = WORKFLOW_PATH.read_text(encoding="utf-8")
    trigger_block = _indented_block(document, "on:", 0)
    dispatch_block = _indented_block(trigger_block, "workflow_dispatch:", 2)
    deploy_block = _indented_block(document, "deploy:", 2)

    assert "  push:" in trigger_block
    assert "  workflow_dispatch:" in trigger_block
    assert "    inputs:" in dispatch_block
    assert "      deploy:" in dispatch_block
    assert "        required: true" in dispatch_block
    assert "        type: boolean" in dispatch_block
    assert "        default: false" in dispatch_block
    assert "    needs: build" in deploy_block


def test_remote_deploy_uses_lowercase_literal_secret_contract(
    deploy_harness: DeployHarness,
) -> None:
    """Clone under HOME and deploy one lowercase image without expanding secrets."""
    document = WORKFLOW_PATH.read_text(encoding="utf-8")
    script = _deploy_script(document)
    image_assignment = next(
        line for line in script.splitlines() if line.startswith("IMAGE_NAME=")
    )
    expected_assignment = (
        "IMAGE_NAME=\"$(printf '%s' "
        '"ghcr.io/${{ github.repository }}:main" '
        "| tr '[:upper:]' '[:lower:]')\""
    )

    assert image_assignment == expected_assignment
    assert script.splitlines()[0] == "set -eu"
    assert script.index('export IMAGE="$IMAGE_NAME"') < script.index(
        "docker compose pull"
    )
    assert script.index("docker compose pull") < script.index(
        "docker compose up --no-build --wait --wait-timeout 180"
    )
    assert 'PROJECT_DIR="$HOME/$PROJECT_NAME"' in script
    assert "cat <<'ENV_FILE' > .env" in script

    result = _run_deploy_script(script, deploy_harness)

    assert result.returncode == 0, result.stderr
    project_directory = deploy_harness.home / "Mixed-Repository"
    assert project_directory.is_dir()
    assert deploy_harness.git_log.read_text(encoding="utf-8").splitlines() == [
        (f"clone https://github.com/MixedOwner/Mixed-Repository {project_directory}"),
        "pull",
    ]
    assert deploy_harness.docker_log.read_text(encoding="utf-8").splitlines() == [
        ("IMAGE=ghcr.io/mixedowner/mixed-repository:main|compose pull"),
        (
            "IMAGE=ghcr.io/mixedowner/mixed-repository:main|"
            "compose up --no-build --wait --wait-timeout 180"
        ),
        ("IMAGE=ghcr.io/mixedowner/mixed-repository:main|image prune -f"),
    ]
    environment_document = (project_directory / ".env").read_text(encoding="utf-8")
    for secret_value in SECRET_CANARIES.values():
        assert secret_value in environment_document


def test_remote_deploy_stops_after_failed_pull(
    deploy_harness: DeployHarness,
) -> None:
    """Propagate a pull failure without starting a stale image or pruning."""
    document = WORKFLOW_PATH.read_text(encoding="utf-8")
    script = _deploy_script(document)

    result = _run_deploy_script(
        script,
        deploy_harness,
        FAKE_DOCKER_PULL_EXIT="23",
    )

    assert result.returncode == 23
    assert deploy_harness.docker_log.read_text(encoding="utf-8").splitlines() == [
        ("IMAGE=ghcr.io/mixedowner/mixed-repository:main|compose pull")
    ]


@pytest.mark.parametrize(
    ("event_name", "git_ref", "deploy", "expected"),
    [
        ("push", "refs/heads/main", True, False),
        ("push", "refs/tags/v1.0.0", True, False),
        ("workflow_dispatch", "refs/heads/feature", True, False),
        ("workflow_dispatch", "refs/tags/main", True, False),
        ("workflow_dispatch", "refs/heads/main", False, False),
        ("workflow_dispatch", "refs/heads/main", None, False),
        ("workflow_dispatch", "refs/heads/main", True, True),
        ("schedule", "refs/heads/main", True, False),
    ],
)
def test_deploy_guard_requires_manual_main_confirmation(
    event_name: str,
    git_ref: str,
    deploy: bool | None,
    expected: bool,
) -> None:
    document = WORKFLOW_PATH.read_text(encoding="utf-8")
    guard = _deploy_guard(document)

    assert guard == EXPECTED_DEPLOY_GUARD
    assert (
        _evaluate_guard(
            guard,
            event_name=event_name,
            git_ref=git_ref,
            deploy=deploy,
        )
        is expected
    )
