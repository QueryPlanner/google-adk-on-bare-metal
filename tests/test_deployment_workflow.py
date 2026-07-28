"""Deployment workflow safety contract tests."""

import os
import shlex
import shutil
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
VALID_SHA = "a" * 40
VALID_DIGEST = f"sha256:{'b' * 64}"
EXPECTED_IMAGE = f"ghcr.io/mixedowner/mixed-repository@{VALID_DIGEST}"
EXPECTED_ORIGIN = "https://github.com/MixedOwner/Mixed-Repository"
EXPECTED_PROJECT = "adk-mixed-repository"
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
    event_log: Path
    git_log: Path
    git_state: Path
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
    """Replace non-provenance GitHub expressions with deterministic values."""
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


def _write_executable(path: Path, content: str) -> None:
    """Write one executable fake boundary."""
    path.write_text(content, encoding="utf-8")
    path.chmod(0o755)


@pytest.fixture
def deploy_harness(tmp_path: Path) -> DeployHarness:
    """Provide strict fake Git and Docker boundaries on an isolated host."""
    bin_directory = tmp_path / "bin"
    home = tmp_path / "home"
    docker_log = tmp_path / "docker.log"
    event_log = tmp_path / "events.log"
    git_log = tmp_path / "git.log"
    git_state = tmp_path / "git.state"
    bin_directory.mkdir()
    home.mkdir()
    docker_log.touch()
    event_log.touch()
    git_log.touch()

    _write_executable(
        bin_directory / "git",
        """#!/bin/sh
set -eu

printf '%s\\n' "$*" >> "$FAKE_GIT_LOG"
printf 'git|%s\\n' "$*" >> "$FAKE_EVENT_LOG"

case "$1" in
  clone)
    exit_code=${FAKE_GIT_CLONE_EXIT:-0}
    [ "$exit_code" -eq 0 ] || exit "$exit_code"
    mkdir -p "$3/.git"
    ;;
  remote)
    [ "${2:-}" = "get-url" ] && [ "${3:-}" = "origin" ] || exit 99
    exit_code=${FAKE_GIT_REMOTE_EXIT:-0}
    [ "$exit_code" -eq 0 ] || exit "$exit_code"
    printf '%s\\n' "${FAKE_GIT_ORIGIN}"
    ;;
  diff)
    if [ "${2:-}" = "--cached" ]; then
      if [ -f "$FAKE_GIT_STATE" ]; then
        exit "${FAKE_GIT_STAGED_AFTER_EXIT:-0}"
      fi
      exit "${FAKE_GIT_STAGED_BEFORE_EXIT:-0}"
    fi
    if [ -f "$FAKE_GIT_STATE" ]; then
      exit "${FAKE_GIT_UNSTAGED_AFTER_EXIT:-0}"
    fi
    exit "${FAKE_GIT_UNSTAGED_BEFORE_EXIT:-0}"
    ;;
  fetch)
    exit "${FAKE_GIT_FETCH_EXIT:-0}"
    ;;
  checkout)
    exit_code=${FAKE_GIT_CHECKOUT_EXIT:-0}
    [ "$exit_code" -eq 0 ] || exit "$exit_code"
    : > "$FAKE_GIT_STATE"
    ;;
  rev-parse)
    exit_code=${FAKE_GIT_REV_PARSE_EXIT:-0}
    [ "$exit_code" -eq 0 ] || exit "$exit_code"
    printf '%s\\n' "${FAKE_GIT_HEAD:-$DEPLOY_SHA}"
    ;;
  ls-files)
    exit "${FAKE_GIT_LS_FILES_EXIT:-0}"
    ;;
  *)
    exit 99
    ;;
esac
""",
    )

    _write_executable(
        bin_directory / "docker",
        """#!/bin/sh
set -eu

printf 'IMAGE=%s|%s\\n' "${IMAGE:-}" "$*" >> "$FAKE_DOCKER_LOG"
printf 'docker|IMAGE=%s|%s\\n' "${IMAGE:-}" "$*" >> "$FAKE_EVENT_LOG"

if [ "$1" = "compose" ]; then
  operation=${8:-}
  case "$operation" in
    config)
      exit_code=${FAKE_DOCKER_CONFIG_EXIT:-0}
      [ "$exit_code" -eq 0 ] || exit "$exit_code"
      printf '%s\\n' "${FAKE_DOCKER_CONFIG_IMAGE:-$IMAGE}"
      ;;
    pull)
      exit "${FAKE_DOCKER_PULL_EXIT:-0}"
      ;;
    up)
      exit "${FAKE_DOCKER_UP_EXIT:-0}"
      ;;
    ps)
      exit_code=${FAKE_DOCKER_PS_EXIT:-0}
      [ "$exit_code" -eq 0 ] || exit "$exit_code"
      printf '%s' "${FAKE_DOCKER_CONTAINER_ID-agent-container}"
      ;;
    *)
      exit 98
      ;;
  esac
  exit 0
fi

if [ "$1" = "inspect" ]; then
  case "$5" in
    *Config.Image*)
      exit_code=${FAKE_DOCKER_CONFIG_INSPECT_EXIT:-0}
      [ "$exit_code" -eq 0 ] || exit "$exit_code"
      printf '%s\\n' "${FAKE_DOCKER_RUNNING_IMAGE:-$IMAGE}"
      ;;
    *.Image*)
      exit_code=${FAKE_DOCKER_IMAGE_ID_EXIT:-0}
      [ "$exit_code" -eq 0 ] || exit "$exit_code"
      printf '%s\\n' "${FAKE_DOCKER_IMAGE_ID-sha256:platform-image}"
      ;;
    *)
      exit 97
      ;;
  esac
  exit 0
fi

if [ "$1" = "image" ] && [ "$2" = "inspect" ]; then
  exit_code=${FAKE_DOCKER_REVISION_EXIT:-0}
  [ "$exit_code" -eq 0 ] || exit "$exit_code"
  printf '%s\\n' "${FAKE_DOCKER_REVISION:-$DEPLOY_SHA}"
  exit 0
fi

if [ "$1" = "image" ] && [ "$2" = "prune" ]; then
  exit "${FAKE_DOCKER_PRUNE_EXIT:-0}"
fi

exit 96
""",
    )

    environment = {
        "FAKE_DOCKER_LOG": str(docker_log),
        "FAKE_EVENT_LOG": str(event_log),
        "FAKE_GIT_LOG": str(git_log),
        "FAKE_GIT_ORIGIN": EXPECTED_ORIGIN,
        "FAKE_GIT_STATE": str(git_state),
        "HOME": str(home),
        "LANG": "C",
        "PATH": f"{bin_directory}:/usr/bin:/bin",
    }
    return DeployHarness(
        environment,
        docker_log,
        event_log,
        git_log,
        git_state,
        home,
    )


def _run_deploy_script(
    script: str,
    harness: DeployHarness,
    **environment_overrides: str,
) -> subprocess.CompletedProcess[str]:
    """Execute the materialized remote program against synthetic boundaries."""
    environment = (
        harness.environment
        | {
            "DEPLOY_SHA": VALID_SHA,
            "IMAGE_DIGEST": VALID_DIGEST,
        }
        | environment_overrides
    )
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


def _workflow_script() -> str:
    """Read the current remote deployment program."""
    return _deploy_script(WORKFLOW_PATH.read_text(encoding="utf-8"))


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


def test_reusable_quality_call_grants_only_codecov_oidc_permissions() -> None:
    """Keep the called quality workflow authenticated and least privilege."""
    document = WORKFLOW_PATH.read_text(encoding="utf-8")
    quality_block = _indented_block(document, "quality:", 2)
    permissions = _indented_block(quality_block, "permissions:", 4)

    assert permissions.splitlines() == [
        "      contents: read",
        "      id-token: write",
    ]
    assert "    uses: ./.github/workflows/code-quality.yml" in quality_block
    assert "secrets:" not in quality_block
    assert "packages: write" not in quality_block


def test_build_digest_is_passed_as_data_to_serialized_deploy() -> None:
    """Bind the remote deployment to build output without script interpolation."""
    document = WORKFLOW_PATH.read_text(encoding="utf-8")
    build_block = _indented_block(document, "build:", 2)
    deploy_block = _indented_block(document, "deploy:", 2)
    concurrency = _indented_block(deploy_block, "concurrency:", 4)
    script = _deploy_script(document)

    assert "      digest: ${{ steps.build-push.outputs.digest }}" in build_block
    assert "      - name: Record immutable image reference" in build_block
    assert '            >> "$GITHUB_STEP_SUMMARY"' in build_block
    assert "          DEPLOY_SHA: ${{ github.sha }}" in deploy_block
    assert "          IMAGE_DIGEST: ${{ needs.build.outputs.digest }}" in deploy_block
    assert "          envs: DEPLOY_SHA,IMAGE_DIGEST" in deploy_block
    assert concurrency.splitlines() == [
        "      group: production-deployment",
        "      cancel-in-progress: false",
        "      queue: max",
    ]
    assert "${{ github.sha }}" not in script
    assert "${{ needs.build.outputs.digest }}" not in script


def test_remote_script_forbids_mutable_or_destructive_deploy_operations() -> None:
    """Require detached provenance and one explicit Compose configuration."""
    script = _workflow_script()

    for forbidden in (
        "git pull",
        "git reset",
        "git clean",
        "checkout --force",
        "checkout -f",
        "ghcr.io/${{ github.repository }}:main",
    ):
        assert forbidden not in script

    assert 'git checkout --detach "$DEPLOY_SHA"' in script
    assert script.count('docker compose --project-name "$DEPLOY_PROJECT"') == 4
    assert script.count("--env-file .env -f compose.yaml") == 4
    assert script.count("docker inspect --type container") == 2


def test_compose_resolves_one_exact_digest_without_repository_secrets(
    tmp_path: Path,
) -> None:
    """Ask the real Compose parser to resolve the digest in an isolated copy."""
    docker = shutil.which("docker")
    if docker is None:
        pytest.skip("Docker CLI is unavailable")

    compose_path = WORKFLOW_PATH.parents[2] / "compose.yaml"
    (tmp_path / "compose.yaml").write_text(
        compose_path.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (tmp_path / ".env").write_text("", encoding="utf-8")
    environment = {
        "COMPOSE_DISABLE_ENV_FILE": "1",
        "HOME": os.environ.get("HOME", str(tmp_path)),
        "IMAGE": EXPECTED_IMAGE,
        "LANG": "C",
        "PATH": os.environ["PATH"],
    }

    result = subprocess.run(  # noqa: S603 - resolved Docker, fixed arguments
        [
            docker,
            "compose",
            "--project-name",
            EXPECTED_PROJECT,
            "--env-file",
            ".env",
            "-f",
            "compose.yaml",
            "config",
            "--images",
        ],
        cwd=tmp_path,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == [EXPECTED_IMAGE]


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    [
        ("DEPLOY_SHA", ""),
        ("DEPLOY_SHA", "a" * 39),
        ("DEPLOY_SHA", "a" * 41),
        ("DEPLOY_SHA", "A" * 40),
        ("DEPLOY_SHA", "g" * 40),
        ("DEPLOY_SHA", f"{'a' * 39}\n"),
        ("DEPLOY_SHA", "$(touch should-not-run)"),
        ("IMAGE_DIGEST", ""),
        ("IMAGE_DIGEST", f"sha256:{'b' * 63}"),
        ("IMAGE_DIGEST", f"sha256:{'b' * 65}"),
        ("IMAGE_DIGEST", f"sha256:{'B' * 64}"),
        ("IMAGE_DIGEST", f"sha256:{'g' * 64}"),
        ("IMAGE_DIGEST", f"sha256:{'b' * 63}\n"),
        ("IMAGE_DIGEST", f"sha512:{'b' * 64}"),
        ("IMAGE_DIGEST", "b" * 64),
        ("IMAGE_DIGEST", "$(touch should-not-run)"),
    ],
)
def test_invalid_provenance_fails_before_side_effects(
    deploy_harness: DeployHarness,
    field: str,
    invalid_value: str,
) -> None:
    """Reject malformed provenance as data before touching Git or Docker."""
    sentinel = deploy_harness.home / "should-not-run"
    value = invalid_value.replace("should-not-run", str(sentinel))

    result = _run_deploy_script(
        _workflow_script(),
        deploy_harness,
        **{field: value},
    )

    assert result.returncode != 0
    assert "ERROR: Invalid " in result.stdout
    assert deploy_harness.git_log.read_text(encoding="utf-8") == ""
    assert deploy_harness.docker_log.read_text(encoding="utf-8") == ""
    assert not (deploy_harness.home / "Mixed-Repository").exists()
    assert not sentinel.exists()


@pytest.mark.parametrize("existing_checkout", [False, True])
def test_remote_deploy_uses_exact_commit_digest_and_literal_secrets(
    deploy_harness: DeployHarness,
    existing_checkout: bool,
) -> None:
    """Deploy one exact revision and preserve shell-sensitive secret bytes."""
    project_directory = deploy_harness.home / "Mixed-Repository"
    if existing_checkout:
        (project_directory / ".git").mkdir(parents=True)

    result = _run_deploy_script(_workflow_script(), deploy_harness)

    assert result.returncode == 0, result.stderr
    git_commands = deploy_harness.git_log.read_text(encoding="utf-8").splitlines()
    expected_after_clone = [
        "remote get-url origin",
        "diff --quiet --",
        "diff --cached --quiet --",
        f"fetch --no-tags origin {VALID_SHA}",
        f"checkout --detach {VALID_SHA}",
        "rev-parse --verify HEAD",
        "diff --quiet --",
        "diff --cached --quiet --",
        "ls-files --error-unmatch -- compose.yaml",
    ]
    if existing_checkout:
        assert git_commands == expected_after_clone
    else:
        assert git_commands == [
            f"clone {EXPECTED_ORIGIN} {project_directory}",
            *expected_after_clone,
        ]

    compose_prefix = (
        f"compose --project-name {EXPECTED_PROJECT} --env-file .env -f compose.yaml"
    )
    assert deploy_harness.docker_log.read_text(encoding="utf-8").splitlines() == [
        f"IMAGE={EXPECTED_IMAGE}|{compose_prefix} config --images",
        f"IMAGE={EXPECTED_IMAGE}|{compose_prefix} pull agent",
        (
            f"IMAGE={EXPECTED_IMAGE}|{compose_prefix} "
            "up --no-build --wait --wait-timeout 180 agent"
        ),
        f"IMAGE={EXPECTED_IMAGE}|{compose_prefix} ps --quiet agent",
        (
            f"IMAGE={EXPECTED_IMAGE}|inspect --type container "
            "--format {{.Config.Image}} agent-container"
        ),
        (
            f"IMAGE={EXPECTED_IMAGE}|inspect --type container "
            "--format {{.Image}} agent-container"
        ),
        (
            f"IMAGE={EXPECTED_IMAGE}|image inspect --format "
            '{{ index .Config.Labels "org.opencontainers.image.revision" }} '
            "sha256:platform-image"
        ),
        f"IMAGE={EXPECTED_IMAGE}|image prune -f",
    ]
    environment_document = (project_directory / ".env").read_text(encoding="utf-8")
    for secret_value in SECRET_CANARIES.values():
        assert secret_value in environment_document


def test_remote_deploy_checks_out_real_git_commit_detached(
    deploy_harness: DeployHarness,
    tmp_path: Path,
) -> None:
    """Independently prove the tracked program reaches an exact detached commit."""
    real_git = shutil.which("git")
    assert real_git is not None

    remote = tmp_path / "remote.git"
    seed = tmp_path / "seed"
    project = deploy_harness.home / "Mixed-Repository"
    git_environment = {
        "HOME": str(deploy_harness.home),
        "LANG": "C",
        "PATH": "/usr/bin:/bin",
    }

    def run_git(*arguments: str, cwd: Path | None = None) -> str:
        result = subprocess.run(  # noqa: S603 - resolved Git, controlled arguments
            [real_git, *arguments],
            cwd=cwd,
            env=git_environment,
            text=True,
            capture_output=True,
            check=True,
        )
        return result.stdout.strip()

    run_git("init", "--bare", str(remote))
    run_git("init", str(seed))
    run_git("-C", str(seed), "config", "user.email", "test@example.com")
    run_git("-C", str(seed), "config", "user.name", "Test Author")
    tracked_file = seed / "tracked.txt"
    tracked_file.write_text("first\n", encoding="utf-8")
    (seed / "compose.yaml").write_text(
        "services:\n  agent:\n    image: agent\n",
        encoding="utf-8",
    )
    run_git("-C", str(seed), "add", "tracked.txt", "compose.yaml")
    run_git("-C", str(seed), "commit", "-m", "first")
    run_git("-C", str(seed), "remote", "add", "origin", str(remote))
    run_git("-C", str(seed), "push", "origin", "HEAD:main")
    run_git("--git-dir", str(remote), "symbolic-ref", "HEAD", "refs/heads/main")
    run_git("clone", str(remote), str(project))
    run_git("-C", str(project), "remote", "set-url", "origin", EXPECTED_ORIGIN)

    tracked_file.write_text("second\n", encoding="utf-8")
    run_git("-C", str(seed), "add", "tracked.txt")
    run_git("-C", str(seed), "commit", "-m", "second")
    target_sha = run_git("-C", str(seed), "rev-parse", "HEAD")
    run_git("-C", str(seed), "push", "origin", "HEAD:main")

    real_bin = tmp_path / "real-bin"
    real_bin.mkdir()
    shutil.copy2(
        Path(deploy_harness.environment["PATH"].split(":", maxsplit=1)[0]) / "docker",
        real_bin / "docker",
    )
    _write_executable(
        real_bin / "git",
        f"""#!/bin/sh
set -eu
if [ "$1" = "fetch" ] && [ "$2" = "--no-tags" ] && [ "$3" = "origin" ]; then
  exec {shlex.quote(real_git)} fetch --no-tags {shlex.quote(str(remote))} "$4"
fi
exec {shlex.quote(real_git)} "$@"
""",
    )

    result = _run_deploy_script(
        _workflow_script(),
        deploy_harness,
        DEPLOY_SHA=target_sha,
        PATH=f"{real_bin}:/usr/bin:/bin",
    )

    assert result.returncode == 0, result.stderr
    assert run_git("-C", str(project), "rev-parse", "HEAD") == target_sha
    symbolic_ref = subprocess.run(  # noqa: S603 - resolved Git, controlled arguments
        [real_git, "-C", str(project), "symbolic-ref", "--quiet", "HEAD"],
        env=git_environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert symbolic_ref.returncode == 1
    assert (project / "tracked.txt").read_text(encoding="utf-8") == "second\n"


def test_remote_deploy_accepts_equivalent_dot_git_origin(
    deploy_harness: DeployHarness,
) -> None:
    """Allow Git's conventional suffix without accepting another repository."""
    project_directory = deploy_harness.home / "Mixed-Repository"
    (project_directory / ".git").mkdir(parents=True)

    result = _run_deploy_script(
        _workflow_script(),
        deploy_harness,
        FAKE_GIT_ORIGIN=f"{EXPECTED_ORIGIN}.git",
    )

    assert result.returncode == 0, result.stderr


def test_remote_deploy_rejects_non_git_project_path(
    deploy_harness: DeployHarness,
) -> None:
    """Never overwrite an operator-owned path that is not this checkout."""
    project_directory = deploy_harness.home / "Mixed-Repository"
    project_directory.mkdir()
    sentinel = project_directory / "operator-owned"
    sentinel.write_text("preserve\n", encoding="utf-8")

    result = _run_deploy_script(_workflow_script(), deploy_harness)

    assert result.returncode != 0
    assert "path exists without a Git checkout" in result.stdout
    assert deploy_harness.git_log.read_text(encoding="utf-8") == ""
    assert deploy_harness.docker_log.read_text(encoding="utf-8") == ""
    assert sentinel.read_text(encoding="utf-8") == "preserve\n"


def test_remote_deploy_rejects_untracked_only_compose_file(
    deploy_harness: DeployHarness,
) -> None:
    """Do not let a host file replace config deleted by the exact commit."""
    project_directory = deploy_harness.home / "Mixed-Repository"
    (project_directory / ".git").mkdir(parents=True)
    untracked_compose = project_directory / "compose.yaml"
    untracked_compose.write_text("operator-owned\n", encoding="utf-8")

    result = _run_deploy_script(
        _workflow_script(),
        deploy_harness,
        FAKE_GIT_LS_FILES_EXIT="1",
    )

    assert result.returncode != 0
    assert "does not track compose.yaml" in result.stdout
    assert deploy_harness.docker_log.read_text(encoding="utf-8") == ""
    assert not (project_directory / ".env").exists()
    assert untracked_compose.read_text(encoding="utf-8") == "operator-owned\n"


@pytest.mark.parametrize(
    ("environment", "expected_message", "forbidden_event"),
    [
        (
            {"FAKE_GIT_ORIGIN": "https://github.com/other/repository"},
            "unexpected origin",
            "git|fetch",
        ),
        (
            {"FAKE_GIT_UNSTAGED_BEFORE_EXIT": "31"},
            "Tracked worktree changes",
            "git|fetch",
        ),
        (
            {"FAKE_GIT_STAGED_BEFORE_EXIT": "32"},
            "Staged changes",
            "git|fetch",
        ),
        (
            {"FAKE_GIT_HEAD": "c" * 40},
            "Checked-out commit does not match",
            "docker|",
        ),
        (
            {"FAKE_GIT_UNSTAGED_AFTER_EXIT": "33"},
            "Checkout left tracked",
            "docker|",
        ),
        (
            {"FAKE_GIT_STAGED_AFTER_EXIT": "34"},
            "Checkout left staged",
            "docker|",
        ),
    ],
)
def test_checkout_provenance_failures_stop_before_docker(
    deploy_harness: DeployHarness,
    environment: dict[str, str],
    expected_message: str,
    forbidden_event: str,
) -> None:
    """Keep unrelated origins, dirty trees, and SHA drift out of deployment."""
    project_directory = deploy_harness.home / "Mixed-Repository"
    (project_directory / ".git").mkdir(parents=True)
    existing_env = project_directory / ".env"
    existing_env.write_text("operator-owned\n", encoding="utf-8")

    result = _run_deploy_script(
        _workflow_script(),
        deploy_harness,
        **environment,
    )

    assert result.returncode != 0
    assert expected_message in result.stdout
    assert forbidden_event not in deploy_harness.event_log.read_text(encoding="utf-8")
    assert existing_env.read_text(encoding="utf-8") == "operator-owned\n"


@pytest.mark.parametrize(
    ("environment", "forbidden_fragment"),
    [
        ({"FAKE_GIT_CLONE_EXIT": "40"}, "git|remote"),
        ({"FAKE_GIT_FETCH_EXIT": "41"}, "git|checkout"),
        ({"FAKE_GIT_CHECKOUT_EXIT": "42"}, "git|rev-parse"),
        ({"FAKE_GIT_REV_PARSE_EXIT": "43"}, "docker|"),
        ({"FAKE_DOCKER_CONFIG_EXIT": "51"}, " pull agent"),
        ({"FAKE_DOCKER_CONFIG_IMAGE": "ghcr.io/other@sha256:bad"}, " pull agent"),
        ({"FAKE_DOCKER_PULL_EXIT": "52"}, " up --no-build"),
        ({"FAKE_DOCKER_UP_EXIT": "53"}, " ps --quiet agent"),
        ({"FAKE_DOCKER_PS_EXIT": "54"}, "inspect --type container"),
        ({"FAKE_DOCKER_CONTAINER_ID": ""}, "inspect --type container"),
        ({"FAKE_DOCKER_CONFIG_INSPECT_EXIT": "55"}, "format {{.Image}}"),
        (
            {"FAKE_DOCKER_RUNNING_IMAGE": "ghcr.io/other@sha256:bad"},
            "format {{.Image}}",
        ),
        ({"FAKE_DOCKER_IMAGE_ID_EXIT": "56"}, "image inspect"),
        ({"FAKE_DOCKER_IMAGE_ID": ""}, "image inspect"),
        ({"FAKE_DOCKER_REVISION_EXIT": "57"}, "image prune"),
        ({"FAKE_DOCKER_REVISION": "d" * 40}, "image prune"),
    ],
)
def test_deploy_boundary_failure_stops_later_actions(
    deploy_harness: DeployHarness,
    environment: dict[str, str],
    forbidden_fragment: str,
) -> None:
    """Propagate each external failure without continuing or pruning."""
    result = _run_deploy_script(
        _workflow_script(),
        deploy_harness,
        **environment,
    )

    assert result.returncode != 0
    assert forbidden_fragment not in deploy_harness.event_log.read_text(
        encoding="utf-8"
    )
    assert "image prune -f" not in deploy_harness.docker_log.read_text(encoding="utf-8")


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
