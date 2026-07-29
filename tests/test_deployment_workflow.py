"""High-trust OpenSSH deployment workflow and script contract tests."""

from __future__ import annotations

import re
import shlex
import shutil
import stat
import subprocess
import sys
import tomllib
from dataclasses import dataclass, replace
from pathlib import Path
from typing import cast

import pytest
import yaml  # type: ignore[import-untyped]

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_PATH = ROOT / ".github" / "workflows" / "docker-publish.yml"
BOOTSTRAP_PATH = ROOT / "scripts" / "deployment_bootstrap.sh"
CLEANUP_PATH = ROOT / "scripts" / "deployment_cleanup.sh"
EVAL_GATE_PATH = ROOT / "tests" / "eval" / "production_adk_eval.py"
PYPROJECT_PATH = ROOT / "pyproject.toml"
LOCK_PATH = ROOT / "uv.lock"
REVISION = "a" * 40
DIGEST = f"sha256:{'b' * 64}"
REPOSITORY = "MixedOwner/Mixed-Repository"
PROJECT_NAME = "Mixed-Repository"
RUN_ID = "12345"
RUN_ATTEMPT = "2"
ORIGIN = f"https://github.com/{REPOSITORY}"
PRODUCTION_NAMES = {
    "AGENT_NAME",
    "DATABASE_URL",
    "OPENROUTER_API_KEY",
    "GOOGLE_API_KEY",
    "ROOT_AGENT_MODEL",
    "LANGFUSE_PUBLIC_KEY",
    "LANGFUSE_SECRET_KEY",
    "LANGFUSE_BASE_URL",
    "LOG_LEVEL",
    "PORT",
    "HOST",
}
EXPECTED_PINS = {
    "actions/checkout": "11d5960a326750d5838078e36cf38b85af677262",
    "docker/setup-qemu-action": "c7c53464625b32c7a7e944ae62b3e17d2b600130",
    "docker/setup-buildx-action": "8d2750c68a42422c14e847fe6c8ac0403b4cbd6f",
    "docker/login-action": "c94ce9fb468520275223c153574b00df6fe4bcc9",
    "docker/metadata-action": "c299e40c65443455700f0fdfc63efafe5b349051",
    "docker/build-push-action": "ca052bb54ab0790a636c9b5f226502c73d547a25",
    "astral-sh/setup-uv": "d4b2f3b6ecc6e67c4457f6d3e41ec42d3d0fcb86",
}


def _workflow() -> dict[str, object]:
    document = yaml.safe_load(WORKFLOW_PATH.read_text(encoding="utf-8"))
    assert isinstance(document, dict)
    return document


def _job(name: str) -> dict[str, object]:
    jobs = _workflow()["jobs"]
    assert isinstance(jobs, dict)
    job = jobs[name]
    assert isinstance(job, dict)
    return job


def _steps(job: str) -> list[dict[str, object]]:
    steps = _job(job)["steps"]
    assert isinstance(steps, list)
    return cast(list[dict[str, object]], steps)


def _step(job: str, name: str) -> dict[str, object]:
    matches = [step for step in _steps(job) if step.get("name") == name]
    assert len(matches) == 1
    return matches[0]


def _run_text(step: dict[str, object]) -> str:
    value = step["run"]
    assert isinstance(value, str)
    return value


def _run(
    command: list[str],
    *,
    cwd: Path,
    environment: dict[str, str] | None = None,
    input_text: str | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603 - fixed test boundaries in tmp_path
        command,
        cwd=cwd,
        env=environment,
        input=input_text,
        text=True,
        capture_output=True,
        check=False,
        timeout=30,
    )


def _git(
    git: str,
    cwd: Path,
    *arguments: str,
    input_text: str | None = None,
) -> str:
    result = _run(
        [git, *arguments],
        cwd=cwd,
        input_text=input_text,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


def _write_executable(path: Path, body: str) -> None:
    path.write_text(body, encoding="utf-8")
    path.chmod(0o755)


@dataclass(frozen=True, slots=True)
class GitHarness:
    home: Path
    project: Path
    release: Path
    lease: Path
    environment: dict[str, str]
    revision: str
    git_log: Path
    hook_canary: Path
    fsmonitor_canary: Path
    diff_canary: Path
    site_canary: Path
    git: str


def _git_harness(
    tmp_path: Path,
    *,
    worktree_failure: str = "none",
) -> GitHarness:
    git = shutil.which("git")
    assert git is not None
    home = tmp_path / "home"
    project = home / PROJECT_NAME
    binary = tmp_path / "bin"
    home.mkdir()
    project.mkdir()
    binary.mkdir()

    _git(git, project, "init", "-q")
    _git(git, project, "config", "user.name", "Deployment Test")
    _git(git, project, "config", "user.email", "deployment@example.test")
    tracked = {
        ".gitignore": ".env\n__pycache__/\n*.pyc\n",
        ".gitattributes": "compose.yaml diff=canary\n",
        "compose.yaml": "services: {}\n",
        "compose.candidate.yaml": "services: {}\n",
        "src/agent/__init__.py": "",
        "src/agent/compose_env.py": "VALUE = 'compose'\n",
        "src/agent/deployment_adoption.py": "VALUE = 'adoption'\n",
        "src/agent/deployment_promotion.py": "VALUE = 'promotion'\n",
        "src/agent/deployment_retention.py": "VALUE = 'retention'\n",
        "src/agent/deployment_state.py": "VALUE = 'state'\n",
    }
    for relative, content in tracked.items():
        path = project / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    _git(git, project, "add", ".")
    _git(git, project, "commit", "-qm", "test release")
    revision = _git(git, project, "rev-parse", "HEAD")
    _git(git, project, "remote", "add", "origin", ORIGIN)
    environment_path = project / ".env"
    environment_path.write_text('PRIVATE="unchanged"\n', encoding="utf-8")
    environment_path.chmod(0o600)

    hook_canary = tmp_path / "post-checkout-ran"
    fsmonitor_canary = tmp_path / "fsmonitor-ran"
    diff_canary = tmp_path / "external-diff-ran"
    site_canary = tmp_path / "sitecustomize-ran"
    hook = project / ".git" / "hooks" / "post-checkout"
    _write_executable(hook, f"#!/bin/sh\n: > {shlex.quote(str(hook_canary))}\n")
    fsmonitor = tmp_path / "fsmonitor"
    _write_executable(
        fsmonitor,
        f"#!/bin/sh\n: > {shlex.quote(str(fsmonitor_canary))}\nprintf '\\n'\n",
    )
    external_diff = tmp_path / "external-diff"
    _write_executable(
        external_diff,
        f"#!/bin/sh\n: > {shlex.quote(str(diff_canary))}\nexit 1\n",
    )
    _git(git, project, "config", "core.fsmonitor", str(fsmonitor))
    _git(git, project, "config", "diff.canary.command", str(external_diff))

    empty_tree = _git(git, project, "mktree", input_text="")
    malicious = _git(
        git,
        project,
        "commit-tree",
        empty_tree,
        input_text="replacement\n",
    )
    _git(git, project, "replace", revision, malicious)

    sitecustomize = home / "sitecustomize.py"
    sitecustomize.write_text(
        f"from pathlib import Path\nPath({str(site_canary)!r}).write_text('loaded')\n",
        encoding="utf-8",
    )

    git_log = tmp_path / "git.log"
    quoted_git = shlex.quote(git)
    quoted_log = shlex.quote(str(git_log))
    failure_body = ""
    if worktree_failure == "registered":
        failure_body = f"""
if [ "$IS_WORKTREE_ADD" -eq 1 ]; then
  {quoted_git} "$@"
  STATUS=$?
  [ "$STATUS" -eq 0 ] || exit "$STATUS"
  exit 47
fi
"""
    elif worktree_failure == "partial":
        failure_body = """
if [ "$IS_WORKTREE_ADD" -eq 1 ]; then
  LAST=
  PREVIOUS=
  for ARG in "$@"; do
    PREVIOUS=$LAST
    LAST=$ARG
  done
  mkdir -p "$PREVIOUS"
  printf partial > "$PREVIOUS/partial"
  exit 47
fi
"""
    _write_executable(
        binary / "git",
        f"""#!/bin/sh
set -u
printf '%s\\n' "$*" >> {quoted_log}
IS_FETCH=0
IS_WORKTREE=0
IS_WORKTREE_ADD=0
for ARG in "$@"; do
  if [ "$ARG" = fetch ]; then IS_FETCH=1; fi
  if [ "$ARG" = worktree ]; then IS_WORKTREE=1; continue; fi
  if [ "$IS_WORKTREE" -eq 1 ] && [ "$ARG" = add ]; then
    IS_WORKTREE_ADD=1
  fi
done
if [ "$IS_FETCH" -eq 1 ]; then exit 0; fi
{failure_body}
exec {quoted_git} "$@"
""",
    )
    _write_executable(binary / "flock", "#!/bin/sh\nexit 0\n")
    _write_executable(
        binary / "python3",
        f'#!/bin/sh\nexec {shlex.quote(sys.executable)} "$@"\n',
    )
    environment = {
        "HOME": str(home),
        "PATH": f"{binary}:/usr/bin:/bin:/usr/sbin:/sbin",
        "PYTHONPATH": str(home),
        "GIT_DIR": str(tmp_path / "hostile-git-dir"),
        "GIT_WORK_TREE": str(tmp_path / "hostile-work-tree"),
        "GIT_INDEX_FILE": str(tmp_path / "hostile-index"),
        "GIT_OBJECT_DIRECTORY": str(tmp_path / "hostile-objects"),
        "GIT_EXEC_PATH": str(tmp_path / "hostile-exec"),
        "GIT_CONFIG_COUNT": "1",
        "GIT_CONFIG_KEY_0": "core.hooksPath",
        "GIT_CONFIG_VALUE_0": str(project / ".git" / "hooks"),
    }
    release = home / f"{PROJECT_NAME}.release-{RUN_ID}-{RUN_ATTEMPT}"
    return GitHarness(
        home=home,
        project=project,
        release=release,
        lease=Path(f"{release}.lease"),
        environment=environment,
        revision=revision,
        git_log=git_log,
        hook_canary=hook_canary,
        fsmonitor_canary=fsmonitor_canary,
        diff_canary=diff_canary,
        site_canary=site_canary,
        git=git,
    )


def _bootstrap(harness: GitHarness) -> subprocess.CompletedProcess[str]:
    return _run(
        [
            "/bin/sh",
            str(BOOTSTRAP_PATH),
            harness.revision,
            DIGEST,
            RUN_ID,
            RUN_ATTEMPT,
            PROJECT_NAME,
            REPOSITORY,
        ],
        cwd=ROOT,
        environment=harness.environment,
    )


def _cleanup(harness: GitHarness) -> subprocess.CompletedProcess[str]:
    return _run(
        [
            "/bin/sh",
            str(CLEANUP_PATH),
            harness.revision,
            DIGEST,
            RUN_ID,
            RUN_ATTEMPT,
            PROJECT_NAME,
            REPOSITORY,
        ],
        cwd=ROOT,
        environment=harness.environment,
    )


def test_external_actions_are_pinned_to_reviewed_commits() -> None:
    uses: list[str] = []
    for job in ("build", "production_eval", "deploy"):
        for step in _steps(job):
            if isinstance(step.get("uses"), str):
                uses.append(cast(str, step["uses"]))

    assert all(re.fullmatch(r"[^@]+@[0-9a-f]{40}", value) for value in uses)
    assert {
        value.split("@", maxsplit=1)[0]: value.split("@", maxsplit=1)[1]
        for value in uses
    } == EXPECTED_PINS


def test_deploy_is_manual_bounded_read_only_and_serialized() -> None:
    deploy = _job("deploy")
    condition = " ".join(str(deploy["if"]).split())

    assert condition == (
        "github.event_name == 'workflow_dispatch' && "
        "github.ref == 'refs/heads/main' && inputs.deploy && "
        "needs.build.result == 'success' && "
        "needs.production_eval.result == 'success'"
    )
    assert deploy["permissions"] == {"contents": "read"}
    assert deploy["timeout-minutes"] == 50
    assert deploy["concurrency"] == {
        "group": "production-deployment",
        "cancel-in-progress": False,
    }
    assert deploy["needs"] == ["build", "production_eval"]
    assert deploy["env"] == {
        "DEPLOY_SHA": "${{ github.sha }}",
        "IMAGE_DIGEST": "${{ needs.build.outputs.digest }}",
        "DEPLOY_RUN_ID": "${{ github.run_id }}",
        "DEPLOY_RUN_ATTEMPT": "${{ github.run_attempt }}",
    }


def test_production_eval_is_manual_locked_exact_and_fail_closed() -> None:
    job = _job("production_eval")
    condition = " ".join(str(job["if"]).split())
    steps = _steps("production_eval")

    assert condition == (
        "github.event_name == 'workflow_dispatch' && "
        "github.ref == 'refs/heads/main' && inputs.deploy"
    )
    assert job["needs"] == "quality"
    assert job["permissions"] == {"contents": "read"}
    assert job["timeout-minutes"] == 10
    assert "env" not in job
    assert "outputs" not in job
    assert "continue-on-error" not in job
    assert "always()" not in condition

    checkout = _step("production_eval", "Checkout evaluation revision")
    assert checkout["with"] == {
        "ref": "${{ github.sha }}",
        "fetch-depth": 1,
        "persist-credentials": False,
    }
    verification = _step("production_eval", "Verify evaluation revision")
    assert verification["env"] == {"EVALUATION_SHA": "${{ github.sha }}"}
    assert "git rev-parse --verify HEAD" in _run_text(verification)

    install = _step("production_eval", "Install locked evaluation dependencies")
    assert _run_text(install) == ("uv sync --locked --no-default-groups --group eval")
    assert "env" not in install

    evaluation = _step("production_eval", "Run committed ADK compatibility evaluation")
    assert "if" not in evaluation
    assert "continue-on-error" not in evaluation
    assert evaluation["env"] == {
        "ADK_DISABLE_LOAD_DOTENV": "true",
        "GOOGLE_API_KEY": "",
        "MEM0_EMBEDDER_DIMS": "",
        "MEM0_EMBEDDER_MODEL": "__disabled_for_adk_compatibility_eval__",
        "OPENROUTER_API_KEY": "${{ secrets.OPENROUTER_API_KEY }}",
        "OTEL_SDK_DISABLED": "true",
        "PYTEST_ADDOPTS": "",
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
        "PYTEST_PLUGINS": "",
        "ROOT_AGENT_MODEL": "google/gemini-2.5-flash",
    }
    assert " ".join(_run_text(evaluation).split()) == (
        "uv run --locked --no-sync --no-default-groups --group eval "
        "pytest --noconftest --confcutdir=tests/eval "
        "-o addopts= -p no:cacheprovider "
        "tests/eval/production_adk_eval.py "
        "-q --tb=line --disable-warnings --show-capture=no"
    )

    for step in steps:
        if step.get("name") == "Run committed ADK compatibility evaluation":
            continue
        assert "${{ secrets." not in str(step)
    assert (
        sum(str(step).count("${{ secrets.OPENROUTER_API_KEY }}") for step in steps) == 1
    )

    project = tomllib.loads(PYPROJECT_PATH.read_text(encoding="utf-8"))
    assert project["dependency-groups"]["eval"] == [
        "google-adk[eval]==1.36.2",
        "pytest>=8.3.4,<9.0.0",
    ]
    lock = tomllib.loads(LOCK_PATH.read_text(encoding="utf-8"))
    locked_project = next(
        package
        for package in lock["package"]
        if package["name"] == "google-adk-on-bare-metal"
    )
    assert locked_project["dev-dependencies"]["eval"] == [
        {"name": "google-adk", "extra": ["eval"]},
        {"name": "pytest"},
    ]
    assert locked_project["metadata"]["requires-dev"]["eval"] == [
        {"name": "google-adk", "extras": ["eval"], "specifier": "==1.36.2"},
        {"name": "pytest", "specifier": ">=8.3.4,<9.0.0"},
    ]
    rendered = WORKFLOW_PATH.read_text(encoding="utf-8")
    evaluation_job = rendered[
        rendered.index("  production_eval:") : rendered.index("\n  deploy:")
    ]
    for forbidden in (" --with ", "pip install", "uv add", "uv lock", "always()"):
        assert forbidden not in evaluation_job
    assert EVAL_GATE_PATH.name == "production_adk_eval.py"
    assert not EVAL_GATE_PATH.name.startswith("test_")
    assert not EVAL_GATE_PATH.name.endswith("_test.py")


def test_production_secrets_are_scoped_only_to_isolated_serializer() -> None:
    serializer = _step("deploy", "Serialize production environment")
    transport = _step("deploy", "Run locked transactional deployment")
    serializer_environment = cast(dict[str, str], serializer["env"])
    transport_environment = cast(dict[str, str], transport["env"])
    serializer_command = _run_text(serializer)
    transport_script = _run_text(transport)

    assert set(serializer_environment) == PRODUCTION_NAMES | {"TRANSPORT_DIR"}
    assert set(transport_environment) == {
        "TRANSPORT_DIR",
        "SERVER_HOST",
        "SERVER_USER",
        "SSH_PRIVATE_KEY",
        "SERVER_HOST_FINGERPRINT",
    }
    for step in _steps("deploy"):
        if step.get("name") == "Serialize production environment":
            continue
        environment = step.get("env", {})
        assert isinstance(environment, dict)
        assert PRODUCTION_NAMES.isdisjoint(environment)
    assert serializer_command.startswith(
        'python3 -I -S -B "$GITHUB_WORKSPACE/src/agent/compose_env.py"'
    )
    assert "PYTHONPATH" not in serializer_command
    assert "-m agent" not in serializer_command
    prepare_script = _run_text(_step("deploy", "Prepare private deployment transport"))
    assert "src/agent/compose_env.py" in prepare_script
    assert 'git ls-tree "$DEPLOY_SHA" -- "$SOURCE_PATH"' in prepare_script
    assert 'if [ ! -f "$SOURCE_PATH" ] || [ -L "$SOURCE_PATH" ]' in prepare_script
    assert "100644:blob|100755:blob" in prepare_script
    assert "${{ secrets.DATABASE_URL }}" not in transport_script
    assert "${{ secrets.OPENROUTER_API_KEY }}" not in transport_script
    assert PRODUCTION_NAMES.isdisjoint(transport_script.split())
    assert PRODUCTION_NAMES.isdisjoint(
        BOOTSTRAP_PATH.read_text(encoding="utf-8").split()
    )
    assert PRODUCTION_NAMES.isdisjoint(CLEANUP_PATH.read_text(encoding="utf-8").split())


def test_openssh_transport_is_strict_direct_and_lease_gated() -> None:
    script = _run_text(_step("deploy", "Run locked transactional deployment"))
    controller = script[
        script.index('CONTROLLER_COMMAND="') : script.index("CONTROLLER_STATUS=$?")
    ]

    assert "appleboy" not in WORKFLOW_PATH.read_text(encoding="utf-8")
    for option in (
        "-F /dev/null",
        ' -i "$IDENTITY_FILE"',
        "StrictHostKeyChecking=yes",
        '"UserKnownHostsFile=$KNOWN_HOSTS"',
        "GlobalKnownHostsFile=/dev/null",
        "IdentitiesOnly=yes",
        "IdentityAgent=none",
        "ForwardAgent=no",
        "ForwardX11=no",
        "ClearAllForwardings=yes",
        "RequestTTY=no",
    ):
        assert option in script
    assert "unset SSH_PRIVATE_KEY" in script
    assert "unset SERVER_HOST_FINGERPRINT" in script
    assert script.index("unset SSH_PRIVATE_KEY") < script.index("ssh-keyscan")
    assert 'if [ "$MATCHING_KEYS" -ne 1 ]' in script
    assert "scripts/deployment_bootstrap.sh" in script
    assert "scripts/deployment_cleanup.sh" in script
    assert "<<'REMOTE_" not in script
    assert "PHYSICAL_HOME=" in controller
    assert "exec env -i" in controller
    assert "python3 -I -S -B" in controller
    assert 'runpy.run_module(\\"agent.deployment_promotion\\"' in controller
    assert "--release-lease" in controller
    assert "--environment-stdin" in controller
    assert "git " not in controller
    assert "flock" not in controller
    assert "CONTROLLER_FILE" not in script
    assert "REMOTE_CLEANUP_ALLOWED=0" not in controller
    assert "124|137|143|255" in script
    assert script.count("checking the lease") == 3
    assert "timeout --signal=TERM --kill-after=30s 8m" in script
    assert "timeout --signal=TERM --kill-after=30s 30m" in script
    assert "timeout --signal=TERM --kill-after=15s 2m" in script


def test_scripts_are_posix_shell_and_encode_exact_cleanup_order() -> None:
    for path in (BOOTSTRAP_PATH, CLEANUP_PATH):
        result = _run(["/bin/sh", "-n", str(path)], cwd=ROOT)
        assert result.returncode == 0, result.stderr

    bootstrap = BOOTSTRAP_PATH.read_text(encoding="utf-8")
    cleanup = CLEANUP_PATH.read_text(encoding="utf-8")
    assert bootstrap.index("WORKTREE_CLEANUP_ARMED=1") < bootstrap.index("worktree add")
    assert bootstrap.index("Legacy .env permissions block deployment") < (
        bootstrap.index("git_safe clone")
    )
    assert "--no-ext-diff --no-textconv" in bootstrap
    assert "GIT_NO_REPLACE_OBJECTS=1" in bootstrap
    assert "GIT_TEMPLATE_DIR=/dev/null" in bootstrap
    assert "-c core.hooksPath=/dev/null" in bootstrap
    assert "-c core.fsmonitor=false" in bootstrap
    assert "BOOTSTRAP_LEASE_READY:%s:%s" in bootstrap
    assert "worktree remove \\\n          --force" in bootstrap
    assert "shutil.rmtree.avoids_symlink_attacks" in bootstrap
    assert 'ls-tree "$DEPLOY_SHA" -- "$TARGET_PATH"' in bootstrap
    assert "100644|100755" in bootstrap
    assert '[ ! -f "$RELEASE_DIR/$TARGET_PATH" ]' in bootstrap
    assert '[ -L "$RELEASE_DIR/$TARGET_PATH" ]' in bootstrap
    assert 'worktree remove "$RELEASE_DIR"' in cleanup
    assert "worktree remove --force" not in cleanup
    assert "--ignored=matching" in bootstrap
    assert "--ignored=matching" in cleanup
    for forbidden in ("reset --hard", "git clean", "docker compose down", "prune"):
        assert forbidden not in bootstrap
        assert forbidden not in cleanup


def test_real_git_bootstrap_ignores_hooks_site_config_and_replace_refs(
    tmp_path: Path,
) -> None:
    harness = _git_harness(tmp_path)

    result = _bootstrap(harness)

    assert result.returncode == 0, result.stderr
    assert f"BOOTSTRAP_LEASE_READY:{RUN_ID}:{RUN_ATTEMPT}" in result.stdout
    assert harness.release.is_dir()
    assert (harness.release / "compose.yaml").read_text() == "services: {}\n"
    assert _git(harness.git, harness.release, "rev-parse", "HEAD") == harness.revision
    assert stat.S_IMODE(harness.lease.stat().st_mode) == 0o700
    assert stat.S_IMODE((harness.lease / "lock").stat().st_mode) == 0o600
    assert not harness.hook_canary.exists()
    assert not harness.fsmonitor_canary.exists()
    assert not harness.diff_canary.exists()
    assert not harness.site_canary.exists()

    cleanup = _cleanup(harness)

    assert cleanup.returncode == 0, cleanup.stderr
    assert not harness.release.exists()
    assert not harness.lease.exists()
    assert (harness.project / ".env").read_text() == 'PRIVATE="unchanged"\n'


def test_legacy_0644_environment_fails_before_git_mutation(tmp_path: Path) -> None:
    harness = _git_harness(tmp_path)
    (harness.project / ".env").chmod(0o644)

    result = _bootstrap(harness)

    assert result.returncode == 1
    assert "Legacy .env permissions block deployment" in result.stdout
    assert f'chmod 600 "{harness.project}/.env"' in result.stdout
    log = (
        harness.git_log.read_text(encoding="utf-8") if harness.git_log.exists() else ""
    )
    assert " fetch " not in f" {log} "
    assert " worktree " not in f" {log} "
    assert not harness.release.exists()
    assert not harness.lease.exists()


def test_external_diff_is_disabled_on_dirty_primary_checkout(
    tmp_path: Path,
) -> None:
    harness = _git_harness(tmp_path)
    (harness.project / "compose.yaml").write_text(
        "services:\n  changed: {}\n",
        encoding="utf-8",
    )

    result = _bootstrap(harness)

    assert result.returncode == 1
    assert "Tracked worktree changes block deployment" in result.stdout
    assert not harness.diff_canary.exists()
    assert "--no-ext-diff --no-textconv --quiet" in harness.git_log.read_text()
    assert not harness.release.exists()


def test_bootstrap_rejects_tracked_required_symlink(tmp_path: Path) -> None:
    """Reject a clean tracked symlink before Python or Compose can follow it."""
    harness = _git_harness(tmp_path)
    _git(harness.git, harness.project, "replace", "-d", harness.revision)
    outside = tmp_path / "outside-controller.py"
    outside.write_text("PRIVATE_CANARY = 'outside-commit'\n", encoding="utf-8")
    controller = harness.project / "src" / "agent" / "deployment_promotion.py"
    controller.unlink()
    controller.symlink_to(outside)
    _git(harness.git, harness.project, "add", "src/agent/deployment_promotion.py")
    _git(harness.git, harness.project, "commit", "-qm", "tracked symlink")
    selected = replace(
        harness,
        revision=_git(harness.git, harness.project, "rev-parse", "HEAD"),
    )

    result = _bootstrap(selected)

    assert result.returncode == 1
    assert "unsafe required file type" in result.stdout
    assert not selected.release.exists()
    assert not selected.lease.exists()


@pytest.mark.parametrize("failure_mode", ["registered", "partial"])
def test_post_materialization_failure_removes_only_owned_release(
    tmp_path: Path,
    failure_mode: str,
) -> None:
    harness = _git_harness(tmp_path, worktree_failure=failure_mode)

    result = _bootstrap(harness)

    assert result.returncode == 47, result.stderr
    assert not harness.release.exists()
    assert not harness.lease.exists()
    listing = _git(harness.git, harness.project, "worktree", "list", "--porcelain")
    assert str(harness.release) not in listing
    if failure_mode == "registered":
        assert "worktree remove --force" in harness.git_log.read_text()


def test_cleanup_handles_marker_without_materialized_release(tmp_path: Path) -> None:
    harness = _git_harness(tmp_path)
    result = _bootstrap(harness)
    assert result.returncode == 0, result.stderr
    remove = _run(
        [
            harness.git,
            "-c",
            "core.hooksPath=/dev/null",
            "-C",
            str(harness.project),
            "worktree",
            "remove",
            "--force",
            str(harness.release),
        ],
        cwd=ROOT,
    )
    assert remove.returncode == 0, remove.stderr

    cleanup = _cleanup(harness)

    assert cleanup.returncode == 0, cleanup.stderr
    assert not harness.lease.exists()


def test_cleanup_rejects_ignored_bytecode_and_preserves_release(
    tmp_path: Path,
) -> None:
    harness = _git_harness(tmp_path)
    result = _bootstrap(harness)
    assert result.returncode == 0, result.stderr
    cache = harness.release / "src" / "agent" / "__pycache__"
    cache.mkdir()
    (cache / "deployment_promotion.cpython-313.pyc").write_bytes(b"canary")

    cleanup = _cleanup(harness)

    assert cleanup.returncode == 1
    assert "not exact-clean" in cleanup.stdout
    assert harness.release.exists()
    assert harness.lease.exists()


def test_cleanup_is_verified_noop_when_bootstrap_created_nothing(
    tmp_path: Path,
) -> None:
    harness = _git_harness(tmp_path)

    result = _cleanup(harness)

    assert result.returncode == 0, result.stderr
    assert not harness.release.exists()
    assert not harness.lease.exists()
