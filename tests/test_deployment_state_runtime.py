"""Opt-in real-Docker proof for legacy VM deployment-state adoption."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from unittest.mock import create_autospec, patch
from urllib.error import URLError
from urllib.request import ProxyHandler, build_opener

import pytest

from agent.compose_env import write_compose_environment
from agent.deployment_state import DeploymentStateStore

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CANDIDATE_COMPOSE_PATH = REPOSITORY_ROOT / "compose.candidate.yaml"
RUN_ENVIRONMENT_NAME = "RUN_DEPLOYMENT_STATE_INTEGRATION"
PREFIX_ENVIRONMENT_NAME = "DEPLOYMENT_STATE_TEST_PREFIX"
PREFIX_PATTERN = re.compile(r"adk-state-[a-z0-9][a-z0-9-]{0,31}\Z")
IMAGE_ID_PATTERN = re.compile(r"sha256:[0-9a-f]{64}\Z")
IMAGE_REFERENCE_PATTERN = re.compile(
    r"127\.0\.0\.1:[0-9]+/[a-z0-9-]+/agent@sha256:[0-9a-f]{64}\Z"
)
PRIVATE_CANARY = 'state-secret-$-हॅलो-"-\\-canary'
EXPECTED_ORIGIN = "https://github.com/QueryPlanner/google-adk-on-bare-metal"
REGISTRY_IMAGE_REFERENCE = (
    "registry@sha256:1be55279f18a2fe1a74edf2664cac61c1bea305b7b4642dab412e7affdcb3e33"
)
ALLOWED_ENVIRONMENT = (
    "AGENT_NAME",
    "ROOT_AGENT_MODEL",
    "LOG_LEVEL",
    "TELEMETRY_NAMESPACE",
    "K_REVISION",
    "CANDIDATE_ENV_CANARY",
)


def _redact(value: str) -> str:
    redacted = value
    for candidate in (
        PRIVATE_CANARY,
        PRIVATE_CANARY.replace("$", "$$"),
        json.dumps(PRIVATE_CANARY, ensure_ascii=False)[1:-1],
        json.dumps(PRIVATE_CANARY, ensure_ascii=True)[1:-1],
        repr(PRIVATE_CANARY),
        repr(PRIVATE_CANARY.encode()),
    ):
        redacted = redacted.replace(candidate, "[REDACTED]")
    return redacted


def _run(
    command: Sequence[str],
    *,
    environment: Mapping[str, str],
    cwd: Path = REPOSITORY_ROOT,
    check: bool = True,
    timeout: float = 180,
) -> subprocess.CompletedProcess[str]:
    try:
        result = subprocess.run(  # noqa: S603 - resolved/fixed test executables
            list(command),
            cwd=cwd,
            env=environment,
            text=True,
            capture_output=True,
            check=False,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as error:
        stdout = _redact(str(error.stdout or "")[-4_000:])
        stderr = _redact(str(error.stderr or "")[-4_000:])
        raise AssertionError(
            f"{' '.join(command[:3])} exceeded {timeout:g} seconds\n"
            f"stdout:\n{stdout}\nstderr:\n{stderr}"
        ) from None
    if check and result.returncode != 0:
        stdout = _redact(result.stdout[-4_000:])
        stderr = _redact(result.stderr[-4_000:])
        raise AssertionError(
            f"{' '.join(command[:3])} failed with {result.returncode}\n"
            f"stdout:\n{stdout}\nstderr:\n{stderr}"
        )
    return result


def _owned_mutation[MutationResult](
    cleanup_commands: list[list[str]],
    cleanup_command: Sequence[str],
    operation: Callable[[], MutationResult],
) -> MutationResult:
    cleanup_commands.append(list(cleanup_command))
    return operation()


def _execute_exact_cleanup(
    cleanup_commands: Sequence[Sequence[str]],
    environment: Mapping[str, str],
) -> list[str]:
    failures: list[str] = []
    for command in reversed(cleanup_commands):
        try:
            result = _run(
                command,
                environment=environment,
                check=False,
                timeout=60,
            )
        except (AssertionError, OSError) as error:
            failures.append(_redact(str(error)[-1_000:]))
            continue
        if result.returncode != 0 and "No such" not in result.stderr:
            failures.append(_redact(result.stderr[-1_000:]))
    return failures


def _capture_cleanup_operation(
    operation: Callable[[], subprocess.CompletedProcess[str]],
) -> list[str]:
    try:
        result = operation()
    except (AssertionError, OSError) as error:
        return [_redact(str(error)[-1_000:])]
    if result.returncode != 0:
        return [_redact(result.stderr[-1_000:])]
    return []


def _report_cleanup_failures(
    failures: Sequence[str],
    primary_error: BaseException | None,
) -> None:
    if not failures:
        return
    cleanup_message = "deployment-state cleanup failed: " + " | ".join(failures)
    if primary_error is None:
        raise AssertionError(cleanup_message)
    primary_error.add_note(cleanup_message)


def _base_environment() -> dict[str, str]:
    environment = {
        key: value
        for key, value in os.environ.items()
        if key not in {"COMPOSE_FILE", "COMPOSE_PROJECT_NAME", "ENV_FILE", "IMAGE"}
    }
    environment.update(
        {
            "COMPOSE_DISABLE_ENV_FILE": "1",
            "LANG": "C.UTF-8",
        }
    )
    return environment


def _require_docker(environment: Mapping[str, str]) -> tuple[str, str]:
    docker = shutil.which("docker")
    git = shutil.which("git")
    if docker is None or git is None:
        raise AssertionError("Docker and Git executables are required")
    _run([docker, "version"], environment=environment, timeout=30)
    _run([docker, "compose", "version"], environment=environment, timeout=30)
    return docker, git


def _resource_prefix() -> str:
    configured = os.environ.get(PREFIX_ENVIRONMENT_NAME, f"adk-state-{os.getpid()}")
    normalized = configured.lower()
    if PREFIX_PATTERN.fullmatch(normalized) is None:
        raise AssertionError("deployment-state Docker prefix is invalid")
    return f"{normalized}-{uuid.uuid4().hex[:8]}"


def _git(
    git: str,
    environment: Mapping[str, str],
    *arguments: str,
    cwd: Path,
) -> str:
    return _run(
        [git, *arguments],
        environment=environment,
        cwd=cwd,
        timeout=30,
    ).stdout.strip()


def _initialize_checkout(
    git: str,
    environment: Mapping[str, str],
    project_directory: Path,
) -> str:
    project_directory.mkdir(mode=0o700)
    shutil.copy2(CANDIDATE_COMPOSE_PATH, project_directory / "compose.candidate.yaml")
    (project_directory / ".gitignore").write_text(".env\n", encoding="utf-8")
    _git(git, environment, "init", cwd=project_directory)
    _git(
        git,
        environment,
        "config",
        "user.email",
        "runtime@example.invalid",
        cwd=project_directory,
    )
    _git(
        git,
        environment,
        "config",
        "user.name",
        "Runtime Contract",
        cwd=project_directory,
    )
    _git(
        git,
        environment,
        "remote",
        "add",
        "origin",
        EXPECTED_ORIGIN,
        cwd=project_directory,
    )
    _git(
        git,
        environment,
        "add",
        ".gitignore",
        "compose.candidate.yaml",
        cwd=project_directory,
    )
    _git(
        git,
        environment,
        "commit",
        "-m",
        "runtime fixture",
        cwd=project_directory,
    )
    revision = _git(git, environment, "rev-parse", "HEAD", cwd=project_directory)
    assert re.fullmatch(r"[0-9a-f]{40}", revision)
    return revision


def _build_image(
    docker: str,
    environment: Mapping[str, str],
    *,
    tag: str,
    revision: str,
    iid_file: Path,
) -> str:
    _run(
        [
            docker,
            "build",
            "--iidfile",
            str(iid_file),
            "--tag",
            tag,
            "--label",
            f"org.opencontainers.image.revision={revision}",
            str(REPOSITORY_ROOT),
        ],
        environment=environment,
        timeout=600,
    )
    image_id = iid_file.read_text(encoding="ascii").strip()
    assert IMAGE_ID_PATTERN.fullmatch(image_id)
    inspected = _run(
        [docker, "image", "inspect", "--format", "{{.Id}}", tag],
        environment=environment,
        timeout=30,
    ).stdout.strip()
    assert inspected == image_id
    return image_id


def _ensure_registry_image(
    docker: str,
    environment: Mapping[str, str],
    cleanup_commands: list[list[str]],
) -> str:
    existing = _run(
        [
            docker,
            "image",
            "inspect",
            "--format",
            "{{.Id}}",
            REGISTRY_IMAGE_REFERENCE,
        ],
        environment=environment,
        check=False,
        timeout=30,
    )
    created = existing.returncode != 0
    if created:

        def pull_and_inspect() -> str:
            _run(
                [docker, "image", "pull", REGISTRY_IMAGE_REFERENCE],
                environment=environment,
                timeout=180,
            )
            return _image_identity(
                docker,
                environment,
                REGISTRY_IMAGE_REFERENCE,
            )

        image_id = _owned_mutation(
            cleanup_commands,
            [docker, "image", "rm", REGISTRY_IMAGE_REFERENCE],
            pull_and_inspect,
        )
    else:
        image_id = existing.stdout.strip()
    assert IMAGE_ID_PATTERN.fullmatch(image_id)
    return image_id


def _create_registry(
    docker: str,
    environment: Mapping[str, str],
    *,
    name: str,
    image_id: str,
) -> None:
    _run(
        [
            docker,
            "container",
            "create",
            "--name",
            name,
            "--publish",
            "127.0.0.1::5000",
            "--env",
            "OTEL_TRACES_EXPORTER=none",
            image_id,
        ],
        environment=environment,
        timeout=30,
    )


def _start_registry(
    docker: str,
    environment: Mapping[str, str],
    *,
    name: str,
) -> str:
    _run(
        [docker, "container", "start", name],
        environment=environment,
        timeout=30,
    )
    published = _run(
        [docker, "container", "port", name, "5000/tcp"],
        environment=environment,
        timeout=30,
    ).stdout.strip()
    match = re.fullmatch(r"127\.0\.0\.1:([0-9]+)", published)
    assert match is not None
    endpoint = f"127.0.0.1:{match.group(1)}"
    opener = build_opener(ProxyHandler({}))
    deadline = time.monotonic() + 30
    while True:
        try:
            with opener.open(f"http://{endpoint}/v2/", timeout=1) as response:
                if response.status == 200:
                    return endpoint
        except (OSError, URLError):
            pass
        if time.monotonic() >= deadline:
            raise AssertionError("local registry did not become ready")
        time.sleep(0.2)


def _push_exact_image(
    docker: str,
    environment: Mapping[str, str],
    *,
    source_image_id: str,
    repository_tag: str,
) -> str:
    _run(
        [docker, "image", "tag", source_image_id, repository_tag],
        environment=environment,
        timeout=30,
    )
    _run(
        [docker, "image", "push", repository_tag],
        environment=environment,
        timeout=180,
    )
    repo_digests = json.loads(
        _run(
            [
                docker,
                "image",
                "inspect",
                "--format",
                "{{json .RepoDigests}}",
                repository_tag,
            ],
            environment=environment,
            timeout=30,
        ).stdout
    )
    assert isinstance(repo_digests, list)
    exact = [
        value
        for value in repo_digests
        if isinstance(value, str)
        and value.startswith(f"{repository_tag.rsplit(':', 1)[0]}@sha256:")
    ]
    assert len(exact) == 1
    assert IMAGE_REFERENCE_PATTERN.fullmatch(exact[0])
    return exact[0]


def _write_environment(path: Path, revision: str) -> None:
    values = {
        "AGENT_NAME": "deployment-state-runtime",
        "ROOT_AGENT_MODEL": "openrouter/openai/gpt-4.1-mini",
        "LOG_LEVEL": "INFO",
        "TELEMETRY_NAMESPACE": "deployment-state-runtime",
        "K_REVISION": revision,
        "CANDIDATE_ENV_CANARY": PRIVATE_CANARY,
    }
    write_compose_environment(path, ALLOWED_ENVIRONMENT, values)
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


def _compose(
    docker: str,
    environment: Mapping[str, str],
    *,
    project_directory: Path,
    project: str,
    env_file: Path,
    image_reference: str,
    arguments: Sequence[str],
    check: bool = True,
    timeout: float = 180,
) -> subprocess.CompletedProcess[str]:
    if not project.startswith("adk-state-"):
        raise AssertionError("refusing an unowned Compose project")
    selected_environment = dict(environment)
    selected_environment.update(
        {
            "ENV_FILE": str(env_file),
            "IMAGE": image_reference,
        }
    )
    return _run(
        [
            docker,
            "compose",
            "--project-name",
            project,
            "--env-file",
            str(env_file),
            "-f",
            str(project_directory / "compose.candidate.yaml"),
            *arguments,
        ],
        environment=selected_environment,
        cwd=project_directory,
        check=check,
        timeout=timeout,
    )


def _cli(
    environment: Mapping[str, str],
    *arguments: str,
    cwd: Path = REPOSITORY_ROOT,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    return _run(
        [sys.executable, "-m", "agent.deployment_state_cli", *arguments],
        environment=environment,
        cwd=cwd,
        check=check,
        timeout=60,
    )


def _container_identity(
    docker: str,
    environment: Mapping[str, str],
    name: str,
) -> tuple[str, str]:
    document = json.loads(
        _run(
            [docker, "container", "inspect", name],
            environment=environment,
            timeout=30,
        ).stdout
    )
    assert isinstance(document, list) and len(document) == 1
    return document[0]["Id"], document[0]["State"]["Status"]


def _volume_identity(
    docker: str,
    environment: Mapping[str, str],
    name: str,
) -> tuple[str, str, str]:
    document = json.loads(
        _run(
            [docker, "volume", "inspect", name],
            environment=environment,
            timeout=30,
        ).stdout
    )
    assert isinstance(document, list) and len(document) == 1
    return (
        document[0]["Name"],
        document[0]["Mountpoint"],
        document[0]["CreatedAt"],
    )


def _docker_names(
    docker: str,
    environment: Mapping[str, str],
    resource: str,
) -> frozenset[str]:
    result = _run(
        [docker, resource, "ls", "--format", "{{.Name}}"],
        environment=environment,
        timeout=30,
    )
    return frozenset(line for line in result.stdout.splitlines() if line)


def _image_identity(
    docker: str,
    environment: Mapping[str, str],
    reference: str,
) -> str:
    identity = _run(
        [docker, "image", "inspect", "--format", "{{.Id}}", reference],
        environment=environment,
        timeout=30,
    ).stdout.strip()
    assert IMAGE_ID_PATTERN.fullmatch(identity)
    return identity


def _compose_container_id(
    docker: str,
    environment: Mapping[str, str],
    *,
    project: str,
) -> str:
    result = _run(
        [
            docker,
            "container",
            "ls",
            "--all",
            "--filter",
            f"label=com.docker.compose.project={project}",
            "--filter",
            "label=com.docker.compose.service=agent",
            "--format",
            "{{.ID}}",
        ],
        environment=environment,
        timeout=30,
    )
    identities = result.stdout.splitlines()
    assert len(identities) == 1
    assert re.fullmatch(r"[0-9a-f]{12,64}", identities[0])
    return identities[0]


def _assert_container_name_is_unused(
    docker: str,
    environment: Mapping[str, str],
    name: str,
) -> None:
    result = _run(
        [
            docker,
            "container",
            "ls",
            "--all",
            "--filter",
            f"name=^/{name}$",
            "--format",
            "{{.ID}}",
        ],
        environment=environment,
        timeout=30,
    )
    assert result.stdout.strip() == ""


def test_runtime_prefix_rejects_unsafe_external_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Prevent a CI input from broadening Docker cleanup ownership."""
    monkeypatch.setenv(PREFIX_ENVIRONMENT_NAME, "../foreign")
    with pytest.raises(AssertionError, match="prefix is invalid"):
        _resource_prefix()


def test_compose_guard_rejects_unowned_project(tmp_path: Path) -> None:
    """Refuse cleanup or startup outside the unique test namespace."""
    with pytest.raises(AssertionError, match="unowned"):
        _compose(
            "/usr/bin/docker",
            {},
            project_directory=tmp_path,
            project="production",
            env_file=tmp_path / ".env",
            image_reference=IMAGE_REFERENCE_PATTERN.pattern,
            arguments=("down",),
        )


def test_runtime_runner_redacts_timeout_bytes() -> None:
    """Never expose the canary through a timed-out subprocess diagnostic."""
    timeout = subprocess.TimeoutExpired(
        ["docker", "compose", "up"],
        1,
        output=PRIVATE_CANARY.encode(),
        stderr=PRIVATE_CANARY.encode(),
    )
    runner = create_autospec(
        subprocess.run,
        spec_set=True,
        side_effect=timeout,
    )
    with (
        patch.object(subprocess, "run", runner),
        pytest.raises(AssertionError) as error,
    ):
        _run(["docker", "compose", "up"], environment={}, timeout=1)
    assert PRIVATE_CANARY not in str(error.value)


def test_owned_mutation_arms_exact_cleanup_before_timeout() -> None:
    """Register recoverable cleanup before an external mutation can half-succeed."""
    cleanup_commands: list[list[str]] = []
    cleanup_command = ["docker", "container", "rm", "owned-resource"]

    def time_out_after_daemon_acceptance() -> None:
        assert cleanup_commands == [cleanup_command]
        raise subprocess.TimeoutExpired(["docker", "container", "create"], 1)

    with pytest.raises(subprocess.TimeoutExpired):
        _owned_mutation(
            cleanup_commands,
            cleanup_command,
            time_out_after_daemon_acceptance,
        )

    assert cleanup_commands == [cleanup_command]


def test_exact_cleanup_reverses_order_ignores_absence_and_redacts() -> None:
    """Clean only registered resources without leaking failed-command output."""
    first = ["docker", "image", "rm", "first"]
    second = ["docker", "container", "rm", "second"]
    third = ["docker", "volume", "rm", "third"]
    runner = create_autospec(
        subprocess.run,
        spec_set=True,
        side_effect=[
            subprocess.TimeoutExpired(
                third,
                1,
                output=PRIVATE_CANARY.encode(),
                stderr=PRIVATE_CANARY.encode(),
            ),
            subprocess.CompletedProcess(
                second,
                1,
                stdout="",
                stderr="No such container",
            ),
            subprocess.CompletedProcess(
                first,
                29,
                stdout="",
                stderr=PRIVATE_CANARY,
            ),
        ],
    )

    with patch.object(subprocess, "run", runner):
        failures = _execute_exact_cleanup([first, second, third], {})

    assert [selected.args[0] for selected in runner.call_args_list] == [
        third,
        second,
        first,
    ]
    assert len(failures) == 2
    assert all(PRIVATE_CANARY not in failure for failure in failures)
    assert all("[REDACTED]" in failure for failure in failures)


def test_compose_timeout_continues_exact_cleanup_and_preserves_primary() -> None:
    """Preserve the test failure while every registered cleanup still runs."""
    first = ["docker", "image", "rm", "first"]
    second = ["docker", "container", "rm", "second"]
    compose = ["docker", "compose", "down"]
    runner = create_autospec(
        subprocess.run,
        spec_set=True,
        side_effect=[
            subprocess.TimeoutExpired(
                compose,
                1,
                output=PRIVATE_CANARY.encode(),
                stderr=PRIVATE_CANARY.encode(),
            ),
            subprocess.CompletedProcess(second, 0, stdout="", stderr=""),
            subprocess.CompletedProcess(first, 0, stdout="", stderr=""),
        ],
    )
    primary_error = RuntimeError("primary integration failure")

    with patch.object(subprocess, "run", runner):
        cleanup_failures = _capture_cleanup_operation(
            lambda: _run(compose, environment={}, check=False, timeout=1)
        )
        cleanup_failures.extend(_execute_exact_cleanup([first, second], {}))
        _report_cleanup_failures(cleanup_failures, primary_error)

    assert [selected.args[0] for selected in runner.call_args_list] == [
        compose,
        second,
        first,
    ]
    assert str(primary_error) == "primary integration failure"
    assert len(primary_error.__notes__) == 1
    assert "deployment-state cleanup failed" in primary_error.__notes__[0]
    assert PRIVATE_CANARY not in primary_error.__notes__[0]
    assert "[REDACTED]" in primary_error.__notes__[0]


@pytest.mark.parametrize(
    "operation",
    [
        pytest.param(
            lambda: (_ for _ in ()).throw(OSError(PRIVATE_CANARY)),
            id="operating-system-error",
        ),
        pytest.param(
            lambda: subprocess.CompletedProcess(
                ["docker", "compose", "down"],
                1,
                stdout="",
                stderr=PRIVATE_CANARY,
            ),
            id="nonzero-result",
        ),
    ],
)
def test_compose_cleanup_failures_are_redacted(
    operation: Callable[[], subprocess.CompletedProcess[str]],
) -> None:
    """Capture all expected teardown failures without exposing environment data."""
    failures = _capture_cleanup_operation(operation)

    assert len(failures) == 1
    assert PRIVATE_CANARY not in failures[0]
    assert "[REDACTED]" in failures[0]


def test_cleanup_failure_without_primary_error_fails_the_proof() -> None:
    """Fail the integration proof when teardown alone is unsuccessful."""
    with pytest.raises(AssertionError, match="deployment-state cleanup failed"):
        _report_cleanup_failures(["compose teardown failed"], None)


@pytest.mark.skipif(
    os.environ.get(RUN_ENVIRONMENT_NAME) != "1",
    reason="real deployment-state Docker proof is opt-in",
)
def test_real_docker_adopts_exact_healthy_compose_state(tmp_path: Path) -> None:
    """Prove real Git/Docker adoption, locking, durability, and non-mutation."""
    environment = _base_environment()
    docker, git = _require_docker(environment)
    prefix = _resource_prefix()
    registry_name = f"{prefix}-registry"
    compose_project = f"{prefix}-legacy"
    zero_project = f"{prefix}-zero"
    source_tag = f"{prefix}-source:runtime"
    repository_name = f"{prefix}/agent"
    sentinel_image = f"{prefix}-sentinel-image:keep"
    sentinel_container = f"{prefix}-sentinel-container"
    sentinel_volume = f"{prefix}-sentinel-volume"
    sentinel_network = f"{prefix}-sentinel-network"
    project_directory = tmp_path / "legacy-checkout"
    state_directory = tmp_path / "deployment-state"
    zero_state_directory = tmp_path / "zero-state"
    env_file = project_directory / ".env"
    baseline_volumes = _docker_names(docker, environment, "volume")

    cleanup_commands: list[list[str]] = []
    compose_cleanup_required = False
    exact_reference: str | None = None
    primary_error: BaseException | None = None
    try:
        revision = _initialize_checkout(
            git,
            environment,
            project_directory,
        )
        image_id = _owned_mutation(
            cleanup_commands,
            [docker, "image", "rm", source_tag],
            lambda: _build_image(
                docker,
                environment,
                tag=source_tag,
                revision=revision,
                iid_file=tmp_path / "source.iid",
            ),
        )
        registry_image_id = _ensure_registry_image(
            docker,
            environment,
            cleanup_commands,
        )
        _assert_container_name_is_unused(
            docker,
            environment,
            registry_name,
        )
        _owned_mutation(
            cleanup_commands,
            [
                docker,
                "container",
                "rm",
                "--force",
                "--volumes",
                registry_name,
            ],
            lambda: _create_registry(
                docker,
                environment,
                name=registry_name,
                image_id=registry_image_id,
            ),
        )
        endpoint = _start_registry(
            docker,
            environment,
            name=registry_name,
        )
        _owned_mutation(
            cleanup_commands,
            [docker, "image", "rm", sentinel_image],
            lambda: _run(
                [docker, "image", "tag", registry_image_id, sentinel_image],
                environment=environment,
                timeout=30,
            ),
        )
        sentinel_image_identity = _image_identity(
            docker,
            environment,
            sentinel_image,
        )

        repository_tag = f"{endpoint}/{repository_name}:runtime"
        exact_reference = _owned_mutation(
            cleanup_commands,
            [docker, "image", "rm", repository_tag],
            lambda: _push_exact_image(
                docker,
                environment,
                source_image_id=image_id,
                repository_tag=repository_tag,
            ),
        )
        cleanup_commands.append([docker, "image", "rm", exact_reference])
        _write_environment(env_file, revision)
        environment_before = env_file.read_bytes()
        environment_metadata_before = (
            env_file.stat().st_dev,
            env_file.stat().st_ino,
            env_file.stat().st_size,
            env_file.stat().st_mtime_ns,
            env_file.stat().st_mode,
            env_file.stat().st_nlink,
        )

        compose_cleanup_required = True
        _compose(
            docker,
            environment,
            project_directory=project_directory,
            project=compose_project,
            env_file=env_file,
            image_reference=exact_reference,
            arguments=(
                "up",
                "--detach",
                "--no-build",
                "--pull",
                "never",
                "--wait",
                "--wait-timeout",
                "120",
                "agent",
            ),
            timeout=150,
        )
        deployed_container = _compose_container_id(
            docker,
            environment,
            project=compose_project,
        )
        deployed_container_identity = _container_identity(
            docker,
            environment,
            deployed_container,
        )
        deployed_image_identity = _image_identity(
            docker,
            environment,
            exact_reference,
        )
        assert deployed_container_identity[1] == "running"
        assert deployed_image_identity == image_id

        _owned_mutation(
            cleanup_commands,
            [docker, "container", "rm", sentinel_container],
            lambda: _run(
                [
                    docker,
                    "container",
                    "create",
                    "--name",
                    sentinel_container,
                    "--network",
                    "none",
                    "--entrypoint",
                    "/bin/true",
                    image_id,
                ],
                environment=environment,
                timeout=30,
            ),
        )
        _run(
            [docker, "container", "start", "--attach", sentinel_container],
            environment=environment,
            timeout=30,
        )
        sentinel_container_identity = _container_identity(
            docker,
            environment,
            sentinel_container,
        )
        assert sentinel_container_identity[1] == "exited"

        _owned_mutation(
            cleanup_commands,
            [docker, "volume", "rm", sentinel_volume],
            lambda: _run(
                [docker, "volume", "create", sentinel_volume],
                environment=environment,
                timeout=30,
            ),
        )
        sentinel_volume_identity = _volume_identity(
            docker,
            environment,
            sentinel_volume,
        )
        network_result = _owned_mutation(
            cleanup_commands,
            [docker, "network", "rm", sentinel_network],
            lambda: _run(
                [docker, "network", "create", sentinel_network],
                environment=environment,
                timeout=30,
            ),
        )
        sentinel_network_id = network_result.stdout.strip()
        assert re.fullmatch(r"[0-9a-f]{64}", sentinel_network_id)

        adopted_result = _cli(
            environment,
            "adopt",
            "--state-dir",
            str(state_directory),
            "--checkout",
            str(project_directory),
            "--expected-origin",
            EXPECTED_ORIGIN,
            "--compose-project",
            compose_project,
        )
        assert PRIVATE_CANARY not in adopted_result.stdout
        assert PRIVATE_CANARY not in adopted_result.stderr
        adopted = json.loads(adopted_result.stdout)
        assert adopted["status"] == "adopted"
        state = adopted["current"]["state"]
        assert state["source_revision"] == revision
        assert state["image_reference"] == exact_reference
        assert state["image_id"] == image_id
        assert state["oci_revision"] == revision
        assert state["compose_project"] == compose_project
        assert state["compose_service"] == "agent"

        store = DeploymentStateStore(state_directory)
        current = store.read_current()
        assert current is not None
        snapshot = state_directory / current.state.environment_snapshot
        assert snapshot.read_bytes() == environment_before
        assert (
            current.state.environment_sha256
            == hashlib.sha256(environment_before).hexdigest()
        )
        assert PRIVATE_CANARY.encode() not in store.current_path.read_bytes()
        assert (
            PRIVATE_CANARY.encode()
            not in (store.journal_path / "00000000000000000001.json").read_bytes()
        )
        assert stat.S_IMODE(state_directory.stat().st_mode) == 0o700
        for path in (
            store.lock_path,
            store.current_path,
            store.journal_path / "00000000000000000001.json",
            snapshot,
        ):
            assert stat.S_IMODE(path.stat().st_mode) == 0o600

        inspected = _cli(
            environment,
            "inspect",
            "--state-dir",
            str(state_directory),
        )
        assert PRIVATE_CANARY not in inspected.stdout
        assert PRIVATE_CANARY not in inspected.stderr
        assert json.loads(inspected.stdout)["status"] == "recorded"

        with store.transaction():
            busy = _cli(
                environment,
                "inspect",
                "--state-dir",
                str(state_directory),
                check=False,
            )
            assert busy.returncode == 75
            assert "another deployment transaction" in busy.stderr
            assert PRIVATE_CANARY not in busy.stderr

        zero = _cli(
            environment,
            "adopt",
            "--state-dir",
            str(zero_state_directory),
            "--checkout",
            str(project_directory),
            "--expected-origin",
            EXPECTED_ORIGIN,
            "--compose-project",
            zero_project,
        )
        assert PRIVATE_CANARY not in zero.stdout
        assert PRIVATE_CANARY not in zero.stderr
        assert json.loads(zero.stdout)["status"] == "fresh"
        assert DeploymentStateStore(zero_state_directory).read_current() is None

        assert env_file.read_bytes() == environment_before
        assert (
            env_file.stat().st_dev,
            env_file.stat().st_ino,
            env_file.stat().st_size,
            env_file.stat().st_mtime_ns,
            env_file.stat().st_mode,
            env_file.stat().st_nlink,
        ) == environment_metadata_before
        assert (
            _git(
                git,
                environment,
                "rev-parse",
                "HEAD",
                cwd=project_directory,
            )
            == revision
        )
        assert (
            _git(
                git,
                environment,
                "diff",
                "--quiet",
                "--",
                cwd=project_directory,
            )
            == ""
        )
        assert (
            _git(
                git,
                environment,
                "diff",
                "--cached",
                "--quiet",
                "--",
                cwd=project_directory,
            )
            == ""
        )
        assert (
            _container_identity(
                docker,
                environment,
                deployed_container,
            )
            == deployed_container_identity
        )
        assert (
            _image_identity(docker, environment, exact_reference)
            == deployed_image_identity
        )
        assert (
            _image_identity(docker, environment, sentinel_image)
            == sentinel_image_identity
        )

        assert (
            _container_identity(
                docker,
                environment,
                sentinel_container,
            )
            == sentinel_container_identity
        )
        assert (
            _volume_identity(
                docker,
                environment,
                sentinel_volume,
            )
            == sentinel_volume_identity
        )
        assert (
            _run(
                [
                    docker,
                    "network",
                    "inspect",
                    "--format",
                    "{{.Id}}",
                    sentinel_network,
                ],
                environment=environment,
                timeout=30,
            ).stdout.strip()
            == sentinel_network_id
        )
    except BaseException as error:
        primary_error = error
        raise
    finally:
        cleanup_failures: list[str] = []
        if compose_cleanup_required:
            cleanup_failures.extend(
                _capture_cleanup_operation(
                    lambda: _compose(
                        docker,
                        environment,
                        project_directory=project_directory,
                        project=compose_project,
                        env_file=env_file,
                        image_reference=(
                            source_tag if exact_reference is None else exact_reference
                        ),
                        arguments=("down", "--remove-orphans", "--timeout", "30"),
                        check=False,
                        timeout=60,
                    )
                )
            )
        cleanup_failures.extend(_execute_exact_cleanup(cleanup_commands, environment))
        try:
            remaining_volumes = _docker_names(docker, environment, "volume")
        except AssertionError:
            cleanup_failures.append("Docker volume inventory failed after cleanup")
        else:
            if remaining_volumes != baseline_volumes:
                cleanup_failures.append("Docker volume set changed after cleanup")
        _report_cleanup_failures(cleanup_failures, primary_error)
