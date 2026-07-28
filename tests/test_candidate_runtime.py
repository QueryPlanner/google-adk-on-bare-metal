"""Opt-in real-Docker proof for the isolated candidate Compose contract."""

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
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import create_autospec

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CANDIDATE_COMPOSE_PATH = REPOSITORY_ROOT / "compose.candidate.yaml"
RUN_ENVIRONMENT_NAME = "RUN_CANDIDATE_INTEGRATION"
PROJECT_PREFIX_ENVIRONMENT_NAME = "CANDIDATE_TEST_PROJECT_PREFIX"
CANDIDATE_ENV_ALLOWLIST = (
    "AGENT_NAME",
    "ROOT_AGENT_MODEL",
    "LOG_LEVEL",
    "TELEMETRY_NAMESPACE",
    "K_REVISION",
    "CANDIDATE_ENV_CANARY",
)
HOSTILE_CANARY = (
    "dollar$VAR${OTHER} quote\" single' backslash\\ tab\t "
    "hash# ampersand& equals= unicode-हॅलो"
)
IMAGE_ID_PATTERN = re.compile(r"sha256:[0-9a-f]{64}\Z")
PROJECT_PREFIX_PATTERN = re.compile(r"adk-candidate-[a-z0-9][a-z0-9-]{0,19}\Z")
RESOURCE_NAME_PATTERN = re.compile(r"[a-z0-9][a-z0-9_.-]{0,62}\Z")
EXPECTED_ENTRYPOINT = ("/app/entrypoint.sh",)
HEALTHY_COMMAND = ("python", "-m", "agent.server")
UNHEALTHY_COMMAND = ("python", "-c", "import time; time.sleep(300)")
READY_PROBE = """\
import json
import urllib.request

opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
with opener.open("http://127.0.0.1:8080/ready", timeout=3) as response:
    payload = json.load(response)
if payload != {
    "status": "ready",
    "checks": {"database": "not_configured"},
}:
    raise SystemExit("candidate-ready-contract-failed")
print("candidate-ready")
"""
CANARY_HASH_PROBE = """\
import hashlib
import os

value = os.environ["CANDIDATE_ENV_CANARY"].encode("utf-8")
print(hashlib.sha256(value).hexdigest())
"""


@dataclass(frozen=True)
class CandidateHarness:
    """Validated local Docker boundary for one candidate proof."""

    docker: str
    environment: dict[str, str]
    env_file: Path
    resource_prefix: str
    owned_projects: frozenset[str]

    def __post_init__(self) -> None:
        """Validate the immutable ownership boundary at construction."""
        assert RESOURCE_NAME_PATTERN.fullmatch(self.resource_prefix)
        assert self.owned_projects
        for project in self.owned_projects:
            self.assert_owned_project(project)

    def assert_owned_project(self, project: str) -> None:
        """Reject projects outside this exact per-run ownership boundary."""
        assert RESOURCE_NAME_PATTERN.fullmatch(project)
        assert project.startswith(f"{self.resource_prefix}-")
        assert project in self.owned_projects

    def compose_prefix(self, project: str) -> list[str]:
        """Return the standalone exact-project Compose command."""
        self.assert_owned_project(project)
        return [
            self.docker,
            "compose",
            "--project-name",
            project,
            "--env-file",
            str(self.env_file),
            "-f",
            str(CANDIDATE_COMPOSE_PATH),
        ]


def _redact(output: str) -> str:
    """Redact raw and common escaped representations of the synthetic canary."""
    redacted = output
    representations = {
        HOSTILE_CANARY,
        HOSTILE_CANARY.replace("$", "$$"),
        json.dumps(HOSTILE_CANARY, ensure_ascii=False)[1:-1],
        json.dumps(HOSTILE_CANARY, ensure_ascii=True)[1:-1],
        repr(HOSTILE_CANARY)[1:-1],
        repr(HOSTILE_CANARY.encode("utf-8"))[2:-1],
    }
    for representation in sorted(representations, key=len, reverse=True):
        redacted = redacted.replace(representation, "[redacted]")
    return redacted


def _run(
    command: list[str],
    *,
    environment: dict[str, str],
    check: bool = True,
    timeout: float = 180,
) -> subprocess.CompletedProcess[str]:
    """Run one fixed subprocess and redact every surfaced diagnostic."""
    try:
        result = subprocess.run(  # noqa: S603 - fixed resolved executables
            command,
            cwd=REPOSITORY_ROOT,
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


def _base_environment() -> dict[str, str]:
    """Inherit only settings needed to reach the selected Docker daemon."""
    inherited_names = (
        "DOCKER_CONFIG",
        "DOCKER_CONTEXT",
        "DOCKER_HOST",
        "HOME",
        "PATH",
        "SSL_CERT_DIR",
        "SSL_CERT_FILE",
        "XDG_RUNTIME_DIR",
    )
    environment = {
        name: os.environ[name] for name in inherited_names if name in os.environ
    }
    environment.update(
        {
            "COMPOSE_DISABLE_ENV_FILE": "1",
            "LANG": "C.UTF-8",
        }
    )
    return environment


def _require_docker(environment: dict[str, str]) -> str:
    """Fail closed when an opted-in run lacks Compose or a reachable daemon."""
    docker = shutil.which("docker", path=environment.get("PATH"))
    assert docker is not None, "Docker CLI is required for the opted-in proof"
    _run(
        [docker, "compose", "version", "--short"],
        environment=environment,
        timeout=30,
    )
    daemon = _run(
        [docker, "info", "--format", "{{.ServerVersion}}"],
        environment=environment,
        timeout=30,
    )
    assert daemon.stdout.strip(), "Docker returned no server version"
    return docker


def _validated_resource_prefix() -> str:
    """Return one collision-resistant, Docker-safe resource prefix."""
    configured_prefix = os.environ.get(
        PROJECT_PREFIX_ENVIRONMENT_NAME,
        f"adk-candidate-{os.getpid()}",
    ).lower()
    assert PROJECT_PREFIX_PATTERN.fullmatch(configured_prefix), (
        f"{PROJECT_PREFIX_ENVIRONMENT_NAME} is invalid"
    )
    prefix = f"{configured_prefix}-{uuid.uuid4().hex[:8]}"
    assert RESOURCE_NAME_PATTERN.fullmatch(prefix)
    return prefix


def _read_image_id(iid_file: Path) -> str:
    """Read and validate one Docker-built immutable image ID."""
    image_id = iid_file.read_text(encoding="ascii").strip()
    assert IMAGE_ID_PATTERN.fullmatch(image_id), "Docker returned an invalid image ID"
    return image_id


def _build_image(
    docker: str,
    environment: dict[str, str],
    *,
    context: Path,
    iid_file: Path,
    tag: str,
    dockerfile: Path | None = None,
    build_arguments: tuple[str, ...] = (),
) -> str:
    """Build one uniquely tagged local image and return its exact ID."""
    command = [
        docker,
        "build",
        "--iidfile",
        str(iid_file),
        "--tag",
        tag,
    ]
    if dockerfile is not None:
        command.extend(["--file", str(dockerfile)])
    for build_argument in build_arguments:
        command.extend(["--build-arg", build_argument])
    command.append(str(context))
    _run(command, environment=environment, timeout=600)
    try:
        image_id = _read_image_id(iid_file)
        inspected_id = _run(
            [docker, "image", "inspect", "--format", "{{.Id}}", tag],
            environment=environment,
            timeout=30,
        ).stdout.strip()
        assert inspected_id == image_id
        return image_id
    except BaseException as error:
        try:
            cleanup = _run(
                [docker, "image", "rm", tag],
                environment=environment,
                check=False,
                timeout=60,
            )
        except BaseException as cleanup_error:
            error.add_note(
                "failed to clean image after post-build validation: "
                f"{_redact(str(cleanup_error))}"
            )
        else:
            if cleanup.returncode != 0:
                error.add_note(
                    "failed to clean image after post-build validation: "
                    f"{_redact(cleanup.stderr[-2_000:])}"
                )
        raise


def _write_candidate_environment(
    environment: dict[str, str],
    private_directory: Path,
) -> Path:
    """Create the candidate env only through the supported serializer CLI."""
    env_file = private_directory / "candidate.env"
    serializer_environment = environment | {
        "AGENT_NAME": "candidate-runtime-agent",
        "ROOT_AGENT_MODEL": "gemini-2.5-flash",
        "LOG_LEVEL": "INFO",
        "TELEMETRY_NAMESPACE": "candidate-runtime",
        "K_REVISION": "candidate-runtime",
        "CANDIDATE_ENV_CANARY": HOSTILE_CANARY,
    }
    _run(
        [
            sys.executable,
            "-m",
            "agent.compose_env",
            str(env_file),
            *CANDIDATE_ENV_ALLOWLIST,
        ],
        environment=serializer_environment,
        timeout=30,
    )
    assert stat.S_IMODE(env_file.stat().st_mode) == 0o600
    return env_file


def _compose(
    harness: CandidateHarness,
    project: str,
    image_id: str,
    *arguments: str,
    check: bool = True,
    timeout: float = 180,
) -> subprocess.CompletedProcess[str]:
    """Run one standalone, exact-project candidate Compose operation."""
    harness.assert_owned_project(project)
    assert IMAGE_ID_PATTERN.fullmatch(image_id)
    environment = harness.environment | {
        "ENV_FILE": str(harness.env_file),
        "IMAGE": image_id,
    }
    return _run(
        [*harness.compose_prefix(project), *arguments],
        environment=environment,
        check=check,
        timeout=timeout,
    )


def _container_id(
    harness: CandidateHarness,
    project: str,
    image_id: str,
) -> str:
    """Resolve exactly one project-scoped candidate container."""
    result = _compose(
        harness,
        project,
        image_id,
        "ps",
        "--all",
        "--quiet",
        "agent",
        timeout=30,
    )
    container_id = result.stdout.strip()
    assert re.fullmatch(r"[0-9a-f]{12,64}", container_id), (
        "Compose did not return exactly one candidate container"
    )
    return container_id


def _inspect_container_field(
    harness: CandidateHarness,
    container_id: str,
    template: str,
) -> str:
    """Inspect one non-environment container field."""
    return _run(
        [
            harness.docker,
            "container",
            "inspect",
            "--format",
            template,
            container_id,
        ],
        environment=harness.environment,
        timeout=30,
    ).stdout.strip()


def _inspect_image_field(
    harness: CandidateHarness,
    image_id: str,
    template: str,
) -> str:
    """Inspect one immutable local image configuration field."""
    return _run(
        [
            harness.docker,
            "image",
            "inspect",
            "--format",
            template,
            image_id,
        ],
        environment=harness.environment,
        timeout=30,
    ).stdout.strip()


def _assert_runtime_isolation(
    harness: CandidateHarness,
    container_id: str,
    image_id: str,
    *,
    expected_command: tuple[str, ...],
    expected_health_status: str,
) -> None:
    """Prove the Docker daemon applied the tracked isolation boundary."""
    assert (
        _inspect_container_field(
            harness,
            container_id,
            "{{.Image}}",
        )
        == image_id
    )
    assert (
        _inspect_container_field(
            harness,
            container_id,
            "{{.Config.User}}",
        )
        == "1000:1000"
    )
    assert json.loads(
        _inspect_image_field(
            harness,
            image_id,
            "{{json .Config.Entrypoint}}",
        )
    ) == list(EXPECTED_ENTRYPOINT)
    assert json.loads(
        _inspect_image_field(
            harness,
            image_id,
            "{{json .Config.Cmd}}",
        )
    ) == list(expected_command)
    assert json.loads(
        _inspect_container_field(
            harness,
            container_id,
            "{{json .Config.Entrypoint}}",
        )
    ) == list(EXPECTED_ENTRYPOINT)
    assert json.loads(
        _inspect_container_field(
            harness,
            container_id,
            "{{json .Config.Cmd}}",
        )
    ) == list(expected_command)
    assert (
        _inspect_container_field(
            harness,
            container_id,
            "{{.HostConfig.NetworkMode}}",
        )
        == "none"
    )
    assert json.loads(
        _inspect_container_field(
            harness,
            container_id,
            "{{json .HostConfig.PortBindings}}",
        )
    ) in (None, {})
    assert (
        json.loads(
            _inspect_container_field(
                harness,
                container_id,
                "{{json .Mounts}}",
            )
        )
        == []
    )
    assert (
        _inspect_container_field(
            harness,
            container_id,
            "{{.HostConfig.RestartPolicy.Name}}",
        )
        == "no"
    )
    assert json.loads(
        _inspect_container_field(
            harness,
            container_id,
            "{{json .HostConfig.CapDrop}}",
        )
    ) == ["ALL"]
    assert json.loads(
        _inspect_container_field(
            harness,
            container_id,
            "{{json .HostConfig.SecurityOpt}}",
        )
    ) == ["no-new-privileges:true"]
    assert (
        _inspect_container_field(
            harness,
            container_id,
            "{{.State.Status}}",
        )
        == "running"
    )
    assert (
        _inspect_container_field(
            harness,
            container_id,
            "{{.State.Health.Status}}",
        )
        == expected_health_status
    )


def _assert_ready_and_canary(
    harness: CandidateHarness,
    project: str,
    image_id: str,
) -> None:
    """Prove local readiness and the serializer's runtime byte contract."""
    ready = _compose(
        harness,
        project,
        image_id,
        "exec",
        "-T",
        "agent",
        "python",
        "-c",
        READY_PROBE,
        timeout=30,
    )
    assert ready.stdout.strip() == "candidate-ready"

    expected_hash = hashlib.sha256(HOSTILE_CANARY.encode("utf-8")).hexdigest()
    actual_hash = _compose(
        harness,
        project,
        image_id,
        "exec",
        "-T",
        "agent",
        "python",
        "-c",
        CANARY_HASH_PROBE,
        timeout=30,
    )
    assert actual_hash.stdout.strip() == expected_hash
    assert HOSTILE_CANARY not in actual_hash.stdout
    assert HOSTILE_CANARY not in actual_hash.stderr


def _down_candidate(
    harness: CandidateHarness,
    project: str,
    image_id: str,
) -> subprocess.CompletedProcess[str]:
    """Remove only containers owned by one validated candidate project."""
    harness.assert_owned_project(project)
    return _compose(
        harness,
        project,
        image_id,
        "down",
        "--remove-orphans",
        "--timeout",
        "30",
        check=False,
        timeout=60,
    )


def _identity(
    docker: str,
    environment: dict[str, str],
    kind: str,
    name: str,
) -> str:
    """Return one stable identity without inspecting sensitive configuration."""
    templates = {
        "container": "{{.Id}}",
        "image": "{{.Id}}",
        "network": "{{.Id}}",
    }
    return _run(
        [docker, kind, "inspect", "--format", templates[kind], name],
        environment=environment,
        timeout=30,
    ).stdout.strip()


@dataclass(frozen=True)
class VolumeIdentity:
    """Stable sentinel volume metadata used to detect replacement."""

    name: str
    mountpoint: str
    created_at: str


def _volume_identity(
    docker: str,
    environment: dict[str, str],
    name: str,
) -> VolumeIdentity:
    """Capture identity fields that change if a named volume is replaced."""
    payload = json.loads(
        _run(
            [docker, "volume", "inspect", "--format", "{{json .}}", name],
            environment=environment,
            timeout=30,
        ).stdout
    )
    identity = VolumeIdentity(
        name=payload["Name"],
        mountpoint=payload["Mountpoint"],
        created_at=payload["CreatedAt"],
    )
    assert identity.name == name
    assert identity.mountpoint
    assert identity.created_at
    return identity


def test_require_docker_rejects_missing_cli(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An opted-in run must fail rather than silently skip without Docker."""
    monkeypatch.setattr(shutil, "which", lambda *_args, **_kwargs: None)

    with pytest.raises(AssertionError, match="Docker CLI is required"):
        _require_docker({"PATH": "/missing"})


@pytest.mark.parametrize(
    ("compose_exit", "daemon_exit"),
    ((17, 0), (0, 19)),
)
def test_require_docker_rejects_missing_compose_or_daemon(
    tmp_path: Path,
    compose_exit: int,
    daemon_exit: int,
) -> None:
    """Propagate either unavailable boundary as a hard opt-in failure."""
    docker = tmp_path / "docker"
    docker.write_text(
        "#!/bin/sh\n"
        "set -eu\n"
        'if [ "$1" = "compose" ]; then\n'
        '  [ "$FAKE_COMPOSE_EXIT" -eq 0 ] || exit "$FAKE_COMPOSE_EXIT"\n'
        "  printf '2.40.3\\n'\n"
        "  exit 0\n"
        "fi\n"
        'if [ "$1" = "info" ]; then\n'
        '  [ "$FAKE_DAEMON_EXIT" -eq 0 ] || exit "$FAKE_DAEMON_EXIT"\n'
        "  printf '29.1.3\\n'\n"
        "  exit 0\n"
        "fi\n"
        "exit 99\n",
        encoding="utf-8",
    )
    docker.chmod(0o755)
    environment = {
        "FAKE_COMPOSE_EXIT": str(compose_exit),
        "FAKE_DAEMON_EXIT": str(daemon_exit),
        "LANG": "C",
        "PATH": str(tmp_path),
    }

    with pytest.raises(AssertionError):
        _require_docker(environment)


@pytest.mark.parametrize(
    "representation",
    (
        HOSTILE_CANARY,
        HOSTILE_CANARY.replace("$", "$$"),
        json.dumps(HOSTILE_CANARY, ensure_ascii=False)[1:-1],
        json.dumps(HOSTILE_CANARY, ensure_ascii=True)[1:-1],
        repr(HOSTILE_CANARY)[1:-1],
        repr(HOSTILE_CANARY.encode("utf-8"))[2:-1],
    ),
)
def test_redact_covers_common_canary_serializations(representation: str) -> None:
    """Remove each expected shell, JSON, text-repr, and byte-repr form."""
    assert _redact(f"before:{representation}:after") == "before:[redacted]:after"


def test_run_redacts_timeout_byte_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Do not surface the canary when TimeoutExpired carries byte output."""
    canary_bytes = HOSTILE_CANARY.encode("utf-8")
    timeout_error = subprocess.TimeoutExpired(
        cmd=["docker", "info"],
        timeout=1,
        output=b"stdout:" + canary_bytes + b":end",
        stderr=b"stderr:" + canary_bytes + b":end",
    )
    run_mock = create_autospec(
        subprocess.run,
        spec_set=True,
        side_effect=timeout_error,
    )
    monkeypatch.setattr(subprocess, "run", run_mock)

    with pytest.raises(AssertionError, match="exceeded") as raised:
        _run(["docker", "info"], environment={}, timeout=1)

    diagnostic = str(raised.value)
    assert HOSTILE_CANARY not in diagnostic
    assert repr(canary_bytes)[2:-1] not in diagnostic
    assert diagnostic.count("[redacted]") == 2


def test_candidate_harness_rejects_unowned_projects(tmp_path: Path) -> None:
    """Require both exact prefix ownership and explicit project registration."""
    prefix = "adk-candidate-unit-12345678"
    healthy_project = f"{prefix}-healthy"
    unhealthy_project = f"{prefix}-unhealthy"
    harness = CandidateHarness(
        docker="/usr/bin/docker",
        environment={},
        env_file=tmp_path / "candidate.env",
        resource_prefix=prefix,
        owned_projects=frozenset({healthy_project, unhealthy_project}),
    )

    assert healthy_project in harness.compose_prefix(healthy_project)
    with pytest.raises(AssertionError):
        harness.compose_prefix(f"{prefix}-unregistered")
    with pytest.raises(AssertionError):
        harness.compose_prefix("adk-candidate-foreign-12345678-healthy")
    with pytest.raises(AssertionError):
        CandidateHarness(
            docker="/usr/bin/docker",
            environment={},
            env_file=tmp_path / "candidate.env",
            resource_prefix=prefix,
            owned_projects=frozenset({"adk-candidate-foreign-12345678-healthy"}),
        )


@pytest.mark.skipif(
    os.environ.get(RUN_ENVIRONMENT_NAME) != "1",
    reason="real candidate Docker proof is opt-in",
)
def test_candidate_runtime_is_isolated_and_fail_closed(tmp_path: Path) -> None:
    """Prove isolated startup, env fidelity, rejection, and scoped cleanup."""
    environment = _base_environment()
    docker = _require_docker(environment)
    prefix = _validated_resource_prefix()
    private_directory = tmp_path / "candidate-private"
    private_directory.mkdir(mode=0o700)
    private_directory.chmod(0o700)
    assert stat.S_IMODE(private_directory.stat().st_mode) == 0o700

    healthy_project = f"{prefix}-healthy"
    unhealthy_project = f"{prefix}-unhealthy"
    healthy_tag = f"candidate-runtime-healthy:{prefix}"
    unhealthy_tag = f"candidate-runtime-unhealthy:{prefix}"
    sentinel_image_tag = f"candidate-runtime-sentinel:{prefix}"
    sentinel_container = f"{prefix}-sentinel-container"
    sentinel_volume = f"{prefix}-sentinel-volume"
    sentinel_network = f"{prefix}-sentinel-network"

    harness: CandidateHarness | None = None
    cleanup_failures: list[str] = []
    active_projects: dict[str, str] = {}
    created_resources: list[list[str]] = []
    primary_error: BaseException | None = None
    try:
        assert CANDIDATE_COMPOSE_PATH.is_file()
        healthy_iid_file = private_directory / "healthy.iid"
        healthy_image_id = _build_image(
            docker,
            environment,
            context=REPOSITORY_ROOT,
            iid_file=healthy_iid_file,
            tag=healthy_tag,
        )
        created_resources.append([docker, "image", "rm", healthy_tag])
        env_file = _write_candidate_environment(environment, private_directory)
        harness = CandidateHarness(
            docker=docker,
            environment=environment,
            env_file=env_file,
            resource_prefix=prefix,
            owned_projects=frozenset({healthy_project, unhealthy_project}),
        )

        sentinel_context = private_directory / "sentinel-context"
        sentinel_context.mkdir(mode=0o700)
        sentinel_dockerfile = sentinel_context / "Dockerfile"
        sentinel_dockerfile.write_text(
            f'FROM scratch\nLABEL candidate.runtime.sentinel="{prefix}"\n',
            encoding="utf-8",
        )
        sentinel_image_id = _build_image(
            docker,
            environment,
            context=sentinel_context,
            iid_file=private_directory / "sentinel.iid",
            tag=sentinel_image_tag,
        )
        created_resources.append([docker, "image", "rm", sentinel_image_tag])
        sentinel_container_create = _run(
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
                healthy_image_id,
            ],
            environment=environment,
            timeout=30,
        )
        created_resources.append([docker, "container", "rm", sentinel_container])
        sentinel_container_id = sentinel_container_create.stdout.strip()
        assert re.fullmatch(r"[0-9a-f]{12,64}", sentinel_container_id)
        _run(
            [docker, "container", "start", "--attach", sentinel_container],
            environment=environment,
            timeout=30,
        )
        assert (
            _run(
                [
                    docker,
                    "container",
                    "inspect",
                    "--format",
                    "{{.State.Status}}",
                    sentinel_container,
                ],
                environment=environment,
                timeout=30,
            ).stdout.strip()
            == "exited"
        )

        created_volume_result = _run(
            [docker, "volume", "create", sentinel_volume],
            environment=environment,
            timeout=30,
        )
        created_resources.append([docker, "volume", "rm", sentinel_volume])
        created_volume = created_volume_result.stdout.strip()
        assert created_volume == sentinel_volume
        sentinel_volume_identity = _volume_identity(
            docker,
            environment,
            sentinel_volume,
        )
        sentinel_network_create = _run(
            [docker, "network", "create", sentinel_network],
            environment=environment,
            timeout=30,
        )
        created_resources.append([docker, "network", "rm", sentinel_network])
        sentinel_network_id = sentinel_network_create.stdout.strip()
        assert re.fullmatch(r"[0-9a-f]{12,64}", sentinel_network_id)

        unhealthy_context = private_directory / "unhealthy-context"
        unhealthy_context.mkdir(mode=0o700)
        unhealthy_dockerfile = unhealthy_context / "Dockerfile"
        unhealthy_dockerfile.write_text(
            "ARG BASE_IMAGE\n"
            "FROM ${BASE_IMAGE}\n"
            f"CMD {json.dumps(UNHEALTHY_COMMAND)}\n",
            encoding="utf-8",
        )
        unhealthy_image_id = _build_image(
            docker,
            environment,
            context=unhealthy_context,
            dockerfile=unhealthy_dockerfile,
            iid_file=private_directory / "unhealthy.iid",
            tag=unhealthy_tag,
            build_arguments=(f"BASE_IMAGE={healthy_image_id}",),
        )
        created_resources.append([docker, "image", "rm", unhealthy_tag])

        active_projects.update(
            {
                healthy_project: healthy_image_id,
                unhealthy_project: unhealthy_image_id,
            }
        )
        _compose(
            harness,
            healthy_project,
            healthy_image_id,
            "config",
            "--quiet",
            timeout=30,
        )
        _compose(
            harness,
            healthy_project,
            healthy_image_id,
            "up",
            "--detach",
            "--no-build",
            "--pull",
            "never",
            "--wait",
            "--wait-timeout",
            "120",
            "agent",
            timeout=150,
        )
        healthy_container_id = _container_id(
            harness,
            healthy_project,
            healthy_image_id,
        )
        _assert_runtime_isolation(
            harness,
            healthy_container_id,
            healthy_image_id,
            expected_command=HEALTHY_COMMAND,
            expected_health_status="healthy",
        )
        _assert_ready_and_canary(
            harness,
            healthy_project,
            healthy_image_id,
        )
        healthy_down = _down_candidate(
            harness,
            healthy_project,
            healthy_image_id,
        )
        assert healthy_down.returncode == 0, _redact(healthy_down.stderr[-2_000:])
        active_projects.pop(healthy_project)

        started = time.monotonic()
        unhealthy_up = _compose(
            harness,
            unhealthy_project,
            unhealthy_image_id,
            "up",
            "--detach",
            "--no-build",
            "--pull",
            "never",
            "--wait",
            "--wait-timeout",
            "45",
            "agent",
            check=False,
            timeout=60,
        )
        elapsed = time.monotonic() - started
        assert unhealthy_up.returncode != 0, (
            "The intentionally non-ready candidate unexpectedly passed"
        )
        assert elapsed < 60, "The intentionally non-ready candidate failed too slowly"
        unhealthy_container_id = _container_id(
            harness,
            unhealthy_project,
            unhealthy_image_id,
        )
        _assert_runtime_isolation(
            harness,
            unhealthy_container_id,
            unhealthy_image_id,
            expected_command=UNHEALTHY_COMMAND,
            expected_health_status="unhealthy",
        )
        unhealthy_down = _down_candidate(
            harness,
            unhealthy_project,
            unhealthy_image_id,
        )
        assert unhealthy_down.returncode == 0, _redact(unhealthy_down.stderr[-2_000:])
        active_projects.pop(unhealthy_project)

        assert (
            _identity(
                docker,
                environment,
                "container",
                sentinel_container,
            )
            == sentinel_container_id
        )
        assert (
            _identity(
                docker,
                environment,
                "image",
                sentinel_image_tag,
            )
            == sentinel_image_id
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
            _identity(
                docker,
                environment,
                "network",
                sentinel_network,
            )
            == sentinel_network_id
        )
    except BaseException as error:
        primary_error = error
        raise
    finally:
        if harness is not None:
            for project, image_id in active_projects.items():
                try:
                    result = _down_candidate(harness, project, image_id)
                except BaseException as cleanup_error:
                    cleanup_failures.append(
                        "candidate project cleanup failed: "
                        f"{_redact(str(cleanup_error))}"
                    )
                else:
                    if result.returncode != 0:
                        cleanup_failures.append(
                            "candidate project cleanup failed: "
                            f"{_redact(result.stderr[-2_000:])}"
                        )

        for command in reversed(created_resources):
            try:
                result = _run(
                    command,
                    environment=environment,
                    check=False,
                    timeout=60,
                )
            except BaseException as cleanup_error:
                cleanup_failures.append(
                    f"exact resource cleanup failed: {_redact(str(cleanup_error))}"
                )
            else:
                if result.returncode != 0:
                    cleanup_failures.append(
                        "exact resource cleanup failed: "
                        f"{_redact(result.stderr[-2_000:])}"
                    )

        if cleanup_failures:
            cleanup_message = "\n".join(cleanup_failures)
            if primary_error is not None:
                primary_error.add_note(cleanup_message)
            else:
                raise AssertionError(cleanup_message)
