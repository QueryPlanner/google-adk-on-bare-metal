"""Opt-in real-Docker proof for atomic VM promotion and verified rollback."""

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
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from urllib.error import URLError
from urllib.request import ProxyHandler, build_opener

import pytest

from agent.compose_env import (
    serialize_compose_environment,
    write_compose_environment,
)
from agent.deployment_promotion import PRODUCTION_ENVIRONMENT_NAMES
from agent.deployment_state import DeploymentStateStore

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CANDIDATE_COMPOSE_PATH = REPOSITORY_ROOT / "compose.candidate.yaml"
PROMOTION_MODULE_PATH = REPOSITORY_ROOT / "src" / "agent" / "deployment_promotion.py"
RUN_ENVIRONMENT_NAME = "RUN_DEPLOYMENT_PROMOTION_INTEGRATION"
PREFIX_ENVIRONMENT_NAME = "DEPLOYMENT_PROMOTION_TEST_PREFIX"
PREFIX_PATTERN = re.compile(r"adk-promotion-[a-z0-9][a-z0-9-]{0,23}\Z")
REVISION_PATTERN = re.compile(r"[0-9a-f]{40}\Z")
IMAGE_ID_PATTERN = re.compile(r"sha256:[0-9a-f]{64}\Z")
IMAGE_REFERENCE_PATTERN = re.compile(
    r"127\.0\.0\.1:[0-9]+/[a-z0-9-]+/agent@sha256:[0-9a-f]{64}\Z"
)
CONTAINER_ID_PATTERN = re.compile(r"[0-9a-f]{64}\Z")
EXPECTED_ORIGIN = "https://github.com/QueryPlanner/google-adk-on-bare-metal"
REGISTRY_IMAGE_REFERENCE = (
    "registry@sha256:1be55279f18a2fe1a74edf2664cac61c1bea305b7b4642dab412e7affdcb3e33"
)
PRIVATE_CANARIES = (
    'promotion-old-$-हॅलो-"-\\-canary',
    'promotion-good-$-हॅलो-"-\\-canary',
    'promotion-failing-$-हॅलो-"-\\-canary',
)
VOLUME_SENTINEL = "atomic-promotion-volume-sentinel"
PRODUCTION_FAILURE_SENTINEL = "production-only-failure-reached"

PRODUCTION_COMPOSE = """\
services:
  agent:
    image: "${IMAGE:?Set IMAGE to an immutable production image}"
    pull_policy: never
    env_file:
      - "${ENV_FILE:?Set ENV_FILE to the production environment}"
    environment:
      AGENT_DIR: "/app/src"
      HOST: "0.0.0.0"
      PORT: "8080"
      ALLOW_ORIGINS: "[]"
      SERVE_WEB_INTERFACE: "false"
      RELOAD_AGENTS: "false"
      ADK_DISABLE_LOAD_DOTENV: "true"
      OTEL_SDK_DISABLED: "true"
      OTEL_TRACES_EXPORTER: "none"
      OTEL_METRICS_EXPORTER: "none"
      OTEL_LOGS_EXPORTER: "none"
      PROMOTION_TEST_FAIL: "__PROMOTION_TEST_FAILURE__"
    volumes:
      - agent_artifacts:/app/src/.adk
    restart: "no"
    healthcheck:
      test:
        - CMD
        - python
        - -c
        - >-
          import urllib.request;
          urllib.request.build_opener(
          urllib.request.ProxyHandler({})).open(
          "http://127.0.0.1:8080/ready", timeout=1).read()
      interval: 2s
      timeout: 2s
      retries: 10
      start_period: 20s
    stop_grace_period: 10s
    command: ["python", "-m", "agent.server"]

volumes:
  agent_artifacts:
"""

DERIVATIVE_DOCKERFILE = """\
ARG BASE_IMAGE
FROM ${BASE_IMAGE}
ARG SOURCE_REVISION
USER root
COPY promotion-wrapper.sh /usr/local/bin/promotion-wrapper
RUN chmod 0755 /usr/local/bin/promotion-wrapper
LABEL org.opencontainers.image.revision="${SOURCE_REVISION}"
USER app
ENTRYPOINT ["/usr/local/bin/promotion-wrapper"]
"""

PROMOTION_WRAPPER = """\
#!/bin/sh
set -eu
if [ "${PROMOTION_TEST_FAIL:-0}" = "1" ] \\
  && [ "${TELEMETRY_NAMESPACE:-production}" != "candidate" ]; then
  printf %s production-only-failure-reached \\
    > /app/src/.adk/promotion-failure-sentinel
  exit 42
fi
exec /app/entrypoint.sh "$@"
"""


def _redact(value: str) -> str:
    redacted = value
    for secret in PRIVATE_CANARIES:
        for candidate in (
            secret,
            secret.replace("$", "$$"),
            json.dumps(secret, ensure_ascii=False)[1:-1],
            json.dumps(secret, ensure_ascii=True)[1:-1],
            repr(secret),
            repr(secret.encode()),
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
        result = subprocess.run(  # noqa: S603 - fixed/resolved test boundaries
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
    cleanup_command: list[str],
    operation: Callable[[], MutationResult],
) -> MutationResult:
    cleanup_commands.append(cleanup_command)
    return operation()


def _execute_exact_cleanup(
    cleanup_commands: Sequence[Sequence[str]],
    environment: Mapping[str, str],
) -> list[str]:
    failures: list[str] = []
    for command in reversed(cleanup_commands):
        if command[-2:] == ["image", "rm"]:
            continue
        try:
            result = _run(
                command,
                environment=environment,
                check=False,
                timeout=90,
            )
        except (AssertionError, OSError) as error:
            failures.append(_redact(str(error)[-1_000:]))
            continue
        if result.returncode != 0 and not any(
            marker in result.stderr
            for marker in ("No such", "not found", "does not exist")
        ):
            failures.append(_redact(result.stderr[-1_000:]))
    return failures


def _report_cleanup_failures(
    failures: Sequence[str],
    primary_error: BaseException | None,
) -> None:
    if not failures:
        return
    message = "deployment-promotion cleanup failed: " + " | ".join(failures)
    if primary_error is None:
        raise AssertionError(message)
    primary_error.add_note(message)


def _base_environment() -> dict[str, str]:
    environment = {
        key: os.environ[key]
        for key in ("HOME", "PATH", "DOCKER_CONFIG", "DOCKER_HOST", "XDG_RUNTIME_DIR")
        if key in os.environ
    }
    environment.update(
        {
            "COMPOSE_DISABLE_ENV_FILE": "1",
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
            "PYTHONDONTWRITEBYTECODE": "1",
        }
    )
    return environment


def _production_environment(
    base: Mapping[str, str],
    *,
    canary: str,
    log_level: str,
) -> dict[str, str]:
    environment = dict(base)
    environment.update(
        {
            "AGENT_NAME": "deployment-promotion-runtime",
            "DATABASE_URL": "",
            "OPENROUTER_API_KEY": canary,
            "GOOGLE_API_KEY": "",
            "ROOT_AGENT_MODEL": "gemini-2.5-flash",
            "LANGFUSE_PUBLIC_KEY": "",
            "LANGFUSE_SECRET_KEY": "",
            "LANGFUSE_BASE_URL": "",
            "LOG_LEVEL": log_level,
            "PORT": "8080",
            "HOST": "0.0.0.0",  # noqa: S104 - synthetic container bind
        }
    )
    assert all(name in environment for name in PRODUCTION_ENVIRONMENT_NAMES)
    return environment


def _require_boundaries(environment: Mapping[str, str]) -> tuple[str, str]:
    docker = shutil.which("docker", path=environment.get("PATH"))
    git = shutil.which("git", path=environment.get("PATH"))
    if docker is None or git is None:
        raise AssertionError("Docker and Git executables are required")
    _run([docker, "version"], environment=environment, timeout=30)
    _run([docker, "compose", "version"], environment=environment, timeout=30)
    return docker, git


def _resource_prefix() -> str:
    configured = os.environ.get(
        PREFIX_ENVIRONMENT_NAME,
        f"adk-promotion-{os.getpid()}",
    ).lower()
    if PREFIX_PATTERN.fullmatch(configured) is None:
        raise AssertionError("deployment-promotion Docker prefix is invalid")
    return f"{configured}-{uuid.uuid4().hex[:8]}"


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


def _commit_fixture(
    git: str,
    environment: Mapping[str, str],
    checkout: Path,
    *,
    phase: str,
    failure: bool,
) -> str:
    (checkout / "compose.yaml").write_text(
        PRODUCTION_COMPOSE.replace(
            "__PROMOTION_TEST_FAILURE__",
            "1" if failure else "0",
        ),
        encoding="utf-8",
    )
    (checkout / "release.txt").write_text(f"{phase}\n", encoding="ascii")
    _git(git, environment, "add", ".", cwd=checkout)
    _git(
        git,
        environment,
        "commit",
        "-m",
        f"fixture: {phase}",
        cwd=checkout,
    )
    revision = _git(git, environment, "rev-parse", "HEAD", cwd=checkout)
    assert REVISION_PATTERN.fullmatch(revision)
    return revision


def _initialize_repository(
    git: str,
    environment: Mapping[str, str],
    checkout: Path,
) -> tuple[str, str, str]:
    checkout.mkdir(mode=0o700)
    (checkout / "src" / "agent").mkdir(parents=True)
    shutil.copy2(CANDIDATE_COMPOSE_PATH, checkout / "compose.candidate.yaml")
    shutil.copy2(
        PROMOTION_MODULE_PATH,
        checkout / "src" / "agent" / "deployment_promotion.py",
    )
    (checkout / ".gitignore").write_text(".env\n", encoding="ascii")
    _git(git, environment, "init", cwd=checkout)
    _git(
        git,
        environment,
        "config",
        "user.email",
        "runtime@example.invalid",
        cwd=checkout,
    )
    _git(
        git,
        environment,
        "config",
        "user.name",
        "Runtime Contract",
        cwd=checkout,
    )
    _git(
        git,
        environment,
        "remote",
        "add",
        "origin",
        EXPECTED_ORIGIN,
        cwd=checkout,
    )
    old_revision = _commit_fixture(
        git,
        environment,
        checkout,
        phase="old",
        failure=False,
    )
    good_revision = _commit_fixture(
        git,
        environment,
        checkout,
        phase="good",
        failure=False,
    )
    failing_revision = _commit_fixture(
        git,
        environment,
        checkout,
        phase="failing",
        failure=True,
    )
    _git(
        git,
        environment,
        "-c",
        "advice.detachedHead=false",
        "checkout",
        "--detach",
        old_revision,
        cwd=checkout,
    )
    return old_revision, good_revision, failing_revision


def _add_release_worktree(
    git: str,
    environment: Mapping[str, str],
    checkout: Path,
    release_checkout: Path,
    revision: str,
) -> None:
    _git(
        git,
        environment,
        "worktree",
        "add",
        "--detach",
        str(release_checkout),
        revision,
        cwd=checkout,
    )
    assert (
        _git(git, environment, "rev-parse", "--verify", "HEAD", cwd=release_checkout)
        == revision
    )


def _build_base_image(
    docker: str,
    environment: Mapping[str, str],
    *,
    tag: str,
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
            str(REPOSITORY_ROOT),
        ],
        environment=environment,
        timeout=900,
    )
    image_id = iid_file.read_text(encoding="ascii").strip()
    assert IMAGE_ID_PATTERN.fullmatch(image_id)
    return image_id


def _write_derivative_context(path: Path) -> None:
    path.mkdir(mode=0o700)
    (path / "Dockerfile").write_text(DERIVATIVE_DOCKERFILE, encoding="utf-8")
    wrapper = path / "promotion-wrapper.sh"
    wrapper.write_text(PROMOTION_WRAPPER, encoding="utf-8")
    wrapper.chmod(0o755)


def _build_release_image(
    docker: str,
    environment: Mapping[str, str],
    *,
    context: Path,
    base_tag: str,
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
            "--build-arg",
            f"BASE_IMAGE={base_tag}",
            "--build-arg",
            f"SOURCE_REVISION={revision}",
            str(context),
        ],
        environment=environment,
        timeout=180,
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


def _image_identity(
    docker: str,
    environment: Mapping[str, str],
    image: str,
) -> dict[str, object]:
    documents = json.loads(
        _run(
            [docker, "image", "inspect", image],
            environment=environment,
            timeout=30,
        ).stdout
    )
    assert isinstance(documents, list) and len(documents) == 1
    document = documents[0]
    assert isinstance(document, dict)
    return document


def _ensure_registry_image(
    docker: str,
    environment: Mapping[str, str],
    cleanup_commands: list[list[str]],
) -> str:
    existing = _run(
        [docker, "image", "inspect", REGISTRY_IMAGE_REFERENCE],
        environment=environment,
        check=False,
        timeout=30,
    )
    if existing.returncode != 0:
        _owned_mutation(
            cleanup_commands,
            [docker, "image", "rm", REGISTRY_IMAGE_REFERENCE],
            lambda: _run(
                [docker, "image", "pull", REGISTRY_IMAGE_REFERENCE],
                environment=environment,
                timeout=180,
            ),
        )
    document = _image_identity(
        docker,
        environment,
        REGISTRY_IMAGE_REFERENCE,
    )
    image_id = document.get("Id")
    assert isinstance(image_id, str)
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
            image_id,
        ],
        environment=environment,
        timeout=30,
    )
    _run(
        [docker, "container", "start", name],
        environment=environment,
        timeout=30,
    )


def _start_registry(
    docker: str,
    environment: Mapping[str, str],
    *,
    name: str,
) -> str:
    endpoint = _run(
        [docker, "container", "port", name, "5000/tcp"],
        environment=environment,
        timeout=30,
    ).stdout.strip()
    match = re.fullmatch(r"127\.0\.0\.1:([0-9]+)", endpoint)
    assert match is not None
    opener = build_opener(ProxyHandler({}))
    deadline = time.monotonic() + 30
    url = f"http://{endpoint}/v2/"
    while True:
        try:
            with opener.open(url, timeout=2) as response:
                assert response.status == 200
            return endpoint
        except URLError:
            if time.monotonic() >= deadline:
                raise AssertionError(
                    "private test registry did not become ready"
                ) from None
            time.sleep(0.25)


def _push_exact_image(
    docker: str,
    environment: Mapping[str, str],
    cleanup_commands: list[list[str]],
    *,
    source_image_id: str,
    repository_tag: str,
) -> str:
    _owned_mutation(
        cleanup_commands,
        [docker, "image", "rm", repository_tag],
        lambda: _run(
            [docker, "image", "tag", source_image_id, repository_tag],
            environment=environment,
            timeout=30,
        ),
    )
    exact_cleanup = [docker, "image", "rm"]
    cleanup_commands.append(exact_cleanup)
    _run(
        [docker, "image", "push", repository_tag],
        environment=environment,
        timeout=180,
    )
    document = _image_identity(docker, environment, repository_tag)
    repo_digests = document.get("RepoDigests")
    assert isinstance(repo_digests, list)
    repository = repository_tag.rsplit(":", maxsplit=1)[0]
    matching = [
        value
        for value in repo_digests
        if isinstance(value, str) and value.startswith(f"{repository}@sha256:")
    ]
    assert len(matching) == 1
    exact_reference = matching[0]
    assert IMAGE_REFERENCE_PATTERN.fullmatch(exact_reference)
    exact_cleanup.append(exact_reference)
    return exact_reference


def _compose(
    docker: str,
    environment: Mapping[str, str],
    *,
    checkout: Path,
    project: str,
    env_file: Path,
    image_reference: str,
    arguments: Sequence[str],
    timeout: float,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    if not project.startswith("adk-promotion-"):
        raise AssertionError("refusing to mutate an unowned Compose project")
    command_environment = dict(environment)
    command_environment.update(
        {
            "ENV_FILE": str(env_file),
            "IMAGE": image_reference,
        }
    )
    prefix = [
        docker,
        "compose",
        "--project-name",
        project,
        "--env-file",
        str(env_file),
        "-f",
        str(checkout / "compose.yaml"),
    ]
    configured = _run(
        [*prefix, "config", "--images"],
        environment=command_environment,
        cwd=checkout,
        timeout=30,
    ).stdout.splitlines()
    assert configured == [image_reference]
    return _run(
        [*prefix, *arguments],
        environment=command_environment,
        cwd=checkout,
        check=check,
        timeout=timeout,
    )


def _production_cleanup(
    docker: str,
    environment: Mapping[str, str],
    *,
    checkout: Path,
    project: str,
    env_file: Path,
    image_reference: str,
) -> list[str]:
    try:
        result = _compose(
            docker,
            environment,
            checkout=checkout,
            project=project,
            env_file=env_file,
            image_reference=image_reference,
            arguments=(
                "down",
                "--volumes",
                "--remove-orphans",
                "--timeout",
                "30",
            ),
            timeout=90,
            check=False,
        )
    except (AssertionError, OSError) as error:
        return [_redact(str(error)[-1_000:])]
    if result.returncode != 0:
        return [_redact(result.stderr[-1_000:])]
    return []


def _promotion_cli(
    environment: Mapping[str, str],
    *,
    state_directory: Path,
    checkout: Path,
    release_checkout: Path,
    project: str,
    revision: str,
    image_reference: str,
    adopt_existing: bool,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    command = [
        sys.executable,
        "-m",
        "agent.deployment_promotion",
        "promote",
        "--state-dir",
        str(state_directory),
        "--checkout",
        str(checkout),
        "--release-checkout",
        str(release_checkout),
        "--expected-origin",
        EXPECTED_ORIGIN,
        "--compose-project",
        project,
        "--compose-service",
        "agent",
        "--source-revision",
        revision,
        "--image-reference",
        image_reference,
    ]
    if adopt_existing:
        command.append("--adopt-existing")
    return _run(
        command,
        environment=environment,
        check=check,
        timeout=360,
    )


def _container_document(
    docker: str,
    environment: Mapping[str, str],
    *,
    project: str,
) -> dict[str, object]:
    short_id = _run(
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
    ).stdout.splitlines()
    assert len(short_id) == 1
    documents = json.loads(
        _run(
            [docker, "container", "inspect", short_id[0]],
            environment=environment,
            timeout=30,
        ).stdout
    )
    assert isinstance(documents, list) and len(documents) == 1
    document = documents[0]
    assert isinstance(document, dict)
    return document


def _production_volume_identity(
    docker: str,
    environment: Mapping[str, str],
    container: Mapping[str, object],
) -> dict[str, str]:
    mounts = container.get("Mounts")
    assert isinstance(mounts, list)
    volumes = [
        mount
        for mount in mounts
        if isinstance(mount, dict) and mount.get("Type") == "volume"
    ]
    assert len(volumes) == 1
    mount = volumes[0]
    name = mount.get("Name")
    driver = mount.get("Driver")
    destination = mount.get("Destination")
    assert all(isinstance(value, str) for value in (name, driver, destination))
    documents = json.loads(
        _run(
            [docker, "volume", "inspect", str(name)],
            environment=environment,
            timeout=30,
        ).stdout
    )
    assert isinstance(documents, list) and len(documents) == 1
    volume = documents[0]
    assert isinstance(volume, dict)
    identity = {
        "name": str(name),
        "driver": str(driver),
        "destination": str(destination),
        "mountpoint": str(volume.get("Mountpoint")),
        "created_at": str(volume.get("CreatedAt")),
    }
    assert identity["name"] == volume.get("Name")
    assert identity["driver"] == volume.get("Driver")
    assert identity["destination"].startswith("/")
    assert identity["mountpoint"].startswith("/")
    datetime.fromisoformat(identity["created_at"].replace("Z", "+00:00")).astimezone(
        UTC
    )
    return identity


def _write_volume_sentinel(
    docker: str,
    environment: Mapping[str, str],
    container_id: str,
) -> None:
    _run(
        [
            docker,
            "container",
            "exec",
            container_id,
            "sh",
            "-c",
            f"printf %s {VOLUME_SENTINEL} > /app/src/.adk/promotion-sentinel",
        ],
        environment=environment,
        timeout=30,
    )


def _read_volume_sentinel(
    docker: str,
    environment: Mapping[str, str],
    container_id: str,
) -> str:
    return _run(
        [
            docker,
            "container",
            "exec",
            container_id,
            "cat",
            "/app/src/.adk/promotion-sentinel",
        ],
        environment=environment,
        timeout=30,
    ).stdout


def _read_production_failure_sentinel(
    docker: str,
    environment: Mapping[str, str],
    container_id: str,
) -> str:
    return _run(
        [
            docker,
            "container",
            "exec",
            container_id,
            "cat",
            "/app/src/.adk/promotion-failure-sentinel",
        ],
        environment=environment,
        timeout=30,
    ).stdout


def _container_identity(
    docker: str,
    environment: Mapping[str, str],
    container: str,
) -> tuple[str, str, str]:
    document = json.loads(
        _run(
            [docker, "container", "inspect", container],
            environment=environment,
            timeout=30,
        ).stdout
    )[0]
    return document["Id"], document["State"]["Status"], document["Image"]


def _volume_identity(
    docker: str,
    environment: Mapping[str, str],
    volume: str,
) -> dict[str, object]:
    documents = json.loads(
        _run(
            [docker, "volume", "inspect", volume],
            environment=environment,
            timeout=30,
        ).stdout
    )
    assert isinstance(documents, list) and len(documents) == 1
    document = documents[0]
    assert isinstance(document, dict)
    assert all(isinstance(key, str) for key in document)
    return {str(key): value for key, value in document.items()}


def _network_identity(
    docker: str,
    environment: Mapping[str, str],
    network: str,
) -> str:
    return _run(
        [docker, "network", "inspect", "--format", "{{.Id}}", network],
        environment=environment,
        timeout=30,
    ).stdout.strip()


def _assert_safe_output(result: subprocess.CompletedProcess[str]) -> None:
    for canary in PRIVATE_CANARIES:
        assert canary not in result.stdout
        assert canary not in result.stderr


@pytest.mark.skipif(
    os.environ.get(RUN_ENVIRONMENT_NAME) != "1",
    reason="real deployment-promotion Docker proof is opt-in",
)
def test_real_docker_promotes_then_restores_exact_verified_baseline(
    tmp_path: Path,
) -> None:
    """Promote one exact digest, then prove automatic full rollback."""
    base_environment = _base_environment()
    docker, git = _require_boundaries(base_environment)
    prefix = _resource_prefix()
    production_project = f"{prefix}-prod"
    registry_name = f"{prefix}-registry"
    repository_name = f"{prefix}/agent"
    base_tag = f"{prefix}-base:runtime"
    old_tag = f"{prefix}-old:runtime"
    good_tag = f"{prefix}-good:runtime"
    failing_tag = f"{prefix}-failing:runtime"
    sentinel_image = f"{prefix}-sentinel-image:keep"
    sentinel_container = f"{prefix}-sentinel-container"
    sentinel_volume = f"{prefix}-sentinel-volume"
    sentinel_network = f"{prefix}-sentinel-network"
    checkout = tmp_path / "production"
    good_release_checkout = tmp_path / "release-good"
    failing_release_checkout = tmp_path / "release-failing"
    state_directory = tmp_path / "deployment-state"
    derivative_context = tmp_path / "derivative"
    env_file = checkout / ".env"
    cleanup_env_file = tmp_path / "cleanup.env"

    cleanup_commands: list[list[str]] = []
    production_cleanup_armed = False
    cleanup_image_reference = old_tag
    primary_error: BaseException | None = None
    try:
        old_revision, good_revision, failing_revision = _initialize_repository(
            git,
            base_environment,
            checkout,
        )
        _add_release_worktree(
            git,
            base_environment,
            checkout,
            good_release_checkout,
            good_revision,
        )
        _add_release_worktree(
            git,
            base_environment,
            checkout,
            failing_release_checkout,
            failing_revision,
        )
        _write_derivative_context(derivative_context)

        _owned_mutation(
            cleanup_commands,
            [docker, "image", "rm", base_tag],
            lambda: _build_base_image(
                docker,
                base_environment,
                tag=base_tag,
                iid_file=tmp_path / "base.iid",
            ),
        )
        release_images: dict[str, str] = {}
        for phase, tag, revision in (
            ("old", old_tag, old_revision),
            ("good", good_tag, good_revision),
            ("failing", failing_tag, failing_revision),
        ):
            release_images[phase] = _owned_mutation(
                cleanup_commands,
                [docker, "image", "rm", tag],
                partial(
                    _build_release_image,
                    docker,
                    base_environment,
                    context=derivative_context,
                    base_tag=base_tag,
                    tag=tag,
                    revision=revision,
                    iid_file=tmp_path / f"{phase}.iid",
                ),
            )

        registry_image_id = _ensure_registry_image(
            docker,
            base_environment,
            cleanup_commands,
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
                base_environment,
                name=registry_name,
                image_id=registry_image_id,
            ),
        )
        registry_endpoint = _start_registry(
            docker,
            base_environment,
            name=registry_name,
        )

        references: dict[str, str] = {}
        for phase in ("old", "good", "failing"):
            references[phase] = _push_exact_image(
                docker,
                base_environment,
                cleanup_commands,
                source_image_id=release_images[phase],
                repository_tag=(f"{registry_endpoint}/{repository_name}:{phase}"),
            )

        _owned_mutation(
            cleanup_commands,
            [docker, "image", "rm", sentinel_image],
            lambda: _run(
                [docker, "image", "tag", registry_image_id, sentinel_image],
                environment=base_environment,
                timeout=30,
            ),
        )
        sentinel_image_identity = _image_identity(
            docker,
            base_environment,
            sentinel_image,
        )
        _owned_mutation(
            cleanup_commands,
            [
                docker,
                "container",
                "rm",
                "--force",
                "--volumes",
                sentinel_container,
            ],
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
                    registry_image_id,
                ],
                environment=base_environment,
                timeout=30,
            ),
        )
        _run(
            [docker, "container", "start", "--attach", sentinel_container],
            environment=base_environment,
            timeout=30,
        )
        sentinel_container_identity = _container_identity(
            docker,
            base_environment,
            sentinel_container,
        )
        _owned_mutation(
            cleanup_commands,
            [docker, "volume", "rm", sentinel_volume],
            lambda: _run(
                [docker, "volume", "create", sentinel_volume],
                environment=base_environment,
                timeout=30,
            ),
        )
        sentinel_volume_identity = _volume_identity(
            docker,
            base_environment,
            sentinel_volume,
        )
        _owned_mutation(
            cleanup_commands,
            [docker, "network", "rm", sentinel_network],
            lambda: _run(
                [docker, "network", "create", sentinel_network],
                environment=base_environment,
                timeout=30,
            ),
        )
        sentinel_network_identity = _network_identity(
            docker,
            base_environment,
            sentinel_network,
        )

        old_environment = _production_environment(
            base_environment,
            canary=PRIVATE_CANARIES[0],
            log_level="INFO",
        )
        write_compose_environment(
            env_file,
            PRODUCTION_ENVIRONMENT_NAMES,
            old_environment,
        )
        write_compose_environment(
            cleanup_env_file,
            PRODUCTION_ENVIRONMENT_NAMES,
            old_environment,
        )
        assert stat.S_IMODE(env_file.stat().st_mode) == 0o600
        cleanup_image_reference = references["old"]
        production_cleanup_armed = True
        _compose(
            docker,
            base_environment,
            checkout=checkout,
            project=production_project,
            env_file=env_file,
            image_reference=references["old"],
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
        old_container = _container_document(
            docker,
            base_environment,
            project=production_project,
        )
        old_container_id = old_container["Id"]
        old_container_image_id = old_container["Image"]
        old_container_config = old_container["Config"]
        old_container_state = old_container["State"]
        assert isinstance(old_container_id, str)
        assert isinstance(old_container_image_id, str)
        assert isinstance(old_container_config, dict)
        assert isinstance(old_container_state, dict)
        assert CONTAINER_ID_PATTERN.fullmatch(old_container_id)
        assert old_container_config["Image"] == references["old"]
        assert old_container_image_id == release_images["old"]
        assert old_container_state["Status"] == "running"
        assert old_container_state["Health"]["Status"] == "healthy"
        old_volume_identity = _production_volume_identity(
            docker,
            base_environment,
            old_container,
        )
        _write_volume_sentinel(
            docker,
            base_environment,
            old_container_id,
        )

        good_environment = _production_environment(
            base_environment,
            canary=PRIVATE_CANARIES[1],
            log_level="DEBUG",
        )
        good_result = _promotion_cli(
            good_environment,
            state_directory=state_directory,
            checkout=checkout,
            release_checkout=good_release_checkout,
            project=production_project,
            revision=good_revision,
            image_reference=references["good"],
            adopt_existing=True,
        )
        _assert_safe_output(good_result)
        assert good_result.returncode == 0
        assert "PROMOTED:" in good_result.stdout
        cleanup_image_reference = references["good"]

        good_container = _container_document(
            docker,
            base_environment,
            project=production_project,
        )
        good_container_id = good_container["Id"]
        good_container_image_id = good_container["Image"]
        good_container_config = good_container["Config"]
        good_container_state = good_container["State"]
        assert isinstance(good_container_id, str)
        assert isinstance(good_container_image_id, str)
        assert isinstance(good_container_config, dict)
        assert isinstance(good_container_state, dict)
        assert good_container_config["Image"] == references["good"]
        assert good_container_image_id == release_images["good"]
        assert good_container_state["Status"] == "running"
        assert good_container_state["Health"]["Status"] == "healthy"
        good_volume_identity = _production_volume_identity(
            docker,
            base_environment,
            good_container,
        )
        assert good_volume_identity == old_volume_identity
        assert (
            _read_volume_sentinel(
                docker,
                base_environment,
                good_container_id,
            )
            == VOLUME_SENTINEL
        )
        good_environment_bytes = env_file.read_bytes()
        assert (
            good_environment_bytes
            == serialize_compose_environment(
                PRODUCTION_ENVIRONMENT_NAMES,
                good_environment,
            ).encode()
        )
        assert (
            _git(git, base_environment, "rev-parse", "HEAD", cwd=checkout)
            == good_revision
        )

        good_current = DeploymentStateStore(state_directory).read_current()
        assert good_current is not None
        assert good_current.state.source_revision == good_revision
        assert good_current.state.image_reference == references["good"]
        assert good_current.state.image_id == release_images["good"]
        assert good_current.state.oci_revision == good_revision

        failing_environment = _production_environment(
            base_environment,
            canary=PRIVATE_CANARIES[2],
            log_level="WARNING",
        )
        failing_result = _promotion_cli(
            failing_environment,
            state_directory=state_directory,
            checkout=checkout,
            release_checkout=failing_release_checkout,
            project=production_project,
            revision=failing_revision,
            image_reference=references["failing"],
            adopt_existing=False,
            check=False,
        )
        _assert_safe_output(failing_result)
        assert failing_result.returncode == 1
        assert "promotion failed; the recorded baseline was restored" in (
            failing_result.stderr
        )

        restored_container = _container_document(
            docker,
            base_environment,
            project=production_project,
        )
        restored_container_id = restored_container["Id"]
        restored_image_id = restored_container["Image"]
        restored_config = restored_container["Config"]
        restored_state = restored_container["State"]
        assert isinstance(restored_container_id, str)
        assert isinstance(restored_config, dict)
        assert isinstance(restored_state, dict)
        assert restored_state["Status"] == "running"
        assert restored_state["Health"]["Status"] == "healthy"
        assert restored_config["Image"] == references["good"]
        assert restored_image_id == release_images["good"]
        assert (
            _production_volume_identity(
                docker,
                base_environment,
                restored_container,
            )
            == good_volume_identity
        )
        assert (
            _read_volume_sentinel(
                docker,
                base_environment,
                restored_container_id,
            )
            == VOLUME_SENTINEL
        )
        assert (
            _read_production_failure_sentinel(
                docker,
                base_environment,
                restored_container_id,
            )
            == PRODUCTION_FAILURE_SENTINEL
        )
        assert env_file.read_bytes() == good_environment_bytes
        assert hashlib.sha256(env_file.read_bytes()).hexdigest() == (
            good_current.state.environment_sha256
        )
        assert (
            _git(git, base_environment, "rev-parse", "HEAD", cwd=checkout)
            == good_revision
        )
        assert _git(git, base_environment, "diff", "--quiet", "--", cwd=checkout) == ""

        restored_image = _image_identity(
            docker,
            base_environment,
            str(restored_image_id),
        )
        restored_config_document = restored_image["Config"]
        restored_repo_digests = restored_image["RepoDigests"]
        assert isinstance(restored_config_document, dict)
        restored_labels = restored_config_document["Labels"]
        assert isinstance(restored_labels, dict)
        assert isinstance(restored_repo_digests, list)
        assert restored_labels["org.opencontainers.image.revision"] == good_revision
        assert references["good"] in restored_repo_digests

        store = DeploymentStateStore(state_directory)
        restored_current = store.read_current()
        journal = store.read_journal()
        assert restored_current == good_current
        assert [entry.event for entry in journal] == [
            "adopted",
            "promoted",
            "rolled_back",
        ]
        rollback = journal[-1]
        promotion = journal[1]
        assert rollback.state == good_current.state
        assert promotion.transaction_id is not None
        assert rollback.transaction_id is not None
        assert rollback.persistent_volumes[0].as_document() == good_volume_identity
        assert not store.pending_path.exists()
        promotion_intent_path = (
            state_directory / "transactions" / f"{promotion.transaction_id}.json"
        )
        promotion_intent_bytes = promotion_intent_path.read_bytes()
        promotion_intent = json.loads(promotion_intent_bytes)
        assert promotion_intent["candidate"]["image_reference"] == references["good"]
        assert promotion_intent["candidate"]["image_id"] == release_images["good"]
        assert promotion_intent["candidate"]["oci_revision"] == good_revision
        assert CONTAINER_ID_PATTERN.fullmatch(
            promotion_intent["candidate"]["container_id"]
        )
        intent_path = (
            state_directory / "transactions" / f"{rollback.transaction_id}.json"
        )
        intent_bytes = intent_path.read_bytes()
        intent = json.loads(intent_bytes)
        assert intent["candidate"]["image_reference"] == references["failing"]
        assert intent["candidate"]["image_id"] == release_images["failing"]
        assert intent["candidate"]["oci_revision"] == failing_revision
        assert CONTAINER_ID_PATTERN.fullmatch(intent["candidate"]["container_id"])
        assert intent["target"]["source_revision"] == failing_revision
        assert intent["baseline_journal_sequence"] == 2
        for canary in PRIVATE_CANARIES:
            assert canary.encode() not in promotion_intent_bytes
            assert canary.encode() not in intent_bytes
            assert canary.encode() not in store.current_path.read_bytes()
            for journal_path in store.journal_path.glob("*.json"):
                assert canary.encode() not in journal_path.read_bytes()

        assert (
            _container_identity(
                docker,
                base_environment,
                sentinel_container,
            )
            == sentinel_container_identity
        )
        assert (
            _volume_identity(
                docker,
                base_environment,
                sentinel_volume,
            )
            == sentinel_volume_identity
        )
        assert (
            _network_identity(
                docker,
                base_environment,
                sentinel_network,
            )
            == sentinel_network_identity
        )
        assert (
            _image_identity(
                docker,
                base_environment,
                sentinel_image,
            )
            == sentinel_image_identity
        )
    except BaseException as error:
        primary_error = error
        raise
    finally:
        failures: list[str] = []
        if production_cleanup_armed:
            failures.extend(
                _production_cleanup(
                    docker,
                    base_environment,
                    checkout=checkout,
                    project=production_project,
                    env_file=cleanup_env_file,
                    image_reference=cleanup_image_reference,
                )
            )
        failures.extend(_execute_exact_cleanup(cleanup_commands, base_environment))
        _report_cleanup_failures(failures, primary_error)
