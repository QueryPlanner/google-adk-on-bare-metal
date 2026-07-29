"""Opt-in real-Docker proof for atomic VM promotion and verified rollback."""

from __future__ import annotations

import hashlib
import hmac
import importlib.util
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
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from types import ModuleType
from typing import Literal
from unittest.mock import create_autospec, patch
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
PROMOTION_RUNTIME_DRIVER_PATH = (
    REPOSITORY_ROOT / "tests" / "deployment_promotion_runtime_driver.py"
)
RUN_ENVIRONMENT_NAME = "RUN_DEPLOYMENT_PROMOTION_INTEGRATION"
PREFIX_ENVIRONMENT_NAME = "DEPLOYMENT_PROMOTION_TEST_PREFIX"
PREFIX_PATTERN = re.compile(r"adk-promotion-[a-z0-9][a-z0-9-]{0,23}\Z")
REVISION_PATTERN = re.compile(r"[0-9a-f]{40}\Z")
IMAGE_ID_PATTERN = re.compile(r"sha256:[0-9a-f]{64}\Z")
IMAGE_REFERENCE_PATTERN = re.compile(
    r"127\.0\.0\.1:[0-9]+/[a-z0-9-]+/agent@sha256:[0-9a-f]{64}\Z"
)
CONTAINER_ID_PATTERN = re.compile(r"[0-9a-f]{64}\Z")
GITHUB_REPOSITORY = "QueryPlanner/google-adk-on-bare-metal"
EXPECTED_ORIGIN = f"https://github.com/{GITHUB_REPOSITORY}"
REGISTRY_IMAGE_REFERENCE = (
    "registry@sha256:1be55279f18a2fe1a74edf2664cac61c1bea305b7b4642dab412e7affdcb3e33"
)
PRIVATE_CANARIES = (
    'promotion-old-$-हॅलो-"-\\-canary',
    'promotion-good-$-हॅलो-"-\\-canary',
    'promotion-unhealthy-$-हॅलो-"-\\-canary',
    'promotion-failing-$-हॅलो-"-\\-canary',
)
VOLUME_SENTINEL = "atomic-promotion-volume-sentinel"
PRODUCTION_FAILURE_SENTINEL = "production-only-failure-reached"
HEALTHY_RELEASE_COMMAND = ("python", "-m", "agent.server")
UNHEALTHY_RELEASE_COMMAND = (
    "python",
    "-c",
    "import time; time.sleep(300)",
)
CANDIDATE_START_PERIOD_SECONDS = 20
CANDIDATE_FAILURE_BOUND_SECONDS = 180
OWNER_LABEL = "io.queryplanner.adk.promotion-test.owner"

type DockerResourceKind = Literal["container", "image", "network", "volume"]


def _load_runtime_driver() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "deployment_promotion_runtime_driver",
        PROMOTION_RUNTIME_DRIVER_PATH,
    )
    if spec is None or spec.loader is None:
        raise AssertionError("runtime driver could not be loaded")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PROMOTION_RUNTIME_DRIVER = _load_runtime_driver()


@dataclass
class CleanupTarget:
    """One exact Docker resource with an independently verifiable owner."""

    kind: DockerResourceKind
    reference: str | None
    expected_owner: str | None = None
    expected_id: str | None = None

    def __post_init__(self) -> None:
        if (self.expected_owner is None) == (self.expected_id is None):
            raise AssertionError("cleanup target needs exactly one ownership proof")


PRODUCTION_COMPOSE = """\
services:
  agent:
    image: "${IMAGE:?Set IMAGE to an immutable production image}"
    pull_policy: never
    labels:
      io.queryplanner.adk.promotion-test.owner: "__PROMOTION_TEST_OWNER__"
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
    labels:
      io.queryplanner.adk.promotion-test.owner: "__PROMOTION_TEST_OWNER__"

networks:
  default:
    labels:
      io.queryplanner.adk.promotion-test.owner: "__PROMOTION_TEST_OWNER__"
"""

DERIVATIVE_DOCKERFILE = """\
ARG BASE_IMAGE
FROM ${BASE_IMAGE}
ARG SOURCE_REVISION
USER root
COPY promotion-wrapper.sh /usr/local/bin/promotion-wrapper
RUN chmod 0755 /usr/local/bin/promotion-wrapper
LABEL org.opencontainers.image.revision="${SOURCE_REVISION}"
LABEL org.opencontainers.image.source="https://github.com/QueryPlanner/google-adk-on-bare-metal"
LABEL io.queryplanner.adk.repository="QueryPlanner/google-adk-on-bare-metal"
USER app
ENTRYPOINT ["/usr/local/bin/promotion-wrapper"]
# Docker clears an inherited CMD when a child image sets ENTRYPOINT.
__RELEASE_COMMAND__
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


def _secret_representations(secret: str) -> tuple[str, ...]:
    byte_representation = repr(secret.encode())
    candidates = {
        secret,
        secret.replace("$", "$$"),
        json.dumps(secret, ensure_ascii=False)[1:-1],
        json.dumps(secret, ensure_ascii=True)[1:-1],
        repr(secret),
        byte_representation,
        byte_representation[2:-1],
    }
    return tuple(sorted(filter(None, candidates), key=len, reverse=True))


def _redact(value: str) -> str:
    redacted = value
    for secret in PRIVATE_CANARIES:
        for candidate in _secret_representations(secret):
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
    cleanup_targets: list[CleanupTarget],
    cleanup_target: CleanupTarget,
    operation: Callable[[], MutationResult],
) -> MutationResult:
    cleanup_targets.append(cleanup_target)
    return operation()


def _listed_resource_ids(
    docker: str,
    environment: Mapping[str, str],
    target: CleanupTarget,
) -> tuple[str, ...]:
    if target.reference is None:
        return ()
    if target.kind == "image":
        result = _run(
            [
                docker,
                "image",
                "ls",
                "--all",
                "--digests",
                "--no-trunc",
                "--format",
                ("{{.Repository}}:{{.Tag}}\t{{.Repository}}@{{.Digest}}\t{{.ID}}"),
            ],
            environment=environment,
            timeout=30,
        )
        matches: set[str] = set()
        for line in result.stdout.splitlines():
            fields = line.split("\t")
            if len(fields) != 3:
                raise AssertionError("Docker image listing was invalid")
            tag, digest, image_id = fields
            if target.reference in {tag, digest}:
                matches.add(image_id)
        return tuple(sorted(matches))
    if target.kind == "container":
        result = _run(
            [
                docker,
                "container",
                "ls",
                "--all",
                "--no-trunc",
                "--format",
                "{{.ID}}\t{{.Names}}",
            ],
            environment=environment,
            timeout=30,
        )
        matches = set()
        for line in result.stdout.splitlines():
            fields = line.split("\t")
            if len(fields) != 2:
                raise AssertionError("Docker container listing was invalid")
            container_id, names = fields
            if target.reference == container_id or target.reference in names.split(","):
                matches.add(container_id)
        return tuple(sorted(matches))
    if target.kind == "volume":
        volume_names = _run(
            [docker, "volume", "ls", "--format", "{{.Name}}"],
            environment=environment,
            timeout=30,
        ).stdout.splitlines()
        return (target.reference,) if target.reference in volume_names else ()
    result = _run(
        [
            docker,
            "network",
            "ls",
            "--no-trunc",
            "--format",
            "{{.ID}}\t{{.Name}}",
        ],
        environment=environment,
        timeout=30,
    )
    matches = set()
    for line in result.stdout.splitlines():
        fields = line.split("\t")
        if len(fields) != 2:
            raise AssertionError("Docker network listing was invalid")
        network_id, name = fields
        if target.reference in {network_id, name}:
            matches.add(network_id)
    return tuple(sorted(matches))


def _owned_resource_document(
    docker: str,
    environment: Mapping[str, str],
    target: CleanupTarget,
) -> dict[str, object] | None:
    if target.reference is None:
        return None
    matches = _listed_resource_ids(docker, environment, target)
    if not matches:
        return None
    if len(matches) != 1:
        raise AssertionError("cleanup target identity was ambiguous")
    documents = json.loads(
        _run(
            [docker, target.kind, "inspect", matches[0]],
            environment=environment,
            timeout=30,
        ).stdout
    )
    if not isinstance(documents, list) or len(documents) != 1:
        raise AssertionError("cleanup target inspection was ambiguous")
    document = documents[0]
    if not isinstance(document, dict):
        raise AssertionError("cleanup target inspection was invalid")
    actual_id = document.get("Name") if target.kind == "volume" else document.get("Id")
    if not isinstance(actual_id, str):
        raise AssertionError("cleanup target ID was invalid")
    if target.expected_id is not None and actual_id != target.expected_id:
        raise AssertionError("cleanup target identity did not match")
    if target.expected_owner is not None:
        if target.kind in {"container", "image"}:
            config = document.get("Config")
            if not isinstance(config, dict):
                raise AssertionError("cleanup target configuration was invalid")
            labels = config.get("Labels")
        else:
            labels = document.get("Labels")
        if not isinstance(labels, dict) or labels.get(OWNER_LABEL) != (
            target.expected_owner
        ):
            raise AssertionError("cleanup target owner did not match")
    return document


def _cleanup_owned_target(
    docker: str,
    environment: Mapping[str, str],
    target: CleanupTarget,
) -> None:
    document = _owned_resource_document(docker, environment, target)
    if document is None or target.reference is None:
        return
    actual_id = document.get("Name") if target.kind == "volume" else document.get("Id")
    if not isinstance(actual_id, str):
        raise AssertionError("cleanup target ID was invalid")
    if target.kind == "image":
        command = [docker, "image", "rm", target.reference]
    elif target.kind == "container":
        command = [
            docker,
            "container",
            "rm",
            "--force",
            "--volumes",
            actual_id,
        ]
    elif target.kind == "volume":
        command = [docker, "volume", "rm", actual_id]
    else:
        command = [docker, "network", "rm", actual_id]
    result = _run(
        command,
        environment=environment,
        check=False,
        timeout=90,
    )
    if result.returncode != 0:
        raise AssertionError(
            "owned Docker cleanup failed: " + _redact(result.stderr[-1_000:])
        )
    if _listed_resource_ids(docker, environment, target):
        raise AssertionError("owned Docker cleanup left its exact target")


def _execute_exact_cleanup(
    docker: str,
    cleanup_targets: Sequence[CleanupTarget],
    environment: Mapping[str, str],
) -> list[str]:
    failures: list[str] = []
    for target in reversed(cleanup_targets):
        try:
            _cleanup_owned_target(docker, environment, target)
        except (AssertionError, OSError, TypeError, ValueError) as error:
            failures.append(_redact(str(error)[-1_000:]))
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
    return f"{configured}-{uuid.uuid4().hex[:16]}"


def _assert_image_reference_absent(
    docker: str,
    environment: Mapping[str, str],
    reference: str,
) -> None:
    matches = _run(
        [
            docker,
            "image",
            "ls",
            "--all",
            "--filter",
            f"reference={reference}",
            "--format",
            "{{.Repository}}:{{.Tag}}",
        ],
        environment=environment,
        timeout=30,
    ).stdout.splitlines()
    if matches:
        raise AssertionError("generated Docker image reference already exists")


def _assert_container_name_absent(
    docker: str,
    environment: Mapping[str, str],
    name: str,
) -> None:
    names = _run(
        [docker, "container", "ls", "--all", "--format", "{{.Names}}"],
        environment=environment,
        timeout=30,
    ).stdout.splitlines()
    if name in names:
        raise AssertionError("generated Docker container name already exists")


def _assert_volume_name_absent(
    docker: str,
    environment: Mapping[str, str],
    name: str,
) -> None:
    names = _run(
        [docker, "volume", "ls", "--format", "{{.Name}}"],
        environment=environment,
        timeout=30,
    ).stdout.splitlines()
    if name in names:
        raise AssertionError("generated Docker volume name already exists")


def _assert_network_name_absent(
    docker: str,
    environment: Mapping[str, str],
    name: str,
) -> None:
    names = _run(
        [docker, "network", "ls", "--format", "{{.Name}}"],
        environment=environment,
        timeout=30,
    ).stdout.splitlines()
    if name in names:
        raise AssertionError("generated Docker network name already exists")


def _assert_compose_project_absent(
    docker: str,
    environment: Mapping[str, str],
    *,
    project: str,
) -> None:
    project_filter = f"label=com.docker.compose.project={project}"
    commands = (
        [docker, "container", "ls", "--all", "--quiet", "--filter", project_filter],
        [docker, "volume", "ls", "--quiet", "--filter", project_filter],
        [docker, "network", "ls", "--quiet", "--filter", project_filter],
    )
    if any(
        _run(command, environment=environment, timeout=30).stdout.strip()
        for command in commands
    ):
        raise AssertionError("generated Docker Compose project already exists")
    for container_name in (f"{project}-agent-1", f"{project}_agent_1"):
        _assert_container_name_absent(
            docker,
            environment,
            container_name,
        )
    _assert_volume_name_absent(
        docker,
        environment,
        f"{project}_agent_artifacts",
    )
    _assert_network_name_absent(
        docker,
        environment,
        f"{project}_default",
    )


def _compose_project_targets(
    docker: str,
    environment: Mapping[str, str],
    *,
    project: str,
    owner: str,
) -> tuple[CleanupTarget, ...]:
    project_filter = f"label=com.docker.compose.project={project}"
    commands: tuple[tuple[DockerResourceKind, list[str]], ...] = (
        (
            "container",
            [
                docker,
                "container",
                "ls",
                "--all",
                "--no-trunc",
                "--quiet",
                "--filter",
                project_filter,
            ],
        ),
        (
            "volume",
            [docker, "volume", "ls", "--quiet", "--filter", project_filter],
        ),
        (
            "network",
            [
                docker,
                "network",
                "ls",
                "--no-trunc",
                "--quiet",
                "--filter",
                project_filter,
            ],
        ),
    )
    references: set[tuple[DockerResourceKind, str]] = set()
    for kind, command in commands:
        references.update(
            (kind, reference)
            for reference in _run(
                command,
                environment=environment,
                timeout=30,
            ).stdout.splitlines()
            if reference
        )
    exact_names: tuple[tuple[DockerResourceKind, str], ...] = (
        ("container", f"{project}-agent-1"),
        ("container", f"{project}_agent_1"),
        ("volume", f"{project}_agent_artifacts"),
        ("network", f"{project}_default"),
    )
    for kind, reference in exact_names:
        probe = CleanupTarget(
            kind=kind,
            reference=reference,
            expected_owner=owner,
        )
        references.update(
            (kind, matched)
            for matched in _listed_resource_ids(docker, environment, probe)
        )
    return tuple(
        CleanupTarget(
            kind=kind,
            reference=reference,
            expected_owner=owner,
        )
        for kind, reference in sorted(references)
    )


def _assert_compose_project_owned(
    docker: str,
    environment: Mapping[str, str],
    *,
    project: str,
    owner: str,
) -> None:
    targets = _verified_compose_project_targets(
        docker,
        environment,
        project=project,
        owner=owner,
    )
    kinds = {target.kind for target in targets}
    if not {"container", "network", "volume"}.issubset(kinds):
        raise AssertionError("owned Docker Compose project is incomplete")


def _verified_compose_project_targets(
    docker: str,
    environment: Mapping[str, str],
    *,
    project: str,
    owner: str,
) -> tuple[CleanupTarget, ...]:
    targets = _compose_project_targets(
        docker,
        environment,
        project=project,
        owner=owner,
    )
    for target in targets:
        if _owned_resource_document(docker, environment, target) is None:
            raise AssertionError("owned Docker Compose resource disappeared")
    return targets


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
    owner: str,
) -> str:
    compose = PRODUCTION_COMPOSE.replace(
        "__PROMOTION_TEST_OWNER__",
        owner,
    ).replace(
        "__PROMOTION_TEST_FAILURE__",
        "1" if failure else "0",
    )
    assert "__PROMOTION_TEST_" not in compose
    (checkout / "compose.yaml").write_text(compose, encoding="utf-8")
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
    *,
    owner: str,
) -> tuple[str, str, str, str]:
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
        owner=owner,
    )
    good_revision = _commit_fixture(
        git,
        environment,
        checkout,
        phase="good",
        failure=False,
        owner=owner,
    )
    unhealthy_revision = _commit_fixture(
        git,
        environment,
        checkout,
        phase="unhealthy",
        failure=False,
        owner=owner,
    )
    failing_revision = _commit_fixture(
        git,
        environment,
        checkout,
        phase="failing",
        failure=True,
        owner=owner,
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
    return old_revision, good_revision, unhealthy_revision, failing_revision


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
    owner: str,
) -> str:
    _run(
        [
            docker,
            "build",
            "--iidfile",
            str(iid_file),
            "--label",
            f"{OWNER_LABEL}={owner}",
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


def _write_derivative_context(
    path: Path,
    *,
    command: Sequence[str],
) -> None:
    normalized_command = tuple(command)
    if normalized_command not in {
        HEALTHY_RELEASE_COMMAND,
        UNHEALTHY_RELEASE_COMMAND,
    }:
        raise AssertionError("release command is not an approved runtime fixture")
    path.mkdir(mode=0o700)
    command_json = json.dumps(list(normalized_command), separators=(",", ":"))
    dockerfile = DERIVATIVE_DOCKERFILE.replace(
        "__RELEASE_COMMAND__",
        f"CMD {command_json}",
    )
    assert "__RELEASE_COMMAND__" not in dockerfile
    (path / "Dockerfile").write_text(dockerfile, encoding="utf-8")
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
    expected_command: Sequence[str],
    owner: str,
) -> str:
    _run(
        [
            docker,
            "build",
            "--iidfile",
            str(iid_file),
            "--label",
            f"{OWNER_LABEL}={owner}",
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
    configured_command = json.loads(
        _run(
            [
                docker,
                "image",
                "inspect",
                "--format",
                "{{json .Config.Cmd}}",
                tag,
            ],
            environment=environment,
            timeout=30,
        ).stdout
    )
    assert configured_command == list(expected_command)
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
    if not isinstance(documents, list) or len(documents) != 1:
        raise AssertionError("Docker image inspection was ambiguous")
    document = documents[0]
    if not isinstance(document, dict):
        raise AssertionError("Docker image inspection was invalid")
    return document


def _ensure_registry_image(
    docker: str,
    environment: Mapping[str, str],
) -> str:
    existing = _run(
        [docker, "image", "inspect", REGISTRY_IMAGE_REFERENCE],
        environment=environment,
        check=False,
        timeout=30,
    )
    if existing.returncode != 0:
        if "No such image" not in existing.stderr:
            raise AssertionError("pinned registry image presence could not be proven")
        _run(
            [docker, "image", "pull", REGISTRY_IMAGE_REFERENCE],
            environment=environment,
            timeout=180,
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
    owner: str,
) -> None:
    _run(
        [
            docker,
            "container",
            "create",
            "--name",
            name,
            "--label",
            f"{OWNER_LABEL}={owner}",
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
    cleanup_targets: list[CleanupTarget],
    *,
    source_image_id: str,
    source_tag: str,
    repository_tag: str,
) -> str:
    _assert_image_reference_absent(
        docker,
        environment,
        repository_tag,
    )
    _owned_mutation(
        cleanup_targets,
        CleanupTarget(
            kind="image",
            reference=repository_tag,
            expected_id=source_image_id,
        ),
        lambda: _run(
            [docker, "image", "tag", source_image_id, repository_tag],
            environment=environment,
            timeout=30,
        ),
    )
    exact_cleanup = CleanupTarget(
        kind="image",
        reference=None,
        expected_id=source_image_id,
    )
    cleanup_targets.append(exact_cleanup)
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
    exact_cleanup.reference = exact_reference
    _run(
        [docker, "image", "rm", repository_tag],
        environment=environment,
        timeout=30,
    )
    _run(
        [docker, "image", "rm", source_tag],
        environment=environment,
        timeout=30,
    )
    _run(
        [docker, "image", "pull", exact_reference],
        environment=environment,
        timeout=180,
    )
    exact_document = _image_identity(docker, environment, exact_reference)
    assert exact_document.get("Id") == source_image_id
    assert exact_document.get("RepoDigests") == [exact_reference]
    assert exact_document.get("RepoTags") == []
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
    owner: str,
) -> list[str]:
    try:
        targets = _verified_compose_project_targets(
            docker,
            environment,
            project=project,
            owner=owner,
        )
        if not targets:
            return []
        kinds = {target.kind for target in targets}
        failures: list[str] = []
        if {"container", "network", "volume"}.issubset(kinds):
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
            if result.returncode != 0:
                failures.append(_redact(result.stderr[-1_000:]))
        removal_order = {"container": 0, "network": 1, "volume": 2, "image": 3}
        for target in sorted(targets, key=lambda item: removal_order[item.kind]):
            try:
                _cleanup_owned_target(docker, environment, target)
            except (AssertionError, OSError, TypeError, ValueError) as error:
                failures.append(_redact(str(error)[-1_000:]))
        _assert_compose_project_absent(
            docker,
            environment,
            project=project,
        )
    except (AssertionError, OSError, TypeError, ValueError) as error:
        return [_redact(str(error)[-1_000:])]
    return failures


def _promotion_cli(
    environment: Mapping[str, str],
    *,
    state_directory: Path,
    checkout: Path,
    release_checkout: Path,
    project: str,
    revision: str,
    image_reference: str,
    image_repository: str,
    adopt_existing: bool,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    command = [
        sys.executable,
        str(PROMOTION_RUNTIME_DRIVER_PATH),
        "--image-repository",
        image_repository,
        "--expected-origin",
        EXPECTED_ORIGIN,
        "--repository",
        GITHUB_REPOSITORY,
        "promote",
        "--state-dir",
        str(state_directory),
        "--checkout",
        str(checkout),
        "--release-checkout",
        str(release_checkout),
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


def test_runtime_driver_receives_exact_test_identity_and_repository(
    tmp_path: Path,
) -> None:
    """Keep the non-GHCR seam confined to the loopback-only test driver."""
    image_repository = (
        "127.0.0.1:49152/adk-promotion-123456789012345678901234-0123456789abcdef/agent"
    )
    image_reference = f"{image_repository}@sha256:{'a' * 64}"
    completed = subprocess.CompletedProcess([], 0, stdout="PROMOTED\n", stderr="")
    runner = create_autospec(_run, spec_set=True, return_value=completed)

    with patch(f"{__name__}._run", runner):
        result = _promotion_cli(
            {},
            state_directory=tmp_path / "state",
            checkout=tmp_path / "checkout",
            release_checkout=tmp_path / "release",
            project="adk-promotion-proof-prod",
            revision="b" * 40,
            image_reference=image_reference,
            image_repository=image_repository,
            adopt_existing=True,
        )

    assert result is completed
    command = runner.call_args.args[0]
    assert command[:2] == [sys.executable, str(PROMOTION_RUNTIME_DRIVER_PATH)]
    assert command[2:8] == [
        "--image-repository",
        image_repository,
        "--expected-origin",
        EXPECTED_ORIGIN,
        "--repository",
        GITHUB_REPOSITORY,
    ]
    assert command[8] == "promote"
    assert "--adopt-existing" in command
    assert "-m" not in command


def test_runtime_driver_accepts_maximum_generated_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Accept a long run ID plus the exact generated UUID suffix."""
    image_repository = (
        "127.0.0.1:49152/adk-promotion-123456789012345678901234-0123456789abcdef/agent"
    )
    promoter = create_autospec(
        PROMOTION_RUNTIME_DRIVER.promotion_main,
        spec_set=True,
        return_value=0,
    )
    monkeypatch.setattr(
        PROMOTION_RUNTIME_DRIVER,
        "promotion_main",
        promoter,
    )

    result = PROMOTION_RUNTIME_DRIVER.main(
        [
            "--image-repository",
            image_repository,
            "--expected-origin",
            EXPECTED_ORIGIN,
            "--repository",
            GITHUB_REPOSITORY,
            "promote",
            "--state-dir",
            "/private/test-state",
        ]
    )

    assert result == 0
    promoter.assert_called_once_with(
        [
            "promote",
            "--expected-origin",
            EXPECTED_ORIGIN,
            "--repository",
            GITHUB_REPOSITORY,
            "--state-dir",
            "/private/test-state",
        ],
        _image_repository=image_repository,
    )


@pytest.mark.parametrize(
    "image_repository",
    [
        "registry.example/adk-promotion-proof-0123456789abcdef/agent",
        "127.0.0.1:0/adk-promotion-proof-0123456789abcdef/agent",
        "127.0.0.1:65536/adk-promotion-proof-0123456789abcdef/agent",
        "127.0.0.1:49152/adk-promotion-proof/agent",
    ],
)
def test_runtime_driver_rejects_nonisolated_repository(
    monkeypatch: pytest.MonkeyPatch,
    image_repository: str,
) -> None:
    """Reject any override outside the exact loopback fixture shape."""
    promoter = create_autospec(
        PROMOTION_RUNTIME_DRIVER.promotion_main,
        spec_set=True,
    )
    monkeypatch.setattr(
        PROMOTION_RUNTIME_DRIVER,
        "promotion_main",
        promoter,
    )

    with pytest.raises(SystemExit, match="isolated loopback registry"):
        PROMOTION_RUNTIME_DRIVER.main(
            [
                "--image-repository",
                image_repository,
                "--expected-origin",
                EXPECTED_ORIGIN,
                "--repository",
                GITHUB_REPOSITORY,
                "promote",
            ]
        )

    promoter.assert_not_called()


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
    if len(short_id) != 1:
        raise AssertionError("production container lookup was ambiguous")
    documents = json.loads(
        _run(
            [docker, "container", "inspect", short_id[0]],
            environment=environment,
            timeout=30,
        ).stdout
    )
    if not isinstance(documents, list) or len(documents) != 1:
        raise AssertionError("production container inspection was ambiguous")
    document = documents[0]
    if not isinstance(document, dict):
        raise AssertionError("production container inspection was invalid")
    return document


def _production_process_identity(
    docker: str,
    environment: Mapping[str, str],
    container_id: str,
) -> dict[str, bool | int | str]:
    fields = _run(
        [
            docker,
            "container",
            "inspect",
            "--format",
            (
                "{{json .Id}}\n"
                "{{json .Config.Image}}\n"
                "{{json .Image}}\n"
                "{{json .State.Status}}\n"
                "{{json .State.Running}}\n"
                "{{json .State.Health.Status}}\n"
                "{{json .State.Pid}}\n"
                "{{json .State.StartedAt}}\n"
                "{{json .RestartCount}}"
            ),
            container_id,
        ],
        environment=environment,
        timeout=30,
    ).stdout.splitlines()
    if len(fields) != 9:
        raise AssertionError("production process inspection was invalid")
    (
        observed_container_id,
        image_reference,
        image_id,
        status,
        running,
        health_status,
        pid,
        started_at,
        restart_count,
    ) = (json.loads(field) for field in fields)
    if (
        not isinstance(observed_container_id, str)
        or not isinstance(image_reference, str)
        or not isinstance(image_id, str)
        or not isinstance(status, str)
        or not isinstance(running, bool)
        or not isinstance(health_status, str)
        or not isinstance(pid, int)
        or not isinstance(started_at, str)
        or not isinstance(restart_count, int)
        or observed_container_id != container_id
        or CONTAINER_ID_PATTERN.fullmatch(container_id) is None
        or IMAGE_ID_PATTERN.fullmatch(image_id) is None
        or pid <= 0
        or restart_count < 0
    ):
        raise AssertionError("production process values were invalid")
    datetime.fromisoformat(started_at.replace("Z", "+00:00")).astimezone(UTC)
    return {
        "container_id": observed_container_id,
        "health_status": health_status,
        "image_id": image_id,
        "image_reference": image_reference,
        "pid": pid,
        "restart_count": restart_count,
        "running": running,
        "started_at": started_at,
        "status": status,
    }


def _release_image_identity(
    image: Mapping[str, object],
) -> tuple[str, str, tuple[str, ...]]:
    image_id = image.get("Id")
    config = image.get("Config")
    repo_digests = image.get("RepoDigests")
    assert isinstance(image_id, str)
    assert isinstance(config, dict)
    assert isinstance(repo_digests, list)
    labels = config.get("Labels")
    assert isinstance(labels, dict)
    revision = labels.get("org.opencontainers.image.revision")
    assert isinstance(revision, str)
    normalized_digests = tuple(
        sorted(digest for digest in repo_digests if isinstance(digest, str))
    )
    assert IMAGE_ID_PATTERN.fullmatch(image_id)
    assert REVISION_PATTERN.fullmatch(revision)
    assert len(normalized_digests) == len(repo_digests)
    return image_id, revision, normalized_digests


def _private_file_fingerprint(path: Path) -> tuple[int, int, int, int, str]:
    metadata = path.lstat()
    mode = stat.S_IMODE(metadata.st_mode)
    if not stat.S_ISREG(metadata.st_mode) or mode != 0o600:
        raise AssertionError("private environment file is not a regular 0600 file")
    contents = path.read_bytes()
    return (
        metadata.st_ino,
        metadata.st_size,
        metadata.st_mtime_ns,
        mode,
        hashlib.sha256(contents).hexdigest(),
    )


def _assert_private_file_contents(path: Path, expected: bytes) -> None:
    if not hmac.compare_digest(path.read_bytes(), expected):
        raise AssertionError("private environment bytes changed")


def _state_tree_fingerprint(
    root: Path,
) -> dict[str, tuple[str, int, int, str]]:
    assert root.is_dir()
    paths = [root, *sorted(root.rglob("*"))]
    fingerprint: dict[str, tuple[str, int, int, str]] = {}
    for path in paths:
        metadata = path.lstat()
        relative = "." if path == root else path.relative_to(root).as_posix()
        mode = stat.S_IMODE(metadata.st_mode)
        if stat.S_ISDIR(metadata.st_mode):
            fingerprint[relative] = ("directory", mode, 0, "")
        elif stat.S_ISREG(metadata.st_mode):
            contents = path.read_bytes()
            fingerprint[relative] = (
                "file",
                mode,
                metadata.st_size,
                hashlib.sha256(contents).hexdigest(),
            )
        else:
            raise AssertionError("deployment-state tree contains a special path")
    return fingerprint


def _candidate_service_container_ids(
    docker: str,
    environment: Mapping[str, str],
    *,
    image_id: str,
    release_checkout: Path,
) -> tuple[str, ...]:
    assert IMAGE_ID_PATTERN.fullmatch(image_id)
    prefix = [
        docker,
        "container",
        "ls",
        "--all",
        "--no-trunc",
        "--filter",
        "label=com.docker.compose.service=agent",
    ]
    commands = (
        [
            *prefix,
            "--filter",
            f"ancestor={image_id}",
            "--format",
            "{{.ID}}",
        ],
        [
            *prefix,
            "--filter",
            (f"label=com.docker.compose.project.working_dir={release_checkout}"),
            "--format",
            "{{.ID}}",
        ],
    )
    container_ids = tuple(
        sorted(
            {
                container_id
                for command in commands
                for container_id in _run(
                    command,
                    environment=environment,
                    timeout=30,
                ).stdout.splitlines()
                if container_id
            }
        )
    )
    assert all(
        CONTAINER_ID_PATTERN.fullmatch(container_id) for container_id in container_ids
    )
    return container_ids


def _candidate_lifecycle_evidence(
    docker: str,
    environment: Mapping[str, str],
    *,
    image_reference: str,
    release_checkout: Path,
    since: float,
    until: float,
) -> tuple[str, int, int]:
    result = _run(
        [
            docker,
            "events",
            "--since",
            f"{since:.9f}",
            "--until",
            f"{until:.9f}",
            "--filter",
            "type=container",
            "--filter",
            (f"label=com.docker.compose.project.working_dir={release_checkout}"),
            "--format",
            "{{json .}}",
        ],
        environment=environment,
        timeout=30,
    )
    events: list[tuple[str, str, int]] = []
    for line in filter(None, result.stdout.splitlines()):
        document = json.loads(line)
        if not isinstance(document, dict):
            raise AssertionError("candidate Docker event was invalid")
        actor = document.get("Actor")
        if not isinstance(actor, dict):
            raise AssertionError("candidate Docker event actor was invalid")
        attributes = actor.get("Attributes")
        action = document.get("Action", document.get("status"))
        container_id = actor.get("ID", document.get("id"))
        time_nano = document.get("timeNano")
        if not (
            isinstance(attributes, dict)
            and isinstance(action, str)
            and isinstance(container_id, str)
            and isinstance(time_nano, int)
        ):
            raise AssertionError("candidate Docker event fields were invalid")
        project = attributes.get("com.docker.compose.project")
        if (
            not isinstance(project, str)
            or not project.startswith("candidate-")
            or attributes.get("com.docker.compose.service") != "agent"
            or attributes.get("com.docker.compose.project.working_dir")
            != str(release_checkout)
            or attributes.get("image") != image_reference
            or CONTAINER_ID_PATTERN.fullmatch(container_id) is None
        ):
            raise AssertionError("candidate Docker event ownership did not match")
        events.append((action, container_id, time_nano))
    if not events:
        raise AssertionError("candidate Docker lifecycle events were missing")
    container_ids = {container_id for _, container_id, _ in events}
    if len(container_ids) != 1:
        raise AssertionError("candidate Docker lifecycle actor was ambiguous")
    positions: dict[str, tuple[int, int]] = {}
    for expected_action in (
        "create",
        "start",
        "health_status: unhealthy",
        "destroy",
    ):
        matches = [
            (index, time_nano)
            for index, (action, _container_id, time_nano) in enumerate(events)
            if action == expected_action
        ]
        if len(matches) != 1:
            raise AssertionError("candidate Docker lifecycle was incomplete")
        positions[expected_action] = matches[0]
    if not (
        positions["create"][0]
        < positions["start"][0]
        < positions["health_status: unhealthy"][0]
        < positions["destroy"][0]
    ):
        raise AssertionError("candidate Docker lifecycle order was invalid")
    started_at = positions["start"][1]
    unhealthy_at = positions["health_status: unhealthy"][1]
    if unhealthy_at - started_at < CANDIDATE_START_PERIOD_SECONDS * 1_000_000_000:
        raise AssertionError("candidate became unhealthy before its start period")
    return container_ids.pop(), started_at, unhealthy_at


def _cleanup_candidate_containers(
    docker: str,
    environment: Mapping[str, str],
    *,
    image_id: str,
    release_checkout: Path,
) -> list[str]:
    failures: list[str] = []
    try:
        container_ids = _candidate_service_container_ids(
            docker,
            environment,
            image_id=image_id,
            release_checkout=release_checkout,
        )
    except (AssertionError, OSError) as error:
        return [_redact(str(error)[-1_000:])]
    for container_id in container_ids:
        try:
            documents = json.loads(
                _run(
                    [docker, "container", "inspect", container_id],
                    environment=environment,
                    timeout=30,
                ).stdout
            )
            if not isinstance(documents, list) or len(documents) != 1:
                raise AssertionError("candidate cleanup inspection was ambiguous")
            document = documents[0]
            if not isinstance(document, dict):
                raise AssertionError("candidate cleanup inspection was invalid")
            config = document.get("Config")
            if not isinstance(config, dict):
                raise AssertionError("candidate cleanup configuration was invalid")
            labels = config.get("Labels")
            if not isinstance(labels, dict):
                raise AssertionError("candidate cleanup labels were invalid")
            project = labels.get("com.docker.compose.project")
            if (
                document.get("Id") != container_id
                or document.get("Image") != image_id
                or not isinstance(project, str)
                or not project.startswith("candidate-")
                or labels.get("com.docker.compose.service") != "agent"
                or labels.get("com.docker.compose.project.working_dir")
                != str(release_checkout)
                or document.get("Mounts") != []
            ):
                raise AssertionError("candidate cleanup ownership did not match")
            result = _run(
                [
                    docker,
                    "container",
                    "rm",
                    "--force",
                    "--volumes",
                    container_id,
                ],
                environment=environment,
                check=False,
                timeout=60,
            )
            if result.returncode != 0:
                failures.append(_redact(result.stderr[-1_000:]))
        except (AssertionError, OSError, TypeError, ValueError) as error:
            failures.append(_redact(str(error)[-1_000:]))
    try:
        if _candidate_service_container_ids(
            docker,
            environment,
            image_id=image_id,
            release_checkout=release_checkout,
        ):
            failures.append("candidate cleanup left an owned service container")
    except (AssertionError, OSError) as error:
        failures.append(_redact(str(error)[-1_000:]))
    return failures


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
        if any(
            representation in stream
            for representation in _secret_representations(canary)
            for stream in (result.stdout, result.stderr)
        ):
            raise AssertionError("promotion output exposed a private canary")


def test_pushed_fixture_is_rehydrated_by_immutable_digest() -> None:
    """Pull the digest after removing the fixture's final mutable tags."""
    source_image_id = f"sha256:{'a' * 64}"
    source_tag = "fixture-source:phase"
    image_repository = "127.0.0.1:49152/adk-promotion-proof-0123456789abcdef/agent"
    repository_tag = f"{image_repository}:phase"
    exact_reference = f"{image_repository}@sha256:{'b' * 64}"
    cleanup_targets: list[CleanupTarget] = []
    runner = create_autospec(
        _run,
        spec_set=True,
        side_effect=[
            subprocess.CompletedProcess([], 0, stdout="", stderr=""),
            subprocess.CompletedProcess([], 0, stdout="", stderr=""),
            subprocess.CompletedProcess([], 0, stdout="", stderr=""),
            subprocess.CompletedProcess(
                [],
                0,
                stdout=json.dumps(
                    [
                        {
                            "Id": source_image_id,
                            "RepoDigests": [exact_reference],
                            "RepoTags": [source_tag, repository_tag],
                        }
                    ]
                ),
                stderr="",
            ),
            subprocess.CompletedProcess([], 0, stdout="", stderr=""),
            subprocess.CompletedProcess([], 0, stdout="", stderr=""),
            subprocess.CompletedProcess([], 0, stdout="", stderr=""),
            subprocess.CompletedProcess(
                [],
                0,
                stdout=json.dumps(
                    [
                        {
                            "Id": source_image_id,
                            "RepoDigests": [exact_reference],
                            "RepoTags": [],
                        }
                    ]
                ),
                stderr="",
            ),
        ],
    )

    with patch(f"{__name__}._run", runner):
        result = _push_exact_image(
            "docker",
            {},
            cleanup_targets,
            source_image_id=source_image_id,
            source_tag=source_tag,
            repository_tag=repository_tag,
        )

    assert result == exact_reference
    assert cleanup_targets == [
        CleanupTarget(
            kind="image",
            reference=repository_tag,
            expected_id=source_image_id,
        ),
        CleanupTarget(
            kind="image",
            reference=exact_reference,
            expected_id=source_image_id,
        ),
    ]
    commands = [call.args[0] for call in runner.call_args_list]
    assert commands[4:] == [
        ["docker", "image", "rm", repository_tag],
        ["docker", "image", "rm", source_tag],
        ["docker", "image", "pull", exact_reference],
        ["docker", "image", "inspect", exact_reference],
    ]


def test_runtime_redaction_covers_embedded_byte_representations() -> None:
    """Redact byte-rendered canaries without reproducing them in failures."""
    canary = PRIVATE_CANARIES[0]
    rendered = f"prefix:{repr(canary.encode())[2:-1]}:suffix"
    redacted = _redact(rendered)

    if "[REDACTED]" not in redacted or canary in redacted:
        raise AssertionError("embedded byte representation was not redacted")


def test_private_file_evidence_is_secret_safe(tmp_path: Path) -> None:
    """Compare private bytes exactly while exposing only stable fingerprints."""
    private_file = tmp_path / "private.env"
    expected = PRIVATE_CANARIES[0].encode()
    private_file.write_bytes(expected)
    private_file.chmod(0o600)

    fingerprint = _private_file_fingerprint(private_file)
    assert fingerprint[1] == len(expected)
    assert fingerprint[3] == 0o600
    assert fingerprint[4] == hashlib.sha256(expected).hexdigest()
    _assert_private_file_contents(private_file, expected)

    private_file.write_bytes(b"different")
    with pytest.raises(AssertionError, match="private environment bytes changed"):
        _assert_private_file_contents(private_file, expected)
    private_file.chmod(0o644)
    with pytest.raises(AssertionError, match="not a regular 0600 file"):
        _private_file_fingerprint(private_file)


@pytest.mark.parametrize(
    ("target", "listing", "document"),
    (
        (
            CleanupTarget(
                kind="image",
                reference="fixture:tag",
                expected_owner="expected-owner",
            ),
            f"fixture:tag\tfixture@<none>\tsha256:{'a' * 64}\n",
            {
                "Id": f"sha256:{'a' * 64}",
                "Config": {"Labels": {OWNER_LABEL: "replacement-owner"}},
            },
        ),
        (
            CleanupTarget(
                kind="container",
                reference="fixture-container",
                expected_owner="expected-owner",
            ),
            f"{'b' * 64}\tfixture-container\n",
            {
                "Id": "b" * 64,
                "Config": {"Labels": {OWNER_LABEL: "replacement-owner"}},
            },
        ),
        (
            CleanupTarget(
                kind="volume",
                reference="fixture-volume",
                expected_owner="expected-owner",
            ),
            "fixture-volume\n",
            {
                "Name": "fixture-volume",
                "Labels": {OWNER_LABEL: "replacement-owner"},
            },
        ),
        (
            CleanupTarget(
                kind="network",
                reference="fixture-network",
                expected_owner="expected-owner",
            ),
            f"{'c' * 64}\tfixture-network\n",
            {
                "Id": "c" * 64,
                "Labels": {OWNER_LABEL: "replacement-owner"},
            },
        ),
    ),
)
def test_cleanup_refuses_a_replaced_docker_resource(
    target: CleanupTarget,
    listing: str,
    document: dict[str, object],
) -> None:
    """Never delete a reserved name or tag whose owner changed."""
    runner = create_autospec(
        _run,
        spec_set=True,
        side_effect=[
            subprocess.CompletedProcess([], 0, stdout=listing, stderr=""),
            subprocess.CompletedProcess(
                [],
                0,
                stdout=json.dumps([document]),
                stderr="",
            ),
        ],
    )

    with (
        patch(f"{__name__}._run", runner),
        pytest.raises(AssertionError, match="cleanup target owner did not match"),
    ):
        _cleanup_owned_target("docker", {}, target)

    assert runner.call_count == 2


def test_cleanup_removes_and_rechecks_one_owned_container() -> None:
    """Delete only the verified full ID and require the name to disappear."""
    container_id = "d" * 64
    target = CleanupTarget(
        kind="container",
        reference="fixture-container",
        expected_owner="expected-owner",
    )
    runner = create_autospec(
        _run,
        spec_set=True,
        side_effect=[
            subprocess.CompletedProcess(
                [],
                0,
                stdout=f"{container_id}\tfixture-container\n",
                stderr="",
            ),
            subprocess.CompletedProcess(
                [],
                0,
                stdout=json.dumps(
                    [
                        {
                            "Id": container_id,
                            "Config": {"Labels": {OWNER_LABEL: "expected-owner"}},
                        }
                    ]
                ),
                stderr="",
            ),
            subprocess.CompletedProcess([], 0, stdout=container_id, stderr=""),
            subprocess.CompletedProcess([], 0, stdout="", stderr=""),
        ],
    )

    with patch(f"{__name__}._run", runner):
        _cleanup_owned_target("docker", {}, target)

    remove_command = runner.call_args_list[2].args[0]
    assert remove_command == [
        "docker",
        "container",
        "rm",
        "--force",
        "--volumes",
        container_id,
    ]
    assert runner.call_count == 4


def test_production_cleanup_accepts_an_absent_project(tmp_path: Path) -> None:
    """Treat an entirely absent, never-started Compose project as clean."""
    blank = subprocess.CompletedProcess([], 0, stdout="", stderr="")
    runner = create_autospec(
        _run,
        spec_set=True,
        side_effect=[blank for _ in range(7)],
    )

    with patch(f"{__name__}._run", runner):
        failures = _production_cleanup(
            "docker",
            {},
            checkout=tmp_path,
            project="adk-promotion-absent",
            env_file=tmp_path / "production.env",
            image_reference="fixture@sha256:" + ("a" * 64),
            owner="expected-owner",
        )

    assert failures == []
    assert runner.call_count == 7
    assert all(
        "down" not in call.args[0] and "rm" not in call.args[0]
        for call in runner.call_args_list
    )


def test_production_cleanup_removes_an_owned_partial_project(
    tmp_path: Path,
) -> None:
    """Exact-remove owner-verified partial startup resources without Compose."""
    project = "adk-promotion-partial"
    volume_name = f"{project}_agent_artifacts"
    blank = subprocess.CompletedProcess([], 0, stdout="", stderr="")
    volume_listing = subprocess.CompletedProcess(
        [],
        0,
        stdout=f"{volume_name}\n",
        stderr="",
    )
    volume_inspection = subprocess.CompletedProcess(
        [],
        0,
        stdout=json.dumps(
            [
                {
                    "Name": volume_name,
                    "Labels": {OWNER_LABEL: "expected-owner"},
                }
            ]
        ),
        stderr="",
    )
    runner = create_autospec(
        _run,
        spec_set=True,
        side_effect=[
            blank,
            volume_listing,
            blank,
            blank,
            blank,
            volume_listing,
            blank,
            volume_listing,
            volume_inspection,
            volume_listing,
            volume_inspection,
            blank,
            blank,
            blank,
            blank,
            blank,
            blank,
            blank,
            blank,
            blank,
        ],
    )

    with patch(f"{__name__}._run", runner):
        failures = _production_cleanup(
            "docker",
            {},
            checkout=tmp_path,
            project=project,
            env_file=tmp_path / "production.env",
            image_reference="fixture@sha256:" + ("b" * 64),
            owner="expected-owner",
        )

    assert failures == []
    assert runner.call_count == 20
    commands = [call.args[0] for call in runner.call_args_list]
    assert ["docker", "volume", "rm", volume_name] in commands
    assert all("down" not in command for command in commands)


def test_candidate_lifecycle_uses_exact_health_event_timing(tmp_path: Path) -> None:
    """Measure the start period from daemon events, not total CLI duration."""
    container_id = "e" * 64
    image_reference = "127.0.0.1:5000/fixture/agent@sha256:" + ("f" * 64)
    release_checkout = tmp_path / "release-unhealthy"
    attributes = {
        "com.docker.compose.project": "candidate-owned",
        "com.docker.compose.project.working_dir": str(release_checkout),
        "com.docker.compose.service": "agent",
        "image": image_reference,
    }
    actions = (
        ("create", 1),
        ("start", 1_000_000_000),
        ("health_status: unhealthy", 21_000_000_000),
        ("destroy", 22_000_000_000),
    )
    output = "".join(
        json.dumps(
            {
                "Action": action,
                "Actor": {"ID": container_id, "Attributes": attributes},
                "timeNano": time_nano,
            }
        )
        + "\n"
        for action, time_nano in actions
    )
    runner = create_autospec(
        _run,
        spec_set=True,
        return_value=subprocess.CompletedProcess(
            [],
            0,
            stdout=output,
            stderr="",
        ),
    )

    with patch(f"{__name__}._run", runner):
        evidence = _candidate_lifecycle_evidence(
            "docker",
            {},
            image_reference=image_reference,
            release_checkout=release_checkout,
            since=0,
            until=30,
        )

    assert evidence == (container_id, 1_000_000_000, 21_000_000_000)
    command = runner.call_args.args[0]
    assert "type=container" in command
    assert f"label=com.docker.compose.project.working_dir={release_checkout}" in command


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
    unhealthy_tag = f"{prefix}-unhealthy:runtime"
    failing_tag = f"{prefix}-failing:runtime"
    sentinel_image = f"{prefix}-sentinel-image:keep"
    sentinel_container = f"{prefix}-sentinel-container"
    sentinel_volume = f"{prefix}-sentinel-volume"
    sentinel_network = f"{prefix}-sentinel-network"
    checkout = tmp_path / "production"
    good_release_checkout = tmp_path / "release-good"
    unhealthy_release_checkout = tmp_path / "release-unhealthy"
    failing_release_checkout = tmp_path / "release-failing"
    state_directory = tmp_path / "deployment-state"
    derivative_context = tmp_path / "derivative"
    unhealthy_derivative_context = tmp_path / "derivative-unhealthy"
    env_file = checkout / ".env"
    cleanup_env_file = tmp_path / "cleanup.env"

    cleanup_targets: list[CleanupTarget] = []
    release_images: dict[str, str] = {}
    production_cleanup_armed = False
    cleanup_image_reference = old_tag
    primary_error: BaseException | None = None
    try:
        (
            old_revision,
            good_revision,
            unhealthy_revision,
            failing_revision,
        ) = _initialize_repository(
            git,
            base_environment,
            checkout,
            owner=prefix,
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
            unhealthy_release_checkout,
            unhealthy_revision,
        )
        _add_release_worktree(
            git,
            base_environment,
            checkout,
            failing_release_checkout,
            failing_revision,
        )
        assert (unhealthy_release_checkout / "compose.candidate.yaml").read_text(
            encoding="utf-8"
        ).count(f"start_period: {CANDIDATE_START_PERIOD_SECONDS}s") == 1
        _write_derivative_context(
            derivative_context,
            command=HEALTHY_RELEASE_COMMAND,
        )
        _write_derivative_context(
            unhealthy_derivative_context,
            command=UNHEALTHY_RELEASE_COMMAND,
        )

        _assert_image_reference_absent(docker, base_environment, base_tag)
        _owned_mutation(
            cleanup_targets,
            CleanupTarget(
                kind="image",
                reference=base_tag,
                expected_owner=prefix,
            ),
            lambda: _build_base_image(
                docker,
                base_environment,
                tag=base_tag,
                iid_file=tmp_path / "base.iid",
                owner=prefix,
            ),
        )
        for phase, tag, revision, context, expected_command in (
            (
                "old",
                old_tag,
                old_revision,
                derivative_context,
                HEALTHY_RELEASE_COMMAND,
            ),
            (
                "good",
                good_tag,
                good_revision,
                derivative_context,
                HEALTHY_RELEASE_COMMAND,
            ),
            (
                "unhealthy",
                unhealthy_tag,
                unhealthy_revision,
                unhealthy_derivative_context,
                UNHEALTHY_RELEASE_COMMAND,
            ),
            (
                "failing",
                failing_tag,
                failing_revision,
                derivative_context,
                HEALTHY_RELEASE_COMMAND,
            ),
        ):
            _assert_image_reference_absent(docker, base_environment, tag)
            release_images[phase] = _owned_mutation(
                cleanup_targets,
                CleanupTarget(
                    kind="image",
                    reference=tag,
                    expected_owner=prefix,
                ),
                partial(
                    _build_release_image,
                    docker,
                    base_environment,
                    context=context,
                    base_tag=base_tag,
                    tag=tag,
                    revision=revision,
                    iid_file=tmp_path / f"{phase}.iid",
                    expected_command=expected_command,
                    owner=prefix,
                ),
            )

        registry_image_id = _ensure_registry_image(
            docker,
            base_environment,
        )
        _assert_container_name_absent(
            docker,
            base_environment,
            registry_name,
        )
        _owned_mutation(
            cleanup_targets,
            CleanupTarget(
                kind="container",
                reference=registry_name,
                expected_owner=prefix,
            ),
            lambda: _create_registry(
                docker,
                base_environment,
                name=registry_name,
                image_id=registry_image_id,
                owner=prefix,
            ),
        )
        registry_endpoint = _start_registry(
            docker,
            base_environment,
            name=registry_name,
        )
        image_repository = f"{registry_endpoint}/{repository_name}"

        references: dict[str, str] = {}
        for phase, source_tag in (
            ("old", old_tag),
            ("good", good_tag),
            ("unhealthy", unhealthy_tag),
            ("failing", failing_tag),
        ):
            references[phase] = _push_exact_image(
                docker,
                base_environment,
                cleanup_targets,
                source_image_id=release_images[phase],
                source_tag=source_tag,
                repository_tag=f"{image_repository}:{phase}",
            )
        revisions = {
            "old": old_revision,
            "good": good_revision,
            "unhealthy": unhealthy_revision,
            "failing": failing_revision,
        }
        for phase, revision in revisions.items():
            image_document = _image_identity(
                docker,
                base_environment,
                references[phase],
            )
            image_id, oci_revision, repo_digests = _release_image_identity(
                image_document
            )
            image_config = image_document.get("Config")
            assert isinstance(image_config, dict)
            labels = image_config.get("Labels")
            assert isinstance(labels, dict)
            assert image_id == release_images[phase]
            assert oci_revision == revision
            assert repo_digests == (references[phase],)
            assert image_document.get("RepoTags") == []
            assert labels["io.queryplanner.adk.repository"] == GITHUB_REPOSITORY
            assert labels["org.opencontainers.image.source"] == EXPECTED_ORIGIN
        assert len(set(revisions.values())) == 4
        assert len(set(release_images.values())) == 4
        assert len(set(references.values())) == 4
        assert (
            len(
                {
                    checkout.resolve(),
                    good_release_checkout.resolve(),
                    unhealthy_release_checkout.resolve(),
                    failing_release_checkout.resolve(),
                }
            )
            == 4
        )

        _assert_image_reference_absent(
            docker,
            base_environment,
            sentinel_image,
        )
        _owned_mutation(
            cleanup_targets,
            CleanupTarget(
                kind="image",
                reference=sentinel_image,
                expected_id=registry_image_id,
            ),
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
        _assert_container_name_absent(
            docker,
            base_environment,
            sentinel_container,
        )
        _owned_mutation(
            cleanup_targets,
            CleanupTarget(
                kind="container",
                reference=sentinel_container,
                expected_owner=prefix,
            ),
            lambda: _run(
                [
                    docker,
                    "container",
                    "create",
                    "--name",
                    sentinel_container,
                    "--label",
                    f"{OWNER_LABEL}={prefix}",
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
        _assert_volume_name_absent(
            docker,
            base_environment,
            sentinel_volume,
        )
        _owned_mutation(
            cleanup_targets,
            CleanupTarget(
                kind="volume",
                reference=sentinel_volume,
                expected_owner=prefix,
            ),
            lambda: _run(
                [
                    docker,
                    "volume",
                    "create",
                    "--label",
                    f"{OWNER_LABEL}={prefix}",
                    sentinel_volume,
                ],
                environment=base_environment,
                timeout=30,
            ),
        )
        sentinel_volume_identity = _volume_identity(
            docker,
            base_environment,
            sentinel_volume,
        )
        _assert_network_name_absent(
            docker,
            base_environment,
            sentinel_network,
        )
        _owned_mutation(
            cleanup_targets,
            CleanupTarget(
                kind="network",
                reference=sentinel_network,
                expected_owner=prefix,
            ),
            lambda: _run(
                [
                    docker,
                    "network",
                    "create",
                    "--label",
                    f"{OWNER_LABEL}={prefix}",
                    sentinel_network,
                ],
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
        _assert_compose_project_absent(
            docker,
            base_environment,
            project=production_project,
        )
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
        _assert_compose_project_owned(
            docker,
            base_environment,
            project=production_project,
            owner=prefix,
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
        if not (
            isinstance(old_container_id, str)
            and isinstance(old_container_image_id, str)
            and isinstance(old_container_config, dict)
            and isinstance(old_container_state, dict)
        ):
            raise AssertionError("old production container identity was invalid")
        assert CONTAINER_ID_PATTERN.fullmatch(old_container_id)
        assert old_container_config["Image"] == references["old"]
        assert old_container_config["Cmd"] == list(HEALTHY_RELEASE_COMMAND)
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
            image_repository=image_repository,
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
        if not (
            isinstance(good_container_id, str)
            and isinstance(good_container_image_id, str)
            and isinstance(good_container_config, dict)
            and isinstance(good_container_state, dict)
        ):
            raise AssertionError("good production container identity was invalid")
        assert good_container_config["Image"] == references["good"]
        assert good_container_config["Cmd"] == list(HEALTHY_RELEASE_COMMAND)
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
        serialized_good_environment = serialize_compose_environment(
            PRODUCTION_ENVIRONMENT_NAMES,
            good_environment,
        ).encode()
        good_environment_bytes = env_file.read_bytes()
        _assert_private_file_contents(env_file, serialized_good_environment)
        good_environment_fingerprint = _private_file_fingerprint(env_file)
        if (
            good_environment_fingerprint[1] != len(serialized_good_environment)
            or good_environment_fingerprint[4]
            != hashlib.sha256(serialized_good_environment).hexdigest()
        ):
            raise AssertionError("serialized production environment changed")
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

        store = DeploymentStateStore(state_directory)
        good_process_identity = _production_process_identity(
            docker,
            base_environment,
            good_container_id,
        )
        assert good_process_identity["running"] is True
        assert good_process_identity["status"] == "running"
        assert good_process_identity["health_status"] == "healthy"
        good_image_identity = _release_image_identity(
            _image_identity(
                docker,
                base_environment,
                str(good_container_image_id),
            )
        )
        assert good_image_identity[0] == release_images["good"]
        assert good_image_identity[1] == good_revision
        assert references["good"] in good_image_identity[2]
        good_journal = store.read_journal()
        assert [entry.event for entry in good_journal] == ["adopted", "promoted"]
        good_state_tree = _state_tree_fingerprint(state_directory)
        good_candidate_containers = _candidate_service_container_ids(
            docker,
            base_environment,
            image_id=release_images["unhealthy"],
            release_checkout=unhealthy_release_checkout,
        )
        assert good_candidate_containers == ()
        assert not store.pending_path.exists()
        assert (
            _git(git, base_environment, "rev-parse", "HEAD", cwd=checkout)
            == good_revision
        )
        assert (
            _git(
                git,
                base_environment,
                "status",
                "--porcelain=v1",
                "--untracked-files=all",
                cwd=checkout,
            )
            == ""
        )

        unhealthy_environment = _production_environment(
            base_environment,
            canary=PRIVATE_CANARIES[2],
            log_level="ERROR",
        )
        unhealthy_started_wall = time.time_ns()
        unhealthy_started_at = time.monotonic()
        unhealthy_result = _promotion_cli(
            unhealthy_environment,
            state_directory=state_directory,
            checkout=checkout,
            release_checkout=unhealthy_release_checkout,
            project=production_project,
            revision=unhealthy_revision,
            image_reference=references["unhealthy"],
            image_repository=image_repository,
            adopt_existing=False,
            check=False,
        )
        unhealthy_elapsed = time.monotonic() - unhealthy_started_at
        unhealthy_finished_wall = time.time_ns()
        _assert_safe_output(unhealthy_result)
        assert unhealthy_result.returncode == 1
        assert unhealthy_result.stdout == ""
        assert unhealthy_result.stderr == (
            "ERROR: deployment command failed during candidate Compose start (exit 1)\n"
        )
        assert unhealthy_elapsed < CANDIDATE_FAILURE_BOUND_SECONDS
        unhealthy_container_id, candidate_started_at, candidate_unhealthy_at = (
            _candidate_lifecycle_evidence(
                docker,
                base_environment,
                image_reference=references["unhealthy"],
                release_checkout=unhealthy_release_checkout,
                since=(unhealthy_started_wall - 1_000_000_000) / 1_000_000_000,
                until=(unhealthy_finished_wall + 1_000_000_000) / 1_000_000_000,
            )
        )
        assert CONTAINER_ID_PATTERN.fullmatch(unhealthy_container_id)
        assert candidate_unhealthy_at <= unhealthy_finished_wall
        assert (
            candidate_unhealthy_at - candidate_started_at
            >= CANDIDATE_START_PERIOD_SECONDS * 1_000_000_000
        )

        after_unhealthy_container = _container_document(
            docker,
            base_environment,
            project=production_project,
        )
        after_unhealthy_container_id = after_unhealthy_container["Id"]
        assert isinstance(after_unhealthy_container_id, str)
        assert (
            _production_process_identity(
                docker,
                base_environment,
                after_unhealthy_container_id,
            )
            == good_process_identity
        )
        assert (
            _release_image_identity(
                _image_identity(
                    docker,
                    base_environment,
                    str(after_unhealthy_container["Image"]),
                )
            )
            == good_image_identity
        )
        assert _private_file_fingerprint(env_file) == good_environment_fingerprint
        _assert_private_file_contents(env_file, good_environment_bytes)
        assert (
            _production_volume_identity(
                docker,
                base_environment,
                after_unhealthy_container,
            )
            == good_volume_identity
        )
        assert (
            _read_volume_sentinel(
                docker,
                base_environment,
                after_unhealthy_container_id,
            )
            == VOLUME_SENTINEL
        )
        assert (
            _git(git, base_environment, "rev-parse", "HEAD", cwd=checkout)
            == good_revision
        )
        assert (
            _git(
                git,
                base_environment,
                "status",
                "--porcelain=v1",
                "--untracked-files=all",
                cwd=checkout,
            )
            == ""
        )
        assert store.read_current() == good_current
        assert store.read_journal() == good_journal
        assert _state_tree_fingerprint(state_directory) == good_state_tree
        assert not store.pending_path.exists()
        assert (
            _candidate_service_container_ids(
                docker,
                base_environment,
                image_id=release_images["unhealthy"],
                release_checkout=unhealthy_release_checkout,
            )
            == good_candidate_containers
        )
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

        failing_environment = _production_environment(
            base_environment,
            canary=PRIVATE_CANARIES[3],
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
            image_repository=image_repository,
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
        if not (
            isinstance(restored_container_id, str)
            and isinstance(restored_config, dict)
            and isinstance(restored_state, dict)
        ):
            raise AssertionError("restored production container identity was invalid")
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
        restored_environment_fingerprint = _private_file_fingerprint(env_file)
        _assert_private_file_contents(env_file, good_environment_bytes)
        assert restored_environment_fingerprint[1] == good_environment_fingerprint[1]
        assert restored_environment_fingerprint[3] == good_environment_fingerprint[3]
        assert restored_environment_fingerprint[4] == good_environment_fingerprint[4]
        assert (
            restored_environment_fingerprint[4] == good_current.state.environment_sha256
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
        durable_payloads = [
            promotion_intent_bytes,
            intent_bytes,
            store.current_path.read_bytes(),
            *(
                journal_path.read_bytes()
                for journal_path in store.journal_path.glob("*.json")
            ),
        ]
        for canary in PRIVATE_CANARIES:
            if any(canary.encode() in payload for payload in durable_payloads):
                raise AssertionError("durable deployment state exposed a canary")

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
                    owner=prefix,
                )
            )
        unhealthy_image_id = release_images.get("unhealthy")
        if unhealthy_image_id is not None:
            failures.extend(
                _cleanup_candidate_containers(
                    docker,
                    base_environment,
                    image_id=unhealthy_image_id,
                    release_checkout=unhealthy_release_checkout,
                )
            )
        failures.extend(
            _execute_exact_cleanup(
                docker,
                cleanup_targets,
                base_environment,
            )
        )
        _report_cleanup_failures(failures, primary_error)
