"""Read-only proof of an existing Compose deployment before state adoption."""

from __future__ import annotations

import json
import os
import re
import stat
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

_COMPOSE_NAME = re.compile(r"[a-z0-9][a-z0-9_-]{0,62}\Z")
_CONTAINER_ID = re.compile(r"[0-9a-f]{12,64}\Z")
_REVISION = re.compile(r"[0-9a-f]{40}\Z")
_IMAGE_ID = re.compile(r"sha256:[0-9a-f]{64}\Z")
_REPOSITORY_COMPONENT = r"[a-z0-9]+(?:(?:[._]|__|-+)[a-z0-9]+)*"
_REGISTRY_LABEL = r"[a-z0-9](?:[a-z0-9-]*[a-z0-9])?"
_IMAGE_REFERENCE = re.compile(
    rf"(?=.{{1,255}}@sha256:)"
    rf"(?:{_REGISTRY_LABEL}(?:\.{_REGISTRY_LABEL})*(?::[0-9]{{1,5}})?/)?"
    rf"{_REPOSITORY_COMPONENT}(?:/{_REPOSITORY_COMPONENT})*"
    r"@sha256:[0-9a-f]{64}\Z"
)
_GITHUB_ORIGIN = re.compile(r"https://github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+\Z")
_MAX_OUTPUT_BYTES = 256 * 1024
_HOST_ENVIRONMENT_NAMES = (
    "HOME",
    "PATH",
    "DOCKER_CONFIG",
    "DOCKER_HOST",
    "XDG_RUNTIME_DIR",
)


class DeploymentAdoptionError(RuntimeError):
    """Report a secret-free legacy deployment observation failure."""


@dataclass(frozen=True, slots=True)
class DeploymentObservation:
    """Exact read-only facts required to adopt one existing deployment."""

    checkout_path: Path
    origin: str
    revision: str
    compose_project: str
    compose_service: str
    environment_path: Path
    image_reference: str
    image_id: str
    oci_revision: str


def _validated_name(value: str, field: str) -> str:
    if _COMPOSE_NAME.fullmatch(value) is None:
        raise DeploymentAdoptionError(f"adoption input is invalid: {field}")
    return value


def _validated_executable(path: Path, field: str) -> Path:
    try:
        resolved = path.resolve(strict=True)
        metadata = resolved.stat()
    except OSError:
        raise DeploymentAdoptionError(
            f"adoption executable is unavailable: {field}"
        ) from None
    if (
        not path.is_absolute()
        or not stat.S_ISREG(metadata.st_mode)
        or not os.access(resolved, os.X_OK)
    ):
        raise DeploymentAdoptionError(f"adoption executable is invalid: {field}")
    return resolved


def _command_environment(
    environment: Mapping[str, str] | None,
) -> dict[str, str]:
    source = os.environ if environment is None else environment
    selected = {
        name: source[name] for name in _HOST_ENVIRONMENT_NAMES if name in source
    }
    selected.update(
        {
            "GIT_CONFIG_COUNT": "2",
            "GIT_CONFIG_GLOBAL": "/dev/null",
            "GIT_CONFIG_SYSTEM": "/dev/null",
            "GIT_CONFIG_KEY_0": "core.hooksPath",
            "GIT_CONFIG_KEY_1": "core.fsmonitor",
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_NO_REPLACE_OBJECTS": "1",
            "GIT_CONFIG_VALUE_0": "/dev/null",
            "GIT_CONFIG_VALUE_1": "false",
            "GIT_OPTIONAL_LOCKS": "0",
            "GIT_TERMINAL_PROMPT": "0",
            "LC_ALL": "C",
            "LANG": "C",
        }
    )
    return selected


def _run(
    executable: Path,
    arguments: Sequence[str],
    *,
    environment: Mapping[str, str],
    cwd: Path | None = None,
    accepted_returncodes: frozenset[int] = frozenset({0}),
) -> subprocess.CompletedProcess[str]:
    try:
        result = subprocess.run(  # noqa: S603 - resolved executable, fixed arguments
            [str(executable), *arguments],
            cwd=cwd,
            env=environment,
            text=True,
            capture_output=True,
            check=False,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        raise DeploymentAdoptionError(
            "legacy deployment observation command failed"
        ) from None
    if result.returncode not in accepted_returncodes:
        raise DeploymentAdoptionError("legacy deployment observation command failed")
    if len(result.stdout.encode()) > _MAX_OUTPUT_BYTES:
        raise DeploymentAdoptionError(
            "legacy deployment observation output is too large"
        )
    return result


def _one_line(value: str, field: str) -> str:
    selected = value[:-1] if value.endswith("\n") else value
    if not selected or "\n" in selected or "\r" in selected or "\0" in selected:
        raise DeploymentAdoptionError(f"legacy deployment output is invalid: {field}")
    return selected


def _json_document(value: str, field: str) -> dict[str, object]:
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError:
        raise DeploymentAdoptionError(
            f"legacy deployment output is invalid: {field}"
        ) from None
    if not isinstance(decoded, list) or len(decoded) != 1:
        raise DeploymentAdoptionError(f"legacy deployment output is invalid: {field}")
    document = decoded[0]
    if not isinstance(document, dict):
        raise DeploymentAdoptionError(f"legacy deployment output is invalid: {field}")
    return document


def _nested_mapping(
    document: Mapping[str, object],
    key: str,
    field: str,
) -> Mapping[str, object]:
    value = document.get(key)
    if not isinstance(value, dict):
        raise DeploymentAdoptionError(f"legacy deployment output is invalid: {field}")
    return value


def _string_field(
    document: Mapping[str, object],
    key: str,
    field: str,
) -> str:
    value = document.get(key)
    if not isinstance(value, str):
        raise DeploymentAdoptionError(f"legacy deployment output is invalid: {field}")
    return value


def _validate_checkout(path: Path) -> Path:
    if not path.is_absolute():
        raise DeploymentAdoptionError("adoption checkout path must be absolute")
    try:
        resolved = path.resolve(strict=True)
        metadata = resolved.stat()
    except OSError:
        raise DeploymentAdoptionError("adoption checkout path is unavailable") from None
    if not stat.S_ISDIR(metadata.st_mode):
        raise DeploymentAdoptionError("adoption checkout path is not a directory")
    return resolved


def _validate_environment(path: Path, checkout: Path) -> Path:
    expected = checkout / ".env"
    if not path.is_absolute() or path.resolve(strict=False) != expected:
        raise DeploymentAdoptionError("legacy environment path must be checkout .env")
    try:
        metadata = path.lstat()
    except OSError:
        raise DeploymentAdoptionError(
            "legacy environment file is unavailable"
        ) from None
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or stat.S_IMODE(metadata.st_mode) != 0o600
        or metadata.st_nlink != 1
        or metadata.st_size <= 0
    ):
        raise DeploymentAdoptionError("legacy environment file is unsafe")
    return path


def _git_facts(
    git: Path,
    checkout: Path,
    expected_origin: str,
    environment: Mapping[str, str],
) -> tuple[str, str]:
    top_level_value = _one_line(
        _run(
            git,
            ["-C", str(checkout), "rev-parse", "--show-toplevel"],
            environment=environment,
        ).stdout,
        "checkout root",
    )
    try:
        top_level = Path(top_level_value).resolve(strict=True)
    except OSError:
        raise DeploymentAdoptionError("legacy checkout root is unavailable") from None
    if top_level != checkout:
        raise DeploymentAdoptionError("legacy checkout root does not match")

    origin = _one_line(
        _run(
            git,
            ["-C", str(checkout), "remote", "get-url", "origin"],
            environment=environment,
        ).stdout,
        "origin",
    )
    if origin not in {expected_origin, f"{expected_origin}.git"}:
        raise DeploymentAdoptionError("legacy checkout origin does not match")

    for arguments in (
        [
            "-C",
            str(checkout),
            "diff",
            "--no-ext-diff",
            "--no-textconv",
            "--quiet",
            "--",
        ],
        [
            "-C",
            str(checkout),
            "diff",
            "--cached",
            "--no-ext-diff",
            "--no-textconv",
            "--quiet",
            "--",
        ],
    ):
        result = _run(
            git,
            arguments,
            environment=environment,
            accepted_returncodes=frozenset({0, 1}),
        )
        if result.returncode == 1:
            raise DeploymentAdoptionError("legacy checkout has tracked changes")

    revision = _one_line(
        _run(
            git,
            ["-C", str(checkout), "rev-parse", "--verify", "HEAD"],
            environment=environment,
        ).stdout,
        "revision",
    )
    if _REVISION.fullmatch(revision) is None:
        raise DeploymentAdoptionError("legacy checkout revision is invalid")
    return expected_origin, revision


def _container_ids(
    docker: Path,
    project: str,
    service: str,
    environment: Mapping[str, str],
) -> tuple[str, ...]:
    result = _run(
        docker,
        [
            "container",
            "ls",
            "--all",
            "--filter",
            f"label=com.docker.compose.project={project}",
            "--filter",
            f"label=com.docker.compose.service={service}",
            "--format",
            "{{.ID}}",
        ],
        environment=environment,
    )
    lines = tuple(line for line in result.stdout.splitlines() if line)
    if any(_CONTAINER_ID.fullmatch(line) is None for line in lines):
        raise DeploymentAdoptionError("legacy container list is invalid")
    return lines


def _container_facts(
    docker: Path,
    container_id: str,
    checkout: Path,
    project: str,
    service: str,
    environment: Mapping[str, str],
) -> tuple[str, str]:
    document = _json_document(
        _run(
            docker,
            ["container", "inspect", container_id],
            environment=environment,
        ).stdout,
        "container",
    )
    full_id = _string_field(document, "Id", "container ID")
    if (
        len(full_id) != 64
        or _CONTAINER_ID.fullmatch(full_id) is None
        or not full_id.startswith(container_id)
    ):
        raise DeploymentAdoptionError("legacy container identity is invalid")

    state = _nested_mapping(document, "State", "container state")
    if _string_field(state, "Status", "container status") != "running":
        raise DeploymentAdoptionError("legacy container is not running")
    health = _nested_mapping(state, "Health", "container health")
    if _string_field(health, "Status", "container health status") != "healthy":
        raise DeploymentAdoptionError("legacy container is not healthy")

    config = _nested_mapping(document, "Config", "container configuration")
    image_reference = _string_field(config, "Image", "container image reference")
    if _IMAGE_REFERENCE.fullmatch(image_reference) is None:
        raise DeploymentAdoptionError("legacy container image is not immutable")
    labels = _nested_mapping(config, "Labels", "container labels")
    expected_labels = {
        "com.docker.compose.project": project,
        "com.docker.compose.service": service,
        "com.docker.compose.project.working_dir": str(checkout),
    }
    for key, expected in expected_labels.items():
        if labels.get(key) != expected:
            raise DeploymentAdoptionError("legacy container labels do not match")

    image_id = _string_field(document, "Image", "container image ID")
    if _IMAGE_ID.fullmatch(image_id) is None:
        raise DeploymentAdoptionError("legacy container image ID is invalid")
    return image_reference, image_id


def _image_revision(
    docker: Path,
    image_id: str,
    image_reference: str,
    environment: Mapping[str, str],
) -> str:
    document = _json_document(
        _run(
            docker,
            ["image", "inspect", image_id],
            environment=environment,
        ).stdout,
        "image",
    )
    inspected_id = _string_field(document, "Id", "image ID")
    if inspected_id != image_id:
        raise DeploymentAdoptionError("legacy image identity does not match")
    repo_digests = document.get("RepoDigests")
    if (
        not isinstance(repo_digests, list)
        or not all(isinstance(value, str) for value in repo_digests)
        or image_reference not in repo_digests
    ):
        raise DeploymentAdoptionError("legacy image digest is not locally proven")
    config = _nested_mapping(document, "Config", "image configuration")
    labels = _nested_mapping(config, "Labels", "image labels")
    revision = labels.get("org.opencontainers.image.revision")
    if not isinstance(revision, str) or _REVISION.fullmatch(revision) is None:
        raise DeploymentAdoptionError("legacy image OCI revision is invalid")
    return revision


def observe_legacy_deployment(
    *,
    checkout_path: Path,
    expected_origin: str,
    compose_project: str,
    compose_service: str,
    environment_path: Path,
    git_executable: Path,
    docker_executable: Path,
    environment: Mapping[str, str] | None = None,
) -> DeploymentObservation | None:
    """Prove one existing deployment without mutating Git, Docker, or its env."""
    checkout = _validate_checkout(checkout_path)
    if _GITHUB_ORIGIN.fullmatch(expected_origin) is None or expected_origin.endswith(
        ".git"
    ):
        raise DeploymentAdoptionError("expected GitHub origin is invalid")
    project = _validated_name(compose_project, "compose_project")
    service = _validated_name(compose_service, "compose_service")
    git = _validated_executable(git_executable, "git")
    docker = _validated_executable(docker_executable, "docker")
    command_environment = _command_environment(environment)
    origin, revision = _git_facts(
        git,
        checkout,
        expected_origin,
        command_environment,
    )
    container_ids = _container_ids(
        docker,
        project,
        service,
        command_environment,
    )
    if not container_ids:
        return None
    if len(container_ids) != 1:
        raise DeploymentAdoptionError("legacy deployment is ambiguous")

    selected_environment = _validate_environment(environment_path, checkout)
    image_reference, image_id = _container_facts(
        docker,
        container_ids[0],
        checkout,
        project,
        service,
        command_environment,
    )
    oci_revision = _image_revision(
        docker,
        image_id,
        image_reference,
        command_environment,
    )
    return DeploymentObservation(
        checkout_path=checkout,
        origin=origin,
        revision=revision,
        compose_project=project,
        compose_service=service,
        environment_path=selected_environment,
        image_reference=image_reference,
        image_id=image_id,
        oci_revision=oci_revision,
    )
