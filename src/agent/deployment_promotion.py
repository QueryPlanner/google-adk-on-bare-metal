"""Lock-held promotion of one exact image with verified rollback."""

from __future__ import annotations

import argparse
import errno
import fcntl
import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import uuid
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path, PurePosixPath
from typing import BinaryIO, Final

from agent.compose_env import (
    ComposeEnvironmentError,
    parse_compose_environment,
    write_compose_environment,
)
from agent.deployment_adoption import (
    DeploymentAdoptionError,
    DeploymentObservation,
    observe_legacy_deployment,
)
from agent.deployment_state import (
    CandidateReceipt,
    CurrentDeployment,
    DeploymentLockBusyError,
    DeploymentState,
    DeploymentStateError,
    DeploymentStateStore,
    DeploymentStateTransaction,
    DeploymentTerminalCommittedError,
    DeploymentTerminalIndeterminateError,
    PendingPromotion,
    PersistentVolumeIdentity,
)

PRODUCTION_ENVIRONMENT_NAMES: Final = (
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
)

_CANDIDATE_ENVIRONMENT_NAMES: Final = (
    "AGENT_NAME",
    "ROOT_AGENT_MODEL",
    "LOG_LEVEL",
    "TELEMETRY_NAMESPACE",
    "K_REVISION",
    "CANDIDATE_ENV_CANARY",
)
_HOST_ENVIRONMENT_NAMES: Final = (
    "HOME",
    "PATH",
    "DOCKER_CONFIG",
    "DOCKER_HOST",
    "XDG_RUNTIME_DIR",
)
_MAX_SERIALIZED_ENVIRONMENT_BYTES: Final = 64 * 1024
_COMPOSE_NAME = re.compile(r"[a-z0-9][a-z0-9_-]{0,62}\Z")
_REVISION = re.compile(r"[0-9a-f]{40}\Z")
_IMAGE_ID = re.compile(r"sha256:[0-9a-f]{64}\Z")
_CONTAINER_ID = re.compile(r"[0-9a-f]{64}\Z")
_CONTROLLER_TRANSACTION_ID = re.compile(
    r"[0-9a-f]{23}-(?P<baseline_revision>[0-9a-f]{40})\Z"
)
_REPOSITORY_COMPONENT = r"[a-z0-9]+(?:(?:[._]|__|-+)[a-z0-9]+)*"
_REGISTRY_LABEL = r"[a-z0-9](?:[a-z0-9-]*[a-z0-9])?"
_IMAGE_REFERENCE = re.compile(
    rf"(?=.{{1,255}}@sha256:)"
    rf"(?:{_REGISTRY_LABEL}(?:\.{_REGISTRY_LABEL})*(?::[0-9]{{1,5}})?/)?"
    rf"{_REPOSITORY_COMPONENT}(?:/{_REPOSITORY_COMPONENT})*"
    r"@sha256:[0-9a-f]{64}\Z"
)
_EXPECTED_ORIGIN = re.compile(r"https://github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+\Z")
_VOLUME_CREATED_AT = re.compile(
    r"[0-9]{4}-[0-9]{2}-[0-9]{2}T"
    r"[0-9]{2}:[0-9]{2}:[0-9]{2}"
    r"(?:\.[0-9]{1,9})?(?:Z|[+-][0-9]{2}:[0-9]{2})\Z"
)
_MAX_COMMAND_OUTPUT_BYTES: Final = 256 * 1024
_MAX_VOLUME_FIELD_BYTES: Final = 4096
_COMMAND_TIMEOUT_SECONDS: Final = 240


class PromotionError(RuntimeError):
    """Report a deterministic, secret-free promotion failure."""


class PromotionRolledBackError(PromotionError):
    """Report a failed promotion whose exact baseline was restored."""


class PromotionRecoveryRequiredError(PromotionError):
    """Report completed recovery that deliberately requires a fresh invocation."""


class PromotionRecoveryFailedError(PromotionError):
    """Report recovery that could not prove the recorded baseline."""


class CommandOperation(StrEnum):
    """Fixed, secret-free labels for external promotion boundaries."""

    GIT_METADATA = "Git metadata read"
    GIT_DIFF = "Git checkout diff"
    GIT_STATUS = "Git release status"
    GIT_MANIFEST = "Git release manifest"
    GIT_CHECKOUT = "Git production checkout"
    IMAGE_PULL = "Docker image pull"
    IMAGE_INSPECT = "Docker image inspect"
    CONTAINER_LIST = "Docker container list"
    CONTAINER_INSPECT = "Docker container inspect"
    CONTAINER_REMOVE = "Docker container removal"
    VOLUME_INSPECT = "Docker volume inspect"
    CANDIDATE_CONFIG = "candidate Compose validation"
    CANDIDATE_START = "candidate Compose start"
    CANDIDATE_CLEANUP = "candidate Compose cleanup"
    PRODUCTION_CONFIG = "production Compose validation"
    PRODUCTION_START = "production Compose start"


@dataclass(frozen=True, slots=True)
class PromotionConfig:
    """Validated host paths and immutable promotion identity."""

    state_dir: Path
    checkout: Path
    release_checkout: Path
    expected_origin: str
    compose_project: str
    compose_service: str
    source_revision: str
    image_reference: str
    adopt_existing: bool


@dataclass(frozen=True, slots=True)
class Executables:
    """Resolved external command boundaries."""

    git: Path
    docker: Path


@dataclass(frozen=True, slots=True)
class ImageProof:
    """Locally observed immutable image identity."""

    image_reference: str
    image_id: str
    oci_revision: str


@dataclass(frozen=True, slots=True, order=True)
class VolumeProof:
    """Runtime and daemon fields that detect replacement of a named volume."""

    name: str
    driver: str
    destination: str
    mountpoint: str
    created_at: str

    def recorded_identity(self) -> PersistentVolumeIdentity:
        """Return the complete durable daemon and mount identity."""
        return PersistentVolumeIdentity(
            name=self.name,
            driver=self.driver,
            mountpoint=self.mountpoint,
            destination=self.destination,
            created_at=self.created_at,
        )


@dataclass(frozen=True, slots=True)
class RuntimeProof:
    """One independently observed healthy Compose service."""

    observation: DeploymentObservation
    container_id: str
    volumes: tuple[VolumeProof, ...]


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _validated_config(config: PromotionConfig) -> PromotionConfig:
    for path, field in (
        (config.state_dir, "state directory"),
        (config.checkout, "production checkout"),
        (config.release_checkout, "release checkout"),
    ):
        if not path.is_absolute() or path != path.resolve(strict=False):
            raise PromotionError(f"{field} must be an absolute normalized path")
    if config.checkout == config.release_checkout:
        raise PromotionError("production and release checkouts must be distinct")
    for path, field in (
        (config.checkout, "production checkout"),
        (config.release_checkout, "release checkout"),
    ):
        try:
            metadata = path.stat()
        except OSError:
            raise PromotionError(f"{field} is unavailable") from None
        if not stat.S_ISDIR(metadata.st_mode):
            raise PromotionError(f"{field} is not a directory")
    if _EXPECTED_ORIGIN.fullmatch(
        config.expected_origin
    ) is None or config.expected_origin.endswith(".git"):
        raise PromotionError("expected origin is invalid")
    for value, field in (
        (config.compose_project, "Compose project"),
        (config.compose_service, "Compose service"),
    ):
        if _COMPOSE_NAME.fullmatch(value) is None:
            raise PromotionError(f"{field} is invalid")
    if _REVISION.fullmatch(config.source_revision) is None:
        raise PromotionError("source revision is invalid")
    if _IMAGE_REFERENCE.fullmatch(config.image_reference) is None:
        raise PromotionError("image reference is not an immutable digest")
    return config


def _command_environment(source: Mapping[str, str]) -> dict[str, str]:
    selected = {
        name: source[name] for name in _HOST_ENVIRONMENT_NAMES if name in source
    }
    selected.update(
        {
            "COMPOSE_DISABLE_ENV_FILE": "1",
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
            "LANG": "C",
            "LC_ALL": "C",
        }
    )
    return selected


def _resolve_executable(name: str, environment: Mapping[str, str]) -> Path:
    resolved = shutil.which(name, path=environment.get("PATH"))
    if resolved is None:
        raise PromotionError(f"required executable is unavailable: {name}")
    path = Path(resolved)
    try:
        canonical = path.resolve(strict=True)
        metadata = canonical.stat()
    except OSError:
        raise PromotionError(f"required executable is unavailable: {name}") from None
    if not canonical.is_absolute() or not stat.S_ISREG(metadata.st_mode):
        raise PromotionError(f"required executable is invalid: {name}")
    return canonical


def _run(
    executable: Path,
    arguments: Sequence[str],
    *,
    operation: CommandOperation,
    environment: Mapping[str, str],
    cwd: Path | None = None,
    accepted_returncodes: frozenset[int] = frozenset({0}),
    timeout: int = _COMMAND_TIMEOUT_SECONDS,
) -> subprocess.CompletedProcess[str]:
    try:
        result = subprocess.run(  # noqa: S603 - resolved executable, bounded args
            [str(executable), *arguments],
            cwd=cwd,
            env=environment,
            text=True,
            capture_output=True,
            check=False,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        raise PromotionError(
            f"deployment command timed out during {operation.value}"
        ) from None
    except OSError:
        raise PromotionError(
            f"deployment command could not start during {operation.value}"
        ) from None
    if result.returncode not in accepted_returncodes:
        raise PromotionError(
            "deployment command failed during "
            f"{operation.value} (exit {result.returncode})"
        )
    if (
        len(result.stdout.encode()) > _MAX_COMMAND_OUTPUT_BYTES
        or len(result.stderr.encode()) > _MAX_COMMAND_OUTPUT_BYTES
    ):
        raise PromotionError(
            f"deployment command output is too large during {operation.value}"
        )
    return result


def _one_line(value: str, field: str) -> str:
    selected = value[:-1] if value.endswith("\n") else value
    if not selected or any(character in selected for character in "\0\r\n"):
        raise PromotionError(f"deployment command output is invalid: {field}")
    return selected


def _json_object(value: str, field: str) -> dict[str, object]:
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError:
        raise PromotionError(f"deployment command output is invalid: {field}") from None
    if not isinstance(decoded, list) or len(decoded) != 1:
        raise PromotionError(f"deployment command output is invalid: {field}")
    selected = decoded[0]
    if not isinstance(selected, dict):
        raise PromotionError(f"deployment command output is invalid: {field}")
    return selected


def _mapping(
    document: Mapping[str, object],
    key: str,
    field: str,
) -> Mapping[str, object]:
    value = document.get(key)
    if not isinstance(value, dict):
        raise PromotionError(f"deployment command output is invalid: {field}")
    return value


def _string(
    document: Mapping[str, object],
    key: str,
    field: str,
) -> str:
    value = document.get(key)
    if not isinstance(value, str):
        raise PromotionError(f"deployment command output is invalid: {field}")
    return value


def _git_line(
    executables: Executables,
    checkout: Path,
    arguments: Sequence[str],
    environment: Mapping[str, str],
    field: str,
) -> str:
    return _one_line(
        _run(
            executables.git,
            ["-C", str(checkout), *arguments],
            operation=CommandOperation.GIT_METADATA,
            environment=environment,
        ).stdout,
        field,
    )


def _validate_release_checkout(
    config: PromotionConfig,
    executables: Executables,
    environment: Mapping[str, str],
) -> None:
    root_value = _git_line(
        executables,
        config.release_checkout,
        ["rev-parse", "--show-toplevel"],
        environment,
        "release root",
    )
    try:
        root = Path(root_value).resolve(strict=True)
    except OSError:
        raise PromotionError("release checkout root is unavailable") from None
    if root != config.release_checkout:
        raise PromotionError("release checkout root does not match")
    origin = _git_line(
        executables,
        config.release_checkout,
        ["remote", "get-url", "origin"],
        environment,
        "release origin",
    )
    if origin not in {config.expected_origin, f"{config.expected_origin}.git"}:
        raise PromotionError("release checkout origin does not match")
    revision = _git_line(
        executables,
        config.release_checkout,
        ["rev-parse", "--verify", "HEAD"],
        environment,
        "release revision",
    )
    if revision != config.source_revision:
        raise PromotionError("release checkout revision does not match")
    for arguments in (
        ["diff", "--no-ext-diff", "--no-textconv", "--quiet", "--"],
        [
            "diff",
            "--cached",
            "--no-ext-diff",
            "--no-textconv",
            "--quiet",
            "--",
        ],
    ):
        result = _run(
            executables.git,
            ["-C", str(config.release_checkout), *arguments],
            operation=CommandOperation.GIT_DIFF,
            environment=environment,
            accepted_returncodes=frozenset({0, 1}),
        )
        if result.returncode == 1:
            raise PromotionError("release checkout has tracked changes")
    status = _run(
        executables.git,
        [
            "-C",
            str(config.release_checkout),
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
            "--ignored=matching",
        ],
        operation=CommandOperation.GIT_STATUS,
        environment=environment,
    )
    if status.stdout:
        raise PromotionError("release checkout contains untracked or ignored files")
    _run(
        executables.git,
        [
            "-C",
            str(config.release_checkout),
            "ls-files",
            "--error-unmatch",
            "--",
            "compose.yaml",
            "compose.candidate.yaml",
            "src/agent/deployment_promotion.py",
        ],
        operation=CommandOperation.GIT_MANIFEST,
        environment=environment,
    )


def _validate_production_checkout(
    config: PromotionConfig,
    executables: Executables,
    environment: Mapping[str, str],
) -> None:
    root_value = _git_line(
        executables,
        config.checkout,
        ["rev-parse", "--show-toplevel"],
        environment,
        "production root",
    )
    try:
        root = Path(root_value).resolve(strict=True)
    except OSError:
        raise PromotionError("production checkout root is unavailable") from None
    if root != config.checkout:
        raise PromotionError("production checkout root does not match")
    origin = _git_line(
        executables,
        config.checkout,
        ["remote", "get-url", "origin"],
        environment,
        "production origin",
    )
    if origin not in {config.expected_origin, f"{config.expected_origin}.git"}:
        raise PromotionError("production checkout origin does not match")
    for arguments in (
        ["diff", "--no-ext-diff", "--no-textconv", "--quiet", "--"],
        [
            "diff",
            "--cached",
            "--no-ext-diff",
            "--no-textconv",
            "--quiet",
            "--",
        ],
    ):
        result = _run(
            executables.git,
            ["-C", str(config.checkout), *arguments],
            operation=CommandOperation.GIT_DIFF,
            environment=environment,
            accepted_returncodes=frozenset({0, 1}),
        )
        if result.returncode == 1:
            raise PromotionError("production checkout has tracked changes")


def _validate_pending_ownership(
    *,
    pending: PendingPromotion,
    current: CurrentDeployment | None,
    config: PromotionConfig,
    executables: Executables,
    environment: Mapping[str, str],
) -> None:
    target = pending.intent.target
    if (
        target.compose_project != config.compose_project
        or target.compose_service != config.compose_service
    ):
        raise PromotionError("pending promotion Compose identity does not match")
    checkout_revision = _git_line(
        executables,
        config.checkout,
        ["rev-parse", "--verify", "HEAD"],
        environment,
        "pending production revision",
    )
    if _REVISION.fullmatch(checkout_revision) is None:
        raise PromotionError("pending production checkout revision is invalid")
    allowed_revisions = {target.source_revision}
    if current is not None:
        if (
            current.state.compose_project != config.compose_project
            or current.state.compose_service != config.compose_service
        ):
            raise PromotionError("pending baseline Compose identity does not match")
        allowed_revisions.add(current.state.source_revision)
    else:
        allowed_revisions.add(_transaction_baseline_revision(pending))
    if checkout_revision not in allowed_revisions:
        raise PromotionError("pending production checkout ownership does not match")
    ids = _container_ids(
        project=config.compose_project,
        service=config.compose_service,
        executables=executables,
        environment=environment,
    )
    if len(ids) > 1:
        raise PromotionError("pending production deployment is ambiguous")
    if not ids:
        return
    document = _json_object(
        _run(
            executables.docker,
            ["container", "inspect", ids[0]],
            operation=CommandOperation.CONTAINER_INSPECT,
            environment=environment,
        ).stdout,
        "pending production container",
    )
    full_id = _string(document, "Id", "pending production container ID")
    config_document = _mapping(
        document,
        "Config",
        "pending production container configuration",
    )
    labels = _mapping(
        config_document,
        "Labels",
        "pending production container labels",
    )
    allowed_images = {
        (target.image_reference, target.image_id),
    }
    if current is not None:
        allowed_images.add(
            (current.state.image_reference, current.state.image_id),
        )
    if (
        _CONTAINER_ID.fullmatch(full_id) is None
        or not full_id.startswith(ids[0])
        or labels.get("com.docker.compose.project") != config.compose_project
        or labels.get("com.docker.compose.service") != config.compose_service
        or labels.get("com.docker.compose.project.working_dir") != str(config.checkout)
        or (config_document.get("Image"), document.get("Image")) not in allowed_images
    ):
        raise PromotionError("pending production container ownership does not match")


def _inspect_image(
    *,
    proof_reference: str,
    expected_revision: str,
    executables: Executables,
    environment: Mapping[str, str],
) -> ImageProof:
    document = _json_object(
        _run(
            executables.docker,
            ["image", "inspect", proof_reference],
            operation=CommandOperation.IMAGE_INSPECT,
            environment=environment,
        ).stdout,
        "image",
    )
    image_id = _string(document, "Id", "image ID")
    if _IMAGE_ID.fullmatch(image_id) is None:
        raise PromotionError("local image ID is invalid")
    repo_digests = document.get("RepoDigests")
    if (
        not isinstance(repo_digests, list)
        or not all(isinstance(value, str) for value in repo_digests)
        or proof_reference not in repo_digests
    ):
        raise PromotionError("local image digest does not match")
    config = _mapping(document, "Config", "image configuration")
    labels = _mapping(config, "Labels", "image labels")
    oci_revision = labels.get("org.opencontainers.image.revision")
    if oci_revision != expected_revision:
        raise PromotionError("image OCI revision does not match")
    return ImageProof(
        image_reference=proof_reference,
        image_id=image_id,
        oci_revision=expected_revision,
    )


def _pull_and_prove_image(
    config: PromotionConfig,
    executables: Executables,
    environment: Mapping[str, str],
) -> ImageProof:
    _run(
        executables.docker,
        ["image", "pull", config.image_reference],
        operation=CommandOperation.IMAGE_PULL,
        environment=environment,
    )
    return _inspect_image(
        proof_reference=config.image_reference,
        expected_revision=config.source_revision,
        executables=executables,
        environment=environment,
    )


def _container_ids(
    *,
    project: str,
    service: str,
    executables: Executables,
    environment: Mapping[str, str],
) -> tuple[str, ...]:
    result = _run(
        executables.docker,
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
        operation=CommandOperation.CONTAINER_LIST,
        environment=environment,
    )
    values = tuple(line for line in result.stdout.splitlines() if line)
    if any(
        not (12 <= len(value) <= 64) or re.fullmatch(r"[0-9a-f]+", value) is None
        for value in values
    ):
        raise PromotionError("Compose container list is invalid")
    return values


def _volume_proofs(
    *,
    mounts: object,
    executables: Executables,
    environment: Mapping[str, str],
) -> tuple[VolumeProof, ...]:
    if not isinstance(mounts, list):
        raise PromotionError("container mount observation is invalid")
    selected: list[VolumeProof] = []
    for raw_mount in mounts:
        if not isinstance(raw_mount, dict):
            raise PromotionError("container mount observation is invalid")
        if raw_mount.get("Type") != "volume":
            continue
        name = _string(raw_mount, "Name", "volume name")
        destination = _string(raw_mount, "Destination", "volume destination")
        driver = _string(raw_mount, "Driver", "volume driver")
        if (
            not name
            or not driver
            or not destination.startswith("/")
            or destination == "/"
            or str(PurePosixPath(destination)) != destination
        ):
            raise PromotionError("named volume mount identity is invalid")
        document = _json_object(
            _run(
                executables.docker,
                ["volume", "inspect", name],
                operation=CommandOperation.VOLUME_INSPECT,
                environment=environment,
            ).stdout,
            "volume",
        )
        if (
            _string(document, "Name", "volume name") != name
            or _string(document, "Driver", "volume driver") != driver
        ):
            raise PromotionError("named volume daemon identity does not match")
        mountpoint = _string(document, "Mountpoint", "volume mountpoint")
        raw_created_at = _string(document, "CreatedAt", "volume creation time")
        if (
            not mountpoint.startswith("/")
            or mountpoint == "/"
            or str(PurePosixPath(mountpoint)) != mountpoint
            or len(mountpoint.encode()) > _MAX_VOLUME_FIELD_BYTES
            or not raw_created_at
            or len(raw_created_at.encode()) > _MAX_VOLUME_FIELD_BYTES
        ):
            raise PromotionError("named volume daemon identity is invalid")
        try:
            parsed_created_at = datetime.fromisoformat(
                raw_created_at.replace("Z", "+00:00")
            )
        except ValueError:
            raise PromotionError("named volume creation time is invalid") from None
        if (
            _VOLUME_CREATED_AT.fullmatch(raw_created_at) is None
            or parsed_created_at.tzinfo is None
        ):
            raise PromotionError("named volume creation time is invalid")
        selected.append(
            VolumeProof(
                name=name,
                driver=driver,
                destination=destination,
                mountpoint=mountpoint,
                created_at=raw_created_at,
            )
        )
    result = tuple(sorted(selected))
    if len({value.name for value in result}) != len(result) or len(
        {value.destination for value in result}
    ) != len(result):
        raise PromotionError("named volume mount identities are ambiguous")
    return result


def _inspect_container(
    *,
    container_id: str,
    expected_checkout: Path,
    expected_project: str,
    expected_service: str,
    image: ImageProof,
    executables: Executables,
    environment: Mapping[str, str],
    require_no_mounts: bool = False,
) -> tuple[str, tuple[VolumeProof, ...]]:
    document = _json_object(
        _run(
            executables.docker,
            ["container", "inspect", container_id],
            operation=CommandOperation.CONTAINER_INSPECT,
            environment=environment,
        ).stdout,
        "container",
    )
    full_id = _string(document, "Id", "container ID")
    if _CONTAINER_ID.fullmatch(full_id) is None or not full_id.startswith(container_id):
        raise PromotionError("container identity is invalid")
    state = _mapping(document, "State", "container state")
    health = _mapping(state, "Health", "container health")
    if state.get("Status") != "running" or health.get("Status") != "healthy":
        raise PromotionError("container is not running and healthy")
    config = _mapping(document, "Config", "container configuration")
    if (
        config.get("Image") != image.image_reference
        or document.get("Image") != image.image_id
    ):
        raise PromotionError("container image identity does not match")
    labels = _mapping(config, "Labels", "container labels")
    expected_labels = {
        "com.docker.compose.project": expected_project,
        "com.docker.compose.service": expected_service,
        "com.docker.compose.project.working_dir": str(expected_checkout),
    }
    if any(labels.get(name) != value for name, value in expected_labels.items()):
        raise PromotionError("container Compose labels do not match")
    mounts = document.get("Mounts")
    if require_no_mounts:
        host_config = _mapping(document, "HostConfig", "candidate host configuration")
        restart_policy = _mapping(
            host_config,
            "RestartPolicy",
            "candidate restart policy",
        )
        if (
            mounts != []
            or config.get("User") != "1000:1000"
            or host_config.get("NetworkMode") != "none"
            or host_config.get("PortBindings") not in (None, {})
            or restart_policy.get("Name") != "no"
            or host_config.get("CapDrop") != ["ALL"]
            or host_config.get("SecurityOpt") != ["no-new-privileges:true"]
        ):
            raise PromotionError("candidate isolation boundary does not match")
    return full_id, _volume_proofs(
        mounts=mounts,
        executables=executables,
        environment=environment,
    )


def _observe_runtime(
    *,
    config: PromotionConfig,
    image: ImageProof,
    executables: Executables,
    environment: Mapping[str, str],
) -> RuntimeProof | None:
    try:
        observation = observe_legacy_deployment(
            checkout_path=config.checkout,
            expected_origin=config.expected_origin,
            compose_project=config.compose_project,
            compose_service=config.compose_service,
            environment_path=config.checkout / ".env",
            git_executable=executables.git,
            docker_executable=executables.docker,
            environment=environment,
        )
    except DeploymentAdoptionError:
        raise PromotionError("production deployment observation failed") from None
    if observation is None:
        return None
    ids = _container_ids(
        project=config.compose_project,
        service=config.compose_service,
        executables=executables,
        environment=environment,
    )
    if len(ids) != 1:
        raise PromotionError("production deployment is ambiguous")
    container_id, volumes = _inspect_container(
        container_id=ids[0],
        expected_checkout=config.checkout,
        expected_project=config.compose_project,
        expected_service=config.compose_service,
        image=image,
        executables=executables,
        environment=environment,
    )
    if (
        observation.image_reference != image.image_reference
        or observation.image_id != image.image_id
        or observation.oci_revision != image.oci_revision
    ):
        raise PromotionError("production observation image does not match")
    return RuntimeProof(
        observation=observation,
        container_id=container_id,
        volumes=volumes,
    )


def _proof_from_observation(observation: DeploymentObservation) -> ImageProof:
    return ImageProof(
        image_reference=observation.image_reference,
        image_id=observation.image_id,
        oci_revision=observation.oci_revision,
    )


def _environment_sha256(path: Path) -> str:
    try:
        payload = path.read_bytes()
    except OSError:
        raise PromotionError("production environment could not be verified") from None
    if not payload:
        raise PromotionError("production environment could not be verified")
    return hashlib.sha256(payload).hexdigest()


def _matches_state(runtime: RuntimeProof, current: CurrentDeployment) -> bool:
    state = current.state
    observation = runtime.observation
    return (
        observation.compose_project == state.compose_project
        and observation.compose_service == state.compose_service
        and observation.revision == state.source_revision
        and observation.image_reference == state.image_reference
        and observation.image_id == state.image_id
        and observation.oci_revision == state.oci_revision
        and _environment_sha256(observation.environment_path)
        == state.environment_sha256
    )


def _baseline(
    *,
    transaction: DeploymentStateTransaction,
    config: PromotionConfig,
    executables: Executables,
    environment: Mapping[str, str],
) -> tuple[CurrentDeployment | None, RuntimeProof | None, str]:
    checkout_revision = _git_line(
        executables,
        config.checkout,
        ["rev-parse", "--verify", "HEAD"],
        environment,
        "production revision",
    )
    if _REVISION.fullmatch(checkout_revision) is None:
        raise PromotionError("production checkout revision is invalid")
    current = transaction.current()
    try:
        initial = observe_legacy_deployment(
            checkout_path=config.checkout,
            expected_origin=config.expected_origin,
            compose_project=config.compose_project,
            compose_service=config.compose_service,
            environment_path=config.checkout / ".env",
            git_executable=executables.git,
            docker_executable=executables.docker,
            environment=environment,
        )
    except DeploymentAdoptionError:
        raise PromotionError("production deployment observation failed") from None
    if initial is None:
        if current is not None:
            raise PromotionError("recorded production deployment is unavailable")
        if (config.checkout / ".env").exists():
            raise PromotionError("unrecorded production environment blocks install")
        return None, None, checkout_revision

    image = _proof_from_observation(initial)
    runtime = _observe_runtime(
        config=config,
        image=image,
        executables=executables,
        environment=environment,
    )
    if runtime is None:
        raise PromotionError("production deployment disappeared during observation")
    if current is None:
        if not config.adopt_existing:
            raise PromotionError("running production deployment is not recorded")
        current = transaction.adopt(
            compose_project=initial.compose_project,
            compose_service=initial.compose_service,
            source_revision=initial.revision,
            image_reference=initial.image_reference,
            image_id=initial.image_id,
            oci_revision=initial.oci_revision,
            environment_source=initial.environment_path,
        )
    if not _matches_state(runtime, current):
        raise PromotionError("recorded and running production deployments disagree")
    establishing_entry = transaction.journal()[current.journal_sequence - 1]
    if (
        establishing_entry.event == "promoted"
        and tuple(volume.recorded_identity() for volume in runtime.volumes)
        != establishing_entry.persistent_volumes
    ):
        raise PromotionError("recorded persistent volume identity changed")
    return current, runtime, checkout_revision


def _candidate_environment(
    source: Mapping[str, str],
    source_revision: str,
) -> dict[str, str]:
    return {
        "AGENT_NAME": source["AGENT_NAME"],
        "ROOT_AGENT_MODEL": source["ROOT_AGENT_MODEL"],
        "LOG_LEVEL": source["LOG_LEVEL"],
        "TELEMETRY_NAMESPACE": "candidate",
        "K_REVISION": source_revision,
        "CANDIDATE_ENV_CANARY": uuid.uuid4().hex,
    }


def _new_transaction_id(baseline_revision: str) -> str:
    """Bind the pre-cutover Git baseline into the durable transaction identity."""
    if _REVISION.fullmatch(baseline_revision) is None:
        raise PromotionError("promotion baseline revision is invalid")
    return f"{uuid.uuid4().hex[:23]}-{baseline_revision}"


def _transaction_baseline_revision(pending: PendingPromotion) -> str:
    """Recover the exact pre-cutover Git baseline from a controller-owned intent."""
    match = _CONTROLLER_TRANSACTION_ID.fullmatch(pending.transaction_id)
    if match is None:
        raise PromotionRecoveryFailedError(
            "fresh-install pending baseline revision is unavailable"
        )
    return match.group("baseline_revision")


def _compose_prefix(
    *,
    docker: Path,
    project: str,
    env_file: Path,
    compose_file: Path,
) -> tuple[str, ...]:
    return (
        str(docker),
        "compose",
        "--project-name",
        project,
        "--env-file",
        str(env_file),
        "-f",
        str(compose_file),
    )


def _run_candidate(
    *,
    transaction: DeploymentStateTransaction,
    config: PromotionConfig,
    image: ImageProof,
    transaction_id: str,
    executables: Executables,
    environment: Mapping[str, str],
    production_environment: Mapping[str, str],
) -> CandidateReceipt:
    candidate_project = f"candidate-{transaction_id[:24]}"
    candidate_compose = config.release_checkout / "compose.candidate.yaml"
    baseline_journal = transaction.journal()
    baseline_sequence = baseline_journal[-1].sequence if baseline_journal else None
    baseline_sha256 = baseline_journal[-1].sha256 if baseline_journal else None
    with tempfile.TemporaryDirectory(
        prefix=".promotion-candidate-",
        dir=config.state_dir,
    ) as temporary_name:
        temporary = Path(temporary_name)
        candidate_env = temporary / "candidate.env"
        write_compose_environment(
            candidate_env,
            _CANDIDATE_ENVIRONMENT_NAMES,
            _candidate_environment(production_environment, config.source_revision),
        )
        candidate_environment = dict(environment)
        candidate_environment.update(
            {"ENV_FILE": str(candidate_env), "IMAGE": image.image_reference}
        )
        prefix = _compose_prefix(
            docker=executables.docker,
            project=candidate_project,
            env_file=candidate_env,
            compose_file=candidate_compose,
        )
        started = False
        failure: Exception | None = None
        receipt: CandidateReceipt
        try:
            _run(
                executables.docker,
                [*prefix[1:], "config", "--quiet"],
                operation=CommandOperation.CANDIDATE_CONFIG,
                environment=candidate_environment,
            )
            started = True
            _run(
                executables.docker,
                [
                    *prefix[1:],
                    "up",
                    "--detach",
                    "--no-build",
                    "--pull",
                    "never",
                    "--wait",
                    "--wait-timeout",
                    "60",
                    config.compose_service,
                ],
                operation=CommandOperation.CANDIDATE_START,
                environment=candidate_environment,
            )
            ids = _container_ids(
                project=candidate_project,
                service=config.compose_service,
                executables=executables,
                environment=environment,
            )
            if len(ids) != 1:
                raise PromotionError("candidate deployment is ambiguous")
            container_id, _volumes = _inspect_container(
                container_id=ids[0],
                expected_checkout=config.release_checkout,
                expected_project=candidate_project,
                expected_service=config.compose_service,
                image=image,
                executables=executables,
                environment=environment,
                require_no_mounts=True,
            )
            receipt = CandidateReceipt(
                observed_at=_now(),
                compose_project=candidate_project,
                compose_service=config.compose_service,
                container_id=container_id,
                image_reference=image.image_reference,
                image_id=image.image_id,
                oci_revision=image.oci_revision,
                baseline_journal_sequence=baseline_sequence,
                baseline_journal_sha256=baseline_sha256,
            )
        except Exception as error:
            failure = error
        finally:
            if started:
                try:
                    _run(
                        executables.docker,
                        [*prefix[1:], "down", "--remove-orphans"],
                        operation=CommandOperation.CANDIDATE_CLEANUP,
                        environment=candidate_environment,
                    )
                    if _container_ids(
                        project=candidate_project,
                        service=config.compose_service,
                        executables=executables,
                        environment=environment,
                    ):
                        raise PromotionError(
                            "candidate cleanup left a service container"
                        )
                except Exception as cleanup_error:
                    failure = cleanup_error
        if failure is not None:
            raise failure
        return receipt


def _checkout_revision(
    *,
    checkout: Path,
    revision: str,
    executables: Executables,
    environment: Mapping[str, str],
) -> None:
    _run(
        executables.git,
        [
            "-C",
            str(checkout),
            "-c",
            "advice.detachedHead=false",
            "-c",
            "core.hooksPath=/dev/null",
            "-c",
            "core.fsmonitor=false",
            "checkout",
            "--detach",
            revision,
        ],
        operation=CommandOperation.GIT_CHECKOUT,
        environment=environment,
    )
    actual = _git_line(
        executables,
        checkout,
        ["rev-parse", "--verify", "HEAD"],
        environment,
        "production revision",
    )
    if actual != revision:
        raise PromotionError("production checkout revision does not match")
    for arguments in (
        ["diff", "--no-ext-diff", "--no-textconv", "--quiet", "--"],
        [
            "diff",
            "--cached",
            "--no-ext-diff",
            "--no-textconv",
            "--quiet",
            "--",
        ],
    ):
        result = _run(
            executables.git,
            ["-C", str(checkout), *arguments],
            operation=CommandOperation.GIT_DIFF,
            environment=environment,
            accepted_returncodes=frozenset({0, 1}),
        )
        if result.returncode == 1:
            raise PromotionError("production checkout has tracked changes")


def _compose_up(
    *,
    config: PromotionConfig,
    image_reference: str,
    executables: Executables,
    environment: Mapping[str, str],
    before_up: Callable[[], None] | None = None,
) -> None:
    env_file = config.checkout / ".env"
    compose_file = config.checkout / "compose.yaml"
    compose_environment = dict(environment)
    compose_environment.update({"ENV_FILE": str(env_file), "IMAGE": image_reference})
    prefix = _compose_prefix(
        docker=executables.docker,
        project=config.compose_project,
        env_file=env_file,
        compose_file=compose_file,
    )
    configured = _run(
        executables.docker,
        [*prefix[1:], "config", "--images"],
        operation=CommandOperation.PRODUCTION_CONFIG,
        environment=compose_environment,
    ).stdout.splitlines()
    if configured != [image_reference]:
        raise PromotionError("production Compose image does not match")
    if before_up is not None:
        before_up()
    _run(
        executables.docker,
        [
            *prefix[1:],
            "up",
            "--detach",
            "--no-build",
            "--force-recreate",
            "--no-deps",
            "--pull",
            "never",
            "--wait",
            "--wait-timeout",
            "180",
            config.compose_service,
        ],
        operation=CommandOperation.PRODUCTION_START,
        environment=compose_environment,
    )


def _remove_owned_service_container(
    *,
    target: DeploymentState,
    config: PromotionConfig,
    executables: Executables,
    environment: Mapping[str, str],
    target_environment_proven: bool,
    verify_target_environment: Callable[[], bool],
) -> None:
    ids = _container_ids(
        project=config.compose_project,
        service=config.compose_service,
        executables=executables,
        environment=environment,
    )
    if len(ids) > 1:
        raise PromotionRecoveryFailedError(
            "fresh-install recovery found ambiguous service containers"
        )
    if not ids:
        return
    document = _json_object(
        _run(
            executables.docker,
            ["container", "inspect", ids[0]],
            operation=CommandOperation.CONTAINER_INSPECT,
            environment=environment,
        ).stdout,
        "fresh-install container",
    )
    full_id = _string(document, "Id", "fresh-install container ID")
    config_document = _mapping(
        document,
        "Config",
        "fresh-install container configuration",
    )
    labels = _mapping(
        config_document,
        "Labels",
        "fresh-install container labels",
    )
    expected_labels = {
        "com.docker.compose.project": config.compose_project,
        "com.docker.compose.service": config.compose_service,
        "com.docker.compose.project.working_dir": str(config.checkout),
    }
    if (
        _CONTAINER_ID.fullmatch(full_id) is None
        or not full_id.startswith(ids[0])
        or any(labels.get(name) != value for name, value in expected_labels.items())
        or config_document.get("Image") != target.image_reference
        or document.get("Image") != target.image_id
    ):
        raise PromotionRecoveryFailedError(
            "fresh-install service container ownership is invalid"
        )
    if not target_environment_proven or not verify_target_environment():
        raise PromotionRecoveryFailedError(
            "fresh-install target environment is unavailable"
        )
    _run(
        executables.docker,
        ["container", "rm", "--force", full_id],
        operation=CommandOperation.CONTAINER_REMOVE,
        environment=environment,
    )


def _abort_fresh_install(
    *,
    transaction: DeploymentStateTransaction,
    pending: PendingPromotion,
    config: PromotionConfig,
    executables: Executables,
    environment: Mapping[str, str],
    baseline_revision: str,
) -> None:
    environment_path = config.checkout / ".env"
    target_environment_proven = transaction.verify_installed_environment(
        pending.intent.target,
        environment_path,
        allow_missing=True,
    )
    _remove_owned_service_container(
        target=pending.intent.target,
        config=config,
        executables=executables,
        environment=environment,
        target_environment_proven=target_environment_proven,
        verify_target_environment=lambda: transaction.verify_installed_environment(
            pending.intent.target,
            environment_path,
            allow_missing=True,
        ),
    )
    transaction.remove_environment(pending.intent.target, environment_path)
    actual_revision = _git_line(
        executables,
        config.checkout,
        ["rev-parse", "--verify", "HEAD"],
        environment,
        "fresh-install recovery revision",
    )
    if actual_revision != baseline_revision:
        _checkout_revision(
            checkout=config.checkout,
            revision=baseline_revision,
            executables=executables,
            environment=environment,
        )
    environment_still_installed = transaction.verify_installed_environment(
        pending.intent.target,
        environment_path,
        allow_missing=True,
    )
    if environment_still_installed or _container_ids(
        project=config.compose_project,
        service=config.compose_service,
        executables=executables,
        environment=environment,
    ):
        raise PromotionRecoveryFailedError(
            "fresh-install baseline could not be verified"
        )
    transaction.record_abort(
        pending.transaction_id,
        persistent_volumes=(),
    )


def _verify_target(
    *,
    transaction: DeploymentStateTransaction,
    pending: PendingPromotion,
    baseline_runtime: RuntimeProof | None,
    config: PromotionConfig,
    executables: Executables,
    environment: Mapping[str, str],
) -> RuntimeProof:
    image = ImageProof(
        image_reference=pending.intent.target.image_reference,
        image_id=pending.intent.target.image_id,
        oci_revision=pending.intent.target.oci_revision,
    )
    runtime = _observe_runtime(
        config=config,
        image=image,
        executables=executables,
        environment=environment,
    )
    if runtime is None:
        raise PromotionError("promoted production deployment is unavailable")
    target = pending.intent.target
    if (
        runtime.observation.revision != target.source_revision
        or _environment_sha256(runtime.observation.environment_path)
        != target.environment_sha256
    ):
        raise PromotionError("promoted production deployment does not match intent")
    if baseline_runtime is not None and runtime.volumes != baseline_runtime.volumes:
        raise PromotionError("persistent volume identity changed during promotion")
    if baseline_runtime is None and not runtime.volumes:
        raise PromotionError("first production install has no persistent volume")
    recorded = tuple(volume.recorded_identity() for volume in runtime.volumes)
    if baseline_runtime is not None and recorded != pending.intent.persistent_volumes:
        raise PromotionError("persistent volume intent does not match runtime")
    transaction.read_environment(target)
    return runtime


def _restore_baseline(
    *,
    transaction: DeploymentStateTransaction,
    pending: PendingPromotion,
    current: CurrentDeployment | None,
    baseline_runtime: RuntimeProof | None,
    config: PromotionConfig,
    executables: Executables,
    environment: Mapping[str, str],
) -> RuntimeProof:
    if current is None:
        raise PromotionRecoveryFailedError(
            "automatic recovery has no recorded production baseline"
        )
    state = current.state
    image = _inspect_image(
        proof_reference=state.image_reference,
        expected_revision=state.oci_revision,
        executables=executables,
        environment=environment,
    )
    if image.image_id != state.image_id:
        raise PromotionRecoveryFailedError(
            "automatic recovery baseline image identity does not match"
        )
    _checkout_revision(
        checkout=config.checkout,
        revision=state.source_revision,
        executables=executables,
        environment=environment,
    )
    transaction.install_environment(state, config.checkout / ".env")
    _compose_up(
        config=config,
        image_reference=state.image_reference,
        executables=executables,
        environment=environment,
    )
    restored = _observe_runtime(
        config=config,
        image=image,
        executables=executables,
        environment=environment,
    )
    if restored is None or not _matches_state(restored, current):
        raise PromotionRecoveryFailedError(
            "automatic recovery did not restore the recorded baseline"
        )
    recorded = tuple(volume.recorded_identity() for volume in restored.volumes)
    if recorded != pending.intent.persistent_volumes:
        raise PromotionRecoveryFailedError(
            "automatic recovery changed persistent volume identity"
        )
    if baseline_runtime is not None and restored.volumes != baseline_runtime.volumes:
        raise PromotionRecoveryFailedError(
            "automatic recovery changed persistent volume identity"
        )
    return restored


def _recover_pending(
    *,
    transaction: DeploymentStateTransaction,
    pending: PendingPromotion,
    config: PromotionConfig,
    executables: Executables,
    environment: Mapping[str, str],
) -> None:
    current = transaction.current()
    if current is None:
        _abort_fresh_install(
            transaction=transaction,
            pending=pending,
            config=config,
            executables=executables,
            environment=environment,
            baseline_revision=_transaction_baseline_revision(pending),
        )
        raise PromotionRecoveryRequiredError(
            "aborted an interrupted fresh install; rerun promotion"
        )
    current_revision = _git_line(
        executables,
        config.checkout,
        ["rev-parse", "--verify", "HEAD"],
        environment,
        "pending production revision",
    )
    if current_revision == current.state.source_revision:
        image = _inspect_image(
            proof_reference=current.state.image_reference,
            expected_revision=current.state.oci_revision,
            executables=executables,
            environment=environment,
        )
        if image.image_id != current.state.image_id:
            raise PromotionRecoveryFailedError(
                "pending baseline image identity does not match"
            )
        try:
            unchanged = _observe_runtime(
                config=config,
                image=image,
                executables=executables,
                environment=environment,
            )
        except PromotionError:
            unchanged = None
        if unchanged is not None and _matches_state(unchanged, current):
            unchanged_volumes = tuple(
                volume.recorded_identity() for volume in unchanged.volumes
            )
            if unchanged_volumes != pending.intent.persistent_volumes:
                raise PromotionRecoveryFailedError(
                    "pending baseline persistent volume identity changed"
                )
            transaction.record_abort(
                pending.transaction_id,
                persistent_volumes=unchanged_volumes,
            )
            raise PromotionRecoveryRequiredError(
                "confirmed an interrupted pre-cutover deployment; rerun promotion"
            )
    restored = _restore_baseline(
        transaction=transaction,
        pending=pending,
        current=current,
        baseline_runtime=None,
        config=config,
        executables=executables,
        environment=environment,
    )
    transaction.record_rollback(
        pending.transaction_id,
        persistent_volumes=tuple(
            volume.recorded_identity() for volume in restored.volumes
        ),
    )
    raise PromotionRecoveryRequiredError(
        "recovered an interrupted deployment; rerun promotion"
    )


def _promote_locked(
    *,
    transaction: DeploymentStateTransaction,
    config: PromotionConfig,
    executables: Executables,
    command_environment: Mapping[str, str],
    source_environment: Mapping[str, str],
) -> CurrentDeployment:
    recovered_terminal = transaction.recovered_terminal()
    if recovered_terminal is not None:
        raise PromotionRecoveryRequiredError(
            "reconciled a committed deployment outcome; rerun promotion"
        )
    _validate_production_checkout(config, executables, command_environment)
    pending_on_entry = transaction.pending()
    if pending_on_entry is not None:
        _validate_pending_ownership(
            pending=pending_on_entry,
            current=transaction.current(),
            config=config,
            executables=executables,
            environment=command_environment,
        )
        _recover_pending(
            transaction=transaction,
            pending=pending_on_entry,
            config=config,
            executables=executables,
            environment=command_environment,
        )

    _validate_release_checkout(
        config,
        executables,
        command_environment,
    )
    try:
        production_environment = {
            name: source_environment[name] for name in PRODUCTION_ENVIRONMENT_NAMES
        }
    except KeyError:
        raise PromotionError("required production environment is incomplete") from None
    for value in production_environment.values():
        if any(character in value for character in "\0\r\n"):
            raise PromotionError("production environment contains an unsafe value")

    current, baseline_runtime, original_revision = _baseline(
        transaction=transaction,
        config=config,
        executables=executables,
        environment=command_environment,
    )
    image = _pull_and_prove_image(
        config,
        executables,
        command_environment,
    )
    transaction_id = _new_transaction_id(original_revision)
    receipt = _run_candidate(
        transaction=transaction,
        config=config,
        image=image,
        transaction_id=transaction_id,
        executables=executables,
        environment=command_environment,
        production_environment=production_environment,
    )
    with tempfile.TemporaryDirectory(
        prefix=".promotion-target-",
        dir=config.state_dir,
    ) as temporary_name:
        environment_source = Path(temporary_name) / "target.env"
        write_compose_environment(
            environment_source,
            PRODUCTION_ENVIRONMENT_NAMES,
            production_environment,
        )
        volumes = (
            ()
            if baseline_runtime is None
            else tuple(
                volume.recorded_identity() for volume in baseline_runtime.volumes
            )
        )
        pending = transaction.begin_promotion(
            compose_project=config.compose_project,
            compose_service=config.compose_service,
            source_revision=config.source_revision,
            image_reference=image.image_reference,
            image_id=image.image_id,
            oci_revision=image.oci_revision,
            environment_source=environment_source,
            candidate=receipt,
            persistent_volumes=volumes,
            transaction_id=transaction_id,
        )

    production_start_attempted = False

    def mark_production_start() -> None:
        nonlocal production_start_attempted
        production_start_attempted = True

    try:
        _checkout_revision(
            checkout=config.checkout,
            revision=config.source_revision,
            executables=executables,
            environment=command_environment,
        )
        transaction.install_environment(
            pending.intent.target,
            config.checkout / ".env",
        )
        _compose_up(
            config=config,
            image_reference=image.image_reference,
            executables=executables,
            environment=command_environment,
            before_up=mark_production_start,
        )
        verified_target = _verify_target(
            transaction=transaction,
            pending=pending,
            baseline_runtime=baseline_runtime,
            config=config,
            executables=executables,
            environment=command_environment,
        )
        return transaction.commit_promotion(
            transaction_id,
            persistent_volumes=tuple(
                volume.recorded_identity() for volume in verified_target.volumes
            ),
        )
    except (
        DeploymentTerminalCommittedError,
        DeploymentTerminalIndeterminateError,
    ):
        raise
    except Exception as primary:
        try:
            if current is None:
                _abort_fresh_install(
                    transaction=transaction,
                    pending=pending,
                    config=config,
                    executables=executables,
                    environment=command_environment,
                    baseline_revision=original_revision,
                )
            else:
                restored_runtime = _restore_baseline(
                    transaction=transaction,
                    pending=pending,
                    current=current,
                    baseline_runtime=baseline_runtime,
                    config=config,
                    executables=executables,
                    environment=command_environment,
                )
                transaction.record_rollback(
                    transaction_id,
                    persistent_volumes=tuple(
                        volume.recorded_identity()
                        for volume in restored_runtime.volumes
                    ),
                )
        except (
            DeploymentTerminalCommittedError,
            DeploymentTerminalIndeterminateError,
        ):
            raise
        except Exception:
            raise PromotionRecoveryFailedError(
                "promotion failed and automatic recovery could not be verified"
            ) from None
        outcome = (
            "fresh install was safely aborted"
            if current is None
            else "the recorded baseline was restored"
        )
        if current is None and production_start_attempted:
            outcome += " and its exact service container was removed"
        raise PromotionRolledBackError(f"promotion failed; {outcome}") from primary


def promote(
    config: PromotionConfig,
    *,
    environment: Mapping[str, str] | None = None,
) -> CurrentDeployment:
    """Execute one complete promotion while holding one state transaction."""
    selected_config = _validated_config(config)
    source = os.environ if environment is None else environment
    command_environment = _command_environment(source)
    executables = Executables(
        git=_resolve_executable("git", command_environment),
        docker=_resolve_executable("docker", command_environment),
    )
    store = DeploymentStateStore(selected_config.state_dir)
    with store.transaction() as transaction:
        return _promote_locked(
            transaction=transaction,
            config=selected_config,
            executables=executables,
            command_environment=command_environment,
            source_environment=source,
        )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m agent.deployment_promotion",
        description="Promote and verify one exact VM deployment.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    promote_parser = subparsers.add_parser("promote")
    promote_parser.add_argument("--state-dir", required=True, type=Path)
    promote_parser.add_argument("--checkout", required=True, type=Path)
    promote_parser.add_argument("--release-checkout", required=True, type=Path)
    promote_parser.add_argument("--expected-origin", required=True)
    promote_parser.add_argument("--compose-project", required=True)
    promote_parser.add_argument("--compose-service", required=True)
    promote_parser.add_argument("--source-revision", required=True)
    promote_parser.add_argument("--image-reference", required=True)
    promote_parser.add_argument("--adopt-existing", action="store_true")
    promote_parser.add_argument("--environment-stdin", action="store_true")
    promote_parser.add_argument("--release-lease", type=Path)
    return parser


def _stdin_environment(stream: BinaryIO) -> dict[str, str]:
    """Read one bounded UTF-8 canonical production environment payload."""
    try:
        payload = stream.read(_MAX_SERIALIZED_ENVIRONMENT_BYTES + 1)
    except OSError:
        raise ComposeEnvironmentError(
            "serialized environment could not be read"
        ) from None
    if len(payload) > _MAX_SERIALIZED_ENVIRONMENT_BYTES:
        raise ComposeEnvironmentError("serialized environment is too large")
    try:
        decoded = payload.decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        raise ComposeEnvironmentError("serialized environment is not UTF-8") from None
    return parse_compose_environment(decoded, PRODUCTION_ENVIRONMENT_NAMES)


@contextmanager
def _release_lease(path: Path) -> Iterator[None]:
    """Hold one pre-created private release lease across the whole controller."""
    if not path.is_absolute() or path != path.resolve(strict=False):
        raise PromotionError("release lease must be an absolute normalized path")
    parent = path.parent
    try:
        parent_metadata = parent.stat()
    except OSError:
        raise PromotionError("release lease directory is unavailable") from None
    if (
        not stat.S_ISDIR(parent_metadata.st_mode)
        or parent_metadata.st_uid != os.geteuid()
        or stat.S_IMODE(parent_metadata.st_mode) != 0o700
    ):
        raise PromotionError("release lease directory is unsafe")
    flags = os.O_RDWR | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError:
        raise PromotionError("release lease is unavailable") from None
    try:
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
            or metadata.st_nlink != 1
            or stat.S_IMODE(metadata.st_mode) != 0o600
        ):
            raise PromotionError("release lease is unsafe")
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            if error.errno in {errno.EACCES, errno.EAGAIN}:
                raise DeploymentLockBusyError(
                    "another release controller holds the lease"
                ) from None
            raise PromotionError("release lease could not be acquired") from None
        try:
            yield
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
    finally:
        os.close(descriptor)


def main(
    argv: Sequence[str] | None = None,
    environment: Mapping[str, str] | None = None,
    input_stream: BinaryIO | None = None,
) -> int:
    """Run the promotion CLI and return a stable process status."""
    arguments = _parser().parse_args(sys.argv[1:] if argv is None else argv)
    config = PromotionConfig(
        state_dir=arguments.state_dir,
        checkout=arguments.checkout,
        release_checkout=arguments.release_checkout,
        expected_origin=arguments.expected_origin,
        compose_project=arguments.compose_project,
        compose_service=arguments.compose_service,
        source_revision=arguments.source_revision,
        image_reference=arguments.image_reference,
        adopt_existing=arguments.adopt_existing,
    )
    try:
        if (arguments.release_lease is None) != (not arguments.environment_stdin):
            raise PromotionError(
                "release lease and serialized environment must be used together"
            )
        if arguments.release_lease is None:
            selected_environment = os.environ if environment is None else environment
            current = promote(config, environment=selected_environment)
        else:
            with _release_lease(arguments.release_lease):
                source = os.environ if environment is None else environment
                stream = sys.stdin.buffer if input_stream is None else input_stream
                production_environment = _stdin_environment(stream)
                selected_environment = {
                    name: source[name]
                    for name in _HOST_ENVIRONMENT_NAMES
                    if name in source
                }
                selected_environment.update(production_environment)
                current = promote(config, environment=selected_environment)
    except PromotionRecoveryRequiredError as error:
        print(f"RECOVERED: {error}", file=sys.stderr)
        return 3
    except (
        DeploymentTerminalCommittedError,
        DeploymentTerminalIndeterminateError,
    ) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 4
    except (
        ComposeEnvironmentError,
        DeploymentLockBusyError,
        DeploymentStateError,
        PromotionError,
        OSError,
    ) as error:
        if isinstance(error, OSError):
            message = "deployment host operation failed"
        else:
            message = str(error)
        print(f"ERROR: {message}", file=sys.stderr)
        return 1
    print(
        f"PROMOTED: revision={current.state.source_revision} "
        f"transaction={current.state.deployment_id}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
