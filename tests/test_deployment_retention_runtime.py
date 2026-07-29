"""Opt-in real-Docker proof for bounded VM image retention."""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import re
import shutil
import stat
import subprocess
import time
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from unittest.mock import create_autospec, patch
from urllib.error import URLError
from urllib.request import ProxyHandler, build_opener

import pytest
import yaml  # type: ignore[import-untyped]

from agent.deployment_retention import (
    RetentionPlan,
    apply_image_retention,
    enforce_pull_admission,
    plan_image_retention,
)
from agent.deployment_state import (
    CandidateReceipt,
    DeploymentStateStore,
    DeploymentStateTransaction,
    PersistentVolumeIdentity,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_PATH = REPOSITORY_ROOT / ".github" / "workflows" / "test-docker-compose.yml"
RUN_ENVIRONMENT_NAME = "RUN_DEPLOYMENT_RETENTION_INTEGRATION"
PREFIX_ENVIRONMENT_NAME = "DEPLOYMENT_RETENTION_TEST_PREFIX"
PREFIX_PATTERN = re.compile(r"adk-retention-[a-z0-9][a-z0-9-]{0,30}\Z")
RESOURCE_NAME_PATTERN = re.compile(r"[a-z0-9][a-z0-9_.-]{0,62}\Z")
IMAGE_ID_PATTERN = re.compile(r"sha256:[0-9a-f]{64}\Z")
IMAGE_REFERENCE_PATTERN = re.compile(
    r"127\.0\.0\.1:[0-9]+/[a-z0-9-]+/agent@sha256:[0-9a-f]{64}\Z"
)
CONTAINER_ID_PATTERN = re.compile(r"[0-9a-f]{64}\Z")
MANAGED_REPOSITORY = "QueryPlanner/google-adk-on-bare-metal"
MANAGED_SOURCE = f"https://github.com/{MANAGED_REPOSITORY}"
MANAGED_LABEL = "io.queryplanner.adk.repository"
OWNER_LABEL = "io.queryplanner.adk.retention-test.owner"
PRIVATE_CANARY = 'retention-private-$-हॅलो-"-\\-canary'
VOLUME_SENTINEL = "retention-volume-sentinel"
REGISTRY_IMAGE_REFERENCE = (
    "registry@sha256:1be55279f18a2fe1a74edf2664cac61c1bea305b7b4642dab412e7affdcb3e33"
)
DERIVATIVE_DOCKERFILE = """\
ARG BASE_IMAGE
FROM ${BASE_IMAGE}
ARG MANAGED_REPOSITORY
ARG MANAGED_SOURCE
ARG OWNER
ARG REVISION
LABEL io.queryplanner.adk.repository="${MANAGED_REPOSITORY}"
LABEL org.opencontainers.image.source="${MANAGED_SOURCE}"
LABEL org.opencontainers.image.revision="${REVISION}"
LABEL io.queryplanner.adk.retention-test.owner="${OWNER}"
"""

type ResourceKind = Literal["container", "image", "network", "volume"]
type BuilderCacheIdentity = tuple[
    str,
    tuple[str, ...],
    str,
    bool,
    bool,
    str,
    str,
    str,
    int,
]


@dataclass(frozen=True, slots=True)
class ManagedImage:
    """One exact managed image identity published to the local registry."""

    reference: str
    image_id: str
    revision: str


@dataclass(frozen=True, slots=True)
class CleanupTarget:
    """One exact Docker resource with an independent ownership proof."""

    kind: ResourceKind
    reference: str
    expected_owner: str | None = None
    expected_id: str | None = None

    def __post_init__(self) -> None:
        if (self.expected_owner is None) == (self.expected_id is None):
            raise AssertionError("cleanup target needs exactly one ownership proof")


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
    for representation in _secret_representations(PRIVATE_CANARY):
        redacted = redacted.replace(representation, "[REDACTED]")
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
        result = subprocess.run(  # noqa: S603 - fixed/resolved test executables
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


def _base_environment() -> dict[str, str]:
    inherited = (
        "DOCKER_CONFIG",
        "DOCKER_CONTEXT",
        "DOCKER_HOST",
        "HOME",
        "PATH",
        "SSL_CERT_DIR",
        "SSL_CERT_FILE",
        "XDG_RUNTIME_DIR",
    )
    environment = {name: os.environ[name] for name in inherited if name in os.environ}
    environment.update({"LANG": "C.UTF-8", "LC_ALL": "C.UTF-8"})
    return environment


def _require_docker(environment: Mapping[str, str]) -> str:
    docker = shutil.which("docker", path=environment.get("PATH"))
    if docker is None:
        raise AssertionError("Docker CLI is required for the opted-in proof")
    selected = Path(docker).resolve(strict=True)
    if not stat.S_ISREG(selected.stat().st_mode):
        raise AssertionError("Docker CLI is not a regular file")
    _run([str(selected), "version"], environment=environment, timeout=30)
    return str(selected)


def _resource_prefix() -> str:
    configured = os.environ.get(
        PREFIX_ENVIRONMENT_NAME,
        f"adk-retention-{os.getpid()}",
    ).lower()
    if PREFIX_PATTERN.fullmatch(configured) is None:
        raise AssertionError("deployment-retention Docker prefix is invalid")
    selected = f"{configured}-{uuid.uuid4().hex[:12]}"
    if RESOURCE_NAME_PATTERN.fullmatch(selected) is None:
        raise AssertionError("deployment-retention resource prefix is invalid")
    return selected


def _inspect_optional(
    docker: str,
    environment: Mapping[str, str],
    kind: ResourceKind,
    reference: str,
) -> dict[str, object] | None:
    result = _run(
        [docker, kind, "inspect", reference],
        environment=environment,
        check=False,
        timeout=30,
    )
    if result.returncode != 0:
        exact_absence_errors = {
            "container": (
                f"Error response from daemon: No such container: {reference}\n"
            ),
            "image": f"Error response from daemon: No such image: {reference}\n",
            "network": f"Error response from daemon: network {reference} not found\n",
            "volume": (
                f"Error response from daemon: get {reference}: no such volume\n"
            ),
        }
        if (
            result.returncode == 1
            and result.stdout in {"[]", "[]\n"}
            and result.stderr == exact_absence_errors[kind]
        ):
            return None
        raise AssertionError(f"Docker {kind} inspection failed")
    documents = json.loads(result.stdout)
    if not isinstance(documents, list) or len(documents) != 1:
        raise AssertionError(f"Docker {kind} inspection was ambiguous")
    document = documents[0]
    if not isinstance(document, dict):
        raise AssertionError(f"Docker {kind} inspection was invalid")
    return document


def _resource_identity(kind: ResourceKind, document: Mapping[str, object]) -> str:
    identity = document.get("Name") if kind == "volume" else document.get("Id")
    if not isinstance(identity, str):
        raise AssertionError(f"Docker {kind} identity was invalid")
    return identity


def _resource_labels(
    kind: ResourceKind,
    document: Mapping[str, object],
) -> Mapping[str, object]:
    if kind in {"container", "image"}:
        config = document.get("Config")
        if not isinstance(config, dict):
            raise AssertionError(f"Docker {kind} configuration was invalid")
        labels = config.get("Labels")
    else:
        labels = document.get("Labels")
    if not isinstance(labels, dict):
        raise AssertionError(f"Docker {kind} labels were invalid")
    return labels


def _owned_document(
    docker: str,
    environment: Mapping[str, str],
    target: CleanupTarget,
) -> dict[str, object] | None:
    document = _inspect_optional(
        docker,
        environment,
        target.kind,
        target.reference,
    )
    if document is None:
        return None
    identity = _resource_identity(target.kind, document)
    if target.expected_id is not None and identity != target.expected_id:
        raise AssertionError("cleanup target identity changed")
    if target.expected_owner is not None:
        labels = _resource_labels(target.kind, document)
        if labels.get(OWNER_LABEL) != target.expected_owner:
            raise AssertionError("cleanup target owner changed")
    return document


def _cleanup_target(
    docker: str,
    environment: Mapping[str, str],
    target: CleanupTarget,
) -> None:
    document = _owned_document(docker, environment, target)
    if document is None:
        return
    identity = _resource_identity(target.kind, document)
    if target.kind == "container":
        state = document.get("State")
        if not isinstance(state, dict):
            raise AssertionError("cleanup container state was invalid")
        if state.get("Running") is True:
            _run(
                [docker, "container", "stop", "--time", "10", identity],
                environment=environment,
                timeout=30,
            )
        if _owned_document(docker, environment, target) is None:
            return
        command = [docker, "container", "rm", "--volumes", identity]
    elif target.kind == "image":
        command = [
            docker,
            "image",
            "rm",
            "--no-prune",
            target.reference,
        ]
    elif target.kind == "network":
        command = [docker, "network", "rm", identity]
    else:
        command = [docker, "volume", "rm", identity]
    _run(command, environment=environment, timeout=90)
    if _owned_document(docker, environment, target) is not None:
        raise AssertionError("exact cleanup left its owned target")


def _execute_cleanup(
    docker: str,
    environment: Mapping[str, str],
    targets: Sequence[CleanupTarget],
) -> list[str]:
    priorities = {"container": 0, "network": 1, "volume": 2, "image": 3}
    failures: list[str] = []
    ordered = sorted(
        enumerate(targets),
        key=lambda item: (priorities[item[1].kind], -item[0]),
    )
    for _index, target in ordered:
        try:
            _cleanup_target(docker, environment, target)
        except (AssertionError, OSError, TypeError, ValueError) as error:
            failures.append(_redact(str(error)[-1_000:]))
    return failures


def _report_cleanup_failures(
    failures: Sequence[str],
    primary_error: BaseException | None,
) -> None:
    if not failures:
        return
    message = "deployment-retention cleanup failed: " + " | ".join(failures)
    if primary_error is None:
        raise AssertionError(message)
    primary_error.add_note(message)


def _assert_absent(
    docker: str,
    environment: Mapping[str, str],
    kind: ResourceKind,
    reference: str,
) -> None:
    if _inspect_optional(docker, environment, kind, reference) is not None:
        raise AssertionError(f"generated Docker {kind} already exists")


def _ensure_registry_image(
    docker: str,
    environment: Mapping[str, str],
    cleanup_targets: list[CleanupTarget],
) -> str:
    document = _inspect_optional(
        docker,
        environment,
        "image",
        REGISTRY_IMAGE_REFERENCE,
    )
    introduced = document is None
    if introduced:
        _run(
            [docker, "image", "pull", REGISTRY_IMAGE_REFERENCE],
            environment=environment,
            timeout=180,
        )
        document = _inspect_optional(
            docker,
            environment,
            "image",
            REGISTRY_IMAGE_REFERENCE,
        )
    if document is None:
        raise AssertionError("pinned registry image is unavailable")
    image_id = _resource_identity("image", document)
    if IMAGE_ID_PATTERN.fullmatch(image_id) is None:
        raise AssertionError("pinned registry image ID was invalid")
    if introduced:
        cleanup_targets.append(
            CleanupTarget(
                kind="image",
                reference=REGISTRY_IMAGE_REFERENCE,
                expected_id=image_id,
            )
        )
    return image_id


def _start_registry(
    docker: str,
    environment: Mapping[str, str],
    cleanup_targets: list[CleanupTarget],
    *,
    name: str,
    image_id: str,
    owner: str,
) -> str:
    _assert_absent(docker, environment, "container", name)
    target = CleanupTarget(
        kind="container",
        reference=name,
        expected_owner=owner,
    )
    cleanup_targets.append(target)
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
    port = _run(
        [
            docker,
            "container",
            "inspect",
            "--format",
            '{{(index (index .NetworkSettings.Ports "5000/tcp") 0).HostPort}}',
            name,
        ],
        environment=environment,
        timeout=30,
    ).stdout.strip()
    if re.fullmatch(r"[0-9]{1,5}", port) is None:
        raise AssertionError("private registry port was invalid")
    endpoint = f"127.0.0.1:{port}"
    opener = build_opener(ProxyHandler({}))
    deadline = time.monotonic() + 30
    while True:
        try:
            with opener.open(f"http://{endpoint}/v2/", timeout=2) as response:
                if response.status != 200:
                    raise AssertionError("private registry response was invalid")
            return endpoint
        except URLError:
            if time.monotonic() >= deadline:
                raise AssertionError("private registry did not become ready") from None
            time.sleep(0.25)


def _write_derivative_context(context: Path) -> None:
    context.mkdir(mode=0o700)
    (context / "Dockerfile").write_text(
        DERIVATIVE_DOCKERFILE,
        encoding="utf-8",
    )


def _managed_image_document(
    docker: str,
    environment: Mapping[str, str],
    reference: str,
    *,
    owner: str,
    revision: str,
) -> dict[str, object]:
    document = _inspect_optional(docker, environment, "image", reference)
    if document is None:
        raise AssertionError("managed image was unavailable")
    image_id = _resource_identity("image", document)
    if IMAGE_ID_PATTERN.fullmatch(image_id) is None:
        raise AssertionError("managed image ID was invalid")
    config = document.get("Config")
    if not isinstance(config, dict):
        raise AssertionError("managed image configuration was invalid")
    labels = config.get("Labels")
    expected_labels = {
        MANAGED_LABEL: MANAGED_REPOSITORY,
        "org.opencontainers.image.source": MANAGED_SOURCE,
        "org.opencontainers.image.revision": revision,
        OWNER_LABEL: owner,
    }
    if not isinstance(labels, dict) or any(
        labels.get(name) != value for name, value in expected_labels.items()
    ):
        raise AssertionError("managed image labels did not match")
    return document


def _publish_managed_image(
    docker: str,
    environment: Mapping[str, str],
    cleanup_targets: list[CleanupTarget],
    *,
    context: Path,
    repository: str,
    phase: int,
    owner: str,
    keep_local: bool,
) -> ManagedImage:
    revision = hashlib.sha1(  # noqa: S324 - Git-shaped fixture identity
        f"{owner}-revision-{phase}".encode()
    ).hexdigest()
    tag = f"{repository}:fixture-{phase}"
    _assert_absent(docker, environment, "image", tag)
    cleanup_targets.append(
        CleanupTarget(
            kind="image",
            reference=tag,
            expected_owner=owner,
        )
    )
    iid_file = context / f"phase-{phase}.iid"
    _run(
        [
            docker,
            "build",
            "--iidfile",
            str(iid_file),
            "--tag",
            tag,
            "--build-arg",
            f"BASE_IMAGE={REGISTRY_IMAGE_REFERENCE}",
            "--build-arg",
            f"MANAGED_REPOSITORY={MANAGED_REPOSITORY}",
            "--build-arg",
            f"MANAGED_SOURCE={MANAGED_SOURCE}",
            "--build-arg",
            f"OWNER={owner}",
            "--build-arg",
            f"REVISION={revision}",
            str(context),
        ],
        environment=environment,
        timeout=300,
    )
    image_id = iid_file.read_text(encoding="ascii").strip()
    if IMAGE_ID_PATTERN.fullmatch(image_id) is None:
        raise AssertionError("Docker build returned an invalid image ID")
    built = _managed_image_document(
        docker,
        environment,
        tag,
        owner=owner,
        revision=revision,
    )
    if _resource_identity("image", built) != image_id:
        raise AssertionError("built image ID changed")
    _run(
        [docker, "image", "push", tag],
        environment=environment,
        timeout=180,
    )
    pushed = _managed_image_document(
        docker,
        environment,
        tag,
        owner=owner,
        revision=revision,
    )
    digests = pushed.get("RepoDigests")
    if not isinstance(digests, list):
        raise AssertionError("pushed image digests were invalid")
    exact = [
        value
        for value in digests
        if isinstance(value, str) and value.startswith(f"{repository}@sha256:")
    ]
    if len(exact) != 1 or IMAGE_REFERENCE_PATTERN.fullmatch(exact[0]) is None:
        raise AssertionError("pushed image digest was ambiguous")
    reference = exact[0]
    cleanup_targets.append(
        CleanupTarget(
            kind="image",
            reference=reference,
            expected_owner=owner,
        )
    )
    _cleanup_target(
        docker,
        environment,
        CleanupTarget(
            kind="image",
            reference=tag,
            expected_owner=owner,
        ),
    )
    if keep_local:
        _run(
            [docker, "image", "pull", reference],
            environment=environment,
            timeout=180,
        )
        local = _managed_image_document(
            docker,
            environment,
            reference,
            owner=owner,
            revision=revision,
        )
        if _resource_identity("image", local) != image_id:
            raise AssertionError("pulled image ID changed")
        if local.get("RepoTags") not in (None, []):
            raise AssertionError("managed image retained a mutable tag")
        if local.get("RepoDigests") != [reference]:
            raise AssertionError("managed image digest set was ambiguous")
    elif _inspect_optional(docker, environment, "image", reference) is not None:
        _cleanup_target(
            docker,
            environment,
            CleanupTarget(
                kind="image",
                reference=reference,
                expected_owner=owner,
            ),
        )
    return ManagedImage(reference=reference, image_id=image_id, revision=revision)


def _volume_document(
    docker: str,
    environment: Mapping[str, str],
    name: str,
) -> dict[str, object]:
    document = _inspect_optional(docker, environment, "volume", name)
    if document is None:
        raise AssertionError("sentinel volume was unavailable")
    return document


def _volume_identity(document: Mapping[str, object]) -> dict[str, object]:
    return {
        name: document.get(name)
        for name in (
            "Name",
            "Driver",
            "Mountpoint",
            "CreatedAt",
            "Labels",
            "Options",
            "Scope",
        )
    }


def _recorded_volume(
    docker: str,
    environment: Mapping[str, str],
    name: str,
) -> PersistentVolumeIdentity:
    document = _volume_document(docker, environment, name)
    driver = document.get("Driver")
    mountpoint = document.get("Mountpoint")
    created_at = document.get("CreatedAt")
    if not all(isinstance(value, str) for value in (driver, mountpoint, created_at)):
        raise AssertionError("sentinel volume identity was invalid")
    return PersistentVolumeIdentity(
        name=name,
        driver=str(driver),
        mountpoint=str(mountpoint),
        destination="/sentinel",
        created_at=str(created_at),
    )


def _container_document(
    docker: str,
    environment: Mapping[str, str],
    name: str,
) -> dict[str, object]:
    document = _inspect_optional(docker, environment, "container", name)
    if document is None:
        raise AssertionError("sentinel container was unavailable")
    return document


def _container_identity(document: Mapping[str, object]) -> dict[str, object]:
    state = document.get("State")
    config = document.get("Config")
    network = document.get("NetworkSettings")
    mounts = document.get("Mounts")
    if not (
        isinstance(state, dict)
        and isinstance(config, dict)
        and isinstance(network, dict)
        and isinstance(mounts, list)
    ):
        raise AssertionError("sentinel container identity was invalid")
    return {
        "Id": document.get("Id"),
        "Name": document.get("Name"),
        "Image": document.get("Image"),
        "Created": document.get("Created"),
        "RestartCount": document.get("RestartCount"),
        "Config": config,
        "State": {
            name: state.get(name)
            for name in (
                "Status",
                "Running",
                "Pid",
                "StartedAt",
                "FinishedAt",
            )
        },
        "Mounts": mounts,
        "Networks": network.get("Networks"),
    }


def _network_identity(document: Mapping[str, object]) -> dict[str, object]:
    return {
        name: document.get(name)
        for name in (
            "Name",
            "Id",
            "Created",
            "Scope",
            "Driver",
            "EnableIPv6",
            "IPAM",
            "Internal",
            "Attachable",
            "Ingress",
            "ConfigOnly",
            "Containers",
            "Options",
            "Labels",
        )
    }


def _image_identity(document: Mapping[str, object]) -> dict[str, object]:
    config = document.get("Config")
    if not isinstance(config, dict):
        raise AssertionError("sentinel image configuration was invalid")
    return {
        "Id": document.get("Id"),
        "RepoTags": document.get("RepoTags"),
        "RepoDigests": document.get("RepoDigests"),
        "Created": document.get("Created"),
        "Config": config,
    }


def _write_private(path: Path, payload: bytes) -> None:
    path.write_bytes(payload)
    path.chmod(0o600)


def _candidate(
    transaction: DeploymentStateTransaction,
    image: ManagedImage,
    *,
    phase: int,
) -> CandidateReceipt:
    journal = transaction.journal()
    tail = journal[-1] if journal else None
    return CandidateReceipt(
        observed_at=f"2026-07-29T00:00:0{phase}.000000Z",
        compose_project=f"candidate-retention-{phase}",
        compose_service="agent",
        container_id=f"{phase + 1:x}" * 64,
        image_reference=image.reference,
        image_id=image.image_id,
        oci_revision=image.revision,
        baseline_journal_sequence=None if tail is None else tail.sequence,
        baseline_journal_sha256=None if tail is None else tail.sha256,
    )


def _record_three_generations(
    state_dir: Path,
    environment_dir: Path,
    images: Sequence[ManagedImage],
    volume: PersistentVolumeIdentity,
    *,
    compose_project: str,
) -> DeploymentStateStore:
    store = DeploymentStateStore(state_dir)
    adopted_environment = environment_dir / "adopted.env"
    _write_private(
        adopted_environment,
        f'PRIVATE="{PRIVATE_CANARY}-adopted"\n'.encode(),
    )
    with store.transaction() as transaction:
        transaction.adopt(
            compose_project=compose_project,
            compose_service="agent",
            source_revision=images[0].revision,
            image_reference=images[0].reference,
            image_id=images[0].image_id,
            oci_revision=images[0].revision,
            environment_source=adopted_environment,
            deployment_id="retention-adopted",
            recorded_at="2026-07-29T00:00:00.000000Z",
        )
    for phase, image in enumerate(images[1:3], start=1):
        environment_path = environment_dir / f"promoted-{phase}.env"
        _write_private(
            environment_path,
            f'PRIVATE="{PRIVATE_CANARY}-promoted-{phase}"\n'.encode(),
        )
        with store.transaction() as transaction:
            pending = transaction.begin_promotion(
                compose_project=compose_project,
                compose_service="agent",
                source_revision=image.revision,
                image_reference=image.reference,
                image_id=image.image_id,
                oci_revision=image.revision,
                environment_source=environment_path,
                candidate=_candidate(transaction, image, phase=phase),
                persistent_volumes=(volume,),
                transaction_id=f"retention-promoted-{phase}",
                recorded_at=f"2026-07-29T00:00:0{phase}.000000Z",
            )
            transaction.commit_promotion(
                pending.transaction_id,
                persistent_volumes=(volume,),
            )
    return store


def _state_tree(path: Path) -> dict[str, tuple[object, ...]]:
    result: dict[str, tuple[object, ...]] = {}
    for selected in sorted((path, *path.rglob("*"))):
        relative = "." if selected == path else selected.relative_to(path).as_posix()
        metadata = selected.lstat()
        mode = stat.S_IMODE(metadata.st_mode)
        if stat.S_ISDIR(metadata.st_mode):
            result[relative] = (
                "directory",
                mode,
                metadata.st_ino,
                metadata.st_mtime_ns,
            )
        elif stat.S_ISREG(metadata.st_mode):
            payload = selected.read_bytes()
            result[relative] = (
                "file",
                mode,
                metadata.st_ino,
                metadata.st_mtime_ns,
                len(payload),
                hashlib.sha256(payload).hexdigest(),
            )
        else:
            raise AssertionError("deployment state contains a special path")
    return result


def _present_references(
    docker: str,
    environment: Mapping[str, str],
    images: Sequence[ManagedImage],
) -> frozenset[str]:
    return frozenset(
        image.reference
        for image in images
        if _inspect_optional(
            docker,
            environment,
            "image",
            image.reference,
        )
        is not None
    )


def _builder_cache_identity(
    docker: str,
    environment: Mapping[str, str],
) -> tuple[BuilderCacheIdentity, ...]:
    result = _run(
        [docker, "buildx", "du", "--format=json"],
        environment=environment,
        timeout=60,
    )
    lines = tuple(line for line in result.stdout.splitlines() if line)
    if not lines:
        raise AssertionError("BuildKit cache sentinel was unavailable")
    expected_fields = {
        "CreatedAt",
        "Description",
        "ID",
        "LastUsedAt",
        "Mutable",
        "Parents",
        "Reclaimable",
        "Shared",
        "Size",
        "Type",
        "UsageCount",
    }
    identities: list[BuilderCacheIdentity] = []
    for line in lines:
        try:
            document = json.loads(line)
        except json.JSONDecodeError:
            raise AssertionError("BuildKit cache identity was invalid") from None
        if not isinstance(document, dict) or set(document) != expected_fields:
            raise AssertionError("BuildKit cache identity was invalid")
        identifier = document["ID"]
        parents = document["Parents"]
        parent_identities = () if parents is None else parents
        created_at = document["CreatedAt"]
        mutable = document["Mutable"]
        reclaimable = document["Reclaimable"]
        size = document["Size"]
        record_type = document["Type"]
        description = document["Description"]
        usage_count = document["UsageCount"]
        last_used_at = document["LastUsedAt"]
        if (
            not isinstance(identifier, str)
            or not identifier
            or not isinstance(parent_identities, (list, tuple))
            or not all(
                isinstance(parent, str) and parent for parent in parent_identities
            )
            or not isinstance(created_at, str)
            or not created_at
            or not isinstance(mutable, bool)
            or not isinstance(reclaimable, bool)
            or not isinstance(document["Shared"], bool)
            or not isinstance(size, str)
            or not isinstance(record_type, str)
            or not record_type
            or not isinstance(description, str)
            or type(usage_count) is not int
            or not (
                last_used_at is None
                or isinstance(last_used_at, str)
                and bool(last_used_at)
            )
        ):
            raise AssertionError("BuildKit cache identity was invalid")
        identities.append(
            (
                identifier,
                tuple(parent_identities),
                created_at,
                mutable,
                reclaimable,
                size,
                record_type,
                description,
                usage_count,
            )
        )
    if len({identity[0] for identity in identities}) != len(identities):
        raise AssertionError("BuildKit cache identity was ambiguous")
    return tuple(sorted(identities))


def _apply_retention(
    store: DeploymentStateStore,
    docker: str,
    environment: Mapping[str, str],
    *,
    image_repository: str,
    target_reference: str,
) -> tuple[RetentionPlan, RetentionPlan]:
    with store.transaction() as transaction:
        planned = plan_image_retention(
            transaction,
            docker=Path(docker),
            environment=environment,
            repository=MANAGED_REPOSITORY,
            target_reference=target_reference,
            image_repository=image_repository,
        )
        applied, admitted, _removed_decisions = apply_image_retention(
            transaction,
            docker=Path(docker),
            environment=environment,
            repository=MANAGED_REPOSITORY,
            target_reference=target_reference,
            image_repository=image_repository,
        )
    if not admitted:
        raise AssertionError("retention did not reserve the target pull slot")
    return planned, applied


def _workflow_job() -> dict[str, object]:
    document = yaml.safe_load(WORKFLOW_PATH.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise AssertionError("Docker workflow was invalid")
    jobs = document.get("jobs")
    if not isinstance(jobs, dict):
        raise AssertionError("Docker workflow jobs were invalid")
    job = jobs.get("deployment-retention")
    if not isinstance(job, dict):
        raise AssertionError("deployment-retention job was unavailable")
    return job


@pytest.mark.parametrize(
    ("kind", "stderr"),
    [
        (
            "container",
            "Error response from daemon: No such container: fixture\n",
        ),
        ("image", "Error response from daemon: No such image: fixture\n"),
        ("network", "Error response from daemon: network fixture not found\n"),
        ("volume", "Error response from daemon: get fixture: no such volume\n"),
    ],
)
def test_optional_inspection_accepts_exact_docker_absence(
    kind: ResourceKind,
    stderr: str,
) -> None:
    """Match the reference-bound absence output emitted by Docker inspect."""
    result = subprocess.CompletedProcess(
        ["docker", kind, "inspect", "fixture"],
        1,
        stdout="[]\n",
        stderr=stderr,
    )
    runner = create_autospec(_run, spec_set=True, return_value=result)

    with patch(f"{__name__}._run", runner):
        assert _inspect_optional("docker", {}, kind, "fixture") is None

    runner.assert_called_once_with(
        ["docker", kind, "inspect", "fixture"],
        environment={},
        check=False,
        timeout=30,
    )


@pytest.mark.parametrize(
    ("returncode", "stdout", "stderr"),
    [
        (1, "", "Error response from daemon: network fixture not found\n"),
        (1, "[]\n", "Error response from daemon: network other not found\n"),
        (1, "[]\n", "Error response from daemon: No such network: fixture\n"),
        (2, "[]\n", "Error response from daemon: network fixture not found\n"),
    ],
)
def test_optional_inspection_rejects_ambiguous_absence(
    returncode: int,
    stdout: str,
    stderr: str,
) -> None:
    """Fail closed when Docker's status, identity, or JSON output disagrees."""
    result = subprocess.CompletedProcess(
        ["docker", "network", "inspect", "fixture"],
        returncode,
        stdout=stdout,
        stderr=stderr,
    )
    runner = create_autospec(_run, spec_set=True, return_value=result)

    with (
        patch(f"{__name__}._run", runner),
        pytest.raises(AssertionError, match="network inspection failed"),
    ):
        _inspect_optional("docker", {}, "network", "fixture")


def test_builder_cache_identity_ignores_relative_last_used_age() -> None:
    """Compare stable cache records without comparing Docker's relative clock."""
    document = {
        "CreatedAt": "2026-07-29 01:07:59.391031871 +0000 UTC",
        "Description": "local source for context",
        "ID": "cache-record",
        "LastUsedAt": "Less than a second ago",
        "Mutable": False,
        "Parents": None,
        "Reclaimable": True,
        "Shared": False,
        "Size": "0B",
        "Type": "source.local",
        "UsageCount": 1,
    }
    later_document = {**document, "LastUsedAt": "2 seconds ago"}
    runner = create_autospec(
        _run,
        spec_set=True,
        side_effect=[
            subprocess.CompletedProcess(
                [],
                0,
                stdout=f"{json.dumps(document)}\n",
                stderr="",
            ),
            subprocess.CompletedProcess(
                [],
                0,
                stdout=f"{json.dumps(later_document)}\n",
                stderr="",
            ),
        ],
    )

    with patch(f"{__name__}._run", runner):
        initial = _builder_cache_identity("docker", {})
        later = _builder_cache_identity("docker", {})

    expected = (
        (
            "cache-record",
            (),
            "2026-07-29 01:07:59.391031871 +0000 UTC",
            False,
            True,
            "0B",
            "source.local",
            "local source for context",
            1,
        ),
    )
    assert initial == expected
    assert later == expected


@pytest.mark.parametrize("last_used_at", ["", 0, []])
def test_builder_cache_identity_rejects_invalid_last_used_age(
    last_used_at: object,
) -> None:
    """Keep validating Docker's dynamic field even though it is not compared."""
    document = {
        "CreatedAt": "2026-07-29 01:07:59.391031871 +0000 UTC",
        "Description": "local source for context",
        "ID": "cache-record",
        "LastUsedAt": last_used_at,
        "Mutable": False,
        "Parents": None,
        "Reclaimable": True,
        "Shared": False,
        "Size": "0B",
        "Type": "source.local",
        "UsageCount": 1,
    }
    runner = create_autospec(
        _run,
        spec_set=True,
        return_value=subprocess.CompletedProcess(
            [],
            0,
            stdout=f"{json.dumps(document)}\n",
            stderr="",
        ),
    )

    with (
        patch(f"{__name__}._run", runner),
        pytest.raises(AssertionError, match="cache identity was invalid"),
    ):
        _builder_cache_identity("docker", {})


def test_hosted_retention_job_is_exact_and_isolated() -> None:
    """Run only the opted-in real-Docker node through an isolated pytest boundary."""
    job = _workflow_job()
    steps = job.get("steps")
    assert isinstance(steps, list)
    matches = [
        step
        for step in steps
        if isinstance(step, dict)
        and step.get("name") == "Run bounded VM image-retention proof"
    ]
    assert len(matches) == 1
    execution = matches[0]
    assert set(job) == {"name", "runs-on", "timeout-minutes", "env", "steps"}
    assert job["name"] == "Validate bounded VM image retention"
    assert job["runs-on"] == "ubuntu-latest"
    assert job["timeout-minutes"] == 30
    assert job["env"] == {
        RUN_ENVIRONMENT_NAME: "1",
        PREFIX_ENVIRONMENT_NAME: "adk-retention-${{ github.run_id }}",
    }
    assert [step.get("name") for step in steps if isinstance(step, dict)] == [
        "Checkout repository",
        "Install uv",
        "Install Python",
        "Install locked dependencies",
        "Run bounded VM image-retention proof",
    ]
    assert set(execution) == {"name", "env", "run"}
    assert execution["env"] == {
        "PYTEST_ADDOPTS": "",
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
        "PYTEST_PLUGINS": "",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    run = execution.get("run")
    assert isinstance(run, str)
    assert " ".join(run.split()) == (
        "uv run --locked --no-sync pytest "
        "--noconftest --confcutdir=tests "
        "-o addopts= -p no:cacheprovider "
        "tests/test_deployment_retention_runtime.py::"
        "test_real_docker_retention_reserves_one_exact_pull_slot "
        "-q --tb=line --disable-warnings --show-capture=no"
    )


def test_runtime_cleanup_is_exact_owned_and_never_forced() -> None:
    """Keep test cleanup inside the exact resources created by this proof."""
    cleanup = inspect.getsource(_cleanup_target)
    ownership = inspect.getsource(_owned_document)

    assert cleanup.index("_owned_document(") < cleanup.index("command =")
    assert "_owned_document(docker, environment, target)" in cleanup
    assert "expected_owner" in ownership
    assert "expected_id" in ownership
    assert '"--no-prune"' in cleanup
    assert '"--force"' not in cleanup
    assert '"image", "prune"' not in cleanup
    assert '"builder", "prune"' not in cleanup
    assert '"buildx", "prune"' not in cleanup
    assert '"system", "prune"' not in cleanup


@pytest.mark.skipif(
    os.environ.get(RUN_ENVIRONMENT_NAME) != "1",
    reason="real deployment-retention Docker proof is opt-in",
)
def test_real_docker_retention_reserves_one_exact_pull_slot(
    tmp_path: Path,
) -> None:
    """Delete one owned digest, pull the target, and preserve every sentinel."""
    environment = _base_environment()
    docker = _require_docker(environment)
    prefix = _resource_prefix()
    registry_name = f"{prefix}-registry"
    running_container = f"{prefix}-running"
    stopped_container = f"{prefix}-stopped"
    sentinel_network = f"{prefix}-network"
    sentinel_volume = f"{prefix}-volume"
    foreign_tag = f"{prefix}-foreign:keep"
    derivative_context = tmp_path / "derivative"
    state_dir = tmp_path / "deployment-state"
    environment_dir = tmp_path / "environments"
    environment_dir.mkdir(mode=0o700)
    cleanup_targets: list[CleanupTarget] = []
    primary_error: BaseException | None = None

    try:
        registry_image_id = _ensure_registry_image(
            docker,
            environment,
            cleanup_targets,
        )
        endpoint = _start_registry(
            docker,
            environment,
            cleanup_targets,
            name=registry_name,
            image_id=registry_image_id,
            owner=prefix,
        )
        repository = f"{endpoint}/{prefix}/agent"
        _write_derivative_context(derivative_context)
        images = tuple(
            _publish_managed_image(
                docker,
                environment,
                cleanup_targets,
                context=derivative_context,
                repository=repository,
                phase=phase,
                owner=prefix,
                keep_local=phase < 8,
            )
            for phase in range(9)
        )
        assert len({image.image_id for image in images}) == 9
        assert len({image.reference for image in images}) == 9
        assert len({image.revision for image in images}) == 9
        assert _present_references(docker, environment, images) == frozenset(
            image.reference for image in images[:8]
        )

        _assert_absent(docker, environment, "image", foreign_tag)
        _run(
            [docker, "image", "tag", registry_image_id, foreign_tag],
            environment=environment,
            timeout=30,
        )
        cleanup_targets.append(
            CleanupTarget(
                kind="image",
                reference=foreign_tag,
                expected_id=registry_image_id,
            )
        )

        sentinel_resources: tuple[tuple[ResourceKind, str], ...] = (
            ("network", sentinel_network),
            ("volume", sentinel_volume),
            ("container", running_container),
            ("container", stopped_container),
        )
        for kind, name in sentinel_resources:
            _assert_absent(docker, environment, kind, name)
        cleanup_targets.append(
            CleanupTarget(
                kind="network",
                reference=sentinel_network,
                expected_owner=prefix,
            )
        )
        _run(
            [
                docker,
                "network",
                "create",
                "--label",
                f"{OWNER_LABEL}={prefix}",
                sentinel_network,
            ],
            environment=environment,
            timeout=30,
        )
        cleanup_targets.append(
            CleanupTarget(
                kind="volume",
                reference=sentinel_volume,
                expected_owner=prefix,
            )
        )
        _run(
            [
                docker,
                "volume",
                "create",
                "--label",
                f"{OWNER_LABEL}={prefix}",
                sentinel_volume,
            ],
            environment=environment,
            timeout=30,
        )
        cleanup_targets.append(
            CleanupTarget(
                kind="container",
                reference=stopped_container,
                expected_owner=prefix,
            )
        )
        _run(
            [
                docker,
                "container",
                "create",
                "--name",
                stopped_container,
                "--label",
                f"{OWNER_LABEL}={prefix}",
                "--network",
                "none",
                "--entrypoint",
                "/bin/true",
                images[6].reference,
            ],
            environment=environment,
            timeout=30,
        )
        cleanup_targets.append(
            CleanupTarget(
                kind="container",
                reference=running_container,
                expected_owner=prefix,
            )
        )
        _run(
            [
                docker,
                "container",
                "create",
                "--name",
                running_container,
                "--label",
                f"{OWNER_LABEL}={prefix}",
                "--network",
                sentinel_network,
                "--mount",
                f"type=volume,source={sentinel_volume},target=/sentinel",
                "--entrypoint",
                "/bin/sh",
                images[7].reference,
                "-c",
                (f"printf %s {VOLUME_SENTINEL} > /sentinel/value; exec sleep 600"),
            ],
            environment=environment,
            timeout=30,
        )
        _run(
            [docker, "container", "start", running_container],
            environment=environment,
            timeout=30,
        )
        running_document = _container_document(
            docker,
            environment,
            running_container,
        )
        running_state = running_document.get("State")
        if (
            not isinstance(running_state, dict)
            or running_state.get("Running") is not True
        ):
            raise AssertionError("managed sentinel container did not start")
        if (
            _run(
                [
                    docker,
                    "container",
                    "exec",
                    running_container,
                    "cat",
                    "/sentinel/value",
                ],
                environment=environment,
                timeout=30,
            ).stdout
            != VOLUME_SENTINEL
        ):
            raise AssertionError("managed sentinel volume data was invalid")

        volume = _recorded_volume(
            docker,
            environment,
            sentinel_volume,
        )
        store = _record_three_generations(
            state_dir,
            environment_dir,
            images,
            volume,
            compose_project=f"retention-{uuid.uuid4().hex[:12]}",
        )
        with store.transaction() as transaction:
            current = transaction.current()
            assert current is not None
            assert current.state.image_reference == images[2].reference
            assert [entry.event for entry in transaction.journal()] == [
                "adopted",
                "promoted",
                "promoted",
            ]
            assert transaction.pending() is None

        state_identity = _state_tree(state_dir)
        running_identity = _container_identity(running_document)
        stopped_identity = _container_identity(
            _container_document(docker, environment, stopped_container)
        )
        network_document = _inspect_optional(
            docker,
            environment,
            "network",
            sentinel_network,
        )
        if network_document is None:
            raise AssertionError("sentinel network was unavailable")
        network_identity = _network_identity(network_document)
        volume_identity = _volume_identity(
            _volume_document(docker, environment, sentinel_volume)
        )
        foreign_document = _inspect_optional(
            docker,
            environment,
            "image",
            foreign_tag,
        )
        if foreign_document is None:
            raise AssertionError("foreign sentinel image was unavailable")
        foreign_identity = _image_identity(foreign_document)
        registry_identity = _container_identity(
            _container_document(docker, environment, registry_name)
        )

        protected_images = (*images[:3], images[6], images[7])
        protected_image_identities = {
            image.reference: _image_identity(
                _managed_image_document(
                    docker,
                    environment,
                    image.reference,
                    owner=prefix,
                    revision=image.revision,
                )
            )
            for image in protected_images
        }
        before = _present_references(docker, environment, images)
        builder_cache_identity = _builder_cache_identity(docker, environment)
        expected_deleted = min(image.reference for image in images[3:6])
        planned, applied = _apply_retention(
            store,
            docker,
            environment,
            image_repository=repository,
            target_reference=images[8].reference,
        )
        assert planned.delete_references == (expected_deleted,)
        assert planned.admitted_count == 9
        assert not planned.admitted
        assert applied.delete_references == ()
        assert applied.admitted_count == 8
        assert applied.admitted
        assert _builder_cache_identity(docker, environment) == builder_cache_identity
        after_retention = _present_references(docker, environment, images)
        assert before - after_retention == {expected_deleted}
        assert after_retention - before == set()
        assert len(after_retention) == 7
        for protected in (*images[:3], images[6], images[7]):
            assert protected.reference in after_retention
        assert images[8].reference not in after_retention

        _run(
            [docker, "image", "pull", images[8].reference],
            environment=environment,
            timeout=180,
        )
        target_document = _managed_image_document(
            docker,
            environment,
            images[8].reference,
            owner=prefix,
            revision=images[8].revision,
        )
        assert _resource_identity("image", target_document) == images[8].image_id
        expected_present = {image.reference for image in images}
        expected_present.remove(expected_deleted)
        assert _present_references(
            docker,
            environment,
            images,
        ) == frozenset(expected_present)
        assert len(_present_references(docker, environment, images)) == 8

        before_admission = _state_tree(state_dir)
        with store.transaction() as transaction:
            plan = enforce_pull_admission(
                transaction,
                docker=Path(docker),
                environment=environment,
                repository=MANAGED_REPOSITORY,
                target_reference=images[8].reference,
                image_repository=repository,
            )
        assert isinstance(plan, RetentionPlan)
        assert _state_tree(state_dir) == before_admission == state_identity

        assert {
            image.reference: _image_identity(
                _managed_image_document(
                    docker,
                    environment,
                    image.reference,
                    owner=prefix,
                    revision=image.revision,
                )
            )
            for image in protected_images
        } == protected_image_identities
        assert (
            _container_identity(
                _container_document(docker, environment, running_container)
            )
            == running_identity
        )
        assert (
            _container_identity(
                _container_document(docker, environment, stopped_container)
            )
            == stopped_identity
        )
        final_network = _inspect_optional(
            docker,
            environment,
            "network",
            sentinel_network,
        )
        assert final_network is not None
        assert _network_identity(final_network) == network_identity
        assert (
            _volume_identity(_volume_document(docker, environment, sentinel_volume))
            == volume_identity
        )
        assert (
            _run(
                [
                    docker,
                    "container",
                    "exec",
                    running_container,
                    "cat",
                    "/sentinel/value",
                ],
                environment=environment,
                timeout=30,
            ).stdout
            == VOLUME_SENTINEL
        )
        final_foreign = _inspect_optional(
            docker,
            environment,
            "image",
            foreign_tag,
        )
        assert final_foreign is not None
        assert _image_identity(final_foreign) == foreign_identity
        assert (
            _container_identity(_container_document(docker, environment, registry_name))
            == registry_identity
        )
    except BaseException as error:
        primary_error = error
        raise
    finally:
        _report_cleanup_failures(
            _execute_cleanup(docker, environment, cleanup_targets),
            primary_error,
        )
