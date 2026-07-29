"""Bound exact project-owned Docker image references before a VM pull."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import stat
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from agent.deployment_state import (
    DeploymentLockBusyError,
    DeploymentState,
    DeploymentStateError,
    DeploymentStateStore,
    DeploymentStateTransaction,
)

MAX_MANAGED_REFERENCES: Final = 8
MAX_DELETIONS_PER_INVOCATION: Final = 5
INCOMPLETE_EXIT_STATUS: Final = 3
_MAX_COMMAND_OUTPUT_BYTES: Final = 256 * 1024
_MAX_INVENTORY_IDENTITIES: Final = 512
_COMMAND_TIMEOUT_SECONDS: Final = 60
_HOST_ENVIRONMENT_NAMES: Final = (
    "HOME",
    "PATH",
    "DOCKER_CONFIG",
    "DOCKER_HOST",
    "XDG_RUNTIME_DIR",
)
_GITHUB_REPOSITORY = re.compile(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+\Z")
_IMAGE_ID = re.compile(r"sha256:[0-9a-f]{64}\Z")
_CONTAINER_ID = re.compile(r"[0-9a-f]{64}\Z")
_REVISION = re.compile(r"[0-9a-f]{40}\Z")
_REPOSITORY_COMPONENT = r"[a-z0-9]+(?:(?:[._]|__|-+)[a-z0-9]+)*"
_IMAGE_REPOSITORY = re.compile(
    rf"(?=.{{1,255}}\Z)"
    rf"(?:[a-z0-9](?:[a-z0-9.-]*[a-z0-9])?(?::[0-9]{{1,5}})?/)?"
    rf"{_REPOSITORY_COMPONENT}(?:/{_REPOSITORY_COMPONENT})*\Z"
)


class ImageRetentionError(RuntimeError):
    """Report a deterministic, secret-free image-retention failure."""


@dataclass(frozen=True, slots=True)
class ManagedImage:
    """One locally proven, alias-free project image reference."""

    reference: str
    image_id: str
    oci_revision: str

    def as_document(self) -> dict[str, str]:
        """Return bounded secret-free image identity."""
        return {
            "reference": self.reference,
            "image_id": self.image_id,
            "oci_revision": self.oci_revision,
        }


@dataclass(frozen=True, slots=True)
class ReferenceDecision:
    """One deterministic keep/delete decision for an occupied project digest."""

    reference: str
    action: str
    reasons: tuple[str, ...]
    image_id: str | None

    def as_document(self) -> dict[str, object]:
        """Return the decision without inventing an unproven image identity."""
        document: dict[str, object] = {
            "reference": self.reference,
            "action": self.action,
            "reasons": list(self.reasons),
        }
        if self.image_id is not None:
            document["image_id"] = self.image_id
        return document


@dataclass(frozen=True, slots=True)
class RetentionPlan:
    """One immutable capacity plan computed from verified state and Docker."""

    repository: str
    target_reference: str
    target_present: bool
    reserved_references: int
    occupied_references: tuple[str, ...]
    ambiguous_references: tuple[str, ...]
    managed_references: tuple[ManagedImage, ...]
    protected_references: tuple[str, ...]
    missing_generation_references: tuple[str, ...]
    delete_references: tuple[str, ...]
    reference_decisions: tuple[ReferenceDecision, ...]

    @property
    def admitted_count(self) -> int:
        """Return occupied references including an absent target reservation."""
        return len(self.occupied_references) + self.reserved_references

    @property
    def required_deletions(self) -> int:
        """Return deletions needed before the requested target can be pulled."""
        return len(self.delete_references)

    @property
    def admitted(self) -> bool:
        """Return whether a target pull stays within the fixed ceiling."""
        return self.admitted_count <= MAX_MANAGED_REFERENCES

    def as_document(self, *, status: str) -> dict[str, object]:
        """Return a bounded, secret-free plan or result document."""
        return {
            "status": status,
            "repository": self.repository,
            "target_reference": self.target_reference,
            "target_present": self.target_present,
            "reserved_references": self.reserved_references,
            "occupied_reference_count": len(self.occupied_references),
            "managed_reference_count": len(self.managed_references),
            "managed_references": [
                image.as_document() for image in self.managed_references
            ],
            "admitted_reference_count": self.admitted_count,
            "managed_reference_limit": MAX_MANAGED_REFERENCES,
            "maximum_deletions": MAX_DELETIONS_PER_INVOCATION,
            "protected_references": list(self.protected_references),
            "ambiguous_references": list(self.ambiguous_references),
            "missing_generation_references": list(self.missing_generation_references),
            "delete_references": list(self.delete_references),
            "reference_decisions": [
                decision.as_document() for decision in self.reference_decisions
            ],
        }


@dataclass(frozen=True, slots=True)
class _RecordedGeneration:
    state: DeploymentState
    last_sequence: int


@dataclass(frozen=True, slots=True)
class _ImageDocument:
    image_id: str
    repo_digests: tuple[str, ...]
    repo_tags: tuple[str, ...]
    labels: Mapping[str, str]


def _validated_identity(
    repository: str,
    target_reference: str,
    image_repository: str | None,
) -> tuple[str, str]:
    if _GITHUB_REPOSITORY.fullmatch(repository) is None:
        raise ImageRetentionError("GitHub repository identity is invalid")
    selected_repository = (
        f"ghcr.io/{repository.lower()}"
        if image_repository is None
        else image_repository
    )
    if _IMAGE_REPOSITORY.fullmatch(selected_repository) is None:
        raise ImageRetentionError("image repository identity is invalid")
    target_pattern = re.compile(
        rf"{re.escape(selected_repository)}@sha256:[0-9a-f]{{64}}\Z"
    )
    if target_pattern.fullmatch(target_reference) is None:
        raise ImageRetentionError("target image reference is invalid")
    return selected_repository, f"https://github.com/{repository}"


def _validated_docker(path: Path) -> Path:
    try:
        canonical = path.resolve(strict=True)
        metadata = canonical.stat()
    except OSError:
        raise ImageRetentionError("Docker executable is unavailable") from None
    if (
        not canonical.is_absolute()
        or canonical != path
        or not stat.S_ISREG(metadata.st_mode)
    ):
        raise ImageRetentionError("Docker executable is invalid")
    return canonical


def _run(
    docker: Path,
    arguments: Sequence[str],
    *,
    environment: Mapping[str, str],
    accepted_returncodes: frozenset[int] = frozenset({0}),
) -> subprocess.CompletedProcess[str]:
    try:
        result = subprocess.run(  # noqa: S603 - resolved executable, fixed arguments
            [str(docker), *arguments],
            env=environment,
            text=True,
            capture_output=True,
            check=False,
            timeout=_COMMAND_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        raise ImageRetentionError("Docker image-retention command timed out") from None
    except OSError:
        raise ImageRetentionError(
            "Docker image-retention command could not start"
        ) from None
    if result.returncode not in accepted_returncodes:
        raise ImageRetentionError(
            f"Docker image-retention command failed (exit {result.returncode})"
        )
    if (
        len(result.stdout.encode()) > _MAX_COMMAND_OUTPUT_BYTES
        or len(result.stderr.encode()) > _MAX_COMMAND_OUTPUT_BYTES
    ):
        raise ImageRetentionError("Docker image-retention output is too large")
    return result


def _identity_lines(
    value: str,
    pattern: re.Pattern[str],
    field: str,
) -> tuple[str, ...]:
    if "\0" in value or "\r" in value:
        raise ImageRetentionError(f"Docker {field} inventory is invalid")
    lines = tuple(line for line in value.splitlines() if line)
    if len(lines) > _MAX_INVENTORY_IDENTITIES or any(
        pattern.fullmatch(line) is None for line in lines
    ):
        raise ImageRetentionError(f"Docker {field} inventory is invalid")
    return tuple(sorted(set(lines)))


def _decode_single_document(value: str, field: str) -> Mapping[str, object]:
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError:
        raise ImageRetentionError(f"Docker {field} inspection is invalid") from None
    if (
        not isinstance(decoded, list)
        or len(decoded) != 1
        or not isinstance(decoded[0], dict)
    ):
        raise ImageRetentionError(f"Docker {field} inspection is invalid")
    return decoded[0]


def _string_sequence(value: object, field: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ImageRetentionError(f"Docker image {field} is invalid")
    return tuple(value)


def _image_mount_selector(value: object) -> str:
    if not isinstance(value, str):
        raise ImageRetentionError(
            "Docker image mount selector is not an immutable digest"
        )
    repository, separator, digest = value.rpartition("@")
    if (
        separator != "@"
        or _IMAGE_REPOSITORY.fullmatch(repository) is None
        or _IMAGE_ID.fullmatch(digest) is None
    ):
        raise ImageRetentionError(
            "Docker image mount selector is not an immutable digest"
        )
    return value


def _image_document(value: str) -> _ImageDocument:
    document = _decode_single_document(value, "image")
    image_id = document.get("Id")
    if not isinstance(image_id, str) or _IMAGE_ID.fullmatch(image_id) is None:
        raise ImageRetentionError("Docker image identity is invalid")
    config = document.get("Config")
    if not isinstance(config, dict):
        raise ImageRetentionError("Docker image configuration is invalid")
    raw_labels = config.get("Labels")
    if raw_labels is None:
        labels: Mapping[str, str] = {}
    elif isinstance(raw_labels, dict) and all(
        isinstance(key, str) and isinstance(item, str)
        for key, item in raw_labels.items()
    ):
        labels = raw_labels
    else:
        raise ImageRetentionError("Docker image labels are invalid")
    return _ImageDocument(
        image_id=image_id,
        repo_digests=_string_sequence(document.get("RepoDigests"), "digests"),
        repo_tags=_string_sequence(document.get("RepoTags"), "tags"),
        labels=labels,
    )


def _inspect_image(
    docker: Path,
    identifier: str,
    environment: Mapping[str, str],
) -> _ImageDocument | None:
    result = _run(
        docker,
        ["image", "inspect", identifier],
        environment=environment,
        accepted_returncodes=frozenset({0, 1}),
    )
    if result.returncode == 1:
        expected_error = f"Error response from daemon: No such image: {identifier}\n"
        if result.stdout not in {"[]", "[]\n"} or result.stderr != expected_error:
            raise ImageRetentionError("Docker image absence could not be proven")
        return None
    return _image_document(result.stdout)


def _managed_image(
    document: _ImageDocument,
    *,
    repository: str,
    image_repository: str,
    expected_source: str,
) -> ManagedImage | None:
    prefix = f"{image_repository}@sha256:"
    project_digests = tuple(
        reference for reference in document.repo_digests if reference.startswith(prefix)
    )
    if (
        len(document.repo_digests) != 1
        or len(project_digests) != 1
        or document.repo_tags
        or document.labels.get("io.queryplanner.adk.repository") != repository
        or document.labels.get("org.opencontainers.image.source") != expected_source
    ):
        return None
    revision = document.labels.get("org.opencontainers.image.revision")
    if revision is None or _REVISION.fullmatch(revision) is None:
        return None
    return ManagedImage(
        reference=project_digests[0],
        image_id=document.image_id,
        oci_revision=revision,
    )


def _local_inventory(
    *,
    docker: Path,
    environment: Mapping[str, str],
    repository: str,
    image_repository: str,
    expected_source: str,
) -> tuple[
    dict[str, ManagedImage],
    frozenset[str],
    frozenset[str],
    frozenset[str],
]:
    image_ids = _identity_lines(
        _run(
            docker,
            ["image", "ls", "--all", "--no-trunc", "--quiet"],
            environment=environment,
        ).stdout,
        _IMAGE_ID,
        "image",
    )
    managed: dict[str, ManagedImage] = {}
    observed_images: dict[str, _ImageDocument] = {}
    local_digest_references: set[str] = set()
    occupied_image_ids: dict[str, set[str]] = {}
    ambiguous_references: set[str] = set()
    exact_reference = re.compile(
        rf"{re.escape(image_repository)}@sha256:[0-9a-f]{{64}}\Z"
    )
    project_prefix = f"{image_repository}@"
    for image_id in image_ids:
        document = _inspect_image(docker, image_id, environment)
        if document is None or document.image_id != image_id:
            raise ImageRetentionError(
                "Docker image inventory changed during inspection"
            )
        observed_images[image_id] = document
        local_digest_references.update(document.repo_digests)
        project_references: list[str] = []
        for reference in document.repo_digests:
            if not reference.startswith(project_prefix):
                continue
            if exact_reference.fullmatch(reference) is None:
                raise ImageRetentionError(
                    "Docker project image digest reference is malformed"
                )
            project_references.append(reference)
            occupied_image_ids.setdefault(reference, set()).add(image_id)
        has_project_tag = any(
            reference.startswith(f"{image_repository}:")
            for reference in document.repo_tags
        )
        if has_project_tag and not project_references:
            raise ImageRetentionError(
                "project image tag lacks a canonical digest reference"
            )
        if (
            document.labels.get("io.queryplanner.adk.repository") == repository
            and not project_references
        ):
            raise ImageRetentionError(
                "labeled project image lacks a canonical digest reference"
            )
        selected = _managed_image(
            document,
            repository=repository,
            image_repository=image_repository,
            expected_source=expected_source,
        )
        if selected is None:
            ambiguous_references.update(project_references)
            continue
        by_reference = _inspect_image(docker, selected.reference, environment)
        if by_reference is None:
            raise ImageRetentionError("managed image reference disappeared")
        confirmed = _managed_image(
            by_reference,
            repository=repository,
            image_repository=image_repository,
            expected_source=expected_source,
        )
        if confirmed != selected:
            raise ImageRetentionError(
                "managed image identity changed during inspection"
            )
        existing = managed.get(selected.reference)
        if existing is not None and existing != selected:
            raise ImageRetentionError("managed image reference is ambiguous")
        managed[selected.reference] = selected
    for reference, owners in occupied_image_ids.items():
        if len(owners) != 1:
            ambiguous_references.add(reference)
            managed.pop(reference, None)
    ambiguous_references.update(set(occupied_image_ids) - set(managed))

    container_ids = _identity_lines(
        _run(
            docker,
            ["container", "ls", "--all", "--no-trunc", "--quiet"],
            environment=environment,
        ).stdout,
        _CONTAINER_ID,
        "container",
    )
    used_image_ids: set[str] = set()
    for container_id in container_ids:
        result = _run(
            docker,
            ["container", "inspect", container_id],
            environment=environment,
        )
        container_document = _decode_single_document(result.stdout, "container")
        full_id = container_document.get("Id")
        container_image_id = container_document.get("Image")
        mounts = container_document.get("Mounts")
        if (
            not isinstance(full_id, str)
            or full_id != container_id
            or _CONTAINER_ID.fullmatch(full_id) is None
            or not isinstance(container_image_id, str)
            or _IMAGE_ID.fullmatch(container_image_id) is None
            or not isinstance(mounts, list)
        ):
            raise ImageRetentionError("Docker container identity is invalid")
        used_image_ids.add(container_image_id)
        image_mount_selectors: list[str] = []
        for mount in mounts:
            if not isinstance(mount, dict):
                raise ImageRetentionError("Docker container mount is invalid")
            if mount.get("Type") != "image":
                continue
            image_mount_selectors.append(_image_mount_selector(mount.get("Name")))
        for selector in sorted(set(image_mount_selectors)):
            mounted = _inspect_image(docker, selector, environment)
            if mounted is None:
                raise ImageRetentionError("Docker image mount is unavailable")
            if (
                selector not in mounted.repo_digests
                or observed_images.get(mounted.image_id) != mounted
            ):
                raise ImageRetentionError("Docker image mount identity is ambiguous")
            used_image_ids.add(mounted.image_id)
    return (
        managed,
        frozenset(used_image_ids),
        frozenset(local_digest_references),
        frozenset(ambiguous_references),
    )


def _recorded_generations(
    transaction: DeploymentStateTransaction,
) -> tuple[
    dict[str, _RecordedGeneration],
    tuple[str, ...],
]:
    if transaction.pending() is not None:
        raise ImageRetentionError("image retention is blocked by pending recovery")
    if transaction.recovered_terminal() is not None:
        raise ImageRetentionError("image retention requires a fresh recovery check")

    journal = transaction.journal()
    current = transaction.current()
    recorded: dict[str, _RecordedGeneration] = {}
    for entry in journal:
        if entry.event not in {"adopted", "promoted"}:
            continue
        if entry.state is None:  # pragma: no cover - state loader invariant
            raise ImageRetentionError("deployment generation state is unavailable")
        reference = entry.state.image_reference
        existing = recorded.get(reference)
        if existing is not None and (
            existing.state.image_id != entry.state.image_id
            or existing.state.oci_revision != entry.state.oci_revision
            or existing.state.source_revision != entry.state.source_revision
        ):
            raise ImageRetentionError("recorded image generation identity conflicts")
        recorded[reference] = _RecordedGeneration(
            state=entry.state,
            last_sequence=entry.sequence,
        )

    if current is None:
        if recorded:  # pragma: no cover - current is reconciled from this journal
            raise ImageRetentionError("deployment current state is unavailable")
        return recorded, ()

    index = current.journal_sequence - 1
    if index < 0 or index >= len(journal):  # pragma: no cover - state invariant
        raise ImageRetentionError("deployment current journal anchor is invalid")
    anchor = journal[index]
    if (  # pragma: no cover - current is reconciled from this exact entry
        anchor.event not in {"adopted", "promoted"}
        or anchor.sha256 != current.journal_sha256
        or anchor.state != current.state
    ):
        raise ImageRetentionError("deployment current journal anchor does not match")

    selected = [current.state.image_reference]
    seen = set(selected)
    for entry in reversed(journal[:index]):
        if entry.event not in {"adopted", "promoted"}:
            continue
        if entry.state is None:  # pragma: no cover - state loader invariant
            raise ImageRetentionError("deployment generation state is unavailable")
        reference = entry.state.image_reference
        if reference in seen:
            continue
        selected.append(reference)
        seen.add(reference)
        if len(selected) == 3:
            break
    return recorded, tuple(selected)


def plan_image_retention(
    transaction: DeploymentStateTransaction,
    *,
    docker: Path,
    environment: Mapping[str, str],
    repository: str,
    target_reference: str,
    image_repository: str | None = None,
) -> RetentionPlan:
    """Plan exact deletions while an existing state transaction holds its lock."""
    image_repository, expected_source = _validated_identity(
        repository,
        target_reference,
        image_repository,
    )
    selected_docker = _validated_docker(docker)
    recorded, generations = _recorded_generations(transaction)
    (
        managed,
        used_image_ids,
        local_digest_references,
        ambiguous_references,
    ) = _local_inventory(
        docker=selected_docker,
        environment=environment,
        repository=repository,
        image_repository=image_repository,
        expected_source=expected_source,
    )
    for reference, generation in recorded.items():
        image = managed.get(reference)
        if image is not None and (
            image.image_id != generation.state.image_id
            or image.oci_revision != generation.state.oci_revision
        ):
            raise ImageRetentionError(
                "local image does not match its recorded generation"
            )

    target_document = _inspect_image(selected_docker, target_reference, environment)
    target_present = target_document is not None
    if target_document is not None:
        target = _managed_image(
            target_document,
            repository=repository,
            image_repository=image_repository,
            expected_source=expected_source,
        )
        if target is None or managed.get(target_reference) != target:
            raise ImageRetentionError("local target image is not exactly managed")

    protected = set(generations)
    protected.add(target_reference)
    protected.update(ambiguous_references)
    protected.update(
        image.reference
        for image in managed.values()
        if image.image_id in used_image_ids
    )
    missing_generations = tuple(
        reference
        for reference in generations
        if reference not in local_digest_references
    )
    reserved = 0 if target_present else 1
    occupied_references = frozenset(
        reference
        for reference in local_digest_references
        if reference.startswith(f"{image_repository}@")
    )
    required = max(
        0,
        len(occupied_references) + reserved - MAX_MANAGED_REFERENCES,
    )

    unrecorded = sorted(
        (
            image
            for image in managed.values()
            if image.reference not in recorded and image.reference not in protected
        ),
        key=lambda image: image.reference,
    )
    historical = sorted(
        (
            image
            for image in managed.values()
            if image.reference in recorded and image.reference not in protected
        ),
        key=lambda image: (
            recorded[image.reference].last_sequence,
            image.reference,
        ),
    )
    candidates = (*unrecorded, *historical)
    if required > len(candidates):
        raise ImageRetentionError(
            "managed image ceiling is unreachable without protected references"
        )
    deletions = tuple(image.reference for image in candidates[:required])
    deletion_set = set(deletions)
    current_reference = None if not generations else generations[0]
    prior_references = set(generations[1:])
    container_references = {
        image.reference
        for image in managed.values()
        if image.image_id in used_image_ids
    }
    decisions: list[ReferenceDecision] = []
    for reference in sorted(occupied_references):
        image = managed.get(reference)
        if reference in deletion_set:
            reason = (
                "planned_historical_deletion"
                if reference in recorded
                else "planned_unrecorded_deletion"
            )
            decisions.append(
                ReferenceDecision(
                    reference=reference,
                    action="delete",
                    reasons=(reason,),
                    image_id=None if image is None else image.image_id,
                )
            )
            continue
        reasons: list[str] = []
        if reference == target_reference:
            reasons.append("requested_target")
        if reference == current_reference:
            reasons.append("current_generation")
        elif reference in prior_references:
            reasons.append("prior_generation")
        if reference in container_references:
            reasons.append("container_use")
        if reference in ambiguous_references:
            reasons.append("ambiguous_ownership")
        if not reasons:
            reasons.append("available_capacity")
        decisions.append(
            ReferenceDecision(
                reference=reference,
                action="keep",
                reasons=tuple(reasons),
                image_id=None if image is None else image.image_id,
            )
        )
    return RetentionPlan(
        repository=repository,
        target_reference=target_reference,
        target_present=target_present,
        reserved_references=reserved,
        occupied_references=tuple(sorted(occupied_references)),
        ambiguous_references=tuple(sorted(ambiguous_references)),
        managed_references=tuple(
            sorted(managed.values(), key=lambda image: image.reference)
        ),
        protected_references=tuple(sorted(protected)),
        missing_generation_references=missing_generations,
        delete_references=deletions,
        reference_decisions=tuple(decisions),
    )


def enforce_pull_admission(
    transaction: DeploymentStateTransaction,
    *,
    docker: Path,
    environment: Mapping[str, str],
    repository: str,
    target_reference: str,
    image_repository: str | None = None,
) -> RetentionPlan:
    """Repeat read-only pull admission while the promotion lock is held."""
    plan = plan_image_retention(
        transaction,
        docker=docker,
        environment=environment,
        repository=repository,
        target_reference=target_reference,
        image_repository=image_repository,
    )
    if not plan.admitted:
        raise ImageRetentionError("managed image capacity is not admitted")
    return plan


def _delete_image(
    image: ManagedImage,
    *,
    docker: Path,
    environment: Mapping[str, str],
    repository: str,
    image_repository: str,
    expected_source: str,
) -> None:
    current = _inspect_image(docker, image.reference, environment)
    confirmed = (
        None
        if current is None
        else _managed_image(
            current,
            repository=repository,
            image_repository=image_repository,
            expected_source=expected_source,
        )
    )
    if confirmed != image:
        raise ImageRetentionError("managed image changed before exact removal")
    _run(
        docker,
        ["image", "rm", "--no-prune", image.reference],
        environment=environment,
    )
    remaining_reference = _inspect_image(docker, image.reference, environment)
    remaining_image = _inspect_image(docker, image.image_id, environment)
    if remaining_reference is not None or remaining_image is not None:
        raise ImageRetentionError("managed image remained after exact removal")


def apply_image_retention(
    transaction: DeploymentStateTransaction,
    *,
    docker: Path,
    environment: Mapping[str, str],
    repository: str,
    target_reference: str,
    image_repository: str | None = None,
) -> tuple[RetentionPlan, bool, tuple[ReferenceDecision, ...]]:
    """Apply at most one deterministic five-reference retention batch."""
    initial = plan_image_retention(
        transaction,
        docker=docker,
        environment=environment,
        repository=repository,
        target_reference=target_reference,
        image_repository=image_repository,
    )
    if initial.admitted:
        return initial, True, ()
    image_repository, expected_source = _validated_identity(
        repository,
        target_reference,
        image_repository,
    )
    selected_docker = _validated_docker(docker)
    removed_decisions: list[ReferenceDecision] = []
    for delete_index in range(MAX_DELETIONS_PER_INVOCATION):
        current = plan_image_retention(
            transaction,
            docker=selected_docker,
            environment=environment,
            repository=repository,
            target_reference=target_reference,
            image_repository=image_repository,
        )
        if current.admitted:
            return current, True, tuple(removed_decisions)
        expected = initial.delete_references[delete_index : delete_index + 1]
        if current.delete_references[:1] != expected:
            raise ImageRetentionError(
                "managed image deletion order changed before exact removal"
            )
        indexed = {image.reference: image for image in current.managed_references}
        reference = current.delete_references[0]
        decision = next(
            decision
            for decision in current.reference_decisions
            if decision.reference == reference
        )
        _delete_image(
            indexed[reference],
            docker=selected_docker,
            environment=environment,
            repository=repository,
            image_repository=image_repository,
            expected_source=expected_source,
        )
        removed_decisions.append(decision)
    final = plan_image_retention(
        transaction,
        docker=selected_docker,
        environment=environment,
        repository=repository,
        target_reference=target_reference,
        image_repository=image_repository,
    )
    return final, final.admitted, tuple(removed_decisions)


def _command_environment(source: Mapping[str, str]) -> dict[str, str]:
    selected = {
        name: source[name] for name in _HOST_ENVIRONMENT_NAMES if name in source
    }
    selected.update({"LANG": "C", "LC_ALL": "C"})
    return selected


def _resolved_docker(environment: Mapping[str, str]) -> Path:
    selected = shutil.which("docker", path=environment.get("PATH"))
    if selected is None:
        raise ImageRetentionError("Docker executable is unavailable")
    return _validated_docker(Path(selected).resolve(strict=True))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="deployment-retention",
        description="Plan or enforce exact VM image-reference retention.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    enforce = subparsers.add_parser("enforce")
    enforce.add_argument("--state-dir", type=Path, required=True)
    enforce.add_argument("--repository", required=True)
    enforce.add_argument("--target-reference", required=True)
    enforce.add_argument("--apply", action="store_true")
    return parser


def _emit(document: Mapping[str, object]) -> None:
    print(
        json.dumps(
            document,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    )


def main(
    argv: Sequence[str] | None = None,
    environment: Mapping[str, str] | None = None,
) -> int:
    """Run the image-retention CLI and return a stable process status."""
    arguments = _parser().parse_args(argv)
    source = os.environ if environment is None else environment
    command_environment = _command_environment(source)
    try:
        docker = _resolved_docker(command_environment)
        store = DeploymentStateStore(arguments.state_dir)
        with store.transaction() as transaction:
            if not arguments.apply:
                plan = plan_image_retention(
                    transaction,
                    docker=docker,
                    environment=command_environment,
                    repository=arguments.repository,
                    target_reference=arguments.target_reference,
                    image_repository=None,
                )
                _emit(plan.as_document(status="dry-run"))
                return 0
            plan, admitted, removed_decisions = apply_image_retention(
                transaction,
                docker=docker,
                environment=command_environment,
                repository=arguments.repository,
                target_reference=arguments.target_reference,
                image_repository=None,
            )
            document = plan.as_document(
                status="admitted" if admitted else "incomplete",
            )
            document["removed_reference_decisions"] = [
                decision.as_document() for decision in removed_decisions
            ]
            _emit(document)
            return 0 if admitted else INCOMPLETE_EXIT_STATUS
    except DeploymentLockBusyError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 75
    except (DeploymentStateError, ImageRetentionError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    except OSError:
        print("ERROR: image-retention host operation failed", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
