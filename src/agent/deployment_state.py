"""Crash-consistent private state for one POSIX VM deployment transaction."""

from __future__ import annotations

import errno
import fcntl
import hashlib
import json
import os
import re
import stat
import tempfile
import uuid
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Final, cast

SCHEMA_VERSION: Final = 1
PROMOTION_SCHEMA_VERSION: Final = 2
MAX_ENVIRONMENT_BYTES: Final = 1024 * 1024
MAX_JSON_BYTES: Final = 64 * 1024
MAX_PERSISTENT_VOLUMES: Final = 64
MAX_VOLUME_PATH_BYTES: Final = 4096

_COMPOSE_NAME = re.compile(r"[a-z0-9][a-z0-9_-]{0,62}\Z")
_DEPLOYMENT_ID = re.compile(r"[a-z0-9][a-z0-9-]{0,63}\Z")
_REVISION = re.compile(r"[0-9a-f]{40}\Z")
_SHA256_HEX = re.compile(r"[0-9a-f]{64}\Z")
_IMAGE_ID = re.compile(r"sha256:[0-9a-f]{64}\Z")
_CONTAINER_ID = re.compile(r"[0-9a-f]{64}\Z")
_VOLUME_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,254}\Z")
_VOLUME_DRIVER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")
_REPOSITORY_COMPONENT = r"[a-z0-9]+(?:(?:[._]|__|-+)[a-z0-9]+)*"
_REGISTRY_LABEL = r"[a-z0-9](?:[a-z0-9-]*[a-z0-9])?"
_IMAGE_REFERENCE = re.compile(
    rf"(?=.{{1,255}}@sha256:)"
    rf"(?:{_REGISTRY_LABEL}(?:\.{_REGISTRY_LABEL})*(?::[0-9]{{1,5}})?/)?"
    rf"{_REPOSITORY_COMPONENT}(?:/{_REPOSITORY_COMPONENT})*"
    r"@sha256:[0-9a-f]{64}\Z"
)
_RECORDED_AT = re.compile(
    r"[0-9]{4}-[0-9]{2}-[0-9]{2}T"
    r"[0-9]{2}:[0-9]{2}:[0-9]{2}\.[0-9]{6}Z\Z"
)
_VOLUME_CREATED_AT = re.compile(
    r"[0-9]{4}-[0-9]{2}-[0-9]{2}T"
    r"[0-9]{2}:[0-9]{2}:[0-9]{2}"
    r"(?:\.[0-9]{1,9})?(?:Z|[+-][0-9]{2}:[0-9]{2})\Z"
)
_JOURNAL_NAME = re.compile(r"([0-9]{20})\.json\Z")
_ENVIRONMENT_NAME = re.compile(r"[a-z0-9][a-z0-9-]{0,63}\.env\Z")
_CURRENT_TEMPORARY_NAME = re.compile(r"\.deployment-current-[a-z0-9._-]+\.tmp\Z")
_JOURNAL_TEMPORARY_NAME = re.compile(r"\.deployment-journal-[a-z0-9._-]+\.tmp\Z")
_ENVIRONMENT_TEMPORARY_NAME = re.compile(
    r"\.deployment-environment-[a-z0-9._-]+\.tmp\Z"
)
_INTENT_NAME = re.compile(r"([a-z0-9][a-z0-9-]{0,63})\.json\Z")
_INTENT_TEMPORARY_NAME = re.compile(r"\.deployment-intent-[a-z0-9._-]+\.tmp\Z")
_PENDING_TEMPORARY_NAME = re.compile(r"\.deployment-pending-[a-z0-9._-]+\.tmp\Z")
_PENDING_NAME = re.compile(r"pending\.json\Z")
_TERMINAL_EVENTS = frozenset({"promoted", "rolled_back", "aborted"})


class DeploymentStateError(RuntimeError):
    """Report a deterministic deployment-state contract failure."""


class DeploymentLockBusyError(DeploymentStateError):
    """Report that another process owns the VM deployment transaction."""


class DeploymentTerminalCommittedError(DeploymentStateError):
    """Report a durable terminal outcome whose derived state needs repair."""

    def __init__(self, event: str, transaction_id: str) -> None:
        self.event = event
        self.transaction_id = transaction_id
        super().__init__(
            f"deployment outcome is committed but reconciliation failed: {event}"
        )


class DeploymentTerminalIndeterminateError(DeploymentStateError):
    """Report a visible terminal record whose directory sync was indeterminate."""

    def __init__(self, event: str, transaction_id: str) -> None:
        self.event = event
        self.transaction_id = transaction_id
        super().__init__(f"deployment outcome publication is indeterminate: {event}")


@dataclass(frozen=True, slots=True)
class DeploymentState:
    """Secret-free metadata needed to identify and restore one deployment."""

    deployment_id: str
    recorded_at: str
    compose_project: str
    compose_service: str
    source_revision: str
    image_reference: str
    image_id: str
    oci_revision: str
    environment_snapshot: str
    environment_sha256: str
    adopted: bool

    def as_document(self) -> dict[str, object]:
        """Return the exact persisted state schema."""
        return {
            "schema_version": (
                SCHEMA_VERSION if self.adopted else PROMOTION_SCHEMA_VERSION
            ),
            "deployment_id": self.deployment_id,
            "recorded_at": self.recorded_at,
            "compose_project": self.compose_project,
            "compose_service": self.compose_service,
            "source_revision": self.source_revision,
            "image_reference": self.image_reference,
            "image_id": self.image_id,
            "oci_revision": self.oci_revision,
            "environment_snapshot": self.environment_snapshot,
            "environment_sha256": self.environment_sha256,
            "adopted": self.adopted,
        }


@dataclass(frozen=True, slots=True)
class JournalEntry:
    """One immutable, hash-chained deployment-state event."""

    sequence: int
    sha256: str
    previous_sha256: str | None
    event: str
    state: DeploymentState | None
    transaction_id: str | None = None
    intent_sha256: str | None = None
    persistent_volumes: tuple[PersistentVolumeIdentity, ...] = ()

    def as_document(self) -> dict[str, object]:
        """Return the exact persisted journal schema, excluding its outer hash."""
        if self.event != "adopted":
            return {
                "schema_version": PROMOTION_SCHEMA_VERSION,
                "sequence": self.sequence,
                "previous_sha256": self.previous_sha256,
                "event": self.event,
                "transaction_id": self.transaction_id,
                "intent_sha256": self.intent_sha256,
                "state": None if self.state is None else self.state.as_document(),
                "persistent_volumes": [
                    volume.as_document() for volume in self.persistent_volumes
                ],
            }
        return {
            "schema_version": SCHEMA_VERSION,
            "sequence": self.sequence,
            "previous_sha256": self.previous_sha256,
            "event": self.event,
            "state": cast(DeploymentState, self.state).as_document(),
        }


@dataclass(frozen=True, slots=True)
class CurrentDeployment:
    """The current pointer bound to one immutable journal record."""

    journal_sequence: int
    journal_sha256: str
    state: DeploymentState

    def as_document(self) -> dict[str, object]:
        """Return the exact persisted current-pointer schema."""
        return {
            "schema_version": (
                SCHEMA_VERSION if self.state.adopted else PROMOTION_SCHEMA_VERSION
            ),
            "journal_sequence": self.journal_sequence,
            "journal_sha256": self.journal_sha256,
            "state": self.state.as_document(),
        }


@dataclass(frozen=True, slots=True)
class CandidateReceipt:
    """Secret-free exact observation of one isolated candidate container."""

    observed_at: str
    compose_project: str
    compose_service: str
    container_id: str
    image_reference: str
    image_id: str
    oci_revision: str
    baseline_journal_sequence: int | None
    baseline_journal_sha256: str | None

    def as_document(self) -> dict[str, object]:
        """Return the exact persisted candidate receipt schema."""
        return {
            "schema_version": PROMOTION_SCHEMA_VERSION,
            "observed_at": self.observed_at,
            "compose_project": self.compose_project,
            "compose_service": self.compose_service,
            "container_id": self.container_id,
            "image_reference": self.image_reference,
            "image_id": self.image_id,
            "oci_revision": self.oci_revision,
            "baseline_journal_sequence": self.baseline_journal_sequence,
            "baseline_journal_sha256": self.baseline_journal_sha256,
        }


@dataclass(frozen=True, slots=True, order=True)
class PersistentVolumeIdentity:
    """Stable Docker volume identity and its production mount destination."""

    name: str
    driver: str
    mountpoint: str
    destination: str
    created_at: str

    def as_document(self) -> dict[str, object]:
        """Return the exact persisted volume identity schema."""
        return {
            "name": self.name,
            "driver": self.driver,
            "mountpoint": self.mountpoint,
            "destination": self.destination,
            "created_at": self.created_at,
        }


@dataclass(frozen=True, slots=True)
class PromotionIntent:
    """Immutable recovery contract published before production mutation."""

    transaction_id: str
    recorded_at: str
    baseline_journal_sequence: int | None
    baseline_journal_sha256: str | None
    baseline_current_sequence: int | None
    baseline_current_sha256: str | None
    target: DeploymentState
    candidate: CandidateReceipt
    persistent_volumes: tuple[PersistentVolumeIdentity, ...]

    def as_document(self) -> dict[str, object]:
        """Return the exact persisted promotion intent schema."""
        return {
            "schema_version": PROMOTION_SCHEMA_VERSION,
            "transaction_id": self.transaction_id,
            "recorded_at": self.recorded_at,
            "baseline_journal_sequence": self.baseline_journal_sequence,
            "baseline_journal_sha256": self.baseline_journal_sha256,
            "baseline_current_sequence": self.baseline_current_sequence,
            "baseline_current_sha256": self.baseline_current_sha256,
            "target": self.target.as_document(),
            "candidate": self.candidate.as_document(),
            "persistent_volumes": [
                volume.as_document() for volume in self.persistent_volumes
            ],
        }


@dataclass(frozen=True, slots=True)
class PendingPromotion:
    """Validated pointer to one immutable unresolved promotion intent."""

    transaction_id: str
    intent_path: str
    intent_sha256: str
    intent: PromotionIntent

    def as_document(self) -> dict[str, object]:
        """Return the exact persisted pending-pointer schema."""
        return {
            "schema_version": PROMOTION_SCHEMA_VERSION,
            "transaction_id": self.transaction_id,
            "intent_path": self.intent_path,
            "intent_sha256": self.intent_sha256,
        }


def _validated_string(
    document: Mapping[str, object],
    key: str,
    pattern: re.Pattern[str],
) -> str:
    value = document.get(key)
    if not isinstance(value, str) or pattern.fullmatch(value) is None:
        raise DeploymentStateError(f"deployment state field is invalid: {key}")
    return value


def _validate_timestamp(value: str, *, field: str = "recorded_at") -> str:
    if _RECORDED_AT.fullmatch(value) is None:
        raise DeploymentStateError(f"deployment state field is invalid: {field}")
    try:
        datetime.strptime(value, "%Y-%m-%dT%H:%M:%S.%fZ").replace(tzinfo=UTC)
    except ValueError:
        raise DeploymentStateError(
            f"deployment state field is invalid: {field}"
        ) from None
    return value


def _validate_snapshot_path(value: object, deployment_id: str) -> str:
    expected = f"environments/{deployment_id}.env"
    if not isinstance(value, str) or value != expected:
        raise DeploymentStateError(
            "deployment state field is invalid: environment_snapshot"
        )
    return value


def _validate_volume_created_at(value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value.encode()) > 35
        or _VOLUME_CREATED_AT.fullmatch(value) is None
    ):
        raise DeploymentStateError("deployment state field is invalid: created_at")
    try:
        datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        raise DeploymentStateError(
            "deployment state field is invalid: created_at"
        ) from None
    return value


def _state_from_document_with_adoption(
    document: object,
    *,
    expected_adopted: bool | None,
) -> DeploymentState:
    if not isinstance(document, dict):
        raise DeploymentStateError("deployment state document must be an object")
    expected_keys = {
        "schema_version",
        "deployment_id",
        "recorded_at",
        "compose_project",
        "compose_service",
        "source_revision",
        "image_reference",
        "image_id",
        "oci_revision",
        "environment_snapshot",
        "environment_sha256",
        "adopted",
    }
    if set(document) != expected_keys:
        raise DeploymentStateError("deployment state schema is invalid")

    deployment_id = _validated_string(document, "deployment_id", _DEPLOYMENT_ID)
    adopted = document.get("adopted")
    if type(adopted) is not bool or (
        expected_adopted is not None and adopted is not expected_adopted
    ):
        raise DeploymentStateError("deployment state field is invalid: adopted")
    expected_schema_version = SCHEMA_VERSION if adopted else PROMOTION_SCHEMA_VERSION
    if document.get("schema_version") != expected_schema_version:
        raise DeploymentStateError("deployment state schema is invalid")
    return DeploymentState(
        deployment_id=deployment_id,
        recorded_at=_validate_timestamp(
            _validated_string(document, "recorded_at", _RECORDED_AT)
        ),
        compose_project=_validated_string(
            document,
            "compose_project",
            _COMPOSE_NAME,
        ),
        compose_service=_validated_string(
            document,
            "compose_service",
            _COMPOSE_NAME,
        ),
        source_revision=_validated_string(
            document,
            "source_revision",
            _REVISION,
        ),
        image_reference=_validated_string(
            document,
            "image_reference",
            _IMAGE_REFERENCE,
        ),
        image_id=_validated_string(document, "image_id", _IMAGE_ID),
        oci_revision=_validated_string(document, "oci_revision", _REVISION),
        environment_snapshot=_validate_snapshot_path(
            document.get("environment_snapshot"),
            deployment_id,
        ),
        environment_sha256=_validated_string(
            document,
            "environment_sha256",
            _SHA256_HEX,
        ),
        adopted=adopted,
    )


def _state_from_document(document: object) -> DeploymentState:
    """Parse the original schema-v1 adopted-state contract."""
    return _state_from_document_with_adoption(document, expected_adopted=True)


def _target_state_from_document(document: object) -> DeploymentState:
    return _state_from_document_with_adoption(document, expected_adopted=False)


def _any_state_from_document(document: object) -> DeploymentState:
    return _state_from_document_with_adoption(document, expected_adopted=None)


def _validated_optional_identity(
    document: Mapping[str, object],
    sequence_key: str,
    sha256_key: str,
) -> tuple[int | None, str | None]:
    sequence = document.get(sequence_key)
    sha256 = document.get(sha256_key)
    if sequence is None and sha256 is None:
        return None, None
    if (
        type(sequence) is not int
        or sequence <= 0
        or not isinstance(sha256, str)
        or _SHA256_HEX.fullmatch(sha256) is None
    ):
        raise DeploymentStateError(
            f"deployment state identity is invalid: {sequence_key}"
        )
    return sequence, sha256


def _candidate_from_document(document: object) -> CandidateReceipt:
    if not isinstance(document, dict):
        raise DeploymentStateError("candidate receipt must be an object")
    expected_keys = {
        "schema_version",
        "observed_at",
        "compose_project",
        "compose_service",
        "container_id",
        "image_reference",
        "image_id",
        "oci_revision",
        "baseline_journal_sequence",
        "baseline_journal_sha256",
    }
    if (
        set(document) != expected_keys
        or document.get("schema_version") != PROMOTION_SCHEMA_VERSION
    ):
        raise DeploymentStateError("candidate receipt schema is invalid")
    baseline_sequence, baseline_sha256 = _validated_optional_identity(
        document,
        "baseline_journal_sequence",
        "baseline_journal_sha256",
    )
    return CandidateReceipt(
        observed_at=_validate_timestamp(
            _validated_string(document, "observed_at", _RECORDED_AT),
            field="observed_at",
        ),
        compose_project=_validated_string(
            document,
            "compose_project",
            _COMPOSE_NAME,
        ),
        compose_service=_validated_string(
            document,
            "compose_service",
            _COMPOSE_NAME,
        ),
        container_id=_validated_string(document, "container_id", _CONTAINER_ID),
        image_reference=_validated_string(
            document,
            "image_reference",
            _IMAGE_REFERENCE,
        ),
        image_id=_validated_string(document, "image_id", _IMAGE_ID),
        oci_revision=_validated_string(document, "oci_revision", _REVISION),
        baseline_journal_sequence=baseline_sequence,
        baseline_journal_sha256=baseline_sha256,
    )


def _volume_from_document(document: object) -> PersistentVolumeIdentity:
    if not isinstance(document, dict) or set(document) != {
        "name",
        "driver",
        "mountpoint",
        "destination",
        "created_at",
    }:
        raise DeploymentStateError("persistent volume identity schema is invalid")
    name = _validated_string(document, "name", _VOLUME_NAME)
    driver = _validated_string(document, "driver", _VOLUME_DRIVER)
    mountpoint = document.get("mountpoint")
    destination = document.get("destination")
    for field, value in (
        ("mountpoint", mountpoint),
        ("destination", destination),
    ):
        if (
            not isinstance(value, str)
            or len(value.encode()) > MAX_VOLUME_PATH_BYTES
            or not value.startswith("/")
            or value.startswith("//")
            or value == "/"
            or str(PurePosixPath(value)) != value
            or ".." in PurePosixPath(value).parts
            or "\\" in value
            or any(ord(character) < 32 or ord(character) == 127 for character in value)
        ):
            raise DeploymentStateError(f"persistent volume {field} is invalid")
    return PersistentVolumeIdentity(
        name=name,
        driver=driver,
        mountpoint=cast(str, mountpoint),
        destination=cast(str, destination),
        created_at=_validate_volume_created_at(document.get("created_at")),
    )


def _validated_volume_identities(
    values: Sequence[PersistentVolumeIdentity],
) -> tuple[PersistentVolumeIdentity, ...]:
    if len(values) > MAX_PERSISTENT_VOLUMES or any(
        not isinstance(value, PersistentVolumeIdentity) for value in values
    ):
        raise DeploymentStateError("persistent volume identities are invalid")
    volumes = tuple(
        sorted(_volume_from_document(value.as_document()) for value in values)
    )
    if (
        len({volume.name for volume in volumes}) != len(volumes)
        or len({volume.destination for volume in volumes}) != len(volumes)
        or len({volume.mountpoint for volume in volumes}) != len(volumes)
    ):
        raise DeploymentStateError("persistent volume identities are invalid")
    return volumes


def _volume_identities_from_document(
    document: object,
) -> tuple[PersistentVolumeIdentity, ...]:
    if not isinstance(document, list) or len(document) > MAX_PERSISTENT_VOLUMES:
        raise DeploymentStateError("persistent volume identities are invalid")
    volumes = tuple(_volume_from_document(value) for value in document)
    if (
        tuple(sorted(volumes)) != volumes
        or len({volume.name for volume in volumes}) != len(volumes)
        or len({volume.destination for volume in volumes}) != len(volumes)
        or len({volume.mountpoint for volume in volumes}) != len(volumes)
    ):
        raise DeploymentStateError("persistent volume identities are invalid")
    return volumes


def _intent_from_document(document: object) -> PromotionIntent:
    if not isinstance(document, dict):
        raise DeploymentStateError("promotion intent must be an object")
    expected_keys = {
        "schema_version",
        "transaction_id",
        "recorded_at",
        "baseline_journal_sequence",
        "baseline_journal_sha256",
        "baseline_current_sequence",
        "baseline_current_sha256",
        "target",
        "candidate",
        "persistent_volumes",
    }
    if (
        set(document) != expected_keys
        or document.get("schema_version") != PROMOTION_SCHEMA_VERSION
    ):
        raise DeploymentStateError("promotion intent schema is invalid")
    baseline_sequence, baseline_sha256 = _validated_optional_identity(
        document,
        "baseline_journal_sequence",
        "baseline_journal_sha256",
    )
    current_sequence, current_sha256 = _validated_optional_identity(
        document,
        "baseline_current_sequence",
        "baseline_current_sha256",
    )
    recorded_at = _validate_timestamp(
        _validated_string(document, "recorded_at", _RECORDED_AT)
    )
    target = _target_state_from_document(document.get("target"))
    candidate = _candidate_from_document(document.get("candidate"))
    volumes = _volume_identities_from_document(document.get("persistent_volumes"))
    transaction_id = _validated_string(
        document,
        "transaction_id",
        _DEPLOYMENT_ID,
    )
    if target.deployment_id != transaction_id:
        raise DeploymentStateError("promotion target identity does not match")
    if target.recorded_at != recorded_at or candidate.observed_at > recorded_at:
        raise DeploymentStateError("promotion intent timestamps do not match")
    if current_sequence is not None and (
        baseline_sequence is None or current_sequence > baseline_sequence
    ):
        raise DeploymentStateError("promotion current baseline is invalid")
    if (
        candidate.baseline_journal_sequence != baseline_sequence
        or candidate.baseline_journal_sha256 != baseline_sha256
    ):
        raise DeploymentStateError("candidate receipt baseline does not match")
    if (
        candidate.compose_project == target.compose_project
        or candidate.compose_service != target.compose_service
        or candidate.image_reference != target.image_reference
        or candidate.image_id != target.image_id
        or candidate.oci_revision != target.oci_revision
        or target.source_revision != target.oci_revision
    ):
        raise DeploymentStateError("candidate receipt target does not match")
    return PromotionIntent(
        transaction_id=transaction_id,
        recorded_at=recorded_at,
        baseline_journal_sequence=baseline_sequence,
        baseline_journal_sha256=baseline_sha256,
        baseline_current_sequence=current_sequence,
        baseline_current_sha256=current_sha256,
        target=target,
        candidate=candidate,
        persistent_volumes=volumes,
    )


def _reject_duplicate_keys(pairs: Sequence[tuple[str, object]]) -> dict[str, object]:
    document: dict[str, object] = {}
    for key, value in pairs:
        if key in document:
            raise DeploymentStateError("deployment state JSON contains duplicate keys")
        document[key] = value
    return document


def _canonical_json(document: Mapping[str, object]) -> bytes:
    return (
        json.dumps(
            document,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode()


def _decode_json(payload: bytes) -> dict[str, object]:
    if not payload or len(payload) > MAX_JSON_BYTES:
        raise DeploymentStateError("deployment state JSON size is invalid")
    try:
        decoded = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
        )
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise DeploymentStateError("deployment state JSON is invalid") from None
    if not isinstance(decoded, dict):
        raise DeploymentStateError("deployment state JSON must be an object")
    if _canonical_json(decoded) != payload:
        raise DeploymentStateError("deployment state JSON is not canonical")
    return decoded


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _validate_secure_metadata(
    metadata: os.stat_result,
    *,
    expected_mode: int,
    directory: bool,
    require_one_link: bool,
) -> None:
    expected_type = stat.S_ISDIR if directory else stat.S_ISREG
    if not expected_type(metadata.st_mode):
        raise DeploymentStateError("deployment state path has an unsafe file type")
    if metadata.st_uid != os.geteuid():
        raise DeploymentStateError("deployment state path has an unsafe owner")
    if stat.S_IMODE(metadata.st_mode) != expected_mode:
        raise DeploymentStateError("deployment state path has unsafe permissions")
    if require_one_link and metadata.st_nlink != 1:
        raise DeploymentStateError("deployment state file has unsafe links")


def _validate_secure_path(
    path: Path,
    *,
    expected_mode: int,
    directory: bool,
    require_one_link: bool = True,
) -> os.stat_result:
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        raise DeploymentStateError("deployment state path is missing") from None
    _validate_secure_metadata(
        metadata,
        expected_mode=expected_mode,
        directory=directory,
        require_one_link=require_one_link,
    )
    return metadata


def _ensure_secure_directory(path: Path) -> None:
    missing: list[Path] = []
    cursor = path
    while True:
        try:
            existing = cursor.lstat()
        except FileNotFoundError:
            missing.append(cursor)
            cursor = cursor.parent
            continue
        if not stat.S_ISDIR(existing.st_mode):
            raise DeploymentStateError(
                "deployment state ancestor has an unsafe file type"
            )
        break

    for directory in reversed(missing):
        with suppress(FileExistsError):
            directory.mkdir(mode=0o700)
        _validate_secure_path(
            directory,
            expected_mode=0o700,
            directory=True,
            require_one_link=False,
        )
        _sync_directory(directory.parent)

    _validate_secure_path(
        path,
        expected_mode=0o700,
        directory=True,
        require_one_link=False,
    )


def _write_all(descriptor: int, payload: bytes) -> None:
    remaining = memoryview(payload)
    while remaining:
        written = os.write(descriptor, remaining)
        if written <= 0:
            raise OSError("deployment state write made no progress")
        remaining = remaining[written:]


def _sync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _publish_private_file(
    path: Path,
    payload: bytes,
    *,
    replace: bool,
    prefix: str,
) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=prefix,
        suffix=".tmp",
        dir=path.parent,
    )
    temporary_path = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        _write_all(descriptor, payload)
        os.fsync(descriptor)
        descriptor_to_close = descriptor
        descriptor = -1
        os.close(descriptor_to_close)
        if replace:
            temporary_path.replace(path)
        else:
            os.link(temporary_path, path, follow_symlinks=False)
            temporary_path.unlink()
        _sync_directory(path.parent)
    finally:
        try:
            if descriptor >= 0:
                os.close(descriptor)
        finally:
            temporary_path.unlink(missing_ok=True)


def _metadata_identity(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_uid,
        metadata.st_nlink,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _read_private_file(
    path: Path,
    *,
    maximum_bytes: int,
    require_one_link: bool = True,
) -> bytes:
    before = _validate_secure_path(
        path,
        expected_mode=0o600,
        directory=False,
        require_one_link=require_one_link,
    )
    if before.st_size <= 0 or before.st_size > maximum_bytes:
        raise DeploymentStateError("private deployment file size is invalid")

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        opened = os.fstat(descriptor)
        _validate_secure_metadata(
            opened,
            expected_mode=0o600,
            directory=False,
            require_one_link=require_one_link,
        )
        identity = _metadata_identity(opened)
        if _metadata_identity(before) != identity:
            raise DeploymentStateError("private deployment file changed before read")
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(descriptor, min(64 * 1024, maximum_bytes + 1 - total))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if total > maximum_bytes:
                raise DeploymentStateError("private deployment file size is invalid")
        after = os.fstat(descriptor)
        _validate_secure_metadata(
            after,
            expected_mode=0o600,
            directory=False,
            require_one_link=require_one_link,
        )
        if _metadata_identity(after) != identity:
            raise DeploymentStateError("private deployment file changed while read")
        pathname_after = _validate_secure_path(
            path,
            expected_mode=0o600,
            directory=False,
            require_one_link=require_one_link,
        )
        if _metadata_identity(pathname_after) != identity:
            raise DeploymentStateError("private deployment file changed while read")
    finally:
        os.close(descriptor)

    payload = b"".join(chunks)
    if len(payload) != before.st_size or not payload:
        raise DeploymentStateError("private deployment file size is invalid")
    return payload


def _journal_from_document(
    document: Mapping[str, object],
    *,
    payload_sha256: str,
) -> JournalEntry:
    schema_version = document.get("schema_version")
    persistent_volumes: tuple[PersistentVolumeIdentity, ...] = ()
    v1_keys = {
        "schema_version",
        "sequence",
        "previous_sha256",
        "event",
        "state",
    }
    v2_keys = v1_keys | {
        "transaction_id",
        "intent_sha256",
        "persistent_volumes",
    }
    if schema_version == SCHEMA_VERSION and set(document) == v1_keys:
        if document.get("event") != "adopted":
            raise DeploymentStateError("deployment journal event is invalid")
        event = "adopted"
        transaction_id = None
        intent_sha256 = None
        state: DeploymentState | None = _state_from_document(document.get("state"))
    elif (
        schema_version == PROMOTION_SCHEMA_VERSION
        and set(document) == v2_keys
        and document.get("event") in _TERMINAL_EVENTS
    ):
        raw_event = document.get("event")
        event = cast(str, raw_event)
        transaction_id = _validated_string(
            document,
            "transaction_id",
            _DEPLOYMENT_ID,
        )
        intent_sha256 = _validated_string(
            document,
            "intent_sha256",
            _SHA256_HEX,
        )
        raw_state = document.get("state")
        state = None if raw_state is None else _any_state_from_document(raw_state)
        persistent_volumes = _volume_identities_from_document(
            document.get("persistent_volumes")
        )
    else:
        raise DeploymentStateError("deployment journal schema is invalid")
    sequence = document.get("sequence")
    if type(sequence) is not int or sequence <= 0:
        raise DeploymentStateError("deployment journal sequence is invalid")
    previous = document.get("previous_sha256")
    if previous is not None and (
        not isinstance(previous, str) or _SHA256_HEX.fullmatch(previous) is None
    ):
        raise DeploymentStateError("deployment journal predecessor is invalid")
    return JournalEntry(
        sequence=sequence,
        sha256=payload_sha256,
        previous_sha256=previous,
        event=event,
        state=state,
        transaction_id=transaction_id,
        intent_sha256=intent_sha256,
        persistent_volumes=persistent_volumes,
    )


def _current_from_document(document: Mapping[str, object]) -> CurrentDeployment:
    expected_keys = {
        "schema_version",
        "journal_sequence",
        "journal_sha256",
        "state",
    }
    if set(document) != expected_keys:
        raise DeploymentStateError("current deployment schema is invalid")
    state = _any_state_from_document(document.get("state"))
    expected_schema_version = (
        SCHEMA_VERSION if state.adopted else PROMOTION_SCHEMA_VERSION
    )
    if document.get("schema_version") != expected_schema_version:
        raise DeploymentStateError("current deployment schema is invalid")
    sequence = document.get("journal_sequence")
    if type(sequence) is not int or sequence <= 0:
        raise DeploymentStateError("current deployment sequence is invalid")
    journal_sha256 = document.get("journal_sha256")
    if (
        not isinstance(journal_sha256, str)
        or _SHA256_HEX.fullmatch(journal_sha256) is None
    ):
        raise DeploymentStateError("current deployment journal hash is invalid")
    return CurrentDeployment(
        journal_sequence=sequence,
        journal_sha256=journal_sha256,
        state=state,
    )


def _pending_from_document(
    document: object,
    *,
    intents: Mapping[str, tuple[str, PromotionIntent]],
) -> PendingPromotion:
    if not isinstance(document, dict):
        raise DeploymentStateError("pending promotion must be an object")
    expected_keys = {
        "schema_version",
        "transaction_id",
        "intent_path",
        "intent_sha256",
    }
    if (
        set(document) != expected_keys
        or document.get("schema_version") != PROMOTION_SCHEMA_VERSION
    ):
        raise DeploymentStateError("pending promotion schema is invalid")
    transaction_id = _validated_string(
        document,
        "transaction_id",
        _DEPLOYMENT_ID,
    )
    expected_path = f"transactions/{transaction_id}.json"
    intent_path = document.get("intent_path")
    if intent_path != expected_path:
        raise DeploymentStateError("pending promotion intent path is invalid")
    intent_sha256 = _validated_string(
        document,
        "intent_sha256",
        _SHA256_HEX,
    )
    selected = intents.get(transaction_id)
    if selected is None:
        raise DeploymentStateError("pending promotion intent is missing")
    actual_sha256, intent = selected
    if actual_sha256 != intent_sha256:
        raise DeploymentStateError("pending promotion intent hash is invalid")
    return PendingPromotion(
        transaction_id=transaction_id,
        intent_path=expected_path,
        intent_sha256=intent_sha256,
        intent=intent,
    )


class DeploymentStateStore:
    """Own the private durable state paths for one Compose deployment."""

    def __init__(self, path: Path) -> None:
        if not path.is_absolute() or path != path.resolve(strict=False):
            raise DeploymentStateError(
                "deployment state directory must be an absolute normalized path"
            )
        self.path = path
        self.journal_path = path / "journal"
        self.environments_path = path / "environments"
        self.transactions_path = path / "transactions"
        self.lock_path = path / "deploy.lock"
        self.current_path = path / "current.json"
        self.pending_path = path / "pending.json"

    def _prepare_layout(self) -> None:
        _ensure_secure_directory(self.path)
        _ensure_secure_directory(self.journal_path)
        _ensure_secure_directory(self.environments_path)

    def _open_lock(self) -> int:
        flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(self.lock_path, flags, 0o600)
        try:
            _validate_secure_metadata(
                os.fstat(descriptor),
                expected_mode=0o600,
                directory=False,
                require_one_link=True,
            )
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as error:
                if error.errno in {errno.EACCES, errno.EAGAIN}:
                    raise DeploymentLockBusyError(
                        "another deployment transaction holds the VM lock"
                    ) from None
                raise
        except BaseException:
            os.close(descriptor)
            raise
        return descriptor

    @contextmanager
    def transaction(self) -> Iterator[DeploymentStateTransaction]:
        """Hold the exclusive VM lock for one complete transaction."""
        self._prepare_layout()
        descriptor = self._open_lock()
        transaction = DeploymentStateTransaction(self)
        try:
            transaction._load_and_reconcile()
            yield transaction
        finally:
            transaction._closed = True
            try:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            finally:
                os.close(descriptor)

    def read_current(self) -> CurrentDeployment | None:
        """Read and reconcile the current state under the VM lock."""
        with self.transaction() as transaction:
            return transaction.current()

    def read_journal(self) -> tuple[JournalEntry, ...]:
        """Read the verified immutable journal under the VM lock."""
        with self.transaction() as transaction:
            return transaction.journal()


class DeploymentStateTransaction:
    """Lock-scoped operations over one verified deployment state store."""

    def __init__(self, store: DeploymentStateStore) -> None:
        self._store = store
        self._closed = False
        self._journal: tuple[JournalEntry, ...] = ()
        self._current: CurrentDeployment | None = None
        self._intents: dict[str, tuple[str, PromotionIntent]] = {}
        self._pending: PendingPromotion | None = None
        self._indeterminate_terminal: tuple[str, str] | None = None
        self._recovered_terminal: JournalEntry | None = None

    def _require_open(self) -> None:
        if self._closed:
            raise DeploymentStateError("deployment state transaction is closed")

    def _journal_files(self) -> list[tuple[int, Path]]:
        files: list[tuple[int, Path]] = []
        for path in self._store.journal_path.iterdir():
            match = _JOURNAL_NAME.fullmatch(path.name)
            if match is not None:
                files.append((int(match.group(1)), path))
                continue
            raise DeploymentStateError("deployment journal contains an unknown path")
        return sorted(files)

    def _intent_files(self) -> list[tuple[str, Path]]:
        try:
            _validate_secure_path(
                self._store.transactions_path,
                expected_mode=0o700,
                directory=True,
                require_one_link=False,
            )
        except DeploymentStateError:
            try:
                self._store.transactions_path.lstat()
            except FileNotFoundError:
                return []
            raise
        files: list[tuple[str, Path]] = []
        for path in self._store.transactions_path.iterdir():
            match = _INTENT_NAME.fullmatch(path.name)
            if match is None:
                raise DeploymentStateError(
                    "deployment transactions contain an unknown path"
                )
            files.append((match.group(1), path))
        return sorted(files)

    def _recover_temporary_files(
        self,
        directory: Path,
        *,
        temporary_pattern: re.Pattern[str],
        final_pattern: re.Pattern[str] | None = None,
        classify_terminal: bool = False,
    ) -> None:
        for temporary_path in sorted(directory.iterdir()):
            if temporary_pattern.fullmatch(temporary_path.name) is None:
                continue
            temporary_metadata = _validate_secure_path(
                temporary_path,
                expected_mode=0o600,
                directory=False,
                require_one_link=False,
            )
            terminal_entry: JournalEntry | None = None
            if temporary_metadata.st_nlink == 2 and final_pattern is not None:
                matching_finals: list[Path] = []
                for candidate in directory.iterdir():
                    if (
                        candidate == temporary_path
                        or final_pattern.fullmatch(candidate.name) is None
                    ):
                        continue
                    candidate_metadata = candidate.lstat()
                    if (
                        candidate_metadata.st_dev,
                        candidate_metadata.st_ino,
                    ) == (
                        temporary_metadata.st_dev,
                        temporary_metadata.st_ino,
                    ):
                        _validate_secure_metadata(
                            candidate_metadata,
                            expected_mode=0o600,
                            directory=False,
                            require_one_link=False,
                        )
                        matching_finals.append(candidate)
                if len(matching_finals) != 1:
                    raise DeploymentStateError(
                        "deployment state temporary hard link is ambiguous"
                    )
                if classify_terminal:
                    payload = _read_private_file(
                        matching_finals[0],
                        maximum_bytes=MAX_JSON_BYTES,
                        require_one_link=False,
                    )
                    candidate_entry = _journal_from_document(
                        _decode_json(payload),
                        payload_sha256=hashlib.sha256(payload).hexdigest(),
                    )
                    if candidate_entry.event in _TERMINAL_EVENTS:
                        terminal_entry = candidate_entry
            elif temporary_metadata.st_nlink != 1:
                raise DeploymentStateError(
                    "deployment state temporary file has unsafe links"
                )
            confirmed_metadata = _validate_secure_path(
                temporary_path,
                expected_mode=0o600,
                directory=False,
                require_one_link=False,
            )
            if _metadata_identity(confirmed_metadata) != _metadata_identity(
                temporary_metadata
            ):
                raise DeploymentStateError(
                    "deployment state temporary file changed during recovery"
                )
            try:
                temporary_path.unlink()
                _sync_directory(directory)
            except OSError as error:
                if terminal_entry is None:
                    raise
                raise DeploymentTerminalIndeterminateError(
                    terminal_entry.event,
                    cast(str, terminal_entry.transaction_id),
                ) from error

    def _load_intents(self) -> dict[str, tuple[str, PromotionIntent]]:
        intents: dict[str, tuple[str, PromotionIntent]] = {}
        for transaction_id, path in self._intent_files():
            payload = _read_private_file(path, maximum_bytes=MAX_JSON_BYTES)
            payload_sha256 = hashlib.sha256(payload).hexdigest()
            intent = _intent_from_document(_decode_json(payload))
            if intent.transaction_id != transaction_id:
                raise DeploymentStateError(
                    "promotion intent filename and identity disagree"
                )
            environment_payload = _read_private_file(
                self._store.path / intent.target.environment_snapshot,
                maximum_bytes=MAX_ENVIRONMENT_BYTES,
            )
            if (
                hashlib.sha256(environment_payload).hexdigest()
                != intent.target.environment_sha256
            ):
                raise DeploymentStateError(
                    "promotion target environment snapshot hash is invalid"
                )
            intents[transaction_id] = (payload_sha256, intent)
        return intents

    def _load_journal(
        self,
        intents: Mapping[str, tuple[str, PromotionIntent]],
    ) -> tuple[JournalEntry, ...]:
        entries: list[JournalEntry] = []
        previous_sha256: str | None = None
        established: CurrentDeployment | None = None
        completed_transactions: set[str] = set()
        for expected_sequence, (sequence, path) in enumerate(
            self._journal_files(),
            start=1,
        ):
            if sequence != expected_sequence:
                raise DeploymentStateError("deployment journal sequence has a gap")
            payload = _read_private_file(path, maximum_bytes=MAX_JSON_BYTES)
            payload_sha256 = hashlib.sha256(payload).hexdigest()
            entry = _journal_from_document(
                _decode_json(payload),
                payload_sha256=payload_sha256,
            )
            if entry.sequence != sequence:
                raise DeploymentStateError(
                    "deployment journal filename and sequence disagree"
                )
            if entry.previous_sha256 != previous_sha256:
                raise DeploymentStateError("deployment journal hash chain is invalid")
            if entry.event != "adopted":
                transaction_id = cast(str, entry.transaction_id)
                intent_sha256 = cast(str, entry.intent_sha256)
                if transaction_id in completed_transactions:
                    raise DeploymentStateError(
                        "deployment transaction has duplicate outcomes"
                    )
                selected = intents.get(transaction_id)
                if selected is None:
                    raise DeploymentStateError("deployment journal intent is missing")
                actual_intent_sha256, intent = selected
                if actual_intent_sha256 != intent_sha256:
                    raise DeploymentStateError(
                        "deployment journal intent hash is invalid"
                    )
                preceding_sequence = entries[-1].sequence if entries else None
                if (
                    intent.baseline_journal_sequence != preceding_sequence
                    or intent.baseline_journal_sha256 != previous_sha256
                ):
                    raise DeploymentStateError(
                        "promotion intent journal baseline is invalid"
                    )
                current_sequence = (
                    None if established is None else established.journal_sequence
                )
                current_sha256 = (
                    None if established is None else established.journal_sha256
                )
                if (
                    intent.baseline_current_sequence != current_sequence
                    or intent.baseline_current_sha256 != current_sha256
                ):
                    raise DeploymentStateError(
                        "promotion intent current baseline is invalid"
                    )
                if established is not None and (
                    intent.target.compose_project != established.state.compose_project
                    or intent.target.compose_service
                    != established.state.compose_service
                ):
                    raise DeploymentStateError(
                        "promotion target Compose identity changed"
                    )
                expected_state = (
                    intent.target
                    if entry.event == "promoted"
                    else None
                    if established is None
                    else established.state
                )
                if entry.state != expected_state:
                    raise DeploymentStateError(
                        "deployment journal outcome state is invalid"
                    )
                if (
                    entry.event != "promoted" or established is not None
                ) and entry.persistent_volumes != intent.persistent_volumes:
                    raise DeploymentStateError(
                        "deployment journal volume identity is invalid"
                    )
                completed_transactions.add(transaction_id)
            if entry.state is not None:
                environment_payload = _read_private_file(
                    self._store.path / entry.state.environment_snapshot,
                    maximum_bytes=MAX_ENVIRONMENT_BYTES,
                )
                if (
                    hashlib.sha256(environment_payload).hexdigest()
                    != entry.state.environment_sha256
                ):
                    raise DeploymentStateError(
                        "deployment environment snapshot hash is invalid"
                    )
            entries.append(entry)
            if entry.event in {"adopted", "promoted"}:
                established_state = cast(DeploymentState, entry.state)
                established = CurrentDeployment(
                    journal_sequence=entry.sequence,
                    journal_sha256=entry.sha256,
                    state=established_state,
                )
            previous_sha256 = payload_sha256
        return tuple(entries)

    def _desired_current(self) -> CurrentDeployment | None:
        current: CurrentDeployment | None = None
        for entry in self._journal:
            if entry.event in {"adopted", "promoted"}:
                established_state = cast(DeploymentState, entry.state)
                current = CurrentDeployment(
                    journal_sequence=entry.sequence,
                    journal_sha256=entry.sha256,
                    state=established_state,
                )
        return current

    def _read_current_file(self) -> CurrentDeployment | None:
        try:
            payload = _read_private_file(
                self._store.current_path,
                maximum_bytes=MAX_JSON_BYTES,
            )
        except DeploymentStateError:
            try:
                self._store.current_path.lstat()
            except FileNotFoundError:
                return None
            raise
        return _current_from_document(_decode_json(payload))

    def _publish_current(self, current: CurrentDeployment) -> None:
        _publish_private_file(
            self._store.current_path,
            _canonical_json(current.as_document()),
            replace=True,
            prefix=".deployment-current-",
        )

    def _reconcile_current(self) -> CurrentDeployment | None:
        desired = self._desired_current()
        current = self._read_current_file()
        if desired is None:
            if current is not None:
                raise DeploymentStateError(
                    "current deployment exists ahead of the journal"
                )
            return None
        if current is None:
            self._publish_current(desired)
            return desired
        if current.journal_sequence > desired.journal_sequence:
            raise DeploymentStateError("current deployment is ahead of the journal")
        indexed = self._journal[current.journal_sequence - 1]
        if indexed.event not in {"adopted", "promoted"} or indexed.state is None:
            raise DeploymentStateError("current deployment does not establish state")
        expected_existing = CurrentDeployment(
            journal_sequence=indexed.sequence,
            journal_sha256=indexed.sha256,
            state=indexed.state,
        )
        if current != expected_existing:
            raise DeploymentStateError("current deployment does not match the journal")
        if current != desired:
            self._publish_current(desired)
            return desired
        return current

    def _read_pending_file(self) -> PendingPromotion | None:
        try:
            payload = _read_private_file(
                self._store.pending_path,
                maximum_bytes=MAX_JSON_BYTES,
            )
        except DeploymentStateError:
            try:
                self._store.pending_path.lstat()
            except FileNotFoundError:
                return None
            raise
        return _pending_from_document(
            _decode_json(payload),
            intents=self._intents,
        )

    def _terminal_for_pending(
        self,
        pending: PendingPromotion,
    ) -> JournalEntry | None:
        matches = [
            entry
            for entry in self._journal
            if entry.transaction_id == pending.transaction_id
            and entry.intent_sha256 == pending.intent_sha256
        ]
        if len(matches) > 1:
            raise DeploymentStateError(
                "pending promotion has duplicate terminal outcomes"
            )
        return None if not matches else matches[0]

    def _validate_unresolved_pending(self, pending: PendingPromotion) -> None:
        tail_sequence = self._journal[-1].sequence if self._journal else None
        tail_sha256 = self._journal[-1].sha256 if self._journal else None
        if (
            pending.intent.baseline_journal_sequence != tail_sequence
            or pending.intent.baseline_journal_sha256 != tail_sha256
        ):
            raise DeploymentStateError("pending promotion journal baseline is stale")
        current_sequence = (
            None if self._current is None else self._current.journal_sequence
        )
        current_sha256 = None if self._current is None else self._current.journal_sha256
        if (
            pending.intent.baseline_current_sequence != current_sequence
            or pending.intent.baseline_current_sha256 != current_sha256
        ):
            raise DeploymentStateError("pending promotion current baseline is stale")
        if self._current is not None and (
            pending.intent.target.compose_project != self._current.state.compose_project
            or pending.intent.target.compose_service
            != self._current.state.compose_service
        ):
            raise DeploymentStateError("promotion target Compose identity changed")

    def _clear_pending_after_terminal(self, entry: JournalEntry) -> None:
        transaction_id = cast(str, entry.transaction_id)
        try:
            self._store.pending_path.unlink()
            _sync_directory(self._store.path)
        except OSError as error:
            raise DeploymentTerminalCommittedError(
                entry.event,
                transaction_id,
            ) from error
        self._pending = None

    def _load_and_reconcile(self) -> None:
        self._recover_temporary_files(
            self._store.environments_path,
            temporary_pattern=_ENVIRONMENT_TEMPORARY_NAME,
            final_pattern=_ENVIRONMENT_NAME,
        )
        if self._store.transactions_path.exists():
            _validate_secure_path(
                self._store.transactions_path,
                expected_mode=0o700,
                directory=True,
                require_one_link=False,
            )
            self._recover_temporary_files(
                self._store.transactions_path,
                temporary_pattern=_INTENT_TEMPORARY_NAME,
                final_pattern=_INTENT_NAME,
            )
        self._recover_temporary_files(
            self._store.journal_path,
            temporary_pattern=_JOURNAL_TEMPORARY_NAME,
            final_pattern=_JOURNAL_NAME,
            classify_terminal=True,
        )
        self._intents = self._load_intents()
        self._journal = self._load_journal(self._intents)

        try:
            self._store.pending_path.lstat()
        except FileNotFoundError:
            pending_marker = False
        else:
            pending_marker = True
        tail = None if not self._journal else self._journal[-1]
        committed_recovery = (
            tail
            if pending_marker and tail is not None and tail.event in _TERMINAL_EVENTS
            else None
        )
        try:
            self._recover_temporary_files(
                self._store.path,
                temporary_pattern=_CURRENT_TEMPORARY_NAME,
            )
            self._recover_temporary_files(
                self._store.path,
                temporary_pattern=_PENDING_TEMPORARY_NAME,
                final_pattern=_PENDING_NAME,
            )
            self._pending = self._read_pending_file()
        except (OSError, DeploymentStateError) as error:
            if committed_recovery is None:
                raise
            transaction_id = cast(str, committed_recovery.transaction_id)
            raise DeploymentTerminalCommittedError(
                committed_recovery.event,
                transaction_id,
            ) from error
        terminal = (
            None if self._pending is None else self._terminal_for_pending(self._pending)
        )
        if terminal is not None and terminal != tail:
            raise DeploymentStateError(
                "pending promotion terminal is not the journal tail"
            )
        try:
            self._current = self._reconcile_current()
        except (OSError, DeploymentStateError) as error:
            if terminal is None:
                raise
            transaction_id = cast(str, terminal.transaction_id)
            raise DeploymentTerminalCommittedError(
                terminal.event,
                transaction_id,
            ) from error
        if self._pending is None:
            return
        if terminal is not None:
            self._clear_pending_after_terminal(terminal)
            self._recovered_terminal = terminal
            return
        self._validate_unresolved_pending(self._pending)

    def current(self) -> CurrentDeployment | None:
        """Return the lock-scoped current deployment."""
        self._require_open()
        return self._current

    def journal(self) -> tuple[JournalEntry, ...]:
        """Return the verified lock-scoped journal."""
        self._require_open()
        return self._journal

    def pending(self) -> PendingPromotion | None:
        """Return the unresolved lock-scoped promotion intent, if any."""
        self._require_open()
        return self._pending

    def recovered_terminal(self) -> JournalEntry | None:
        """Return a terminal outcome reconciled while opening this transaction."""
        self._require_open()
        return self._recovered_terminal

    def _require_no_recovered_terminal(self) -> None:
        if self._recovered_terminal is None:
            return
        raise DeploymentTerminalCommittedError(
            self._recovered_terminal.event,
            cast(str, self._recovered_terminal.transaction_id),
        )

    def _tail_identity(self) -> tuple[int | None, str | None]:
        if not self._journal:
            return None, None
        tail = self._journal[-1]
        return tail.sequence, tail.sha256

    def _current_identity(self) -> tuple[int | None, str | None]:
        if self._current is None:
            return None, None
        return self._current.journal_sequence, self._current.journal_sha256

    def _recorded_state(self, state: DeploymentState) -> bool:
        if any(entry.state == state for entry in self._journal):
            return True
        return any(intent.target == state for _, intent in self._intents.values())

    def read_environment(self, state: DeploymentState) -> bytes:
        """Read exact private bytes belonging to one recorded deployment state."""
        self._require_open()
        if not self._recorded_state(state):
            raise DeploymentStateError("deployment environment state is not recorded")
        payload = _read_private_file(
            self._store.path / state.environment_snapshot,
            maximum_bytes=MAX_ENVIRONMENT_BYTES,
        )
        if hashlib.sha256(payload).hexdigest() != state.environment_sha256:
            raise DeploymentStateError(
                "deployment environment snapshot hash is invalid"
            )
        return payload

    def _environment_destination_parent(self, destination: Path) -> Path:
        if not destination.is_absolute() or destination != destination.resolve(
            strict=False
        ):
            raise DeploymentStateError(
                "deployment environment destination must be absolute and normalized"
            )
        try:
            parent = destination.parent.resolve(strict=True)
            parent_metadata = parent.stat()
        except OSError:
            raise DeploymentStateError(
                "deployment environment destination parent is unavailable"
            ) from None
        if (
            parent != destination.parent
            or not stat.S_ISDIR(parent_metadata.st_mode)
            or parent_metadata.st_uid != os.geteuid()
            or stat.S_IMODE(parent_metadata.st_mode) & 0o022
        ):
            raise DeploymentStateError(
                "deployment environment destination parent is unsafe"
            )
        return parent

    def install_environment(
        self,
        state: DeploymentState,
        destination: Path,
    ) -> None:
        """Atomically install exact recorded environment bytes at a private path."""
        self._require_open()
        self._require_no_recovered_terminal()
        self._environment_destination_parent(destination)
        try:
            destination.lstat()
        except FileNotFoundError:
            pass
        else:
            _validate_secure_path(
                destination,
                expected_mode=0o600,
                directory=False,
            )
        payload = self.read_environment(state)
        _publish_private_file(
            destination,
            payload,
            replace=True,
            prefix=".deployment-install-",
        )
        installed = _read_private_file(
            destination,
            maximum_bytes=MAX_ENVIRONMENT_BYTES,
        )
        if installed != payload:
            raise DeploymentStateError(
                "installed deployment environment does not match"
            )

    def verify_installed_environment(
        self,
        state: DeploymentState,
        destination: Path,
        *,
        allow_missing: bool = False,
    ) -> bool:
        """Prove exact installed private bytes without mutating the destination."""
        self._require_open()
        self._environment_destination_parent(destination)
        expected_payload = self.read_environment(state)
        try:
            destination.lstat()
        except FileNotFoundError:
            if allow_missing:
                return False
            raise DeploymentStateError(
                "installed deployment environment is missing"
            ) from None
        installed_payload = _read_private_file(
            destination,
            maximum_bytes=MAX_ENVIRONMENT_BYTES,
        )
        if (
            installed_payload != expected_payload
            or hashlib.sha256(installed_payload).hexdigest() != state.environment_sha256
        ):
            raise DeploymentStateError(
                "installed deployment environment does not match recorded state"
            )
        return True

    def remove_environment(
        self,
        state: DeploymentState,
        destination: Path,
    ) -> None:
        """Idempotently remove an exact recorded environment and sync its parent."""
        self._require_open()
        self._require_no_recovered_terminal()
        parent = self._environment_destination_parent(destination)
        expected_payload = self.read_environment(state)
        try:
            before = _validate_secure_path(
                destination,
                expected_mode=0o600,
                directory=False,
            )
        except DeploymentStateError:
            try:
                destination.lstat()
            except FileNotFoundError:
                return
            raise
        installed_payload = _read_private_file(
            destination,
            maximum_bytes=MAX_ENVIRONMENT_BYTES,
        )
        if (
            installed_payload != expected_payload
            or hashlib.sha256(installed_payload).hexdigest() != state.environment_sha256
        ):
            raise DeploymentStateError(
                "installed deployment environment does not match recorded state"
            )
        confirmed = _validate_secure_path(
            destination,
            expected_mode=0o600,
            directory=False,
        )
        if _metadata_identity(confirmed) != _metadata_identity(before):
            raise DeploymentStateError(
                "installed deployment environment changed before removal"
            )
        destination.unlink()
        _sync_directory(parent)

    def begin_promotion(
        self,
        *,
        compose_project: str,
        compose_service: str,
        source_revision: str,
        image_reference: str,
        image_id: str,
        oci_revision: str,
        environment_source: Path,
        candidate: CandidateReceipt,
        persistent_volumes: Sequence[PersistentVolumeIdentity],
        transaction_id: str | None = None,
        recorded_at: str | None = None,
    ) -> PendingPromotion:
        """Durably publish one immutable promotion intent before cutover."""
        self._require_open()
        self._require_no_recovered_terminal()
        if self._pending is not None:
            raise DeploymentStateError("a promotion transaction is already pending")

        selected_id = uuid.uuid4().hex if transaction_id is None else transaction_id
        selected_time = _now() if recorded_at is None else recorded_at
        snapshot_relative = f"environments/{selected_id}.env"
        provisional = {
            "schema_version": PROMOTION_SCHEMA_VERSION,
            "deployment_id": selected_id,
            "recorded_at": selected_time,
            "compose_project": compose_project,
            "compose_service": compose_service,
            "source_revision": source_revision,
            "image_reference": image_reference,
            "image_id": image_id,
            "oci_revision": oci_revision,
            "environment_snapshot": snapshot_relative,
            "environment_sha256": "0" * 64,
            "adopted": False,
        }
        validated_target = _target_state_from_document(provisional)
        validated_candidate = _candidate_from_document(candidate.as_document())
        volumes = _validated_volume_identities(persistent_volumes)

        baseline_sequence, baseline_sha256 = self._tail_identity()
        current_sequence, current_sha256 = self._current_identity()
        if (
            validated_candidate.baseline_journal_sequence != baseline_sequence
            or validated_candidate.baseline_journal_sha256 != baseline_sha256
        ):
            raise DeploymentStateError("candidate receipt baseline does not match")
        if (
            validated_candidate.compose_service != validated_target.compose_service
            or validated_candidate.compose_project == validated_target.compose_project
            or validated_candidate.image_reference != validated_target.image_reference
            or validated_candidate.image_id != validated_target.image_id
            or validated_candidate.oci_revision != validated_target.oci_revision
            or validated_target.source_revision != validated_target.oci_revision
        ):
            raise DeploymentStateError("candidate receipt target does not match")
        if self._current is not None and (
            self._current.state.compose_project != validated_target.compose_project
            or self._current.state.compose_service != validated_target.compose_service
        ):
            raise DeploymentStateError("promotion target Compose identity changed")

        environment_payload = _read_private_file(
            environment_source,
            maximum_bytes=MAX_ENVIRONMENT_BYTES,
        )
        environment_sha256 = hashlib.sha256(environment_payload).hexdigest()
        target = DeploymentState(
            deployment_id=validated_target.deployment_id,
            recorded_at=validated_target.recorded_at,
            compose_project=validated_target.compose_project,
            compose_service=validated_target.compose_service,
            source_revision=validated_target.source_revision,
            image_reference=validated_target.image_reference,
            image_id=validated_target.image_id,
            oci_revision=validated_target.oci_revision,
            environment_snapshot=validated_target.environment_snapshot,
            environment_sha256=environment_sha256,
            adopted=False,
        )
        intent = PromotionIntent(
            transaction_id=selected_id,
            recorded_at=selected_time,
            baseline_journal_sequence=baseline_sequence,
            baseline_journal_sha256=baseline_sha256,
            baseline_current_sequence=current_sequence,
            baseline_current_sha256=current_sha256,
            target=target,
            candidate=validated_candidate,
            persistent_volumes=volumes,
        )
        validated_intent = _intent_from_document(intent.as_document())
        intent_payload = _canonical_json(validated_intent.as_document())
        if len(intent_payload) > MAX_JSON_BYTES:
            raise DeploymentStateError("promotion intent exceeds the size limit")

        _ensure_secure_directory(self._store.transactions_path)
        snapshot_path = self._store.path / target.environment_snapshot
        intent_path = self._store.transactions_path / f"{selected_id}.json"
        if snapshot_path.exists() or intent_path.exists():
            raise DeploymentStateError("promotion transaction identity already exists")
        _publish_private_file(
            snapshot_path,
            environment_payload,
            replace=False,
            prefix=".deployment-environment-",
        )
        intent_sha256 = hashlib.sha256(intent_payload).hexdigest()
        _publish_private_file(
            intent_path,
            intent_payload,
            replace=False,
            prefix=".deployment-intent-",
        )
        pending = PendingPromotion(
            transaction_id=selected_id,
            intent_path=f"transactions/{selected_id}.json",
            intent_sha256=intent_sha256,
            intent=validated_intent,
        )
        _publish_private_file(
            self._store.pending_path,
            _canonical_json(pending.as_document()),
            replace=False,
            prefix=".deployment-pending-",
        )
        self._intents[selected_id] = (intent_sha256, validated_intent)
        self._pending = pending
        return pending

    def _selected_pending(
        self,
        transaction_id: str,
    ) -> PendingPromotion:
        self._require_no_recovered_terminal()
        if self._indeterminate_terminal is not None:
            event, indeterminate_id = self._indeterminate_terminal
            raise DeploymentTerminalIndeterminateError(
                event,
                indeterminate_id,
            )
        if _DEPLOYMENT_ID.fullmatch(transaction_id) is None:
            raise DeploymentStateError("promotion transaction identity is invalid")
        if self._pending is None:
            raise DeploymentStateError("no promotion transaction is pending")
        if self._pending.transaction_id != transaction_id:
            raise DeploymentStateError("pending promotion identity does not match")
        terminal = self._terminal_for_pending(self._pending)
        if terminal is not None:
            raise DeploymentTerminalCommittedError(
                terminal.event,
                transaction_id,
            )
        return self._pending

    def _publish_terminal(
        self,
        *,
        event: str,
        transaction_id: str,
        persistent_volumes: Sequence[PersistentVolumeIdentity],
    ) -> CurrentDeployment | None:
        if event not in _TERMINAL_EVENTS:
            raise DeploymentStateError("deployment journal event is invalid")
        pending = self._selected_pending(transaction_id)
        observed_volumes = _validated_volume_identities(persistent_volumes)
        if (
            event != "promoted" or self._current is not None
        ) and observed_volumes != pending.intent.persistent_volumes:
            raise DeploymentStateError(
                "persistent volume identity changed during promotion"
            )
        state = (
            pending.intent.target
            if event == "promoted"
            else None
            if self._current is None
            else self._current.state
        )
        sequence = len(self._journal) + 1
        previous_sha256 = None if not self._journal else self._journal[-1].sha256
        entry = JournalEntry(
            sequence=sequence,
            sha256="0" * 64,
            previous_sha256=previous_sha256,
            event=event,
            state=state,
            transaction_id=transaction_id,
            intent_sha256=pending.intent_sha256,
            persistent_volumes=observed_volumes,
        )
        payload = _canonical_json(entry.as_document())
        sha256 = hashlib.sha256(payload).hexdigest()
        committed_entry = JournalEntry(
            sequence=sequence,
            sha256=sha256,
            previous_sha256=previous_sha256,
            event=event,
            state=state,
            transaction_id=transaction_id,
            intent_sha256=pending.intent_sha256,
            persistent_volumes=observed_volumes,
        )
        journal_path = self._store.journal_path / f"{sequence:020d}.json"
        try:
            _publish_private_file(
                journal_path,
                payload,
                replace=False,
                prefix=".deployment-journal-",
            )
        except OSError as error:
            try:
                journal_path.lstat()
            except FileNotFoundError:
                raise error from None
            except OSError:
                pass
            self._indeterminate_terminal = (event, transaction_id)
            raise DeploymentTerminalIndeterminateError(
                event,
                transaction_id,
            ) from error

        self._journal = (*self._journal, committed_entry)
        if event == "promoted":
            current = CurrentDeployment(
                journal_sequence=sequence,
                journal_sha256=sha256,
                state=pending.intent.target,
            )
            self._current = current
            try:
                self._publish_current(current)
            except OSError as error:
                raise DeploymentTerminalCommittedError(
                    event,
                    transaction_id,
                ) from error
        self._clear_pending_after_terminal(committed_entry)
        return self._current

    def commit_promotion(
        self,
        transaction_id: str,
        *,
        persistent_volumes: Sequence[PersistentVolumeIdentity],
    ) -> CurrentDeployment:
        """Commit a verified target; terminal journal durability is authoritative."""
        self._require_open()
        current = self._publish_terminal(
            event="promoted",
            transaction_id=transaction_id,
            persistent_volumes=persistent_volumes,
        )
        return cast(CurrentDeployment, current)

    def record_rollback(
        self,
        transaction_id: str,
        *,
        persistent_volumes: Sequence[PersistentVolumeIdentity],
    ) -> CurrentDeployment | None:
        """Record independently verified restoration of the prior deployment."""
        self._require_open()
        return self._publish_terminal(
            event="rolled_back",
            transaction_id=transaction_id,
            persistent_volumes=persistent_volumes,
        )

    def record_abort(
        self,
        transaction_id: str,
        *,
        persistent_volumes: Sequence[PersistentVolumeIdentity],
    ) -> CurrentDeployment | None:
        """Record cancellation before production mutation."""
        self._require_open()
        return self._publish_terminal(
            event="aborted",
            transaction_id=transaction_id,
            persistent_volumes=persistent_volumes,
        )

    def adopt(
        self,
        *,
        compose_project: str,
        compose_service: str,
        source_revision: str,
        image_reference: str,
        image_id: str,
        oci_revision: str,
        environment_source: Path,
        deployment_id: str | None = None,
        recorded_at: str | None = None,
    ) -> CurrentDeployment:
        """Record one explicitly observed legacy deployment without overwriting."""
        self._require_open()
        self._require_no_recovered_terminal()
        if self._current is not None or self._journal or self._pending is not None:
            raise DeploymentStateError("deployment state has already been initialized")

        selected_id = uuid.uuid4().hex if deployment_id is None else deployment_id
        selected_time = _now() if recorded_at is None else recorded_at
        snapshot_relative = f"environments/{selected_id}.env"
        provisional = {
            "schema_version": SCHEMA_VERSION,
            "deployment_id": selected_id,
            "recorded_at": selected_time,
            "compose_project": compose_project,
            "compose_service": compose_service,
            "source_revision": source_revision,
            "image_reference": image_reference,
            "image_id": image_id,
            "oci_revision": oci_revision,
            "environment_snapshot": snapshot_relative,
            "environment_sha256": "0" * 64,
            "adopted": True,
        }
        validated = _state_from_document(provisional)
        environment_payload = _read_private_file(
            environment_source,
            maximum_bytes=MAX_ENVIRONMENT_BYTES,
        )
        environment_sha256 = hashlib.sha256(environment_payload).hexdigest()
        state = DeploymentState(
            deployment_id=validated.deployment_id,
            recorded_at=validated.recorded_at,
            compose_project=validated.compose_project,
            compose_service=validated.compose_service,
            source_revision=validated.source_revision,
            image_reference=validated.image_reference,
            image_id=validated.image_id,
            oci_revision=validated.oci_revision,
            environment_snapshot=validated.environment_snapshot,
            environment_sha256=environment_sha256,
            adopted=True,
        )
        snapshot_path = self._store.path / state.environment_snapshot
        _publish_private_file(
            snapshot_path,
            environment_payload,
            replace=False,
            prefix=".deployment-environment-",
        )

        journal_document = {
            "schema_version": SCHEMA_VERSION,
            "sequence": 1,
            "previous_sha256": None,
            "event": "adopted",
            "state": state.as_document(),
        }
        journal_payload = _canonical_json(journal_document)
        journal_sha256 = hashlib.sha256(journal_payload).hexdigest()
        journal_path = self._store.journal_path / "00000000000000000001.json"
        _publish_private_file(
            journal_path,
            journal_payload,
            replace=False,
            prefix=".deployment-journal-",
        )
        entry = JournalEntry(
            sequence=1,
            sha256=journal_sha256,
            previous_sha256=None,
            event="adopted",
            state=state,
        )
        current = CurrentDeployment(
            journal_sequence=1,
            journal_sha256=journal_sha256,
            state=state,
        )
        self._journal = (entry,)
        self._publish_current(current)
        self._current = current
        return current
