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
from pathlib import Path
from typing import Final

SCHEMA_VERSION: Final = 1
MAX_ENVIRONMENT_BYTES: Final = 1024 * 1024
MAX_JSON_BYTES: Final = 64 * 1024

_COMPOSE_NAME = re.compile(r"[a-z0-9][a-z0-9_-]{0,62}\Z")
_DEPLOYMENT_ID = re.compile(r"[a-z0-9][a-z0-9-]{0,63}\Z")
_REVISION = re.compile(r"[0-9a-f]{40}\Z")
_SHA256_HEX = re.compile(r"[0-9a-f]{64}\Z")
_IMAGE_ID = re.compile(r"sha256:[0-9a-f]{64}\Z")
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
_JOURNAL_NAME = re.compile(r"([0-9]{20})\.json\Z")
_ENVIRONMENT_NAME = re.compile(r"[a-z0-9][a-z0-9-]{0,63}\.env\Z")
_CURRENT_TEMPORARY_NAME = re.compile(r"\.deployment-current-[a-z0-9._-]+\.tmp\Z")
_JOURNAL_TEMPORARY_NAME = re.compile(r"\.deployment-journal-[a-z0-9._-]+\.tmp\Z")
_ENVIRONMENT_TEMPORARY_NAME = re.compile(
    r"\.deployment-environment-[a-z0-9._-]+\.tmp\Z"
)


class DeploymentStateError(RuntimeError):
    """Report a deterministic deployment-state contract failure."""


class DeploymentLockBusyError(DeploymentStateError):
    """Report that another process owns the VM deployment transaction."""


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
            "schema_version": SCHEMA_VERSION,
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
    state: DeploymentState

    def as_document(self) -> dict[str, object]:
        """Return the exact persisted journal schema, excluding its outer hash."""
        return {
            "schema_version": SCHEMA_VERSION,
            "sequence": self.sequence,
            "previous_sha256": self.previous_sha256,
            "event": self.event,
            "state": self.state.as_document(),
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
            "schema_version": SCHEMA_VERSION,
            "journal_sequence": self.journal_sequence,
            "journal_sha256": self.journal_sha256,
            "state": self.state.as_document(),
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


def _validate_timestamp(value: str) -> str:
    if _RECORDED_AT.fullmatch(value) is None:
        raise DeploymentStateError("deployment state field is invalid: recorded_at")
    try:
        datetime.strptime(value, "%Y-%m-%dT%H:%M:%S.%fZ").replace(tzinfo=UTC)
    except ValueError:
        raise DeploymentStateError(
            "deployment state field is invalid: recorded_at"
        ) from None
    return value


def _validate_snapshot_path(value: object, deployment_id: str) -> str:
    expected = f"environments/{deployment_id}.env"
    if not isinstance(value, str) or value != expected:
        raise DeploymentStateError(
            "deployment state field is invalid: environment_snapshot"
        )
    return value


def _state_from_document(document: object) -> DeploymentState:
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
    if (
        set(document) != expected_keys
        or document.get("schema_version") != SCHEMA_VERSION
    ):
        raise DeploymentStateError("deployment state schema is invalid")

    deployment_id = _validated_string(document, "deployment_id", _DEPLOYMENT_ID)
    adopted = document.get("adopted")
    if adopted is not True:
        raise DeploymentStateError("deployment state field is invalid: adopted")
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
        adopted=True,
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


def _read_private_file(path: Path, *, maximum_bytes: int) -> bytes:
    before = _validate_secure_path(
        path,
        expected_mode=0o600,
        directory=False,
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
            require_one_link=True,
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
            require_one_link=True,
        )
        if _metadata_identity(after) != identity:
            raise DeploymentStateError("private deployment file changed while read")
        pathname_after = _validate_secure_path(
            path,
            expected_mode=0o600,
            directory=False,
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
    expected_keys = {
        "schema_version",
        "sequence",
        "previous_sha256",
        "event",
        "state",
    }
    if (
        set(document) != expected_keys
        or document.get("schema_version") != SCHEMA_VERSION
    ):
        raise DeploymentStateError("deployment journal schema is invalid")
    sequence = document.get("sequence")
    if type(sequence) is not int or sequence <= 0:
        raise DeploymentStateError("deployment journal sequence is invalid")
    previous = document.get("previous_sha256")
    if previous is not None and (
        not isinstance(previous, str) or _SHA256_HEX.fullmatch(previous) is None
    ):
        raise DeploymentStateError("deployment journal predecessor is invalid")
    event = document.get("event")
    if event != "adopted":
        raise DeploymentStateError("deployment journal event is invalid")
    return JournalEntry(
        sequence=sequence,
        sha256=payload_sha256,
        previous_sha256=previous,
        event="adopted",
        state=_state_from_document(document.get("state")),
    )


def _current_from_document(document: Mapping[str, object]) -> CurrentDeployment:
    expected_keys = {
        "schema_version",
        "journal_sequence",
        "journal_sha256",
        "state",
    }
    if (
        set(document) != expected_keys
        or document.get("schema_version") != SCHEMA_VERSION
    ):
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
        state=_state_from_document(document.get("state")),
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
        self.lock_path = path / "deploy.lock"
        self.current_path = path / "current.json"

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

    def _recover_temporary_files(
        self,
        directory: Path,
        *,
        temporary_pattern: re.Pattern[str],
        final_pattern: re.Pattern[str] | None = None,
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
            temporary_path.unlink()
            _sync_directory(directory)

    def _load_journal(self) -> tuple[JournalEntry, ...]:
        entries: list[JournalEntry] = []
        previous_sha256: str | None = None
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
            previous_sha256 = payload_sha256
        return tuple(entries)

    def _desired_current(self) -> CurrentDeployment | None:
        if not self._journal:
            return None
        latest = self._journal[-1]
        return CurrentDeployment(
            journal_sequence=latest.sequence,
            journal_sha256=latest.sha256,
            state=latest.state,
        )

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

    def _load_and_reconcile(self) -> None:
        self._recover_temporary_files(
            self._store.path,
            temporary_pattern=_CURRENT_TEMPORARY_NAME,
        )
        self._recover_temporary_files(
            self._store.environments_path,
            temporary_pattern=_ENVIRONMENT_TEMPORARY_NAME,
            final_pattern=_ENVIRONMENT_NAME,
        )
        self._recover_temporary_files(
            self._store.journal_path,
            temporary_pattern=_JOURNAL_TEMPORARY_NAME,
            final_pattern=_JOURNAL_NAME,
        )
        self._journal = self._load_journal()
        self._current = self._reconcile_current()

    def current(self) -> CurrentDeployment | None:
        """Return the lock-scoped current deployment."""
        self._require_open()
        return self._current

    def journal(self) -> tuple[JournalEntry, ...]:
        """Return the verified lock-scoped journal."""
        self._require_open()
        return self._journal

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
        if self._current is not None or self._journal:
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
