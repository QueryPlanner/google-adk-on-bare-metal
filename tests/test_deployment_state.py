"""Durable VM deployment state and lock contract tests."""

from __future__ import annotations

import errno
import fcntl
import hashlib
import json
import os
import stat
import subprocess
import sys
from pathlib import Path
from unittest.mock import create_autospec, patch

import pytest

from agent import deployment_state as state_module
from agent.deployment_state import (
    CurrentDeployment,
    DeploymentLockBusyError,
    DeploymentStateError,
    DeploymentStateStore,
)

REVISION = "a" * 40
OCI_REVISION = "b" * 40
IMAGE_ID = f"sha256:{'c' * 64}"
IMAGE_REFERENCE = f"ghcr.io/queryplanner/agent@sha256:{'d' * 64}"
DEPLOYMENT_ID = "adopt-0123456789abcdef"
RECORDED_AT = "2026-07-28T12:34:56.123456Z"
ENVIRONMENT_BYTES = b'API_KEY="secret-$-canary"\nEMPTY=""\n'


def _write_private(path: Path, payload: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    path.chmod(0o600)
    return path


def _environment(tmp_path: Path) -> Path:
    return _write_private(tmp_path / "legacy.env", ENVIRONMENT_BYTES)


def _adopt(
    store: DeploymentStateStore,
    environment: Path,
    *,
    deployment_id: str | None = DEPLOYMENT_ID,
    recorded_at: str | None = RECORDED_AT,
) -> CurrentDeployment:
    with store.transaction() as transaction:
        return transaction.adopt(
            compose_project="adk-template",
            compose_service="agent",
            source_revision=REVISION,
            image_reference=IMAGE_REFERENCE,
            image_id=IMAGE_ID,
            oci_revision=OCI_REVISION,
            environment_source=environment,
            deployment_id=deployment_id,
            recorded_at=recorded_at,
        )


def _journal_path(store: DeploymentStateStore, sequence: int = 1) -> Path:
    return store.journal_path / f"{sequence:020d}.json"


def _read_document(path: Path) -> dict[str, object]:
    decoded = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(decoded, dict)
    return decoded


def _write_document(path: Path, document: dict[str, object]) -> None:
    _write_private(path, state_module._canonical_json(document))


def _append_second_record(store: DeploymentStateStore) -> dict[str, object]:
    first_payload = _journal_path(store).read_bytes()
    first = _read_document(_journal_path(store))
    second = {
        "schema_version": 1,
        "sequence": 2,
        "previous_sha256": hashlib.sha256(first_payload).hexdigest(),
        "event": "adopted",
        "state": first["state"],
    }
    _write_document(_journal_path(store, 2), second)
    return second


def test_adoption_round_trips_private_crash_consistent_state(
    tmp_path: Path,
) -> None:
    """Persist exact recovery bytes while journals remain secret-free."""
    store = DeploymentStateStore(tmp_path / "state")
    environment = _environment(tmp_path)

    with store.transaction() as transaction:
        assert transaction.current() is None
        assert transaction.journal() == ()
        current = transaction.adopt(
            compose_project="adk-template",
            compose_service="agent",
            source_revision=REVISION,
            image_reference=IMAGE_REFERENCE,
            image_id=IMAGE_ID,
            oci_revision=OCI_REVISION,
            environment_source=environment,
            deployment_id=DEPLOYMENT_ID,
            recorded_at=RECORDED_AT,
        )
        assert transaction.current() == current
        assert transaction.journal()[0].sha256 == current.journal_sha256
        assert transaction.journal()[0].state == current.state

    snapshot = store.path / current.state.environment_snapshot
    assert snapshot.read_bytes() == ENVIRONMENT_BYTES
    assert (
        current.state.environment_sha256
        == hashlib.sha256(ENVIRONMENT_BYTES).hexdigest()
    )
    assert store.read_current() == current
    assert store.read_journal()[0].state == current.state

    for directory in (store.path, store.journal_path, store.environments_path):
        assert stat.S_IMODE(directory.stat().st_mode) == 0o700
    for private_file in (
        store.lock_path,
        store.current_path,
        _journal_path(store),
        snapshot,
    ):
        assert stat.S_IMODE(private_file.stat().st_mode) == 0o600
        assert private_file.stat().st_nlink == 1

    public_bytes = store.current_path.read_bytes() + _journal_path(store).read_bytes()
    assert b"secret-$-canary" not in public_bytes
    assert str(environment).encode() not in public_bytes
    assert current.state.environment_snapshot.encode() in public_bytes


def test_default_identity_and_timestamp_are_canonical(tmp_path: Path) -> None:
    """Generate safe defaults without weakening deterministic caller overrides."""
    current = _adopt(
        DeploymentStateStore(tmp_path / "state"),
        _environment(tmp_path),
        deployment_id=None,
        recorded_at=None,
    )

    assert len(current.state.deployment_id) == 32
    assert current.state.deployment_id.isalnum()
    assert current.state.recorded_at.endswith("Z")
    assert len(current.state.recorded_at) == 27


def test_transaction_object_fails_after_lock_release(tmp_path: Path) -> None:
    """Prevent lock-scoped state from being reused after its descriptor closes."""
    store = DeploymentStateStore(tmp_path / "state")
    with store.transaction() as transaction:
        assert transaction.current() is None

    with pytest.raises(DeploymentStateError, match="transaction is closed"):
        transaction.current()
    with pytest.raises(DeploymentStateError, match="transaction is closed"):
        transaction.journal()
    with pytest.raises(DeploymentStateError, match="transaction is closed"):
        transaction.adopt(
            compose_project="adk-template",
            compose_service="agent",
            source_revision=REVISION,
            image_reference=IMAGE_REFERENCE,
            image_id=IMAGE_ID,
            oci_revision=OCI_REVISION,
            environment_source=_environment(tmp_path),
        )


def test_adoption_never_overwrites_initialized_state(tmp_path: Path) -> None:
    """Require promotion logic instead of silently replacing the recovery base."""
    store = DeploymentStateStore(tmp_path / "state")
    environment = _environment(tmp_path)
    original = _adopt(store, environment)

    with (
        store.transaction() as transaction,
        pytest.raises(DeploymentStateError, match="already been initialized"),
    ):
        transaction.adopt(
            compose_project="adk-template",
            compose_service="agent",
            source_revision="e" * 40,
            image_reference=IMAGE_REFERENCE,
            image_id=IMAGE_ID,
            oci_revision=OCI_REVISION,
            environment_source=environment,
        )

    assert store.read_current() == original
    assert len(store.read_journal()) == 1


@pytest.mark.parametrize(
    ("override", "value", "field"),
    [
        ("compose_project", "Invalid", "compose_project"),
        ("compose_service", "-agent", "compose_service"),
        ("source_revision", "A" * 40, "source_revision"),
        ("image_reference", "agent:latest", "image_reference"),
        (
            "image_reference",
            f"ghcr.io/queryplanner/agent:latest@sha256:{'d' * 64}",
            "image_reference",
        ),
        (
            "image_reference",
            f"ghcr.io/queryplanner//agent@sha256:{'d' * 64}",
            "image_reference",
        ),
        ("image_id", "sha256:short", "image_id"),
        ("oci_revision", "g" * 40, "oci_revision"),
        ("deployment_id", "../escape", "deployment_id"),
        ("recorded_at", "2026-02-30T12:34:56.123456Z", "recorded_at"),
    ],
)
def test_adoption_rejects_invalid_metadata_before_snapshot(
    tmp_path: Path,
    override: str,
    value: str,
    field: str,
) -> None:
    """Validate all public metadata before copying private environment bytes."""
    store = DeploymentStateStore(tmp_path / override / "state")
    environment = _environment(tmp_path)
    arguments = {
        "compose_project": "adk-template",
        "compose_service": "agent",
        "source_revision": REVISION,
        "image_reference": IMAGE_REFERENCE,
        "image_id": IMAGE_ID,
        "oci_revision": OCI_REVISION,
        "environment_source": environment,
        "deployment_id": DEPLOYMENT_ID,
        "recorded_at": RECORDED_AT,
    }
    arguments[override] = value

    with (
        store.transaction() as transaction,
        pytest.raises(DeploymentStateError, match=field),
    ):
        transaction.adopt(**arguments)  # type: ignore[arg-type]

    assert list(store.environments_path.iterdir()) == []
    assert list(store.journal_path.iterdir()) == []
    assert not store.current_path.exists()


@pytest.mark.parametrize(
    "path_factory",
    [
        lambda root: Path("relative-state"),
        lambda root: root / "nested" / ".." / "state",
    ],
)
def test_store_requires_absolute_normalized_path(
    tmp_path: Path,
    path_factory: object,
) -> None:
    """Keep transaction identity independent of caller working directory."""
    selected = path_factory(tmp_path)  # type: ignore[operator]
    with pytest.raises(DeploymentStateError, match="absolute normalized"):
        DeploymentStateStore(selected)


def test_nested_state_parent_is_created_privately(tmp_path: Path) -> None:
    """Create every missing state directory with owner-only permissions."""
    state_path = tmp_path / "missing" / "nested" / "state"
    store = DeploymentStateStore(state_path)

    with store.transaction() as transaction:
        assert transaction.current() is None

    for path in (
        tmp_path / "missing",
        tmp_path / "missing" / "nested",
        state_path,
        store.journal_path,
        store.environments_path,
    ):
        assert stat.S_IMODE(path.stat().st_mode) == 0o700


def test_new_state_hierarchy_syncs_every_parent_directory(tmp_path: Path) -> None:
    """Make each newly published directory entry durable before continuing."""
    state_path = tmp_path / "missing" / "nested" / "state"
    real_fsync = os.fsync
    directory_syncs = 0

    def record_fsync(descriptor: int) -> None:
        nonlocal directory_syncs
        if stat.S_ISDIR(os.fstat(descriptor).st_mode):
            directory_syncs += 1
        real_fsync(descriptor)

    fsync_spy = create_autospec(
        os.fsync,
        spec_set=True,
        side_effect=record_fsync,
    )
    with (
        patch("agent.deployment_state.os.fsync", new=fsync_spy),
        DeploymentStateStore(state_path).transaction(),
    ):
        pass

    assert directory_syncs == 5


@pytest.mark.parametrize("unsafe_kind", ["file", "symlink", "mode", "owner"])
def test_state_root_rejects_unsafe_metadata(
    tmp_path: Path,
    unsafe_kind: str,
) -> None:
    """Fail before acquiring a lock through an unsafe state boundary."""
    state_path = tmp_path / "state"
    owner_patch = None
    if unsafe_kind == "file":
        state_path.write_text("not-a-directory", encoding="utf-8")
    elif unsafe_kind == "symlink":
        target = tmp_path / "target"
        target.mkdir(mode=0o700)
        state_path.symlink_to(target, target_is_directory=True)
    else:
        state_path.mkdir(mode=0o700)
        if unsafe_kind == "mode":
            state_path.chmod(0o755)
        else:
            owner_patch = patch(
                "agent.deployment_state.os.geteuid",
                new=create_autospec(os.geteuid, spec_set=True, return_value=1),
            )

    context = owner_patch if owner_patch is not None else patch.dict({}, {})
    with (
        context,
        pytest.raises(DeploymentStateError, match="unsafe|absolute normalized"),
        DeploymentStateStore(state_path).transaction(),
    ):
        pytest.fail("unsafe state root acquired a deployment lock")


@pytest.mark.parametrize("unsafe_kind", ["mode", "hardlink", "symlink"])
def test_lock_file_rejects_unsafe_metadata(
    tmp_path: Path,
    unsafe_kind: str,
) -> None:
    """Reject lock paths that can be observed or redirected by another user."""
    store = DeploymentStateStore(tmp_path / "state")
    with store.transaction():
        pass

    if unsafe_kind == "mode":
        store.lock_path.chmod(0o644)
    elif unsafe_kind == "hardlink":
        os.link(store.lock_path, tmp_path / "second-lock-name")
    else:
        store.lock_path.unlink()
        target = _write_private(tmp_path / "target-lock", b"lock")
        store.lock_path.symlink_to(target)

    with pytest.raises((DeploymentStateError, OSError)), store.transaction():
        pytest.fail("unsafe lock file acquired")


def test_lock_contention_is_nonblocking_and_cross_process(tmp_path: Path) -> None:
    """Prove a separate process cannot enter until the descriptor is released."""
    store = DeploymentStateStore(tmp_path / "state")
    child = (
        "from pathlib import Path\n"
        "from agent.deployment_state import "
        "DeploymentLockBusyError,DeploymentStateStore\n"
        "store=DeploymentStateStore(Path(__import__('sys').argv[1]))\n"
        "try:\n"
        "  with store.transaction(): print('acquired')\n"
        "except DeploymentLockBusyError:\n"
        "  print('busy')\n"
    )

    with store.transaction():
        blocked = subprocess.run(  # noqa: S603 - current interpreter, fixed code
            [sys.executable, "-c", child, str(store.path)],
            text=True,
            capture_output=True,
            check=True,
        )
        assert blocked.stdout.strip() == "busy"

    acquired = subprocess.run(  # noqa: S603 - current interpreter, fixed code
        [sys.executable, "-c", child, str(store.path)],
        text=True,
        capture_output=True,
        check=True,
    )
    assert acquired.stdout.strip() == "acquired"


@pytest.mark.parametrize("error_number", [errno.EAGAIN, errno.EACCES])
def test_lock_busy_error_is_secret_free(
    tmp_path: Path,
    error_number: int,
) -> None:
    """Map both portable nonblocking lock errors to one stable exception."""
    lock_failure = create_autospec(
        fcntl.flock,
        spec_set=True,
        side_effect=OSError(error_number, "secret-kernel-detail"),
    )

    with (
        patch("agent.deployment_state.fcntl.flock", new=lock_failure),
        pytest.raises(DeploymentLockBusyError, match="another deployment"),
        DeploymentStateStore(tmp_path / "state").transaction(),
    ):
        pytest.fail("busy lock unexpectedly acquired")


def test_unexpected_lock_error_propagates_and_closes_descriptor(
    tmp_path: Path,
) -> None:
    """Do not misclassify an actual filesystem or kernel failure as contention."""
    lock_failure = create_autospec(
        fcntl.flock,
        spec_set=True,
        side_effect=OSError(errno.EIO, "synthetic lock I/O failure"),
    )

    with (
        patch("agent.deployment_state.fcntl.flock", new=lock_failure),
        pytest.raises(OSError, match="synthetic lock"),
        DeploymentStateStore(tmp_path / "state").transaction(),
    ):
        pytest.fail("failed lock unexpectedly acquired")


def test_missing_current_is_recovered_from_durable_journal(tmp_path: Path) -> None:
    """Treat the journal as authoritative after a crash before pointer publish."""
    store = DeploymentStateStore(tmp_path / "state")
    adopted = _adopt(store, _environment(tmp_path))
    store.current_path.unlink()

    recovered = store.read_current()

    assert recovered == adopted
    assert _read_document(store.current_path) == adopted.as_document()


def test_stale_current_is_advanced_to_latest_valid_journal(tmp_path: Path) -> None:
    """Recover a committed journal record that became durable before current."""
    store = DeploymentStateStore(tmp_path / "state")
    _adopt(store, _environment(tmp_path))
    second = _append_second_record(store)

    recovered = store.read_current()

    assert recovered is not None
    assert recovered.journal_sequence == 2
    assert recovered.state.as_document() == second["state"]
    assert len(store.read_journal()) == 2


def test_orphaned_recognized_journal_temp_is_removed(tmp_path: Path) -> None:
    """Remove a private unpublished temp file before journal validation."""
    store = DeploymentStateStore(tmp_path / "state")
    adopted = _adopt(store, _environment(tmp_path))
    temporary_path = _write_private(
        store.journal_path / ".deployment-journal-orphan.tmp",
        b"partial",
    )

    assert store.read_current() == adopted
    assert len(store.read_journal()) == 1
    assert not temporary_path.exists()


@pytest.mark.parametrize("record_kind", ["journal", "environment"])
def test_published_hard_link_crash_residue_is_recovered(
    tmp_path: Path,
    record_kind: str,
) -> None:
    """Remove the stale temp name left between link and directory fsync."""
    store = DeploymentStateStore(tmp_path / "state")
    adopted = _adopt(store, _environment(tmp_path))
    if record_kind == "journal":
        final_path = _journal_path(store)
        temporary_path = store.journal_path / ".deployment-journal-crash.tmp"
    else:
        final_path = store.path / adopted.state.environment_snapshot
        temporary_path = store.environments_path / ".deployment-environment-crash.tmp"
    os.link(final_path, temporary_path)
    assert final_path.stat().st_nlink == 2

    assert store.read_current() == adopted
    assert not temporary_path.exists()
    assert final_path.stat().st_nlink == 1


def test_orphaned_current_temp_is_removed(tmp_path: Path) -> None:
    """Discard a pre-replace current temp without trusting its contents."""
    store = DeploymentStateStore(tmp_path / "state")
    adopted = _adopt(store, _environment(tmp_path))
    temporary_path = _write_private(
        store.path / ".deployment-current-orphan.tmp",
        b"partial",
    )

    assert store.read_current() == adopted
    assert not temporary_path.exists()


def test_current_temp_hard_link_is_rejected(tmp_path: Path) -> None:
    """Do not unlink a temp hard link without an exact published counterpart."""
    store = DeploymentStateStore(tmp_path / "state")
    _adopt(store, _environment(tmp_path))
    target = _write_private(tmp_path / "current-temp-target", b"private")
    temporary_path = store.path / ".deployment-current-linked.tmp"
    os.link(target, temporary_path)

    with pytest.raises(DeploymentStateError, match="unsafe links"):
        store.read_current()


def test_temporary_file_change_during_recovery_is_rejected(
    tmp_path: Path,
) -> None:
    """Bind cleanup to the exact private inode metadata that was validated."""
    store = DeploymentStateStore(tmp_path / "state")
    _adopt(store, _environment(tmp_path))
    temporary_path = _write_private(
        store.path / ".deployment-current-changing.tmp",
        b"partial",
    )
    real_lstat = Path.lstat
    temporary_lstats = 0

    def change_after_first_lstat(selected: Path) -> os.stat_result:
        nonlocal temporary_lstats
        metadata = real_lstat(selected)
        if selected == temporary_path:
            temporary_lstats += 1
            if temporary_lstats == 1:
                temporary_path.write_bytes(b"changed-after-validation")
        return metadata

    lstat_boundary = create_autospec(
        Path.lstat,
        spec_set=True,
        side_effect=change_after_first_lstat,
    )
    with (
        patch("pathlib.Path.lstat", new=lstat_boundary),
        pytest.raises(DeploymentStateError, match="changed during recovery"),
    ):
        store.read_current()
    assert temporary_lstats == 2


@pytest.mark.parametrize(
    "unsafe_kind",
    ["directory", "symlink", "mode", "unrelated-hardlink"],
)
def test_recognized_journal_temp_rejects_unsafe_metadata(
    tmp_path: Path,
    unsafe_kind: str,
) -> None:
    """Never ignore a temp-shaped path with untrusted metadata or identity."""
    store = DeploymentStateStore(tmp_path / "state")
    _adopt(store, _environment(tmp_path))
    temporary_path = store.journal_path / ".deployment-journal-unsafe.tmp"
    if unsafe_kind == "directory":
        temporary_path.mkdir(mode=0o700)
    elif unsafe_kind == "symlink":
        target = _write_private(tmp_path / "temporary-target", b"private")
        temporary_path.symlink_to(target)
    elif unsafe_kind == "mode":
        _write_private(temporary_path, b"private").chmod(0o644)
    else:
        target = _write_private(tmp_path / "unrelated-target", b"private")
        os.link(target, temporary_path)

    with pytest.raises(DeploymentStateError, match="unsafe|ambiguous"):
        store.read_current()


def test_unknown_journal_path_fails_closed(tmp_path: Path) -> None:
    """Do not silently ignore operator files or attacker-controlled records."""
    store = DeploymentStateStore(tmp_path / "state")
    _adopt(store, _environment(tmp_path))
    _write_private(store.journal_path / "notes.txt", b"unknown")

    with pytest.raises(DeploymentStateError, match="unknown path"):
        store.read_journal()


def test_journal_gap_fails_closed(tmp_path: Path) -> None:
    """Reject deletion or reordering in the immutable history."""
    store = DeploymentStateStore(tmp_path / "state")
    _adopt(store, _environment(tmp_path))
    _journal_path(store).rename(_journal_path(store, 2))

    with pytest.raises(DeploymentStateError, match="sequence has a gap"):
        store.read_journal()


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda doc: doc.update(sequence=2), "filename and sequence"),
        (lambda doc: doc.update(previous_sha256="0" * 64), "hash chain"),
        (lambda doc: doc.update(previous_sha256="invalid"), "predecessor"),
        (lambda doc: doc.update(event="promoted"), "event"),
        (lambda doc: doc.update(schema_version=2), "schema"),
        (lambda doc: doc.update(sequence=True), "sequence"),
    ],
)
def test_corrupt_journal_record_fails_closed(
    tmp_path: Path,
    mutation: object,
    message: str,
) -> None:
    """Reject schema, sequence, event, and hash-chain tampering."""
    store = DeploymentStateStore(tmp_path / "state")
    _adopt(store, _environment(tmp_path))
    document = _read_document(_journal_path(store))
    mutation(document)  # type: ignore[operator]
    _write_document(_journal_path(store), document)

    with pytest.raises(DeploymentStateError, match=message):
        store.read_journal()


def test_environment_snapshot_tampering_fails_closed(tmp_path: Path) -> None:
    """Bind the journal to the exact private rollback bytes."""
    store = DeploymentStateStore(tmp_path / "state")
    current = _adopt(store, _environment(tmp_path))
    snapshot = store.path / current.state.environment_snapshot
    snapshot.write_bytes(b"changed-secret\n")
    snapshot.chmod(0o600)

    with pytest.raises(DeploymentStateError, match="snapshot hash"):
        store.read_current()


def test_missing_environment_snapshot_fails_closed(tmp_path: Path) -> None:
    """Never claim rollback material exists when its private file is gone."""
    store = DeploymentStateStore(tmp_path / "state")
    current = _adopt(store, _environment(tmp_path))
    (store.path / current.state.environment_snapshot).unlink()

    with pytest.raises(DeploymentStateError, match="path is missing"):
        store.read_journal()


def test_journal_entry_document_matches_persisted_schema(tmp_path: Path) -> None:
    """Expose a stable secret-free journal representation to future tooling."""
    store = DeploymentStateStore(tmp_path / "state")
    _adopt(store, _environment(tmp_path))
    entry = store.read_journal()[0]

    assert entry.as_document() == _read_document(_journal_path(store))


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda document: None, "must be an object"),
        (lambda document: document.update(extra="unknown"), "schema"),
        (lambda document: document.update(schema_version=2), "schema"),
        (lambda document: document.update(adopted=False), "adopted"),
        (
            lambda document: document.update(
                environment_snapshot="environments/other.env"
            ),
            "environment_snapshot",
        ),
    ],
)
def test_state_document_parser_rejects_schema_ambiguity(
    tmp_path: Path,
    mutation: object,
    message: str,
) -> None:
    """Reject unknown fields, unsafe snapshots, and non-adoption state."""
    store = DeploymentStateStore(tmp_path / "state")
    current = _adopt(store, _environment(tmp_path))
    document: object = current.state.as_document()
    if message == "must be an object":
        document = None
    else:
        mutation(document)  # type: ignore[operator]

    with pytest.raises(DeploymentStateError, match=message):
        state_module._state_from_document(document)


def test_timestamp_validator_rejects_wrong_shape() -> None:
    """Keep public timestamps in one canonical UTC representation."""
    with pytest.raises(DeploymentStateError, match="recorded_at"):
        state_module._validate_timestamp("2026-07-28T12:34:56Z")


@pytest.mark.parametrize("mode", [0o000, 0o640])
def test_nonprivate_environment_source_is_rejected(
    tmp_path: Path,
    mode: int,
) -> None:
    """Require an exact owner-only rollback source before copying bytes."""
    environment = _environment(tmp_path)
    environment.chmod(mode)
    store = DeploymentStateStore(tmp_path / "state")

    with (
        store.transaction() as transaction,
        pytest.raises(DeploymentStateError, match="unsafe permissions"),
    ):
        transaction.adopt(
            compose_project="adk-template",
            compose_service="agent",
            source_revision=REVISION,
            image_reference=IMAGE_REFERENCE,
            image_id=IMAGE_ID,
            oci_revision=OCI_REVISION,
            environment_source=environment,
            deployment_id=DEPLOYMENT_ID,
            recorded_at=RECORDED_AT,
        )


@pytest.mark.parametrize(
    "unsafe_kind", ["missing", "empty", "directory", "symlink", "hardlink"]
)
def test_unsafe_environment_source_is_rejected(
    tmp_path: Path,
    unsafe_kind: str,
) -> None:
    """Reject absent, empty, linked, or non-regular recovery inputs."""
    environment = tmp_path / "legacy.env"
    if unsafe_kind == "empty":
        _write_private(environment, b"")
    elif unsafe_kind == "directory":
        environment.mkdir(mode=0o700)
    elif unsafe_kind == "symlink":
        target = _write_private(tmp_path / "target.env", ENVIRONMENT_BYTES)
        environment.symlink_to(target)
    elif unsafe_kind == "hardlink":
        target = _write_private(tmp_path / "target.env", ENVIRONMENT_BYTES)
        os.link(target, environment)

    store = DeploymentStateStore(tmp_path / "state")
    with (
        store.transaction() as transaction,
        pytest.raises(DeploymentStateError),
    ):
        transaction.adopt(
            compose_project="adk-template",
            compose_service="agent",
            source_revision=REVISION,
            image_reference=IMAGE_REFERENCE,
            image_id=IMAGE_ID,
            oci_revision=OCI_REVISION,
            environment_source=environment,
            deployment_id=DEPLOYMENT_ID,
            recorded_at=RECORDED_AT,
        )


def test_oversized_environment_source_is_rejected(tmp_path: Path) -> None:
    """Bound private recovery memory and disk consumption."""
    environment = _write_private(
        tmp_path / "legacy.env",
        b"x" * (state_module.MAX_ENVIRONMENT_BYTES + 1),
    )
    store = DeploymentStateStore(tmp_path / "state")

    with (
        store.transaction() as transaction,
        pytest.raises(DeploymentStateError, match="size is invalid"),
    ):
        transaction.adopt(
            compose_project="adk-template",
            compose_service="agent",
            source_revision=REVISION,
            image_reference=IMAGE_REFERENCE,
            image_id=IMAGE_ID,
            oci_revision=OCI_REVISION,
            environment_source=environment,
            deployment_id=DEPLOYMENT_ID,
            recorded_at=RECORDED_AT,
        )


def test_environment_change_during_read_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Detect a non-cooperating writer racing the adoption snapshot."""
    environment = _environment(tmp_path)
    real_fstat = os.fstat
    source_descriptor: int | None = None
    calls = 0

    def changed_fstat(descriptor: int) -> os.stat_result:
        nonlocal calls, source_descriptor
        metadata = real_fstat(descriptor)
        if stat.S_ISREG(metadata.st_mode) and metadata.st_size == len(
            ENVIRONMENT_BYTES
        ):
            source_descriptor = descriptor
            calls += 1
            if calls == 2:
                values = list(metadata)
                values[8] = metadata.st_mtime + 1
                return os.stat_result(values)
        return metadata

    monkeypatch.setattr(os, "fstat", changed_fstat)
    store = DeploymentStateStore(tmp_path / "state")

    with (
        store.transaction() as transaction,
        pytest.raises(DeploymentStateError, match="changed while read"),
    ):
        transaction.adopt(
            compose_project="adk-template",
            compose_service="agent",
            source_revision=REVISION,
            image_reference=IMAGE_REFERENCE,
            image_id=IMAGE_ID,
            oci_revision=OCI_REVISION,
            environment_source=environment,
            deployment_id=DEPLOYMENT_ID,
            recorded_at=RECORDED_AT,
        )
    assert source_descriptor is not None


def test_environment_replacement_before_open_is_rejected(tmp_path: Path) -> None:
    """Bind the validated pathname inode to the descriptor that is read."""
    environment = _environment(tmp_path)
    replacement = _write_private(
        tmp_path / "replacement.env",
        b"X" * len(ENVIRONMENT_BYTES),
    )
    real_open = os.open
    replaced = False

    def replace_then_open(
        path: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal replaced
        if Path(os.fsdecode(path)) == environment and not replaced:
            environment.unlink()
            replacement.replace(environment)
            replaced = True
        return real_open(path, flags, mode, dir_fd=dir_fd)

    open_boundary = create_autospec(
        os.open,
        spec_set=True,
        side_effect=replace_then_open,
    )
    with (
        patch("agent.deployment_state.os.open", new=open_boundary),
        pytest.raises(DeploymentStateError, match="changed before read"),
    ):
        state_module._read_private_file(
            environment,
            maximum_bytes=state_module.MAX_ENVIRONMENT_BYTES,
        )
    assert replaced


def test_environment_path_replacement_after_read_is_rejected(
    tmp_path: Path,
) -> None:
    """Recheck that the pathname still identifies the descriptor after reading."""
    environment = _environment(tmp_path)
    original_path = tmp_path / "original.env"
    replacement = _write_private(
        tmp_path / "replacement.env",
        b"X" * len(ENVIRONMENT_BYTES),
    )
    real_lstat = Path.lstat
    environment_lstats = 0

    def replace_before_final_lstat(selected: Path) -> os.stat_result:
        nonlocal environment_lstats
        if selected == environment:
            environment_lstats += 1
        if selected == environment and environment_lstats == 2:
            environment.replace(original_path)
            replacement.replace(environment)
        return real_lstat(selected)

    lstat_boundary = create_autospec(
        Path.lstat,
        spec_set=True,
        side_effect=replace_before_final_lstat,
    )
    with (
        patch("pathlib.Path.lstat", new=lstat_boundary),
        pytest.raises(DeploymentStateError, match="changed while read"),
    ):
        state_module._read_private_file(
            environment,
            maximum_bytes=state_module.MAX_ENVIRONMENT_BYTES,
        )
    assert environment_lstats == 2


def test_environment_hard_link_created_during_read_is_rejected(
    tmp_path: Path,
) -> None:
    """Revalidate private-file link count after all bytes have been read."""
    environment = _environment(tmp_path)
    extra_link = tmp_path / "linked.env"
    real_read = os.read
    linked = False

    def read_then_link(descriptor: int, length: int) -> bytes:
        nonlocal linked
        payload = real_read(descriptor, length)
        if payload and not linked:
            os.link(environment, extra_link)
            linked = True
        return payload

    read_boundary = create_autospec(
        os.read,
        spec_set=True,
        side_effect=read_then_link,
    )
    with (
        patch("agent.deployment_state.os.read", new=read_boundary),
        pytest.raises(DeploymentStateError, match="unsafe links"),
    ):
        state_module._read_private_file(
            environment,
            maximum_bytes=state_module.MAX_ENVIRONMENT_BYTES,
        )
    assert linked


def test_environment_growth_beyond_bound_is_rejected(
    tmp_path: Path,
) -> None:
    """Fail if a source grows after its pre-open size validation."""
    environment = _environment(tmp_path)
    oversized_read = create_autospec(
        os.read,
        spec_set=True,
        return_value=b"x" * (state_module.MAX_ENVIRONMENT_BYTES + 1),
    )

    with (
        patch("agent.deployment_state.os.read", new=oversized_read),
        pytest.raises(DeploymentStateError, match="size is invalid"),
    ):
        state_module._read_private_file(
            environment,
            maximum_bytes=state_module.MAX_ENVIRONMENT_BYTES,
        )


def test_environment_truncation_during_read_is_rejected(
    tmp_path: Path,
) -> None:
    """Fail when the payload length no longer matches its validated metadata."""
    environment = _environment(tmp_path)
    empty_read = create_autospec(os.read, spec_set=True, return_value=b"")

    with (
        patch("agent.deployment_state.os.read", new=empty_read),
        pytest.raises(DeploymentStateError, match="size is invalid"),
    ):
        state_module._read_private_file(
            environment,
            maximum_bytes=state_module.MAX_ENVIRONMENT_BYTES,
        )


def test_successful_short_writes_publish_complete_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise the complete-write loop without exposing a partial pathname."""
    real_write = os.write
    calls = 0

    def one_byte(descriptor: int, payload: bytes | memoryview) -> int:
        nonlocal calls
        calls += 1
        return real_write(descriptor, bytes(payload[:1]))

    monkeypatch.setattr(os, "write", one_byte)
    store = DeploymentStateStore(tmp_path / "state")
    current = _adopt(store, _environment(tmp_path))

    assert calls > 4
    assert store.read_current() == current


def test_stalled_write_leaves_no_published_record(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Remove private temporary files when a write stops making progress."""
    real_write = os.write
    calls = 0

    def partial_then_zero(descriptor: int, payload: bytes | memoryview) -> int:
        nonlocal calls
        calls += 1
        if calls == 1:
            return real_write(descriptor, bytes(payload[:1]))
        return 0

    monkeypatch.setattr(os, "write", partial_then_zero)
    store = DeploymentStateStore(tmp_path / "state")
    with (
        store.transaction() as transaction,
        pytest.raises(OSError, match="made no progress"),
    ):
        transaction.adopt(
            compose_project="adk-template",
            compose_service="agent",
            source_revision=REVISION,
            image_reference=IMAGE_REFERENCE,
            image_id=IMAGE_ID,
            oci_revision=OCI_REVISION,
            environment_source=_environment(tmp_path),
            deployment_id=DEPLOYMENT_ID,
            recorded_at=RECORDED_AT,
        )

    assert list(store.environments_path.iterdir()) == []
    assert list(store.journal_path.iterdir()) == []
    assert not store.current_path.exists()


def test_journal_publish_failure_does_not_advance_current(
    tmp_path: Path,
) -> None:
    """Leave only recoverable orphan env bytes when journal durability fails."""
    store = DeploymentStateStore(tmp_path / "state")
    real_link = os.link

    def reject_journal_link(
        source: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        destination: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        *,
        src_dir_fd: int | None = None,
        dst_dir_fd: int | None = None,
        follow_symlinks: bool = True,
    ) -> None:
        if str(destination).endswith(".json"):
            raise OSError("synthetic journal link failure")
        real_link(
            source,
            destination,
            src_dir_fd=src_dir_fd,
            dst_dir_fd=dst_dir_fd,
            follow_symlinks=follow_symlinks,
        )

    with (
        patch("agent.deployment_state.os.link", new=reject_journal_link),
        store.transaction() as transaction,
        pytest.raises(OSError, match="journal link"),
    ):
        transaction.adopt(
            compose_project="adk-template",
            compose_service="agent",
            source_revision=REVISION,
            image_reference=IMAGE_REFERENCE,
            image_id=IMAGE_ID,
            oci_revision=OCI_REVISION,
            environment_source=_environment(tmp_path),
            deployment_id=DEPLOYMENT_ID,
            recorded_at=RECORDED_AT,
        )

    assert len(list(store.environments_path.iterdir())) == 1
    assert list(store.journal_path.iterdir()) == []
    assert not store.current_path.exists()


def test_current_publish_failure_recovers_from_committed_journal(
    tmp_path: Path,
) -> None:
    """Recover after a crash-equivalent failure between journal and pointer."""
    store = DeploymentStateStore(tmp_path / "state")
    real_replace = Path.replace

    def reject_current(source: Path, target: Path) -> Path:
        if target.name == "current.json":
            raise OSError("synthetic current replace failure")
        return real_replace(source, target)

    with (
        patch("pathlib.Path.replace", new=reject_current),
        store.transaction() as transaction,
        pytest.raises(OSError, match="current replace"),
    ):
        transaction.adopt(
            compose_project="adk-template",
            compose_service="agent",
            source_revision=REVISION,
            image_reference=IMAGE_REFERENCE,
            image_id=IMAGE_ID,
            oci_revision=OCI_REVISION,
            environment_source=_environment(tmp_path),
            deployment_id=DEPLOYMENT_ID,
            recorded_at=RECORDED_AT,
        )

    assert _journal_path(store).exists()
    assert not store.current_path.exists()
    assert store.read_current() is not None


def test_prepublication_fsync_failure_leaves_no_environment_snapshot(
    tmp_path: Path,
) -> None:
    """Do not link a private temp file before its contents are synchronized."""
    sync_failure = create_autospec(
        os.fsync,
        spec_set=True,
        side_effect=OSError("synthetic file fsync failure"),
    )
    store = DeploymentStateStore(tmp_path / "state")
    with store.transaction():
        pass

    with (
        patch("agent.deployment_state.os.fsync", new=sync_failure),
        store.transaction() as transaction,
        pytest.raises(OSError, match="file fsync"),
    ):
        transaction.adopt(
            compose_project="adk-template",
            compose_service="agent",
            source_revision=REVISION,
            image_reference=IMAGE_REFERENCE,
            image_id=IMAGE_ID,
            oci_revision=OCI_REVISION,
            environment_source=_environment(tmp_path),
            deployment_id=DEPLOYMENT_ID,
            recorded_at=RECORDED_AT,
        )

    assert list(store.environments_path.iterdir()) == []


def test_prepublication_close_failure_leaves_no_environment_snapshot(
    tmp_path: Path,
) -> None:
    """Remove the unpublished temp file when its descriptor cannot be closed."""
    store = DeploymentStateStore(tmp_path / "state")
    with store.transaction():
        pass
    close_failure = create_autospec(
        os.close,
        spec_set=True,
        side_effect=OSError("synthetic file close failure"),
    )

    with (
        store.transaction() as transaction,
        patch("agent.deployment_state.os.close", new=close_failure),
        pytest.raises(OSError, match="file close"),
    ):
        transaction.adopt(
            compose_project="adk-template",
            compose_service="agent",
            source_revision=REVISION,
            image_reference=IMAGE_REFERENCE,
            image_id=IMAGE_ID,
            oci_revision=OCI_REVISION,
            environment_source=_environment(tmp_path),
            deployment_id=DEPLOYMENT_ID,
            recorded_at=RECORDED_AT,
        )

    assert list(store.environments_path.iterdir()) == []
    assert list(store.journal_path.iterdir()) == []
    assert not store.current_path.exists()


@pytest.mark.parametrize(
    ("failed_directory_sync", "committed_journal"),
    [(1, False), (2, True), (3, True)],
)
def test_directory_fsync_failures_never_advance_beyond_journal(
    tmp_path: Path,
    failed_directory_sync: int,
    committed_journal: bool,
) -> None:
    """Recover safely across snapshot, journal, and current directory failures."""
    store = DeploymentStateStore(tmp_path / "state")
    with store.transaction():
        pass
    real_fsync = os.fsync
    directory_syncs = 0

    def fail_selected_directory(descriptor: int) -> None:
        nonlocal directory_syncs
        if stat.S_ISDIR(os.fstat(descriptor).st_mode):
            directory_syncs += 1
            if directory_syncs == failed_directory_sync:
                raise OSError(
                    f"synthetic directory fsync {failed_directory_sync} failure"
                )
        real_fsync(descriptor)

    fsync_boundary = create_autospec(
        os.fsync,
        spec_set=True,
        side_effect=fail_selected_directory,
    )
    with (
        patch("agent.deployment_state.os.fsync", new=fsync_boundary),
        store.transaction() as transaction,
        pytest.raises(
            OSError,
            match=f"directory fsync {failed_directory_sync}",
        ),
    ):
        transaction.adopt(
            compose_project="adk-template",
            compose_service="agent",
            source_revision=REVISION,
            image_reference=IMAGE_REFERENCE,
            image_id=IMAGE_ID,
            oci_revision=OCI_REVISION,
            environment_source=_environment(tmp_path),
            deployment_id=DEPLOYMENT_ID,
            recorded_at=RECORDED_AT,
        )

    assert directory_syncs == failed_directory_sync
    assert _journal_path(store).exists() is committed_journal
    if not committed_journal:
        assert store.read_current() is None
        assert not store.current_path.exists()
        return

    store.current_path.unlink(missing_ok=True)
    recovered = store.read_current()
    assert recovered is not None
    assert recovered.journal_sequence == 1
    assert _read_document(store.current_path) == recovered.as_document()


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (b"", "size"),
        (b"x" * (state_module.MAX_JSON_BYTES + 1), "size"),
        (b"\xff", "JSON is invalid"),
        (b"{not-json}\n", "JSON is invalid"),
        (b"[]\n", "must be an object"),
        (b'{"a":1,"a":2}\n', "duplicate"),
        (b'{ "a":1 }\n', "not canonical"),
    ],
)
def test_json_decoder_rejects_ambiguous_or_noncanonical_bytes(
    payload: bytes,
    message: str,
) -> None:
    """Keep hashes and schema meaning byte-for-byte deterministic."""
    with pytest.raises(DeploymentStateError, match=message):
        state_module._decode_json(payload)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda doc: doc.update(journal_sequence=2), "ahead"),
        (lambda doc: doc.update(journal_sha256="0" * 64), "does not match"),
        (lambda doc: doc.update(schema_version=2), "schema"),
        (lambda doc: doc.update(journal_sequence=True), "sequence"),
        (lambda doc: doc.update(journal_sha256="bad"), "journal hash"),
    ],
)
def test_corrupt_current_pointer_fails_closed(
    tmp_path: Path,
    mutation: object,
    message: str,
) -> None:
    """Reject a pointer that cannot be proven against immutable history."""
    store = DeploymentStateStore(tmp_path / "state")
    _adopt(store, _environment(tmp_path))
    document = _read_document(store.current_path)
    mutation(document)  # type: ignore[operator]
    _write_document(store.current_path, document)

    with pytest.raises(DeploymentStateError, match=message):
        store.read_current()


def test_current_without_journal_fails_closed(tmp_path: Path) -> None:
    """Never treat a mutable pointer as authoritative on its own."""
    store = DeploymentStateStore(tmp_path / "state")
    adopted = _adopt(store, _environment(tmp_path))
    _journal_path(store).unlink()

    with pytest.raises(DeploymentStateError, match="ahead of the journal"):
        store.read_current()
    assert adopted.journal_sequence == 1


def test_noncanonical_current_file_fails_closed(tmp_path: Path) -> None:
    """Do not repair a corrupt pointer unless its prior journal identity is proven."""
    store = DeploymentStateStore(tmp_path / "state")
    _adopt(store, _environment(tmp_path))
    document = _read_document(store.current_path)
    store.current_path.write_text(
        json.dumps(document, indent=2) + "\n",
        encoding="utf-8",
    )
    store.current_path.chmod(0o600)

    with pytest.raises(DeploymentStateError, match="not canonical"):
        store.read_current()


def test_insecure_existing_current_file_is_not_silently_repaired(
    tmp_path: Path,
) -> None:
    """Re-raise unsafe metadata when the current pathname still exists."""
    store = DeploymentStateStore(tmp_path / "state")
    _adopt(store, _environment(tmp_path))
    store.current_path.chmod(0o644)

    with pytest.raises(DeploymentStateError, match="unsafe permissions"):
        store.read_current()


def test_dangling_current_symlink_is_not_silently_repaired(tmp_path: Path) -> None:
    """Treat a dangling pointer symlink as unsafe rather than absent."""
    store = DeploymentStateStore(tmp_path / "state")
    _adopt(store, _environment(tmp_path))
    missing_target = tmp_path / "missing-current-target"
    store.current_path.unlink()
    store.current_path.symlink_to(missing_target)

    with pytest.raises(DeploymentStateError, match="unsafe file type"):
        store.read_current()
    assert store.current_path.is_symlink()
    assert not missing_target.exists()
