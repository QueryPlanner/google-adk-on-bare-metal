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
from typing import Any
from unittest.mock import create_autospec, patch

import pytest

from agent import deployment_state as state_module
from agent.deployment_state import (
    CandidateReceipt,
    CurrentDeployment,
    DeploymentLockBusyError,
    DeploymentStateError,
    DeploymentStateStore,
    DeploymentTerminalCommittedError,
    DeploymentTerminalIndeterminateError,
    PendingPromotion,
    PersistentVolumeIdentity,
)

REVISION = "a" * 40
OCI_REVISION = "b" * 40
IMAGE_ID = f"sha256:{'c' * 64}"
IMAGE_REFERENCE = f"ghcr.io/queryplanner/agent@sha256:{'d' * 64}"
DEPLOYMENT_ID = "adopt-0123456789abcdef"
RECORDED_AT = "2026-07-28T12:34:56.123456Z"
ENVIRONMENT_BYTES = b'API_KEY="secret-$-canary"\nEMPTY=""\n'
TARGET_REVISION = "e" * 40
TARGET_IMAGE_ID = f"sha256:{'f' * 64}"
TARGET_IMAGE_REFERENCE = f"ghcr.io/queryplanner/agent@sha256:{'1' * 64}"
TARGET_ENVIRONMENT_BYTES = b'API_KEY="next-secret-$-canary"\nEMPTY=""\n'
TRANSACTION_ID = "promote-0123456789abcdef"
PROMOTION_RECORDED_AT = "2026-07-28T12:35:56.123456Z"
CANDIDATE_OBSERVED_AT = "2026-07-28T12:35:55.123456Z"
CANDIDATE_CONTAINER_ID = "2" * 64
PERSISTENT_VOLUMES = (
    PersistentVolumeIdentity(
        name="adk-template-data",
        driver="local",
        mountpoint="/var/lib/docker/volumes/adk-template-data/_data",
        destination="/app/data",
        created_at="2026-07-27T10:11:12.123456Z",
    ),
)


def _write_private(path: Path, payload: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    path.chmod(0o600)
    return path


def _environment(tmp_path: Path) -> Path:
    return _write_private(tmp_path / "legacy.env", ENVIRONMENT_BYTES)


def _promotion_environment(tmp_path: Path) -> Path:
    return _write_private(tmp_path / "target.env", TARGET_ENVIRONMENT_BYTES)


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


def _candidate(
    transaction: state_module.DeploymentStateTransaction,
    **overrides: object,
) -> CandidateReceipt:
    journal = transaction.journal()
    tail = None if not journal else journal[-1]
    values: dict[str, object] = {
        "observed_at": CANDIDATE_OBSERVED_AT,
        "compose_project": "adk-template-candidate",
        "compose_service": "agent",
        "container_id": CANDIDATE_CONTAINER_ID,
        "image_reference": TARGET_IMAGE_REFERENCE,
        "image_id": TARGET_IMAGE_ID,
        "oci_revision": TARGET_REVISION,
        "baseline_journal_sequence": None if tail is None else tail.sequence,
        "baseline_journal_sha256": None if tail is None else tail.sha256,
    }
    values.update(overrides)
    return CandidateReceipt(**values)  # type: ignore[arg-type]


def _begin_promotion(
    transaction: state_module.DeploymentStateTransaction,
    environment: Path,
    *,
    transaction_id: str = TRANSACTION_ID,
    candidate_overrides: dict[str, object] | None = None,
    persistent_volumes: tuple[PersistentVolumeIdentity, ...] = PERSISTENT_VOLUMES,
) -> PendingPromotion:
    return transaction.begin_promotion(
        compose_project="adk-template",
        compose_service="agent",
        source_revision=TARGET_REVISION,
        image_reference=TARGET_IMAGE_REFERENCE,
        image_id=TARGET_IMAGE_ID,
        oci_revision=TARGET_REVISION,
        environment_source=environment,
        candidate=_candidate(transaction, **(candidate_overrides or {})),
        persistent_volumes=persistent_volumes,
        transaction_id=transaction_id,
        recorded_at=PROMOTION_RECORDED_AT,
    )


def _read_document(path: Path) -> dict[str, object]:
    decoded = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(decoded, dict)
    return decoded


def _write_document(path: Path, document: dict[str, object]) -> None:
    _write_private(path, state_module._canonical_json(document))


def _append_later_adoption(store: DeploymentStateStore) -> None:
    previous_payload = _journal_path(store).read_bytes()
    deployment_id = "adopt-later-00000002"
    environment_snapshot = f"environments/{deployment_id}.env"
    _write_private(store.path / environment_snapshot, ENVIRONMENT_BYTES)
    state = {
        "schema_version": 1,
        "deployment_id": deployment_id,
        "recorded_at": RECORDED_AT,
        "compose_project": "adk-template",
        "compose_service": "agent",
        "source_revision": REVISION,
        "image_reference": IMAGE_REFERENCE,
        "image_id": IMAGE_ID,
        "oci_revision": OCI_REVISION,
        "environment_snapshot": environment_snapshot,
        "environment_sha256": hashlib.sha256(ENVIRONMENT_BYTES).hexdigest(),
        "adopted": True,
    }
    later = {
        "schema_version": 1,
        "sequence": 2,
        "previous_sha256": hashlib.sha256(previous_payload).hexdigest(),
        "event": "adopted",
        "state": state,
    }
    _write_document(_journal_path(store, 2), later)


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
    baseline_bytes = store.current_path.read_bytes()
    with store.transaction() as transaction:
        pending = _begin_promotion(transaction, _promotion_environment(tmp_path))
        promoted = transaction.commit_promotion(
            pending.transaction_id,
            persistent_volumes=PERSISTENT_VOLUMES,
        )
    _write_private(store.current_path, baseline_bytes)

    recovered = store.read_current()

    assert recovered == promoted
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


@pytest.mark.parametrize(
    "operation",
    ["read-current", "read-journal", "transaction"],
)
def test_noninitial_schema_v1_adoption_fails_closed(
    tmp_path: Path,
    operation: str,
) -> None:
    """Reject a validly encoded adoption that attempts to reset journal history."""
    store = DeploymentStateStore(tmp_path / operation / "state")
    _adopt(store, _environment(tmp_path))
    _append_later_adoption(store)
    current_bytes = store.current_path.read_bytes()

    with pytest.raises(
        DeploymentStateError,
        match="deployment journal adoption is not initial",
    ) as error:
        if operation == "read-current":
            store.read_current()
        elif operation == "read-journal":
            store.read_journal()
        else:
            with store.transaction():
                pytest.fail("noninitial adoption opened a transaction")

    assert str(error.value) == "deployment journal adoption is not initial"
    assert store.current_path.read_bytes() == current_bytes


def test_adoption_after_empty_baseline_terminal_fails_closed(tmp_path: Path) -> None:
    """Reject later adoption even when no earlier event established current state."""
    store = DeploymentStateStore(tmp_path / "state")
    with store.transaction() as transaction:
        pending = _begin_promotion(
            transaction,
            _promotion_environment(tmp_path),
            persistent_volumes=(),
        )
        transaction.record_abort(
            pending.transaction_id,
            persistent_volumes=(),
        )
    assert store.read_current() is None
    _append_later_adoption(store)

    with pytest.raises(DeploymentStateError) as error:
        store.read_journal()

    assert str(error.value) == "deployment journal adoption is not initial"


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


def test_schema_v1_adoption_bytes_remain_exact_and_unchanged(
    tmp_path: Path,
) -> None:
    """Keep the original adoption, journal, and current formats byte-exact."""
    store = DeploymentStateStore(tmp_path / "state")
    adopted = _adopt(store, _environment(tmp_path))
    environment_sha256 = hashlib.sha256(ENVIRONMENT_BYTES).hexdigest()
    state_json = (
        f'{{"adopted":true,"compose_project":"adk-template",'
        f'"compose_service":"agent","deployment_id":"{DEPLOYMENT_ID}",'
        f'"environment_sha256":"{environment_sha256}",'
        f'"environment_snapshot":"environments/{DEPLOYMENT_ID}.env",'
        f'"image_id":"{IMAGE_ID}","image_reference":"{IMAGE_REFERENCE}",'
        f'"oci_revision":"{OCI_REVISION}","recorded_at":"{RECORDED_AT}",'
        f'"schema_version":1,"source_revision":"{REVISION}"}}'
    )
    expected_journal = (
        '{"event":"adopted","previous_sha256":null,"schema_version":1,'
        f'"sequence":1,"state":{state_json}}}\n'
    ).encode()
    journal_sha256 = hashlib.sha256(expected_journal).hexdigest()
    expected_current = (
        f'{{"journal_sequence":1,"journal_sha256":"{journal_sha256}",'
        f'"schema_version":1,"state":{state_json}}}\n'
    ).encode()

    assert _journal_path(store).read_bytes() == expected_journal
    assert store.current_path.read_bytes() == expected_current
    assert adopted.state.as_document()["schema_version"] == 1

    assert store.read_current() == adopted
    assert store.read_journal() == (store.read_journal()[0],)
    assert _journal_path(store).read_bytes() == expected_journal
    assert store.current_path.read_bytes() == expected_current


def test_promotion_round_trips_v2_state_and_exact_environment(
    tmp_path: Path,
) -> None:
    """Persist a complete secret-free intent and make its terminal authoritative."""
    store = DeploymentStateStore(tmp_path / "state")
    baseline = _adopt(store, _environment(tmp_path))
    source = _promotion_environment(tmp_path)
    installed = tmp_path / "installed.env"

    with store.transaction() as transaction:
        pending = _begin_promotion(transaction, source)
        assert transaction.pending() == pending
        assert pending.intent.baseline_journal_sequence == 1
        assert pending.intent.baseline_journal_sha256 == baseline.journal_sha256
        assert pending.intent.baseline_current_sequence == 1
        assert pending.intent.baseline_current_sha256 == baseline.journal_sha256
        assert pending.intent.target.adopted is False
        assert pending.intent.target.as_document()["schema_version"] == 2
        assert pending.intent.persistent_volumes == PERSISTENT_VOLUMES
        assert transaction.read_environment(pending.intent.target) == (
            TARGET_ENVIRONMENT_BYTES
        )
        transaction.install_environment(pending.intent.target, installed)

        promoted = transaction.commit_promotion(
            pending.transaction_id,
            persistent_volumes=PERSISTENT_VOLUMES,
        )

        assert promoted.state == pending.intent.target
        assert promoted.journal_sequence == 2
        assert promoted.as_document()["schema_version"] == 2
        assert transaction.current() == promoted
        assert transaction.pending() is None
        assert [entry.event for entry in transaction.journal()] == [
            "adopted",
            "promoted",
        ]

    intent_path = store.transactions_path / f"{TRANSACTION_ID}.json"
    target_snapshot = store.path / promoted.state.environment_snapshot
    assert installed.read_bytes() == TARGET_ENVIRONMENT_BYTES
    assert stat.S_IMODE(installed.stat().st_mode) == 0o600
    assert target_snapshot.read_bytes() == TARGET_ENVIRONMENT_BYTES
    for private_file in (
        intent_path,
        target_snapshot,
        _journal_path(store, 2),
        store.current_path,
    ):
        assert stat.S_IMODE(private_file.stat().st_mode) == 0o600
        assert private_file.stat().st_nlink == 1
    assert stat.S_IMODE(store.transactions_path.stat().st_mode) == 0o700
    assert not store.pending_path.exists()

    public_bytes = (
        intent_path.read_bytes()
        + _journal_path(store, 2).read_bytes()
        + store.current_path.read_bytes()
    )
    assert b"next-secret-$-canary" not in public_bytes
    assert str(source).encode() not in public_bytes
    assert store.read_current() == promoted
    assert store.read_journal()[-1].intent_sha256 == pending.intent_sha256


def test_first_install_can_promote_without_an_adopted_baseline(
    tmp_path: Path,
) -> None:
    """Support a fresh VM while binding the first terminal to an empty baseline."""
    store = DeploymentStateStore(tmp_path / "state")
    with store.transaction() as transaction:
        pending = _begin_promotion(
            transaction,
            _promotion_environment(tmp_path),
            persistent_volumes=(),
        )
        assert pending.intent.baseline_journal_sequence is None
        assert pending.intent.baseline_current_sequence is None

        current = transaction.commit_promotion(
            pending.transaction_id,
            persistent_volumes=PERSISTENT_VOLUMES,
        )

    assert current.journal_sequence == 1
    assert current.state.adopted is False
    assert store.read_current() == current
    terminal = store.read_journal()[0]
    assert terminal.previous_sha256 is None
    assert terminal.event == "promoted"
    assert terminal.persistent_volumes == PERSISTENT_VOLUMES


@pytest.mark.parametrize(
    ("method_name", "event"),
    [
        ("record_rollback", "rolled_back"),
        ("record_abort", "aborted"),
    ],
)
def test_nonpromoted_terminal_preserves_baseline_current_bytes(
    tmp_path: Path,
    method_name: str,
    event: str,
) -> None:
    """Append history without moving or rewriting the established pointer."""
    store = DeploymentStateStore(tmp_path / event / "state")
    baseline = _adopt(store, _environment(tmp_path))
    baseline_bytes = store.current_path.read_bytes()

    with store.transaction() as transaction:
        pending = _begin_promotion(transaction, _promotion_environment(tmp_path))
        terminal_method = getattr(transaction, method_name)
        result = terminal_method(
            pending.transaction_id,
            persistent_volumes=PERSISTENT_VOLUMES,
        )

        assert result == baseline
        assert transaction.current() == baseline
        assert transaction.pending() is None

    assert store.current_path.read_bytes() == baseline_bytes
    assert store.read_current() == baseline
    journal = store.read_journal()
    assert [entry.event for entry in journal] == ["adopted", event]
    assert journal[-1].state == baseline.state


@pytest.mark.parametrize("method_name", ["record_rollback", "record_abort"])
def test_empty_baseline_terminal_keeps_current_absent(
    tmp_path: Path,
    method_name: str,
) -> None:
    """Represent a verified non-install outcome without inventing current state."""
    store = DeploymentStateStore(tmp_path / method_name / "state")
    with store.transaction() as transaction:
        pending = _begin_promotion(
            transaction,
            _promotion_environment(tmp_path),
            persistent_volumes=(),
        )
        result = getattr(transaction, method_name)(
            pending.transaction_id,
            persistent_volumes=(),
        )

    assert result is None
    assert store.read_current() is None
    assert not store.current_path.exists()
    assert store.read_journal()[0].state is None


def test_future_promotion_uses_journal_tail_but_last_established_current(
    tmp_path: Path,
) -> None:
    """Fold current across rollback while chaining the next intent to journal tail."""
    store = DeploymentStateStore(tmp_path / "state")
    baseline = _adopt(store, _environment(tmp_path))
    with store.transaction() as transaction:
        first = _begin_promotion(transaction, _promotion_environment(tmp_path))
        assert (
            transaction.record_rollback(
                first.transaction_id,
                persistent_volumes=PERSISTENT_VOLUMES,
            )
            == baseline
        )

    second_id = "promote-fedcba9876543210"
    second_source = _write_private(
        tmp_path / "second-target.env",
        b'API_KEY="third-secret"\n',
    )
    with store.transaction() as transaction:
        second = _begin_promotion(
            transaction,
            second_source,
            transaction_id=second_id,
        )
        assert second.intent.baseline_journal_sequence == 2
        assert second.intent.baseline_current_sequence == 1
        promoted = transaction.commit_promotion(
            second.transaction_id,
            persistent_volumes=PERSISTENT_VOLUMES,
        )

    assert promoted.journal_sequence == 3
    assert [entry.event for entry in store.read_journal()] == [
        "adopted",
        "rolled_back",
        "promoted",
    ]
    assert store.read_current() == promoted


def test_unresolved_pending_survives_restart_and_blocks_new_state(
    tmp_path: Path,
) -> None:
    """Expose recovery work instead of silently replacing an active intent."""
    store = DeploymentStateStore(tmp_path / "state")
    _adopt(store, _environment(tmp_path))
    with store.transaction() as transaction:
        pending = _begin_promotion(transaction, _promotion_environment(tmp_path))

    with store.transaction() as recovered:
        assert recovered.pending() == pending
        with pytest.raises(DeploymentStateError, match="already pending"):
            _begin_promotion(
                recovered,
                _promotion_environment(tmp_path),
                transaction_id="promote-a23456789abcdef0",
            )
        with pytest.raises(DeploymentStateError, match="already been initialized"):
            recovered.adopt(
                compose_project="adk-template",
                compose_service="agent",
                source_revision=REVISION,
                image_reference=IMAGE_REFERENCE,
                image_id=IMAGE_ID,
                oci_revision=OCI_REVISION,
                environment_source=_environment(tmp_path),
            )


def test_full_volume_identity_rejects_same_name_driver_and_destination(
    tmp_path: Path,
) -> None:
    """Detect Docker volume replacement even when its friendly fields are reused."""
    store = DeploymentStateStore(tmp_path / "state")
    _adopt(store, _environment(tmp_path))
    replaced_volume = PersistentVolumeIdentity(
        name=PERSISTENT_VOLUMES[0].name,
        driver=PERSISTENT_VOLUMES[0].driver,
        mountpoint=PERSISTENT_VOLUMES[0].mountpoint,
        destination=PERSISTENT_VOLUMES[0].destination,
        created_at="2026-07-28T10:11:12.123456Z",
    )

    with store.transaction() as transaction:
        pending = _begin_promotion(transaction, _promotion_environment(tmp_path))
        with pytest.raises(DeploymentStateError, match="identity changed"):
            transaction.commit_promotion(
                pending.transaction_id,
                persistent_volumes=(replaced_volume,),
            )
        assert len(transaction.journal()) == 1
        assert transaction.pending() == pending
        transaction.record_abort(
            pending.transaction_id,
            persistent_volumes=PERSISTENT_VOLUMES,
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("name", "../data", "name"),
        ("driver", "", "driver"),
        ("mountpoint", "relative", "mountpoint"),
        ("mountpoint", "/var/lib/../data", "mountpoint"),
        ("destination", "/", "destination"),
        ("destination", "/app/../data", "destination"),
        ("created_at", "2026-02-30T10:11:12Z", "created_at"),
    ],
)
def test_begin_rejects_invalid_volume_identity_before_snapshot(
    tmp_path: Path,
    field: str,
    value: str,
    message: str,
) -> None:
    """Validate daemon volume observations before publishing private bytes."""
    store = DeploymentStateStore(tmp_path / field / "state")
    _adopt(store, _environment(tmp_path))
    values = PERSISTENT_VOLUMES[0].as_document()
    values[field] = value
    invalid = PersistentVolumeIdentity(**values)  # type: ignore[arg-type]

    with (
        store.transaction() as transaction,
        pytest.raises(DeploymentStateError, match=message),
    ):
        _begin_promotion(
            transaction,
            _promotion_environment(tmp_path),
            persistent_volumes=(invalid,),
        )

    assert not store.pending_path.exists()
    assert not store.transactions_path.exists()


@pytest.mark.parametrize(
    ("stage", "expected_snapshot", "expected_intent"),
    [
        ("snapshot", False, False),
        ("intent", True, False),
        ("pending", True, True),
    ],
)
def test_begin_publication_failure_never_returns_without_durable_pending(
    tmp_path: Path,
    stage: str,
    expected_snapshot: bool,
    expected_intent: bool,
) -> None:
    """Leave only pre-mutation residue when a create-only link fails."""
    store = DeploymentStateStore(tmp_path / stage / "state")
    _adopt(store, _environment(tmp_path))
    source = _promotion_environment(tmp_path)
    snapshot_path = store.environments_path / f"{TRANSACTION_ID}.env"
    intent_path = store.transactions_path / f"{TRANSACTION_ID}.json"
    destinations = {
        "snapshot": snapshot_path,
        "intent": intent_path,
        "pending": store.pending_path,
    }
    real_link = os.link

    def fail_selected_link(
        source_path: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        destination_path: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        *,
        src_dir_fd: int | None = None,
        dst_dir_fd: int | None = None,
        follow_symlinks: bool = True,
    ) -> None:
        if Path(os.fsdecode(destination_path)) == destinations[stage]:
            raise OSError(f"synthetic {stage} link failure")
        real_link(
            source_path,
            destination_path,
            src_dir_fd=src_dir_fd,
            dst_dir_fd=dst_dir_fd,
            follow_symlinks=follow_symlinks,
        )

    link_boundary = create_autospec(
        os.link,
        spec_set=True,
        side_effect=fail_selected_link,
    )
    with (
        store.transaction() as transaction,
        patch("agent.deployment_state.os.link", new=link_boundary),
        pytest.raises(OSError, match=f"{stage} link"),
    ):
        _begin_promotion(transaction, source)

    assert snapshot_path.exists() is expected_snapshot
    assert intent_path.exists() is expected_intent
    assert not store.pending_path.exists()
    assert len(store.read_journal()) == 1


def test_pending_hard_link_crash_residue_recovers_to_active_pointer(
    tmp_path: Path,
) -> None:
    """Remove the create-only temp alias while preserving the pending final."""
    store = DeploymentStateStore(tmp_path / "state")
    _adopt(store, _environment(tmp_path))
    with store.transaction() as transaction:
        pending = _begin_promotion(transaction, _promotion_environment(tmp_path))
    temporary = store.path / ".deployment-pending-crash.tmp"
    os.link(store.pending_path, temporary)
    assert store.pending_path.stat().st_nlink == 2

    with store.transaction() as recovered:
        assert recovered.pending() == pending

    assert not temporary.exists()
    assert store.pending_path.stat().st_nlink == 1


def test_definite_terminal_link_failure_keeps_pending_for_compensation(
    tmp_path: Path,
) -> None:
    """Allow rollback only when the terminal final pathname was never visible."""
    store = DeploymentStateStore(tmp_path / "state")
    baseline = _adopt(store, _environment(tmp_path))
    real_link = os.link

    with store.transaction() as transaction:
        pending = _begin_promotion(transaction, _promotion_environment(tmp_path))
        terminal_path = _journal_path(store, 2)

        def reject_terminal_link(
            source_path: str | bytes | os.PathLike[str] | os.PathLike[bytes],
            destination_path: str | bytes | os.PathLike[str] | os.PathLike[bytes],
            *,
            src_dir_fd: int | None = None,
            dst_dir_fd: int | None = None,
            follow_symlinks: bool = True,
        ) -> None:
            if Path(os.fsdecode(destination_path)) == terminal_path:
                raise OSError("synthetic terminal link failure")
            real_link(
                source_path,
                destination_path,
                src_dir_fd=src_dir_fd,
                dst_dir_fd=dst_dir_fd,
                follow_symlinks=follow_symlinks,
            )

        link_boundary = create_autospec(
            os.link,
            spec_set=True,
            side_effect=reject_terminal_link,
        )
        with (
            patch("agent.deployment_state.os.link", new=link_boundary),
            pytest.raises(OSError, match="terminal link"),
        ):
            transaction.commit_promotion(
                pending.transaction_id,
                persistent_volumes=PERSISTENT_VOLUMES,
            )

        assert not terminal_path.exists()
        assert transaction.pending() == pending
        assert (
            transaction.record_rollback(
                pending.transaction_id,
                persistent_volumes=PERSISTENT_VOLUMES,
            )
            == baseline
        )


def test_visible_terminal_directory_fsync_failure_is_indeterminate(
    tmp_path: Path,
) -> None:
    """Never permit compensation after a link-visible, fsync-uncertain terminal."""
    store = DeploymentStateStore(tmp_path / "state")
    _adopt(store, _environment(tmp_path))
    real_fsync = os.fsync

    with store.transaction() as transaction:
        pending = _begin_promotion(transaction, _promotion_environment(tmp_path))
        journal_identity = (
            store.journal_path.stat().st_dev,
            store.journal_path.stat().st_ino,
        )

        def reject_journal_directory_sync(descriptor: int) -> None:
            metadata = os.fstat(descriptor)
            if (metadata.st_dev, metadata.st_ino) == journal_identity:
                raise OSError("synthetic terminal directory fsync failure")
            real_fsync(descriptor)

        sync_boundary = create_autospec(
            os.fsync,
            spec_set=True,
            side_effect=reject_journal_directory_sync,
        )
        with (
            patch("agent.deployment_state.os.fsync", new=sync_boundary),
            pytest.raises(
                DeploymentTerminalIndeterminateError,
                match="publication is indeterminate",
            ) as captured,
        ):
            transaction.commit_promotion(
                pending.transaction_id,
                persistent_volumes=PERSISTENT_VOLUMES,
            )

        assert captured.value.event == "promoted"
        assert captured.value.transaction_id == TRANSACTION_ID
        assert _journal_path(store, 2).exists()
        with pytest.raises(DeploymentTerminalIndeterminateError):
            transaction.record_rollback(
                pending.transaction_id,
                persistent_volumes=(),
            )

    recovered = store.read_current()
    assert recovered is not None
    assert recovered.journal_sequence == 2
    assert recovered.state.adopted is False
    assert not store.pending_path.exists()


def test_current_replace_failure_reports_committed_and_reconciles(
    tmp_path: Path,
) -> None:
    """Treat the terminal as final even when promoted current publication fails."""
    store = DeploymentStateStore(tmp_path / "state")
    baseline = _adopt(store, _environment(tmp_path))
    real_replace = Path.replace

    def reject_current_replace(source: Path, destination: Path) -> Path:
        if destination == store.current_path:
            raise OSError("synthetic promoted current replace failure")
        return real_replace(source, destination)

    replace_boundary = create_autospec(
        Path.replace,
        spec_set=True,
        side_effect=reject_current_replace,
    )
    with store.transaction() as transaction:
        pending = _begin_promotion(transaction, _promotion_environment(tmp_path))
        with (
            patch("pathlib.Path.replace", new=replace_boundary),
            pytest.raises(
                DeploymentTerminalCommittedError,
                match="outcome is committed",
            ) as captured,
        ):
            transaction.commit_promotion(
                pending.transaction_id,
                persistent_volumes=PERSISTENT_VOLUMES,
            )
        assert captured.value.event == "promoted"
        assert captured.value.transaction_id == TRANSACTION_ID
        with pytest.raises(DeploymentTerminalCommittedError):
            transaction.record_rollback(
                pending.transaction_id,
                persistent_volumes=(),
            )

    assert store.current_path.read_bytes() == state_module._canonical_json(
        baseline.as_document()
    )
    assert store.pending_path.exists()
    recovered = store.read_current()
    assert recovered is not None
    assert recovered.journal_sequence == 2
    assert recovered.state.adopted is False
    assert not store.pending_path.exists()


def test_pending_unlink_failure_reports_committed_and_reconciles(
    tmp_path: Path,
) -> None:
    """Keep a stale pointer recoverable after the authoritative terminal commit."""
    store = DeploymentStateStore(tmp_path / "state")
    _adopt(store, _environment(tmp_path))
    real_unlink = Path.unlink

    def reject_pending_unlink(selected: Path, missing_ok: bool = False) -> None:
        if selected == store.pending_path:
            raise OSError("synthetic pending unlink failure")
        real_unlink(selected, missing_ok=missing_ok)

    unlink_boundary = create_autospec(
        Path.unlink,
        spec_set=True,
        side_effect=reject_pending_unlink,
    )
    with store.transaction() as transaction:
        pending = _begin_promotion(transaction, _promotion_environment(tmp_path))
        with (
            patch("pathlib.Path.unlink", new=unlink_boundary),
            pytest.raises(DeploymentTerminalCommittedError),
        ):
            transaction.commit_promotion(
                pending.transaction_id,
                persistent_volumes=PERSISTENT_VOLUMES,
            )
        with pytest.raises(DeploymentTerminalCommittedError):
            transaction.record_abort(
                pending.transaction_id,
                persistent_volumes=(),
            )

    assert store.pending_path.exists()
    assert store.read_current() is not None
    assert not store.pending_path.exists()


def test_reconciled_terminal_is_exposed_and_blocks_same_transaction_mutation(
    tmp_path: Path,
) -> None:
    """Require a fresh controller invocation after clearing a stale pointer."""
    store = DeploymentStateStore(tmp_path / "state")
    _adopt(store, _environment(tmp_path))
    source = _promotion_environment(tmp_path)
    with store.transaction() as transaction:
        assert transaction.recovered_terminal() is None
        pending = _begin_promotion(transaction, source)
        stale_pointer = store.pending_path.read_bytes()
        transaction.record_abort(
            pending.transaction_id,
            persistent_volumes=PERSISTENT_VOLUMES,
        )
    _write_private(store.pending_path, stale_pointer)

    with store.transaction() as recovered:
        marker = recovered.recovered_terminal()
        assert marker is not None
        assert marker.event == "aborted"
        assert marker.transaction_id == pending.transaction_id
        assert recovered.pending() is None
        with pytest.raises(
            DeploymentTerminalCommittedError,
            match="outcome is committed",
        ):
            _begin_promotion(recovered, source)
        with pytest.raises(DeploymentTerminalCommittedError):
            recovered.install_environment(
                pending.intent.target,
                tmp_path / "runtime.env",
            )
        with pytest.raises(DeploymentTerminalCommittedError):
            recovered.remove_environment(
                pending.intent.target,
                tmp_path / "runtime.env",
            )
        with pytest.raises(DeploymentTerminalCommittedError):
            recovered.record_rollback(
                pending.transaction_id,
                persistent_volumes=PERSISTENT_VOLUMES,
            )

    assert not store.pending_path.exists()
    assert [entry.event for entry in store.read_journal()] == [
        "adopted",
        "aborted",
    ]


def test_pending_cleanup_fsync_failure_is_committed_with_no_pointer(
    tmp_path: Path,
) -> None:
    """Classify failure after pending unlink as committed, never rollback-safe."""
    store = DeploymentStateStore(tmp_path / "state")
    _adopt(store, _environment(tmp_path))
    real_fsync = os.fsync

    with store.transaction() as transaction:
        pending = _begin_promotion(transaction, _promotion_environment(tmp_path))
        root_identity = (store.path.stat().st_dev, store.path.stat().st_ino)
        root_syncs = 0

        def fail_second_root_sync(descriptor: int) -> None:
            nonlocal root_syncs
            metadata = os.fstat(descriptor)
            if (metadata.st_dev, metadata.st_ino) == root_identity:
                root_syncs += 1
                if root_syncs == 2:
                    raise OSError("synthetic pending cleanup fsync failure")
            real_fsync(descriptor)

        sync_boundary = create_autospec(
            os.fsync,
            spec_set=True,
            side_effect=fail_second_root_sync,
        )
        with (
            patch("agent.deployment_state.os.fsync", new=sync_boundary),
            pytest.raises(DeploymentTerminalCommittedError),
        ):
            transaction.commit_promotion(
                pending.transaction_id,
                persistent_volumes=PERSISTENT_VOLUMES,
            )

    assert root_syncs == 2
    assert not store.pending_path.exists()
    assert store.read_current() is not None


@pytest.mark.parametrize("derived_failure", ["current", "current-temp"])
def test_restart_classifies_derived_repair_failure_as_committed(
    tmp_path: Path,
    derived_failure: str,
) -> None:
    """Preserve terminal classification while a stale pending proves recovery."""
    store = DeploymentStateStore(tmp_path / derived_failure / "state")
    _adopt(store, _environment(tmp_path))
    real_replace = Path.replace

    def reject_current_replace(source: Path, destination: Path) -> Path:
        if destination == store.current_path:
            raise OSError("synthetic current replace failure")
        return real_replace(source, destination)

    with store.transaction() as transaction:
        pending = _begin_promotion(transaction, _promotion_environment(tmp_path))
        with (
            patch(
                "pathlib.Path.replace",
                new=create_autospec(
                    Path.replace,
                    spec_set=True,
                    side_effect=reject_current_replace,
                ),
            ),
            pytest.raises(DeploymentTerminalCommittedError),
        ):
            transaction.commit_promotion(
                pending.transaction_id,
                persistent_volumes=PERSISTENT_VOLUMES,
            )

    if derived_failure == "current":
        store.current_path.chmod(0o644)
        context: Any = patch.dict({}, {})
    else:
        temporary = _write_private(
            store.path / ".deployment-current-recovery.tmp",
            b"orphan",
        )
        real_unlink = Path.unlink

        def reject_temp_unlink(selected: Path, missing_ok: bool = False) -> None:
            if selected == temporary:
                raise OSError("synthetic current temp unlink failure")
            real_unlink(selected, missing_ok=missing_ok)

        context = patch(
            "pathlib.Path.unlink",
            new=create_autospec(
                Path.unlink,
                spec_set=True,
                side_effect=reject_temp_unlink,
            ),
        )

    with (
        context,
        pytest.raises(DeploymentTerminalCommittedError) as captured,
        store.transaction(),
    ):
        pytest.fail("derived repair unexpectedly completed")
    assert captured.value.event == "promoted"

    if derived_failure == "current":
        store.current_path.chmod(0o600)
    assert store.read_current() is not None
    assert not store.pending_path.exists()


@pytest.mark.parametrize(
    "created_at",
    [
        "2026-07-28T10:11:12Z",
        "2026-07-28T10:11:12.123456789Z",
        "2026-07-28T10:11:12.123456789+05:30",
    ],
)
def test_volume_creation_identity_preserves_docker_rfc3339_text(
    tmp_path: Path,
    created_at: str,
) -> None:
    """Accept representative daemon timestamps without normalizing exact bytes."""
    volume = PersistentVolumeIdentity(
        name=PERSISTENT_VOLUMES[0].name,
        driver=PERSISTENT_VOLUMES[0].driver,
        mountpoint=PERSISTENT_VOLUMES[0].mountpoint,
        destination=PERSISTENT_VOLUMES[0].destination,
        created_at=created_at,
    )
    store = DeploymentStateStore(tmp_path / created_at.replace(":", "-") / "state")
    _adopt(store, _environment(tmp_path))

    with store.transaction() as transaction:
        pending = _begin_promotion(
            transaction,
            _promotion_environment(tmp_path),
            persistent_volumes=(volume,),
        )
        assert pending.intent.persistent_volumes[0].created_at == created_at
        transaction.record_abort(
            pending.transaction_id,
            persistent_volumes=(volume,),
        )

    assert store.read_journal()[-1].persistent_volumes == (volume,)


def test_remove_environment_is_exact_private_and_idempotent(
    tmp_path: Path,
) -> None:
    """Restore absence without exposing bytes and tolerate a prior successful unlink."""
    store = DeploymentStateStore(tmp_path / "state")
    destination = tmp_path / "runtime.env"
    with store.transaction() as transaction:
        pending = _begin_promotion(
            transaction,
            _promotion_environment(tmp_path),
            persistent_volumes=(),
        )
        transaction.install_environment(pending.intent.target, destination)
        assert destination.read_bytes() == TARGET_ENVIRONMENT_BYTES

        transaction.remove_environment(pending.intent.target, destination)
        assert not destination.exists()
        transaction.remove_environment(pending.intent.target, destination)
        transaction.record_abort(
            pending.transaction_id,
            persistent_volumes=(),
        )


def test_verify_installed_environment_is_exact_private_and_read_only(
    tmp_path: Path,
) -> None:
    """Prove exact bytes or absence without changing the runtime path."""
    store = DeploymentStateStore(tmp_path / "state")
    destination = tmp_path / "runtime.env"
    with store.transaction() as transaction:
        pending = _begin_promotion(
            transaction,
            _promotion_environment(tmp_path),
            persistent_volumes=(),
        )
        state = pending.intent.target
        assert (
            transaction.verify_installed_environment(
                state,
                destination,
                allow_missing=True,
            )
            is False
        )
        with pytest.raises(DeploymentStateError, match="environment is missing"):
            transaction.verify_installed_environment(state, destination)

        transaction.install_environment(state, destination)
        before = destination.stat()
        assert transaction.verify_installed_environment(state, destination) is True
        assert destination.stat() == before

        destination.write_bytes(b"TAMPERED=secret\n")
        destination.chmod(0o600)
        with pytest.raises(DeploymentStateError, match="does not match"):
            transaction.verify_installed_environment(state, destination)
        assert destination.read_bytes() == b"TAMPERED=secret\n"

        destination.write_bytes(TARGET_ENVIRONMENT_BYTES)
        destination.chmod(0o600)
        transaction.remove_environment(state, destination)
        transaction.record_abort(
            pending.transaction_id,
            persistent_volumes=(),
        )


def test_remove_environment_rejects_tampered_or_unsafe_destination(
    tmp_path: Path,
) -> None:
    """Never unlink bytes or pathnames that are not the exact recorded environment."""
    store = DeploymentStateStore(tmp_path / "state")
    destination = tmp_path / "runtime.env"
    with store.transaction() as transaction:
        pending = _begin_promotion(
            transaction,
            _promotion_environment(tmp_path),
            persistent_volumes=(),
        )
        state = pending.intent.target
        transaction.install_environment(state, destination)
        destination.write_bytes(b"DIFFERENT=secret\n")
        destination.chmod(0o600)
        with pytest.raises(DeploymentStateError, match="does not match"):
            transaction.remove_environment(state, destination)
        assert destination.exists()

        destination.write_bytes(TARGET_ENVIRONMENT_BYTES)
        destination.chmod(0o644)
        with pytest.raises(DeploymentStateError, match="unsafe permissions"):
            transaction.remove_environment(state, destination)
        assert destination.exists()
        destination.chmod(0o600)
        transaction.remove_environment(state, destination)
        transaction.record_abort(
            pending.transaction_id,
            persistent_volumes=(),
        )


def test_remove_environment_detects_path_change_before_unlink(
    tmp_path: Path,
) -> None:
    """Revalidate the exact inode metadata immediately before deletion."""
    store = DeploymentStateStore(tmp_path / "state")
    destination = tmp_path / "runtime.env"
    real_lstat = Path.lstat
    destination_lstats = 0

    with store.transaction() as transaction:
        pending = _begin_promotion(
            transaction,
            _promotion_environment(tmp_path),
            persistent_volumes=(),
        )
        state = pending.intent.target
        transaction.install_environment(state, destination)

        def change_before_final_lstat(selected: Path) -> os.stat_result:
            nonlocal destination_lstats
            if selected == destination:
                destination_lstats += 1
                if destination_lstats == 4:
                    destination.write_bytes(b"CHANGED=after-read\n")
                    destination.chmod(0o600)
            return real_lstat(selected)

        lstat_boundary = create_autospec(
            Path.lstat,
            spec_set=True,
            side_effect=change_before_final_lstat,
        )
        with (
            patch("pathlib.Path.lstat", new=lstat_boundary),
            pytest.raises(DeploymentStateError, match="changed before removal"),
        ):
            transaction.remove_environment(state, destination)

        assert destination.exists()
        destination.write_bytes(TARGET_ENVIRONMENT_BYTES)
        destination.chmod(0o600)
        transaction.remove_environment(state, destination)
        transaction.record_abort(
            pending.transaction_id,
            persistent_volumes=(),
        )


def test_remove_environment_fsync_failure_retries_as_absent(
    tmp_path: Path,
) -> None:
    """Make a crash after unlink safe for an idempotent abort retry."""
    store = DeploymentStateStore(tmp_path / "state")
    destination = tmp_path / "runtime.env"
    real_fsync = os.fsync
    with store.transaction() as transaction:
        pending = _begin_promotion(
            transaction,
            _promotion_environment(tmp_path),
            persistent_volumes=(),
        )
        state = pending.intent.target
        transaction.install_environment(state, destination)
        parent_identity = (
            destination.parent.stat().st_dev,
            destination.parent.stat().st_ino,
        )

        def reject_parent_sync(descriptor: int) -> None:
            metadata = os.fstat(descriptor)
            if (metadata.st_dev, metadata.st_ino) == parent_identity:
                raise OSError("synthetic removal directory fsync failure")
            real_fsync(descriptor)

        sync_boundary = create_autospec(
            os.fsync,
            spec_set=True,
            side_effect=reject_parent_sync,
        )
        with (
            patch("agent.deployment_state.os.fsync", new=sync_boundary),
            pytest.raises(OSError, match="removal directory fsync"),
        ):
            transaction.remove_environment(state, destination)

        assert not destination.exists()
        transaction.remove_environment(state, destination)
        transaction.record_abort(
            pending.transaction_id,
            persistent_volumes=(),
        )


@pytest.mark.parametrize("tamper", ["mismatch", "duplicate", "unknown-field"])
def test_terminal_replay_rejects_tampered_volume_observation(
    tmp_path: Path,
    tamper: str,
) -> None:
    """Revalidate terminal volume evidence instead of trusting canonical JSON."""
    store = DeploymentStateStore(tmp_path / tamper / "state")
    _adopt(store, _environment(tmp_path))
    with store.transaction() as transaction:
        pending = _begin_promotion(transaction, _promotion_environment(tmp_path))
        transaction.record_rollback(
            pending.transaction_id,
            persistent_volumes=PERSISTENT_VOLUMES,
        )

    terminal = _read_document(_journal_path(store, 2))
    volumes = terminal["persistent_volumes"]
    assert isinstance(volumes, list)
    first = volumes[0]
    assert isinstance(first, dict)
    if tamper == "mismatch":
        first["created_at"] = "2026-07-28T10:11:12Z"
        message = "volume identity"
    elif tamper == "duplicate":
        volumes.append(dict(first))
        message = "volume identities"
    else:
        first["unexpected"] = "value"
        message = "volume identity schema"
    _write_document(_journal_path(store, 2), terminal)

    with pytest.raises(DeploymentStateError, match=message):
        store.read_journal()


@pytest.mark.parametrize(
    ("tamper", "message"),
    [
        ("target-schema", "state schema"),
        ("compose-project", "Compose identity changed"),
        ("intent-timestamp", "timestamps"),
        ("candidate-baseline", "receipt baseline"),
    ],
)
def test_unresolved_intent_revalidates_nested_contracts(
    tmp_path: Path,
    tamper: str,
    message: str,
) -> None:
    """Reject a hash-consistent pointer when nested intent evidence is altered."""
    store = DeploymentStateStore(tmp_path / tamper / "state")
    _adopt(store, _environment(tmp_path))
    with store.transaction() as transaction:
        pending = _begin_promotion(transaction, _promotion_environment(tmp_path))

    intent_path = store.path / pending.intent_path
    intent = _read_document(intent_path)
    target = intent["target"]
    candidate = intent["candidate"]
    assert isinstance(target, dict)
    assert isinstance(candidate, dict)
    if tamper == "target-schema":
        target["schema_version"] = 1
    elif tamper == "compose-project":
        target["compose_project"] = "other-production"
    elif tamper == "intent-timestamp":
        intent["recorded_at"] = "2026-07-28T12:35:57.123456Z"
    else:
        candidate["baseline_journal_sha256"] = "0" * 64
    _write_document(intent_path, intent)
    intent_sha256 = hashlib.sha256(intent_path.read_bytes()).hexdigest()
    pointer = _read_document(store.pending_path)
    pointer["intent_sha256"] = intent_sha256
    _write_document(store.pending_path, pointer)

    with pytest.raises(DeploymentStateError, match=message):
        store.read_current()


def test_promotion_snapshot_tampering_fails_before_pending_is_exposed(
    tmp_path: Path,
) -> None:
    """Bind every recovery intent to its exact private environment bytes."""
    store = DeploymentStateStore(tmp_path / "state")
    _adopt(store, _environment(tmp_path))
    with store.transaction() as transaction:
        pending = _begin_promotion(transaction, _promotion_environment(tmp_path))
    snapshot = store.path / pending.intent.target.environment_snapshot
    snapshot.write_bytes(b'API_KEY="tampered"\n')
    snapshot.chmod(0o600)

    with pytest.raises(DeploymentStateError, match="snapshot hash"):
        store.read_current()


@pytest.mark.parametrize(
    ("tamper", "message"),
    [
        ("schema", "pending promotion schema"),
        ("path", "intent path"),
        ("hash", "intent hash"),
        ("missing-intent", "intent is missing"),
    ],
)
def test_pending_pointer_tampering_fails_closed(
    tmp_path: Path,
    tamper: str,
    message: str,
) -> None:
    """Require the fixed pointer to select one exact immutable intent."""
    store = DeploymentStateStore(tmp_path / tamper / "state")
    _adopt(store, _environment(tmp_path))
    with store.transaction() as transaction:
        pending = _begin_promotion(transaction, _promotion_environment(tmp_path))
    pointer = _read_document(store.pending_path)
    if tamper == "schema":
        pointer["schema_version"] = 1
        _write_document(store.pending_path, pointer)
    elif tamper == "path":
        pointer["intent_path"] = "transactions/other.json"
        _write_document(store.pending_path, pointer)
    elif tamper == "hash":
        pointer["intent_sha256"] = "0" * 64
        _write_document(store.pending_path, pointer)
    else:
        (store.path / pending.intent_path).unlink()

    with pytest.raises(DeploymentStateError, match=message):
        store.read_current()


def test_v2_nested_parsers_reject_invalid_exact_schemas(
    tmp_path: Path,
) -> None:
    """Exercise every closed v2 schema at its own validation boundary."""
    store = DeploymentStateStore(tmp_path / "state")
    with store.transaction() as transaction:
        pending = _begin_promotion(
            transaction,
            _promotion_environment(tmp_path),
            persistent_volumes=(),
        )
        base = pending.intent.as_document()

        with pytest.raises(DeploymentStateError, match="created_at"):
            state_module._validate_volume_created_at("not-rfc3339")
        with pytest.raises(DeploymentStateError, match="identity"):
            state_module._validated_optional_identity(
                {
                    "sequence": True,
                    "sha256": "0" * 64,
                },
                "sequence",
                "sha256",
            )
        with pytest.raises(DeploymentStateError, match="must be an object"):
            state_module._candidate_from_document(None)
        raw_candidate_schema = base["candidate"]
        assert isinstance(raw_candidate_schema, dict)
        candidate_schema = dict(raw_candidate_schema)
        candidate_schema["unknown"] = True
        with pytest.raises(DeploymentStateError, match="receipt schema"):
            state_module._candidate_from_document(candidate_schema)
        with pytest.raises(DeploymentStateError, match="volume identities"):
            state_module._validated_volume_identities((object(),))  # type: ignore[arg-type]
        with pytest.raises(DeploymentStateError, match="volume identities"):
            state_module._validated_volume_identities(
                (PERSISTENT_VOLUMES[0], PERSISTENT_VOLUMES[0])
            )
        with pytest.raises(DeploymentStateError, match="volume identities"):
            state_module._volume_identities_from_document({})
        with pytest.raises(DeploymentStateError, match="must be an object"):
            state_module._intent_from_document(None)
        intent_schema = json.loads(json.dumps(base))
        intent_schema["unknown"] = True
        with pytest.raises(DeploymentStateError, match="intent schema"):
            state_module._intent_from_document(intent_schema)

        wrong_target_id = json.loads(json.dumps(base))
        target = wrong_target_id["target"]
        assert isinstance(target, dict)
        target["deployment_id"] = "promote-other123456789"
        target["environment_snapshot"] = "environments/promote-other123456789.env"
        with pytest.raises(DeploymentStateError, match="target identity"):
            state_module._intent_from_document(wrong_target_id)

        future_receipt = json.loads(json.dumps(base))
        candidate = future_receipt["candidate"]
        assert isinstance(candidate, dict)
        candidate["observed_at"] = "2026-07-28T12:35:57.123456Z"
        with pytest.raises(DeploymentStateError, match="timestamps"):
            state_module._intent_from_document(future_receipt)

        invalid_current_baseline = json.loads(json.dumps(base))
        invalid_current_baseline["baseline_current_sequence"] = 2
        invalid_current_baseline["baseline_current_sha256"] = "0" * 64
        with pytest.raises(DeploymentStateError, match="current baseline"):
            state_module._intent_from_document(invalid_current_baseline)

        wrong_receipt_baseline = json.loads(json.dumps(base))
        candidate = wrong_receipt_baseline["candidate"]
        assert isinstance(candidate, dict)
        candidate["baseline_journal_sequence"] = 1
        candidate["baseline_journal_sha256"] = "0" * 64
        with pytest.raises(DeploymentStateError, match="receipt baseline"):
            state_module._intent_from_document(wrong_receipt_baseline)

        wrong_receipt_target = json.loads(json.dumps(base))
        candidate = wrong_receipt_target["candidate"]
        assert isinstance(candidate, dict)
        candidate["compose_project"] = "adk-template"
        with pytest.raises(DeploymentStateError, match="receipt target"):
            state_module._intent_from_document(wrong_receipt_target)

        current_document = {
            "schema_version": 2,
            "journal_sequence": 1,
            "journal_sha256": "0" * 64,
            "state": pending.intent.target.as_document(),
            "unknown": True,
        }
        with pytest.raises(DeploymentStateError, match="current deployment schema"):
            state_module._current_from_document(current_document)
        with pytest.raises(DeploymentStateError, match="must be an object"):
            state_module._pending_from_document(None, intents={})

        transaction.record_abort(
            pending.transaction_id,
            persistent_volumes=(),
        )


@pytest.mark.parametrize("unsafe_kind", ["mode", "unknown-path"])
def test_transaction_directory_rejects_unsafe_contents(
    tmp_path: Path,
    unsafe_kind: str,
) -> None:
    """Fail closed on a mutable or ambiguous immutable-intent namespace."""
    store = DeploymentStateStore(tmp_path / unsafe_kind / "state")
    with store.transaction() as transaction:
        _begin_promotion(
            transaction,
            _promotion_environment(tmp_path),
            persistent_volumes=(),
        )
    if unsafe_kind == "mode":
        store.transactions_path.chmod(0o755)
        message = "unsafe permissions"
    else:
        _write_private(store.transactions_path / "unknown.txt", b"unknown")
        message = "unknown path"

    with pytest.raises(DeploymentStateError, match=message):
        store.read_current()


def test_intent_filename_must_match_embedded_transaction_identity(
    tmp_path: Path,
) -> None:
    """Reject a valid intent copied under a different immutable identity."""
    store = DeploymentStateStore(tmp_path / "state")
    with store.transaction() as transaction:
        pending = _begin_promotion(
            transaction,
            _promotion_environment(tmp_path),
            persistent_volumes=(),
        )
    original = store.path / pending.intent_path
    renamed = store.transactions_path / "promote-renamed123456.json"
    original.rename(renamed)

    with pytest.raises(DeploymentStateError, match="filename and identity"):
        store.read_current()


@pytest.mark.parametrize(
    ("tamper", "message"),
    [
        ("intent-hash", "intent hash"),
        ("journal-baseline", "journal baseline"),
        ("current-baseline", "current baseline"),
        ("compose", "Compose identity"),
        ("outcome-state", "outcome state"),
    ],
)
def test_terminal_replay_rejects_transition_tampering(
    tmp_path: Path,
    tamper: str,
    message: str,
) -> None:
    """Recompute attacker-controlled hashes and still reject invalid transitions."""
    store = DeploymentStateStore(tmp_path / tamper / "state")
    _adopt(store, _environment(tmp_path))
    with store.transaction() as transaction:
        pending = _begin_promotion(transaction, _promotion_environment(tmp_path))
        transaction.record_rollback(
            pending.transaction_id,
            persistent_volumes=PERSISTENT_VOLUMES,
        )

    intent_path = store.path / pending.intent_path
    intent = _read_document(intent_path)
    terminal = _read_document(_journal_path(store, 2))
    if tamper == "intent-hash":
        terminal["intent_sha256"] = "0" * 64
    elif tamper == "journal-baseline":
        intent["baseline_journal_sequence"] = 2
        intent["baseline_journal_sha256"] = "0" * 64
        candidate = intent["candidate"]
        assert isinstance(candidate, dict)
        candidate["baseline_journal_sequence"] = 2
        candidate["baseline_journal_sha256"] = "0" * 64
    elif tamper == "current-baseline":
        intent["baseline_current_sha256"] = "0" * 64
    elif tamper == "compose":
        target = intent["target"]
        assert isinstance(target, dict)
        target["compose_project"] = "other-production"
    else:
        terminal["state"] = intent["target"]
    if tamper not in {"intent-hash", "outcome-state"}:
        _write_document(intent_path, intent)
        terminal["intent_sha256"] = hashlib.sha256(intent_path.read_bytes()).hexdigest()
    _write_document(_journal_path(store, 2), terminal)

    with pytest.raises(DeploymentStateError, match=message):
        store.read_journal()


def test_terminal_replay_requires_referenced_intent(
    tmp_path: Path,
) -> None:
    """Keep terminal history inseparable from its immutable recovery contract."""
    store = DeploymentStateStore(tmp_path / "state")
    _adopt(store, _environment(tmp_path))
    with store.transaction() as transaction:
        pending = _begin_promotion(transaction, _promotion_environment(tmp_path))
        transaction.record_abort(
            pending.transaction_id,
            persistent_volumes=PERSISTENT_VOLUMES,
        )
    (store.path / pending.intent_path).unlink()

    with pytest.raises(DeploymentStateError, match="intent is missing"):
        store.read_journal()


def test_duplicate_terminal_outcomes_are_rejected(
    tmp_path: Path,
) -> None:
    """Allow exactly one terminal event for each immutable intent hash."""
    store = DeploymentStateStore(tmp_path / "state")
    baseline = _adopt(store, _environment(tmp_path))
    with store.transaction() as transaction:
        pending = _begin_promotion(transaction, _promotion_environment(tmp_path))
        transaction.record_abort(
            pending.transaction_id,
            persistent_volumes=PERSISTENT_VOLUMES,
        )
    second_payload = _journal_path(store, 2).read_bytes()
    duplicate = {
        "schema_version": 2,
        "sequence": 3,
        "previous_sha256": hashlib.sha256(second_payload).hexdigest(),
        "event": "rolled_back",
        "transaction_id": pending.transaction_id,
        "intent_sha256": pending.intent_sha256,
        "state": baseline.state.as_document(),
        "persistent_volumes": [volume.as_document() for volume in PERSISTENT_VOLUMES],
    }
    _write_document(_journal_path(store, 3), duplicate)

    with pytest.raises(DeploymentStateError, match="duplicate outcomes"):
        store.read_journal()


def test_pending_baselines_must_match_replayed_state(
    tmp_path: Path,
) -> None:
    """Reject unresolved work when either journal or current moved underneath it."""
    journal_store = DeploymentStateStore(tmp_path / "journal" / "state")
    _adopt(journal_store, _environment(tmp_path))
    with journal_store.transaction() as transaction:
        _begin_promotion(transaction, _promotion_environment(tmp_path))
    stale_pending = journal_store.pending_path.read_bytes()
    journal_store.pending_path.unlink()
    with journal_store.transaction() as transaction:
        later = _begin_promotion(
            transaction,
            _write_private(
                tmp_path / "journal-second.env",
                TARGET_ENVIRONMENT_BYTES,
            ),
            transaction_id="promote-journal-second",
        )
        transaction.record_abort(
            later.transaction_id,
            persistent_volumes=PERSISTENT_VOLUMES,
        )
    _write_private(journal_store.pending_path, stale_pending)
    with pytest.raises(DeploymentStateError, match="journal baseline is stale"):
        journal_store.read_current()

    current_store = DeploymentStateStore(tmp_path / "current" / "state")
    _adopt(current_store, _environment(tmp_path))
    with current_store.transaction() as transaction:
        pending = _begin_promotion(transaction, _promotion_environment(tmp_path))
    intent_path = current_store.path / pending.intent_path
    intent = _read_document(intent_path)
    intent["baseline_current_sha256"] = "0" * 64
    _write_document(intent_path, intent)
    pointer = _read_document(current_store.pending_path)
    pointer["intent_sha256"] = hashlib.sha256(intent_path.read_bytes()).hexdigest()
    _write_document(current_store.pending_path, pointer)
    with pytest.raises(DeploymentStateError, match="current baseline is stale"):
        current_store.read_current()


def test_current_pointer_cannot_select_nonestablishing_terminal(
    tmp_path: Path,
) -> None:
    """Restrict current to adopted/promoted records even before final reconciliation."""
    store = DeploymentStateStore(tmp_path / "state")
    baseline = _adopt(store, _environment(tmp_path))
    with store.transaction() as transaction:
        first = _begin_promotion(transaction, _promotion_environment(tmp_path))
        transaction.record_rollback(
            first.transaction_id,
            persistent_volumes=PERSISTENT_VOLUMES,
        )
    with store.transaction() as transaction:
        second = _begin_promotion(
            transaction,
            _write_private(tmp_path / "other.env", b"OTHER=secret\n"),
            transaction_id="promote-second123456789",
        )
        promoted = transaction.commit_promotion(
            second.transaction_id,
            persistent_volumes=PERSISTENT_VOLUMES,
        )
    rollback = store.read_journal()[1]
    bad_current = CurrentDeployment(
        journal_sequence=rollback.sequence,
        journal_sha256=rollback.sha256,
        state=baseline.state,
    )
    _write_document(store.current_path, bad_current.as_document())
    assert promoted.journal_sequence == 3

    with pytest.raises(DeploymentStateError, match="does not establish state"):
        store.read_current()


def test_environment_apis_reject_unrecorded_tampered_and_unsafe_targets(
    tmp_path: Path,
) -> None:
    """Fail closed around recovery byte selection and destination ownership."""
    store = DeploymentStateStore(tmp_path / "state")
    baseline = _adopt(store, _environment(tmp_path))
    with store.transaction() as transaction:
        assert transaction.read_environment(baseline.state) == ENVIRONMENT_BYTES
        unrecorded = state_module.DeploymentState(
            deployment_id="unrecorded-123",
            recorded_at=RECORDED_AT,
            compose_project="adk-template",
            compose_service="agent",
            source_revision=REVISION,
            image_reference=IMAGE_REFERENCE,
            image_id=IMAGE_ID,
            oci_revision=OCI_REVISION,
            environment_snapshot="environments/unrecorded-123.env",
            environment_sha256="0" * 64,
            adopted=True,
        )
        with pytest.raises(DeploymentStateError, match="is not recorded"):
            transaction.read_environment(unrecorded)
        with pytest.raises(DeploymentStateError, match="absolute and normalized"):
            transaction.install_environment(baseline.state, Path("relative.env"))
        with pytest.raises(DeploymentStateError, match="parent is unavailable"):
            transaction.install_environment(
                baseline.state,
                tmp_path / "missing-parent" / "runtime.env",
            )
        unsafe_parent = tmp_path / "unsafe-parent"
        unsafe_parent.mkdir(mode=0o700)
        unsafe_parent.chmod(0o777)
        with pytest.raises(DeploymentStateError, match="parent is unsafe"):
            transaction.install_environment(
                baseline.state,
                unsafe_parent / "runtime.env",
            )
        destination = _write_private(tmp_path / "existing.env", b"old")
        transaction.install_environment(baseline.state, destination)
        assert destination.read_bytes() == ENVIRONMENT_BYTES

        snapshot = store.path / baseline.state.environment_snapshot
        snapshot.write_bytes(b"TAMPERED=secret\n")
        snapshot.chmod(0o600)
        with pytest.raises(DeploymentStateError, match="snapshot hash"):
            transaction.read_environment(baseline.state)


def test_install_environment_verifies_post_replace_bytes(
    tmp_path: Path,
) -> None:
    """Detect corruption between atomic replacement and independent verification."""
    store = DeploymentStateStore(tmp_path / "state")
    baseline = _adopt(store, _environment(tmp_path))
    destination = tmp_path / "runtime.env"
    real_replace = Path.replace

    def corrupt_after_replace(source: Path, target: Path) -> Path:
        replaced = real_replace(source, target)
        if target == destination:
            target.write_bytes(b"CORRUPTED=secret\n")
            target.chmod(0o600)
        return replaced

    replace_boundary = create_autospec(
        Path.replace,
        spec_set=True,
        side_effect=corrupt_after_replace,
    )
    with (
        store.transaction() as transaction,
        patch("pathlib.Path.replace", new=replace_boundary),
        pytest.raises(DeploymentStateError, match="does not match"),
    ):
        transaction.install_environment(baseline.state, destination)


@pytest.mark.parametrize(
    ("candidate_overrides", "begin_overrides", "message"),
    [
        (
            {"baseline_journal_sha256": "0" * 64},
            {},
            "receipt baseline",
        ),
        (
            {"compose_service": "other"},
            {},
            "receipt target",
        ),
        (
            {},
            {"compose_project": "other-production"},
            "Compose identity",
        ),
    ],
)
def test_begin_rejects_runtime_identity_mismatches(
    tmp_path: Path,
    candidate_overrides: dict[str, object],
    begin_overrides: dict[str, object],
    message: str,
) -> None:
    """Validate baseline, candidate, and production identities before snapshotting."""
    store = DeploymentStateStore(tmp_path / message.replace(" ", "-") / "state")
    _adopt(store, _environment(tmp_path))
    with store.transaction() as transaction:
        arguments: dict[str, object] = {
            "compose_project": "adk-template",
            "compose_service": "agent",
            "source_revision": TARGET_REVISION,
            "image_reference": TARGET_IMAGE_REFERENCE,
            "image_id": TARGET_IMAGE_ID,
            "oci_revision": TARGET_REVISION,
            "environment_source": _promotion_environment(tmp_path),
            "candidate": _candidate(transaction, **candidate_overrides),
            "persistent_volumes": PERSISTENT_VOLUMES,
            "transaction_id": TRANSACTION_ID,
            "recorded_at": PROMOTION_RECORDED_AT,
        }
        arguments.update(begin_overrides)
        with pytest.raises(DeploymentStateError, match=message):
            transaction.begin_promotion(**arguments)  # type: ignore[arg-type]

    assert not list(store.environments_path.glob(f"{TRANSACTION_ID}.env"))


def test_oversize_intent_is_rejected_before_any_publication(
    tmp_path: Path,
) -> None:
    """Bound the durable document even with the maximum volume count."""
    store = DeploymentStateStore(tmp_path / "state")
    _adopt(store, _environment(tmp_path))
    long_component = "x" * 3900
    volumes = tuple(
        PersistentVolumeIdentity(
            name=f"volume-{index}",
            driver="local",
            mountpoint=f"/var/lib/{index}-{long_component}",
            destination=f"/app/{index}-{long_component}",
            created_at="2026-07-28T10:11:12Z",
        )
        for index in range(20)
    )

    with (
        store.transaction() as transaction,
        pytest.raises(DeploymentStateError, match="size limit"),
    ):
        _begin_promotion(
            transaction,
            _promotion_environment(tmp_path),
            persistent_volumes=volumes,
        )

    assert not store.transactions_path.exists()
    assert not (store.environments_path / f"{TRANSACTION_ID}.env").exists()
    assert not store.pending_path.exists()


def test_orphan_intent_identity_cannot_be_reused(
    tmp_path: Path,
) -> None:
    """Retain pre-mutation residue while forbidding transaction-ID reuse."""
    store = DeploymentStateStore(tmp_path / "state")
    _adopt(store, _environment(tmp_path))
    real_link = os.link

    def reject_pending_link(
        source_path: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        destination_path: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        *,
        src_dir_fd: int | None = None,
        dst_dir_fd: int | None = None,
        follow_symlinks: bool = True,
    ) -> None:
        if Path(os.fsdecode(destination_path)) == store.pending_path:
            raise OSError("synthetic pending link failure")
        real_link(
            source_path,
            destination_path,
            src_dir_fd=src_dir_fd,
            dst_dir_fd=dst_dir_fd,
            follow_symlinks=follow_symlinks,
        )

    with (
        store.transaction() as transaction,
        patch(
            "agent.deployment_state.os.link",
            new=create_autospec(
                os.link,
                spec_set=True,
                side_effect=reject_pending_link,
            ),
        ),
        pytest.raises(OSError, match="pending link"),
    ):
        _begin_promotion(transaction, _promotion_environment(tmp_path))

    with (
        store.transaction() as transaction,
        pytest.raises(DeploymentStateError, match="already exists"),
    ):
        _begin_promotion(transaction, _promotion_environment(tmp_path))


def test_terminal_methods_reject_invalid_or_missing_pending_identity(
    tmp_path: Path,
) -> None:
    """Require one exact loaded pending identity for every outcome."""
    empty_store = DeploymentStateStore(tmp_path / "empty" / "state")
    with (
        empty_store.transaction() as transaction,
        pytest.raises(DeploymentStateError, match="no promotion"),
    ):
        transaction.record_abort(
            TRANSACTION_ID,
            persistent_volumes=(),
        )

    store = DeploymentStateStore(tmp_path / "pending" / "state")
    with store.transaction() as transaction:
        pending = _begin_promotion(
            transaction,
            _promotion_environment(tmp_path),
            persistent_volumes=(),
        )
        with pytest.raises(DeploymentStateError, match="identity is invalid"):
            transaction.record_abort(
                "../invalid",
                persistent_volumes=(),
            )
        with pytest.raises(DeploymentStateError, match="does not match"):
            transaction.record_abort(
                "promote-other123456789",
                persistent_volumes=(),
            )
        with pytest.raises(DeploymentStateError, match="journal event"):
            transaction._publish_terminal(
                event="unknown",
                transaction_id=pending.transaction_id,
                persistent_volumes=(),
            )
        transaction.record_abort(
            pending.transaction_id,
            persistent_volumes=(),
        )


def test_indeterminate_terminal_when_visibility_check_itself_fails(
    tmp_path: Path,
) -> None:
    """Forbid compensation when the final pathname cannot be inspected."""
    store = DeploymentStateStore(tmp_path / "state")
    _adopt(store, _environment(tmp_path))
    real_link = os.link
    real_lstat = Path.lstat
    with store.transaction() as transaction:
        pending = _begin_promotion(transaction, _promotion_environment(tmp_path))
        terminal_path = _journal_path(store, 2)

        def reject_terminal_link(
            source_path: str | bytes | os.PathLike[str] | os.PathLike[bytes],
            destination_path: str | bytes | os.PathLike[str] | os.PathLike[bytes],
            *,
            src_dir_fd: int | None = None,
            dst_dir_fd: int | None = None,
            follow_symlinks: bool = True,
        ) -> None:
            if Path(os.fsdecode(destination_path)) == terminal_path:
                raise OSError("synthetic terminal link uncertainty")
            real_link(
                source_path,
                destination_path,
                src_dir_fd=src_dir_fd,
                dst_dir_fd=dst_dir_fd,
                follow_symlinks=follow_symlinks,
            )

        def reject_terminal_lstat(selected: Path) -> os.stat_result:
            if selected == terminal_path:
                raise PermissionError("synthetic terminal visibility uncertainty")
            return real_lstat(selected)

        with (
            patch(
                "agent.deployment_state.os.link",
                new=create_autospec(
                    os.link,
                    spec_set=True,
                    side_effect=reject_terminal_link,
                ),
            ),
            patch(
                "pathlib.Path.lstat",
                new=create_autospec(
                    Path.lstat,
                    spec_set=True,
                    side_effect=reject_terminal_lstat,
                ),
            ),
            pytest.raises(DeploymentTerminalIndeterminateError),
        ):
            transaction.commit_promotion(
                pending.transaction_id,
                persistent_volumes=PERSISTENT_VOLUMES,
            )


def test_internal_readers_preserve_unsafe_path_errors(
    tmp_path: Path,
) -> None:
    """Do not reinterpret present-but-unsafe intent or pending paths as absent."""
    store = DeploymentStateStore(tmp_path / "state")
    with store.transaction() as transaction:
        pending = _begin_promotion(
            transaction,
            _promotion_environment(tmp_path),
            persistent_volumes=(),
        )
        store.transactions_path.chmod(0o755)
        with pytest.raises(DeploymentStateError, match="unsafe permissions"):
            transaction._intent_files()
        store.transactions_path.chmod(0o700)

        store.pending_path.chmod(0o644)
        with pytest.raises(DeploymentStateError, match="unsafe permissions"):
            transaction._read_pending_file()
        store.pending_path.chmod(0o600)

        terminal = state_module.JournalEntry(
            sequence=1,
            sha256="0" * 64,
            previous_sha256=None,
            event="aborted",
            state=None,
            transaction_id=pending.transaction_id,
            intent_sha256=pending.intent_sha256,
            persistent_volumes=(),
        )
        original_journal = transaction._journal
        transaction._journal = (terminal, terminal)
        with pytest.raises(DeploymentStateError, match="duplicate terminal"):
            transaction._terminal_for_pending(pending)
        transaction._journal = original_journal
        transaction.record_abort(
            pending.transaction_id,
            persistent_volumes=(),
        )


def test_unclean_pending_terminal_must_remain_journal_tail(
    tmp_path: Path,
) -> None:
    """Reject later history after a terminal whose pending cleanup never completed."""
    store = DeploymentStateStore(tmp_path / "state")
    _adopt(store, _environment(tmp_path))
    real_unlink = Path.unlink

    def reject_pending_unlink(selected: Path, missing_ok: bool = False) -> None:
        if selected == store.pending_path:
            raise OSError("synthetic stale pending cleanup failure")
        real_unlink(selected, missing_ok=missing_ok)

    with store.transaction() as transaction:
        pending = _begin_promotion(transaction, _promotion_environment(tmp_path))
        with (
            patch(
                "pathlib.Path.unlink",
                new=create_autospec(
                    Path.unlink,
                    spec_set=True,
                    side_effect=reject_pending_unlink,
                ),
            ),
            pytest.raises(DeploymentTerminalCommittedError),
        ):
            transaction.record_rollback(
                pending.transaction_id,
                persistent_volumes=PERSISTENT_VOLUMES,
            )

    stale_pending = store.pending_path.read_bytes()
    store.pending_path.unlink()
    with store.transaction() as transaction:
        later = _begin_promotion(
            transaction,
            _write_private(
                tmp_path / "tail-second.env",
                TARGET_ENVIRONMENT_BYTES,
            ),
            transaction_id="promote-tail-second",
        )
        transaction.record_abort(
            later.transaction_id,
            persistent_volumes=PERSISTENT_VOLUMES,
        )
    _write_private(store.pending_path, stale_pending)

    with pytest.raises(DeploymentStateError, match="not the journal tail"):
        store.read_current()


def test_terminal_hard_link_recovery_fsync_failure_is_indeterminate(
    tmp_path: Path,
) -> None:
    """Preserve terminal uncertainty while cleaning a link-visible crash residue."""
    store = DeploymentStateStore(tmp_path / "state")
    with store.transaction() as transaction:
        pending = _begin_promotion(
            transaction,
            _promotion_environment(tmp_path),
            persistent_volumes=(),
        )
        transaction.commit_promotion(
            pending.transaction_id,
            persistent_volumes=PERSISTENT_VOLUMES,
        )
    terminal_path = _journal_path(store)
    temporary = store.journal_path / ".deployment-journal-terminal-crash.tmp"
    os.link(terminal_path, temporary)
    journal_identity = (
        store.journal_path.stat().st_dev,
        store.journal_path.stat().st_ino,
    )
    real_fsync = os.fsync

    def reject_journal_recovery_sync(descriptor: int) -> None:
        metadata = os.fstat(descriptor)
        if (metadata.st_dev, metadata.st_ino) == journal_identity:
            raise OSError("synthetic terminal recovery directory fsync failure")
        real_fsync(descriptor)

    sync_boundary = create_autospec(
        os.fsync,
        spec_set=True,
        side_effect=reject_journal_recovery_sync,
    )
    with (
        patch("agent.deployment_state.os.fsync", new=sync_boundary),
        pytest.raises(DeploymentTerminalIndeterminateError) as captured,
        store.transaction(),
    ):
        pytest.fail("terminal recovery unexpectedly completed")

    assert captured.value.event == "promoted"
    assert captured.value.transaction_id == TRANSACTION_ID
    assert not temporary.exists()
    assert store.read_current() is not None
