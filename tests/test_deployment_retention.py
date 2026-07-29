"""Bounded VM image retention with real deployment state."""

from __future__ import annotations

import json
import runpy
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from unittest.mock import create_autospec, patch

import pytest

from agent import deployment_retention as retention
from agent.deployment_retention import (
    ImageRetentionError,
    apply_image_retention,
    enforce_pull_admission,
    main,
    plan_image_retention,
)
from agent.deployment_state import (
    CandidateReceipt,
    DeploymentStateStore,
    DeploymentTerminalCommittedError,
)

REPOSITORY = "QueryPlanner/google-adk-on-bare-metal"
IMAGE_REPOSITORY = "ghcr.io/queryplanner/google-adk-on-bare-metal"
SOURCE = f"https://github.com/{REPOSITORY}"
ENVIRONMENT = {"PATH": "/usr/bin", "HOME": "/home/agent"}


def _reference(index: int) -> str:
    return f"{IMAGE_REPOSITORY}@sha256:{index:064x}"


def _image_id(index: int) -> str:
    return f"sha256:{index + 1000:064x}"


def _revision(index: int) -> str:
    return f"{index + 2000:040x}"


def _private_environment(tmp_path: Path, name: str) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / f"{name}.env"
    path.write_bytes(f'SECRET="{name}"\n'.encode())
    path.chmod(0o600)
    return path


def _managed_document(
    index: int,
    *,
    reference: str | None = None,
    image_id: str | None = None,
    repository_label: str = REPOSITORY,
    source: str = SOURCE,
    revision: str | None = None,
    extra_digests: tuple[str, ...] = (),
    tags: tuple[str, ...] = (),
) -> dict[str, object]:
    selected_reference = _reference(index) if reference is None else reference
    return {
        "Id": _image_id(index) if image_id is None else image_id,
        "RepoDigests": [selected_reference, *extra_digests],
        "RepoTags": list(tags),
        "Config": {
            "Labels": {
                "io.queryplanner.adk.repository": repository_label,
                "org.opencontainers.image.source": source,
                "org.opencontainers.image.revision": (
                    _revision(index) if revision is None else revision
                ),
            }
        },
    }


def _completed(
    command: list[str],
    *,
    returncode: int = 0,
    stdout: str = "",
    stderr: str = "",
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(
        command,
        returncode,
        stdout=stdout,
        stderr=stderr,
    )


@dataclass(slots=True)
class DockerBoundary:
    """Stateful Docker CLI boundary; application state and planning stay real."""

    images: dict[str, dict[str, object]]
    containers: dict[str, str] = field(default_factory=dict)
    calls: list[tuple[str, ...]] = field(default_factory=list)
    fail_remove_at: int | None = None
    retain_id_after_remove: bool = False
    remove_count: int = 0
    container_mounts: dict[str, list[object]] = field(default_factory=dict)
    container_host_mounts: dict[str, list[object] | None] = field(default_factory=dict)
    scripted: dict[
        tuple[str, ...],
        list[subprocess.CompletedProcess[str] | BaseException],
    ] = field(default_factory=dict)
    late_container: tuple[str, str] | None = None
    container_list_count: int = 0
    remove_images_before_list: dict[int, tuple[str, ...]] = field(default_factory=dict)
    image_list_count: int = 0

    def _image_for(self, identifier: str) -> dict[str, object] | None:
        selected = self.images.get(identifier)
        if selected is not None:
            return selected
        for document in self.images.values():
            digests = document.get("RepoDigests")
            tags = document.get("RepoTags")
            if (
                isinstance(digests, list)
                and identifier in digests
                or isinstance(tags, list)
                and identifier in tags
            ):
                return document
        return None

    def run(
        self,
        command: list[str],
        *,
        env: dict[str, str],
        text: bool,
        capture_output: bool,
        check: bool,
        timeout: int,
    ) -> subprocess.CompletedProcess[str]:
        """Return deterministic Docker observations and exact removals."""
        assert env["PATH"]
        assert text is True
        assert capture_output is True
        assert check is False
        assert timeout == 60
        selected = tuple(command)
        self.calls.append(selected)
        arguments = command[1:]
        scripted = self.scripted.get(tuple(arguments))
        if scripted:
            response = scripted.pop(0)
            if isinstance(response, BaseException):
                raise response
            return subprocess.CompletedProcess(
                command,
                response.returncode,
                stdout=response.stdout,
                stderr=response.stderr,
            )
        if arguments == ["image", "ls", "--all", "--no-trunc", "--quiet"]:
            self.image_list_count += 1
            for image_id in self.remove_images_before_list.get(
                self.image_list_count, ()
            ):
                self.images.pop(image_id, None)
            return _completed(
                command,
                stdout="".join(f"{image_id}\n" for image_id in sorted(self.images)),
            )
        if arguments == ["container", "ls", "--all", "--no-trunc", "--quiet"]:
            self.container_list_count += 1
            if self.container_list_count >= 2 and self.late_container is not None:
                container_id, image_id = self.late_container
                self.containers[container_id] = image_id
            return _completed(
                command,
                stdout="".join(
                    f"{container_id}\n" for container_id in sorted(self.containers)
                ),
            )
        if arguments[:2] == ["image", "inspect"] and len(arguments) == 3:
            document = self._image_for(arguments[2])
            if document is None:
                return _completed(
                    command,
                    returncode=1,
                    stdout="[]\n",
                    stderr=(
                        f"Error response from daemon: No such image: {arguments[2]}\n"
                    ),
                )
            return _completed(command, stdout=json.dumps([document]))
        if arguments[:2] == ["container", "inspect"] and len(arguments) == 3:
            container_id = arguments[2]
            container_image_id = self.containers.get(container_id)
            if container_image_id is None:
                return _completed(command, returncode=1, stderr="not found")
            return _completed(
                command,
                stdout=json.dumps(
                    [
                        {
                            "Id": container_id,
                            "Image": container_image_id,
                            "Mounts": self.container_mounts.get(container_id, []),
                            "HostConfig": {
                                "Mounts": self.container_host_mounts.get(container_id)
                            },
                        }
                    ]
                ),
            )
        if arguments[:3] == ["image", "rm", "--no-prune"] and len(arguments) == 4:
            self.remove_count += 1
            if self.fail_remove_at == self.remove_count:
                return _completed(command, returncode=1, stderr="injected")
            reference = arguments[3]
            document = self._image_for(reference)
            if document is None:
                return _completed(command, returncode=1, stderr="not found")
            removed_image_id = document["Id"]
            assert isinstance(removed_image_id, str)
            if self.retain_id_after_remove:
                document["RepoDigests"] = []
            else:
                self.images.pop(removed_image_id)
            return _completed(command, stdout=f"Untagged: {reference}\n")
        return _completed(command, returncode=99, stderr="unexpected")


def _docker(tmp_path: Path) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = (tmp_path / "docker").resolve()
    path.write_text("#!/bin/sh\n", encoding="utf-8")
    path.chmod(0o700)
    return path


def _mock_run(boundary: DockerBoundary) -> Any:
    return create_autospec(
        subprocess.run,
        spec_set=True,
        side_effect=boundary.run,
    )


def _adopt(
    store: DeploymentStateStore,
    tmp_path: Path,
    index: int,
) -> None:
    with store.transaction() as transaction:
        transaction.adopt(
            compose_project="adk-template",
            compose_service="agent",
            source_revision=_revision(index),
            image_reference=_reference(index),
            image_id=_image_id(index),
            oci_revision=_revision(index),
            environment_source=_private_environment(tmp_path, f"adopt-{index}"),
            deployment_id=f"adopt-{index}",
            recorded_at=f"2026-07-29T00:00:{index:02d}.000000Z",
        )


def _begin(
    store: DeploymentStateStore,
    tmp_path: Path,
    index: int,
) -> str:
    with store.transaction() as transaction:
        journal = transaction.journal()
        tail = None if not journal else journal[-1]
        sequence = len(journal) + 1
        transaction_id = f"promote-{index}-{sequence}"
        pending = transaction.begin_promotion(
            compose_project="adk-template",
            compose_service="agent",
            source_revision=_revision(index),
            image_reference=_reference(index),
            image_id=_image_id(index),
            oci_revision=_revision(index),
            environment_source=_private_environment(tmp_path, transaction_id),
            candidate=CandidateReceipt(
                observed_at=f"2026-07-29T00:00:{sequence:02d}.000000Z",
                compose_project=f"candidate-{index}-{sequence}",
                compose_service="agent",
                container_id=f"{index * 100 + sequence:064x}",
                image_reference=_reference(index),
                image_id=_image_id(index),
                oci_revision=_revision(index),
                baseline_journal_sequence=None if tail is None else tail.sequence,
                baseline_journal_sha256=None if tail is None else tail.sha256,
            ),
            persistent_volumes=(),
            transaction_id=transaction_id,
            recorded_at=f"2026-07-29T00:00:{sequence:02d}.000000Z",
        )
        return pending.transaction_id


def _terminal(
    store: DeploymentStateStore,
    transaction_id: str,
    event: str,
) -> None:
    with store.transaction() as transaction:
        method = {
            "promoted": transaction.commit_promotion,
            "rolled_back": transaction.record_rollback,
            "aborted": transaction.record_abort,
        }[event]
        method(transaction_id, persistent_volumes=())


def _promote(store: DeploymentStateStore, tmp_path: Path, index: int) -> None:
    _terminal(store, _begin(store, tmp_path, index), "promoted")


def _plan(
    store: DeploymentStateStore,
    docker: Path,
    boundary: DockerBoundary,
    target: str,
) -> retention.RetentionPlan:
    run = _mock_run(boundary)
    with (
        patch("agent.deployment_retention.subprocess.run", new=run),
        store.transaction() as transaction,
    ):
        return plan_image_retention(
            transaction,
            docker=docker,
            environment=ENVIRONMENT,
            repository=REPOSITORY,
            target_reference=target,
        )


def test_fresh_state_reserves_absent_target_and_orders_unrecorded_digests(
    tmp_path: Path,
) -> None:
    """Allow fresh state while reducing eight local references to seven."""
    store = DeploymentStateStore((tmp_path / "state").resolve())
    images = {_image_id(index): _managed_document(index) for index in range(1, 9)}
    boundary = DockerBoundary(images)

    plan = _plan(store, _docker(tmp_path), boundary, _reference(99))

    assert plan.reserved_references == 1
    assert plan.admitted_count == 9
    assert plan.delete_references == (_reference(1),)
    assert plan.protected_references == (_reference(99),)
    assert plan.missing_generation_references == ()
    assert plan.required_deletions == 1
    assert plan.admitted is False
    document = plan.as_document(status="dry-run")
    assert document["managed_reference_limit"] == 8
    assert document["maximum_deletions"] == 5
    decisions = document["reference_decisions"]
    assert isinstance(decisions, list)
    assert [decision["reference"] for decision in decisions] == [
        _reference(index) for index in range(1, 9)
    ]
    assert decisions[0] == {
        "reference": _reference(1),
        "action": "delete",
        "reasons": ["planned_unrecorded_deletion"],
        "image_id": _image_id(1),
    }
    assert decisions[1] == {
        "reference": _reference(2),
        "action": "keep",
        "reasons": ["available_capacity"],
        "image_id": _image_id(2),
    }


def test_present_exact_target_uses_all_eight_slots_and_admits(
    tmp_path: Path,
) -> None:
    """Count a fully proven local target instead of reserving another slot."""
    store = DeploymentStateStore((tmp_path / "state").resolve())
    boundary = DockerBoundary(
        {_image_id(index): _managed_document(index) for index in range(1, 9)}
    )
    docker = _docker(tmp_path)
    run = _mock_run(boundary)

    with (
        patch("agent.deployment_retention.subprocess.run", new=run),
        store.transaction() as transaction,
    ):
        plan = enforce_pull_admission(
            transaction,
            docker=docker,
            environment=ENVIRONMENT,
            repository=REPOSITORY,
            target_reference=_reference(8),
        )

    assert plan.target_present is True
    assert plan.reserved_references == 0
    assert plan.admitted is True
    assert plan.delete_references == ()
    target_decision = next(
        decision
        for decision in plan.reference_decisions
        if decision.reference == _reference(8)
    )
    assert target_decision.action == "keep"
    assert target_decision.reasons == ("requested_target",)


def test_generation_keep_set_ignores_rollback_abort_and_repeated_reference(
    tmp_path: Path,
) -> None:
    """Protect current plus two prior distinct establishing events only."""
    store = DeploymentStateStore((tmp_path / "state").resolve())
    _adopt(store, tmp_path, 1)
    _promote(store, tmp_path, 2)
    _promote(store, tmp_path, 3)
    _terminal(store, _begin(store, tmp_path, 4), "rolled_back")
    _terminal(store, _begin(store, tmp_path, 5), "aborted")
    _promote(store, tmp_path, 2)
    images = {_image_id(index): _managed_document(index) for index in range(1, 13)}

    plan = _plan(
        store,
        _docker(tmp_path),
        DockerBoundary(images),
        _reference(99),
    )

    assert set(plan.protected_references) == {
        _reference(2),
        _reference(3),
        _reference(1),
        _reference(99),
    }
    assert _reference(4) in plan.delete_references
    assert _reference(5) in plan.delete_references


def test_reference_decisions_explain_generations_containers_and_history(
    tmp_path: Path,
) -> None:
    """Emit ordered, exact reasons for every occupied project digest."""
    store = DeploymentStateStore((tmp_path / "state").resolve())
    for index in range(1, 9):
        if index == 1:
            _adopt(store, tmp_path, index)
        else:
            _promote(store, tmp_path, index)
    container_id = f"{123:064x}"
    boundary = DockerBoundary(
        {_image_id(index): _managed_document(index) for index in range(1, 9)},
        containers={container_id: _image_id(7)},
    )

    plan = _plan(store, _docker(tmp_path), boundary, _reference(99))

    assert [decision.reference for decision in plan.reference_decisions] == [
        _reference(index) for index in range(1, 9)
    ]
    decisions = {decision.reference: decision for decision in plan.reference_decisions}
    assert decisions[_reference(1)].action == "delete"
    assert decisions[_reference(1)].reasons == ("planned_historical_deletion",)
    assert decisions[_reference(1)].image_id == _image_id(1)
    assert decisions[_reference(2)].reasons == ("available_capacity",)
    assert decisions[_reference(6)].reasons == ("prior_generation",)
    assert decisions[_reference(7)].reasons == (
        "prior_generation",
        "container_use",
    )
    assert decisions[_reference(8)].reasons == ("current_generation",)


def test_missing_prior_is_reported_without_substituting_older_generation(
    tmp_path: Path,
) -> None:
    """Keep history selection stable when a recent prior is not local."""
    store = DeploymentStateStore((tmp_path / "state").resolve())
    for index in range(1, 5):
        if index == 1:
            _adopt(store, tmp_path, index)
        else:
            _promote(store, tmp_path, index)
    boundary = DockerBoundary(
        {_image_id(index): _managed_document(index) for index in (1, 2, 4, 6, 7, 8, 9)}
    )

    plan = _plan(store, _docker(tmp_path), boundary, _reference(9))

    assert plan.missing_generation_references == (_reference(3),)
    assert _reference(1) not in plan.protected_references
    assert _reference(2) in plan.protected_references
    assert _reference(3) in plan.protected_references
    assert _reference(4) in plan.protected_references


@pytest.mark.parametrize("event", ["rolled_back", "aborted"])
def test_fresh_terminal_without_current_allows_empty_generation_set(
    tmp_path: Path,
    event: str,
) -> None:
    """A verified fresh non-install outcome is valid pre-pull state."""
    store = DeploymentStateStore((tmp_path / event / "state").resolve())
    _terminal(store, _begin(store, tmp_path / event, 1), event)

    plan = _plan(
        store,
        _docker(tmp_path / event),
        DockerBoundary({}),
        _reference(2),
    )

    assert plan.protected_references == (_reference(2),)
    assert plan.managed_references == ()
    assert plan.admitted_count == 1


def test_pending_and_just_recovered_state_block_before_docker(
    tmp_path: Path,
) -> None:
    """Never inspect or mutate Docker across unresolved/reconciled recovery."""
    pending_store = DeploymentStateStore((tmp_path / "pending-state").resolve())
    _begin(pending_store, tmp_path, 1)
    docker = _docker(tmp_path)
    boundary = DockerBoundary({})
    run = _mock_run(boundary)
    with (
        patch("agent.deployment_retention.subprocess.run", new=run),
        pending_store.transaction() as transaction,
        pytest.raises(ImageRetentionError, match="pending recovery"),
    ):
        plan_image_retention(
            transaction,
            docker=docker,
            environment=ENVIRONMENT,
            repository=REPOSITORY,
            target_reference=_reference(2),
        )
    assert boundary.calls == []

    recovered_store = DeploymentStateStore((tmp_path / "recovered-state").resolve())
    _adopt(recovered_store, tmp_path, 2)
    transaction_id = _begin(recovered_store, tmp_path, 3)
    original_unlink = Path.unlink

    def fail_pending_unlink(path: Path, missing_ok: bool = False) -> None:
        if path == recovered_store.pending_path:
            raise OSError("injected")
        original_unlink(path, missing_ok=missing_ok)

    with (
        recovered_store.transaction() as transaction,
        patch.object(Path, "unlink", autospec=True, side_effect=fail_pending_unlink),
        pytest.raises(DeploymentTerminalCommittedError),
    ):
        transaction.record_abort(transaction_id, persistent_volumes=())
    with (
        patch("agent.deployment_retention.subprocess.run", new=run),
        recovered_store.transaction() as transaction,
        pytest.raises(ImageRetentionError, match="fresh recovery"),
    ):
        plan_image_retention(
            transaction,
            docker=docker,
            environment=ENVIRONMENT,
            repository=REPOSITORY,
            target_reference=_reference(4),
        )
    assert boundary.calls == []


def test_container_use_protects_stopped_or_running_image_and_can_block(
    tmp_path: Path,
) -> None:
    """Use inspected image IDs from every container as hard protections."""
    store = DeploymentStateStore((tmp_path / "state").resolve())
    images = {_image_id(index): _managed_document(index) for index in range(1, 10)}
    containers = {f"{index:064x}": _image_id(index) for index in range(1, 9)}
    boundary = DockerBoundary(images, containers=containers)

    with pytest.raises(ImageRetentionError, match="ceiling is unreachable"):
        _plan(store, _docker(tmp_path), boundary, _reference(99))

    assert not any(call[1:3] == ("image", "rm") for call in boundary.calls)


@pytest.mark.parametrize(
    "change",
    [
        {"repository_label": "Other/repo"},
        {"source": "https://github.com/Other/repo"},
        {"revision": "invalid"},
        {"extra_digests": ("ghcr.io/other/repo@sha256:" + "f" * 64,)},
        {"tags": ("ghcr.io/queryplanner/google-adk-on-bare-metal:latest",)},
    ],
)
def test_present_target_must_pass_full_managed_singleton_proof(
    tmp_path: Path,
    change: dict[str, object],
) -> None:
    """Reject present targets that are unlabeled, aliased, or malformed."""
    store = DeploymentStateStore((tmp_path / "state").resolve())
    document = _managed_document(1, **change)  # type: ignore[arg-type]
    boundary = DockerBoundary({_image_id(1): document})

    with pytest.raises(
        ImageRetentionError,
        match="target image is not exactly managed",
    ):
        _plan(store, _docker(tmp_path), boundary, _reference(1))


def test_recorded_identity_conflict_and_local_replacement_fail_closed(
    tmp_path: Path,
) -> None:
    """Reject one reference associated with conflicting durable or local IDs."""
    conflict_store = DeploymentStateStore((tmp_path / "conflict").resolve())
    _adopt(conflict_store, tmp_path, 1)
    with conflict_store.transaction() as transaction:
        journal = transaction.journal()
        tail = journal[-1]
        pending = transaction.begin_promotion(
            compose_project="adk-template",
            compose_service="agent",
            source_revision=_revision(9),
            image_reference=_reference(1),
            image_id=_image_id(9),
            oci_revision=_revision(9),
            environment_source=_private_environment(tmp_path, "conflict"),
            candidate=CandidateReceipt(
                observed_at="2026-07-29T00:00:09.000000Z",
                compose_project="candidate-conflict",
                compose_service="agent",
                container_id="9" * 64,
                image_reference=_reference(1),
                image_id=_image_id(9),
                oci_revision=_revision(9),
                baseline_journal_sequence=tail.sequence,
                baseline_journal_sha256=tail.sha256,
            ),
            persistent_volumes=(),
            transaction_id="promote-conflict",
            recorded_at="2026-07-29T00:00:09.000000Z",
        )
        transaction.commit_promotion(
            pending.transaction_id,
            persistent_volumes=(),
        )
    with pytest.raises(ImageRetentionError, match="generation identity conflicts"):
        _plan(
            conflict_store,
            _docker(tmp_path),
            DockerBoundary({_image_id(9): _managed_document(9)}),
            _reference(10),
        )

    replacement_store = DeploymentStateStore((tmp_path / "replacement").resolve())
    _adopt(replacement_store, tmp_path, 2)
    replacement = _managed_document(
        2,
        image_id=_image_id(8),
        revision=_revision(2),
    )
    with pytest.raises(ImageRetentionError, match="recorded generation"):
        _plan(
            replacement_store,
            _docker(tmp_path),
            DockerBoundary({_image_id(8): replacement}),
            _reference(10),
        )


def test_apply_deletes_unrecorded_before_oldest_success_and_caps_batch(
    tmp_path: Path,
) -> None:
    """Apply one deterministic five-reference batch and report incomplete."""
    store = DeploymentStateStore((tmp_path / "state").resolve())
    for index in range(1, 5):
        if index == 1:
            _adopt(store, tmp_path, index)
        else:
            _promote(store, tmp_path, index)
    images = {_image_id(index): _managed_document(index) for index in range(1, 15)}
    boundary = DockerBoundary(images)
    docker = _docker(tmp_path)
    run = _mock_run(boundary)

    with (
        patch("agent.deployment_retention.subprocess.run", new=run),
        store.transaction() as transaction,
    ):
        final, admitted, removed_decisions = apply_image_retention(
            transaction,
            docker=docker,
            environment=ENVIRONMENT,
            repository=REPOSITORY,
            target_reference=_reference(99),
        )

    removed = [
        call[-1]
        for call in boundary.calls
        if call[1:4] == ("image", "rm", "--no-prune")
    ]
    assert removed == [_reference(index) for index in range(5, 10)]
    assert [decision.reference for decision in removed_decisions] == removed
    assert all(decision.action == "delete" for decision in removed_decisions)
    assert all(
        decision.reasons == ("planned_unrecorded_deletion",)
        for decision in removed_decisions
    )
    assert admitted is False
    assert final.admitted_count == 10

    boundary.calls.clear()
    with (
        patch("agent.deployment_retention.subprocess.run", new=run),
        store.transaction() as transaction,
    ):
        final, admitted, removed_decisions = apply_image_retention(
            transaction,
            docker=docker,
            environment=ENVIRONMENT,
            repository=REPOSITORY,
            target_reference=_reference(99),
        )
    assert admitted is True
    assert final.admitted_count == 8
    assert [decision.reference for decision in removed_decisions] == [
        _reference(10),
        _reference(11),
    ]
    assert [call[-1] for call in boundary.calls if call[1:3] == ("image", "rm")] == [
        _reference(10),
        _reference(11),
    ]


def test_removal_failure_and_reference_only_untagging_stop_batch(
    tmp_path: Path,
) -> None:
    """Never count a failed removal or a remaining image ID as reclaimed."""
    store = DeploymentStateStore((tmp_path / "state").resolve())
    docker = _docker(tmp_path)
    images = {_image_id(index): _managed_document(index) for index in range(1, 10)}
    failure = DockerBoundary(dict(images), fail_remove_at=1)
    with (
        patch(
            "agent.deployment_retention.subprocess.run",
            new=_mock_run(failure),
        ),
        store.transaction() as transaction,
        pytest.raises(ImageRetentionError, match="command failed"),
    ):
        apply_image_retention(
            transaction,
            docker=docker,
            environment=ENVIRONMENT,
            repository=REPOSITORY,
            target_reference=_reference(99),
        )

    retained = DockerBoundary(dict(images), retain_id_after_remove=True)
    with (
        patch(
            "agent.deployment_retention.subprocess.run",
            new=_mock_run(retained),
        ),
        store.transaction() as transaction,
        pytest.raises(ImageRetentionError, match="remained after exact removal"),
    ):
        apply_image_retention(
            transaction,
            docker=docker,
            environment=ENVIRONMENT,
            repository=REPOSITORY,
            target_reference=_reference(99),
        )

    retained.calls.clear()
    with (
        patch(
            "agent.deployment_retention.subprocess.run",
            new=_mock_run(retained),
        ),
        store.transaction() as transaction,
        pytest.raises(ImageRetentionError, match="lacks a canonical digest"),
    ):
        apply_image_retention(
            transaction,
            docker=docker,
            environment=ENVIRONMENT,
            repository=REPOSITORY,
            target_reference=_reference(99),
        )
    assert not any(call[1:3] == ("image", "rm") for call in retained.calls)


@pytest.mark.parametrize(
    "unrelated",
    [
        _managed_document(
            1,
            reference="docker.io/library/alpine@sha256:" + "a" * 64,
            repository_label="Other/repository",
            source="https://github.com/Other/repository",
        ),
        {
            "Id": _image_id(1),
            "RepoDigests": [],
            "RepoTags": ["docker.io/library/alpine:latest"],
            "Config": {"Labels": None},
        },
    ],
)
def test_unrelated_unmanaged_images_do_not_block_admission(
    tmp_path: Path,
    unrelated: dict[str, object],
) -> None:
    """Ignore images that neither use nor claim the project repository."""
    store = DeploymentStateStore((tmp_path / "state").resolve())

    plan = _plan(
        store,
        _docker(tmp_path),
        DockerBoundary({_image_id(1): unrelated}),
        _reference(99),
    )

    assert plan.admitted is True
    assert plan.occupied_references == ()
    assert plan.managed_references == ()


@pytest.mark.parametrize(
    "labels",
    [
        None,
        {
            "io.queryplanner.adk.repository": "Other/repository",
            "org.opencontainers.image.source": "https://github.com/Other/repository",
            "org.opencontainers.image.revision": _revision(1),
        },
    ],
)
def test_project_tag_without_digest_blocks_regardless_of_labels(
    tmp_path: Path,
    labels: dict[str, str] | None,
) -> None:
    """Treat a tag-only project image as unreclaimed ownership evidence."""
    document: dict[str, object] = {
        "Id": _image_id(1),
        "RepoDigests": [],
        "RepoTags": [f"{IMAGE_REPOSITORY}:latest"],
        "Config": {"Labels": labels},
    }
    store = DeploymentStateStore((tmp_path / "state").resolve())

    with pytest.raises(ImageRetentionError, match="tag lacks a canonical digest"):
        _plan(
            store,
            _docker(tmp_path),
            DockerBoundary({_image_id(1): document}),
            _reference(99),
        )


def test_project_tag_with_digest_remains_ambiguous_and_protected(
    tmp_path: Path,
) -> None:
    """Retain a tagged canonical digest without treating it as exact-managed."""
    document = _managed_document(
        1,
        tags=(f"{IMAGE_REPOSITORY}:latest",),
    )
    store = DeploymentStateStore((tmp_path / "state").resolve())

    plan = _plan(
        store,
        _docker(tmp_path),
        DockerBoundary({_image_id(1): document}),
        _reference(99),
    )

    assert plan.ambiguous_references == (_reference(1),)
    assert _reference(1) in plan.protected_references
    assert plan.delete_references == ()


def test_partial_batch_recomputes_from_docker_without_state_mutation(
    tmp_path: Path,
) -> None:
    """Resume deterministically after deletion three fails in a five-item batch."""
    store = DeploymentStateStore((tmp_path / "state").resolve())
    for index in range(1, 5):
        if index == 1:
            _adopt(store, tmp_path, index)
        else:
            _promote(store, tmp_path, index)
    state_before = store.current_path.read_bytes()
    journal_before = {
        path.name: path.read_bytes() for path in sorted(store.journal_path.iterdir())
    }
    boundary = DockerBoundary(
        {_image_id(index): _managed_document(index) for index in range(1, 15)},
        fail_remove_at=3,
    )
    docker = _docker(tmp_path)
    run = _mock_run(boundary)

    with (
        patch("agent.deployment_retention.subprocess.run", new=run),
        store.transaction() as transaction,
        pytest.raises(ImageRetentionError, match="command failed"),
    ):
        apply_image_retention(
            transaction,
            docker=docker,
            environment=ENVIRONMENT,
            repository=REPOSITORY,
            target_reference=_reference(99),
        )

    first_calls = [
        call[-1]
        for call in boundary.calls
        if call[1:4] == ("image", "rm", "--no-prune")
    ]
    assert first_calls == [_reference(5), _reference(6), _reference(7)]
    assert _image_id(5) not in boundary.images
    assert _image_id(6) not in boundary.images
    assert _image_id(7) in boundary.images
    assert all(_image_id(index) in boundary.images for index in (2, 3, 4))

    boundary.fail_remove_at = None
    boundary.calls.clear()
    with (
        patch("agent.deployment_retention.subprocess.run", new=run),
        store.transaction() as transaction,
    ):
        final, admitted, removed_decisions = apply_image_retention(
            transaction,
            docker=docker,
            environment=ENVIRONMENT,
            repository=REPOSITORY,
            target_reference=_reference(99),
        )

    resumed_calls = [
        call[-1]
        for call in boundary.calls
        if call[1:4] == ("image", "rm", "--no-prune")
    ]
    assert resumed_calls == [
        _reference(7),
        _reference(8),
        _reference(9),
        _reference(10),
        _reference(11),
    ]
    assert [decision.reference for decision in removed_decisions] == resumed_calls
    assert admitted is True
    assert final.admitted_count == 8
    assert all(_image_id(index) in boundary.images for index in (2, 3, 4))
    assert store.current_path.read_bytes() == state_before
    assert {
        path.name: path.read_bytes() for path in sorted(store.journal_path.iterdir())
    } == journal_before


def test_admission_is_read_only_when_capacity_requires_cleanup(
    tmp_path: Path,
) -> None:
    """The in-lock controller guard never converts a plan into mutation."""
    store = DeploymentStateStore((tmp_path / "state").resolve())
    boundary = DockerBoundary(
        {_image_id(index): _managed_document(index) for index in range(1, 9)}
    )
    run = _mock_run(boundary)
    with (
        patch("agent.deployment_retention.subprocess.run", new=run),
        store.transaction() as transaction,
        pytest.raises(ImageRetentionError, match="capacity is not admitted"),
    ):
        enforce_pull_admission(
            transaction,
            docker=_docker(tmp_path),
            environment=ENVIRONMENT,
            repository=REPOSITORY,
            target_reference=_reference(99),
        )
    assert not any(call[1:3] == ("image", "rm") for call in boundary.calls)


def test_cli_dry_run_apply_incomplete_and_errors_are_secret_free(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Expose stable CLI modes and statuses without forwarding Docker stderr."""
    state_dir = (tmp_path / "state").resolve()
    images = {_image_id(index): _managed_document(index) for index in range(1, 15)}
    boundary = DockerBoundary(images)
    docker = _docker(tmp_path)
    arguments = [
        "enforce",
        "--state-dir",
        str(state_dir),
        "--repository",
        REPOSITORY,
        "--target-reference",
        _reference(99),
    ]
    run = _mock_run(boundary)
    secret_environment = {
        **ENVIRONMENT,
        "OPENROUTER_API_KEY": "SECRET-retention-value",
    }
    with (
        patch("agent.deployment_retention.subprocess.run", new=run),
        patch("agent.deployment_retention.shutil.which", return_value=str(docker)),
    ):
        assert main(arguments, environment=secret_environment) == 0
        dry_run = json.loads(capsys.readouterr().out)
        assert "SECRET-retention-value" not in json.dumps(dry_run)
        assert dry_run["status"] == "dry-run"
        assert "removed_reference_decisions" not in dry_run
        assert main([*arguments, "--apply"], environment=secret_environment) == 3
        incomplete = json.loads(capsys.readouterr().out)
        assert "SECRET-retention-value" not in json.dumps(incomplete)
        assert incomplete["status"] == "incomplete"
        assert incomplete["removed_reference_decisions"] == [
            {
                "reference": _reference(index),
                "action": "delete",
                "reasons": ["planned_unrecorded_deletion"],
                "image_id": _image_id(index),
            }
            for index in range(1, 6)
        ]

    failure = DockerBoundary(
        {_image_id(index): _managed_document(index) for index in range(1, 10)},
        fail_remove_at=1,
    )
    with (
        patch(
            "agent.deployment_retention.subprocess.run",
            new=_mock_run(failure),
        ),
        patch("agent.deployment_retention.shutil.which", return_value=str(docker)),
    ):
        assert main([*arguments, "--apply"], environment=secret_environment) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "injected" not in captured.err
    assert "SECRET" not in captured.err

    post_failure = DockerBoundary(
        {_image_id(index): _managed_document(index) for index in range(1, 10)},
        retain_id_after_remove=True,
    )
    with (
        patch(
            "agent.deployment_retention.subprocess.run",
            new=_mock_run(post_failure),
        ),
        patch("agent.deployment_retention.shutil.which", return_value=str(docker)),
    ):
        assert main([*arguments, "--apply"], environment=secret_environment) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "remained after exact removal" in captured.err

    ambiguous = DockerBoundary(
        {
            _image_id(index): _managed_document(
                index,
                repository_label="Other/repository",
            )
            for index in range(1, 9)
        }
    )
    with (
        patch(
            "agent.deployment_retention.subprocess.run",
            new=_mock_run(ambiguous),
        ),
        patch("agent.deployment_retention.shutil.which", return_value=str(docker)),
    ):
        assert main([*arguments, "--apply"], environment=secret_environment) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "ceiling is unreachable" in captured.err
    assert not any(call[1:3] == ("image", "rm") for call in ambiguous.calls)


def test_cli_apply_emits_one_verified_removal_decision(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Audit one successful exact removal separately from the final inventory."""
    state_dir = (tmp_path / "state").resolve()
    docker = _docker(tmp_path)
    boundary = DockerBoundary(
        {_image_id(index): _managed_document(index) for index in range(1, 9)}
    )
    arguments = [
        "enforce",
        "--state-dir",
        str(state_dir),
        "--repository",
        REPOSITORY,
        "--target-reference",
        _reference(99),
        "--apply",
    ]

    with (
        patch(
            "agent.deployment_retention.subprocess.run",
            new=_mock_run(boundary),
        ),
        patch("agent.deployment_retention.shutil.which", return_value=str(docker)),
    ):
        assert main(arguments, environment=ENVIRONMENT) == 0

    document = json.loads(capsys.readouterr().out)
    assert document["status"] == "admitted"
    assert document["removed_reference_decisions"] == [
        {
            "reference": _reference(1),
            "action": "delete",
            "reasons": ["planned_unrecorded_deletion"],
            "image_id": _image_id(1),
        }
    ]
    assert _reference(1) not in {
        decision["reference"] for decision in document["reference_decisions"]
    }


def test_ambiguous_project_references_occupy_capacity_and_are_never_deleted(
    tmp_path: Path,
) -> None:
    """Count alias/wrong-label references while excluding them from deletion."""
    store = DeploymentStateStore((tmp_path / "state").resolve())
    images = {
        _image_id(index): _managed_document(
            index,
            repository_label="Other/repository",
        )
        for index in range(1, 9)
    }
    boundary = DockerBoundary(images)
    docker = _docker(tmp_path)

    with (
        patch(
            "agent.deployment_retention.subprocess.run",
            new=_mock_run(boundary),
        ),
        store.transaction() as transaction,
        pytest.raises(ImageRetentionError, match="ceiling is unreachable"),
    ):
        apply_image_retention(
            transaction,
            docker=docker,
            environment=ENVIRONMENT,
            repository=REPOSITORY,
            target_reference=_reference(99),
        )

    assert not any(call[1:3] == ("image", "rm") for call in boundary.calls)

    images[_image_id(1)] = _managed_document(1)
    plan = _plan(store, docker, DockerBoundary(images), _reference(99))
    assert len(plan.occupied_references) == 8
    assert len(plan.ambiguous_references) == 7
    assert plan.delete_references == (_reference(1),)
    assert _reference(2) in plan.protected_references
    ambiguous = next(
        decision
        for decision in plan.reference_decisions
        if decision.reference == _reference(2)
    )
    assert ambiguous.action == "keep"
    assert ambiguous.reasons == ("ambiguous_ownership",)
    assert ambiguous.image_id is None
    assert ambiguous.as_document() == {
        "reference": _reference(2),
        "action": "keep",
        "reasons": ["ambiguous_ownership"],
    }


def test_malformed_project_digest_fails_closed(
    tmp_path: Path,
) -> None:
    """Reject a project-prefixed reference that is not one canonical digest."""
    store = DeploymentStateStore((tmp_path / "state").resolve())
    malformed = _managed_document(1, reference=f"{_reference(1)}suffix")
    with pytest.raises(ImageRetentionError, match="reference is malformed"):
        _plan(
            store,
            _docker(tmp_path),
            DockerBoundary({_image_id(1): malformed}),
            _reference(99),
        )


def test_internal_repository_override_supports_local_registry_proofs(
    tmp_path: Path,
) -> None:
    """Use a runtime-only repository override without broadening the CLI."""
    local_repository = "127.0.0.1:5000/queryplanner/google-adk-on-bare-metal"
    target = f"{local_repository}@sha256:{1:064x}"
    document = _managed_document(1, reference=target)
    store = DeploymentStateStore((tmp_path / "state").resolve())
    docker = _docker(tmp_path)
    boundary = DockerBoundary({_image_id(1): document})
    run = _mock_run(boundary)

    with (
        patch("agent.deployment_retention.subprocess.run", new=run),
        store.transaction() as transaction,
    ):
        plan = plan_image_retention(
            transaction,
            docker=docker,
            environment=ENVIRONMENT,
            repository=REPOSITORY,
            target_reference=target,
            image_repository=local_repository,
        )

    assert plan.target_present is True
    assert plan.occupied_references == (target,)


def test_image_mount_and_root_image_ids_are_protected(
    tmp_path: Path,
) -> None:
    """Resolve Moby's immutable mount Name, never its snapshot Source path."""
    store = DeploymentStateStore((tmp_path / "state").resolve())
    images = {_image_id(index): _managed_document(index) for index in range(1, 10)}
    container_id = f"{123:064x}"
    snapshot_path = "/var/lib/docker/image/mounts/snapshots/8/fs"
    boundary = DockerBoundary(
        images,
        containers={container_id: _image_id(9)},
        container_mounts={
            container_id: [
                {"Type": "bind", "Source": "/operator/data"},
                {
                    "Type": "image",
                    "Name": _reference(8),
                    "Source": snapshot_path,
                    "Destination": "/models",
                },
            ]
        },
        container_host_mounts={
            container_id: [
                {
                    "Type": "image",
                    "Source": _reference(8),
                    "Target": "/models",
                }
            ]
        },
    )

    plan = _plan(store, _docker(tmp_path), boundary, _reference(99))

    assert _reference(8) in plan.protected_references
    assert _reference(9) in plan.protected_references
    assert _reference(8) not in plan.delete_references
    assert _reference(9) not in plan.delete_references
    assert not any(
        call[1:] == ("image", "inspect", snapshot_path) for call in boundary.calls
    )


@pytest.mark.parametrize(
    ("mounts", "message"),
    [
        (["invalid"], "container mount is invalid"),
        (
            [{"Type": "image", "Source": "/var/lib/docker/snapshot"}],
            "selector is not an immutable digest",
        ),
        (
            [
                {
                    "Type": "image",
                    "Name": f"{IMAGE_REPOSITORY}:latest",
                    "Source": "/var/lib/docker/snapshot",
                }
            ],
            "selector is not an immutable digest",
        ),
        (
            [
                {
                    "Type": "image",
                    "Name": _image_id(2),
                    "Source": "/var/lib/docker/snapshot",
                }
            ],
            "selector is not an immutable digest",
        ),
        (
            [
                {
                    "Type": "image",
                    "Name": _reference(2),
                    "Source": "/var/lib/docker/snapshot",
                }
            ],
            "mount is unavailable",
        ),
    ],
)
def test_malformed_or_missing_image_mount_fails_closed(
    tmp_path: Path,
    mounts: list[object],
    message: str,
) -> None:
    """Reject image-mount observations that cannot prove an exact image ID."""
    store = DeploymentStateStore((tmp_path / "state").resolve())
    container_id = f"{123:064x}"
    boundary = DockerBoundary(
        {_image_id(1): _managed_document(1)},
        containers={container_id: _image_id(1)},
        container_mounts={container_id: mounts},
    )
    with pytest.raises(ImageRetentionError, match=message):
        _plan(store, _docker(tmp_path), boundary, _reference(99))


def test_malformed_container_identity_fails_closed(tmp_path: Path) -> None:
    """Reject a container inspection that does not match its listed ID."""
    store = DeploymentStateStore((tmp_path / "state").resolve())
    container_id = f"{123:064x}"
    malformed = {
        "Id": f"{124:064x}",
        "Image": _image_id(1),
        "Mounts": [],
    }
    boundary = DockerBoundary(
        {_image_id(1): _managed_document(1)},
        containers={container_id: _image_id(1)},
        scripted={
            ("container", "inspect", container_id): [
                _completed([], stdout=json.dumps([malformed]))
            ]
        },
    )

    with pytest.raises(ImageRetentionError, match="container identity is invalid"):
        _plan(store, _docker(tmp_path), boundary, _reference(99))


def test_mount_name_is_authoritative_over_legacy_host_config(
    tmp_path: Path,
) -> None:
    """Do not reject an immutable Moby Name because HostConfig is legacy."""
    store = DeploymentStateStore((tmp_path / "state").resolve())
    container_id = f"{123:064x}"
    boundary = DockerBoundary(
        {_image_id(1): _managed_document(1)},
        containers={container_id: _image_id(1)},
        container_mounts={
            container_id: [
                {
                    "Type": "image",
                    "Name": _reference(1),
                    "Source": "/var/lib/docker/image/mounts/snapshots/1/fs",
                }
            ]
        },
        container_host_mounts={
            container_id: [
                {
                    "Type": "image",
                    "Source": f"{IMAGE_REPOSITORY}:legacy",
                }
            ]
        },
    )

    plan = _plan(store, _docker(tmp_path), boundary, _reference(99))

    assert _reference(1) in plan.protected_references


def test_mount_selector_identity_drift_fails_closed(tmp_path: Path) -> None:
    """Require the selector inspection to match the inventoried image ID proof."""
    selector = "docker.io/library/alpine@sha256:" + "a" * 64
    observed: dict[str, object] = {
        "Id": _image_id(1),
        "RepoDigests": [selector],
        "RepoTags": [],
        "Config": {"Labels": None},
    }
    changed: dict[str, object] = {
        **observed,
        "Config": {"Labels": {"drift": "detected"}},
    }
    container_id = f"{123:064x}"
    boundary = DockerBoundary(
        {_image_id(1): observed},
        containers={container_id: _image_id(1)},
        container_mounts={
            container_id: [
                {
                    "Type": "image",
                    "Name": selector,
                    "Source": "/var/lib/docker/image/mounts/snapshots/1/fs",
                }
            ]
        },
        scripted={
            ("image", "inspect", selector): [
                _completed([], stdout=json.dumps([changed]))
            ]
        },
    )
    store = DeploymentStateStore((tmp_path / "state").resolve())

    with pytest.raises(ImageRetentionError, match="mount identity is ambiguous"):
        _plan(store, _docker(tmp_path), boundary, _reference(99))


def test_inventory_disappearance_drift_and_duplicate_ownership_fail_closed(
    tmp_path: Path,
) -> None:
    """Reject races between image listing, ID inspection, and reference proof."""
    store = DeploymentStateStore((tmp_path / "state").resolve())
    docker = _docker(tmp_path)
    absent = _completed(
        [],
        returncode=1,
        stdout="[]\n",
        stderr=f"Error response from daemon: No such image: {_image_id(1)}\n",
    )
    disappeared = DockerBoundary(
        {_image_id(1): _managed_document(1)},
        scripted={("image", "inspect", _image_id(1)): [absent]},
    )
    with pytest.raises(ImageRetentionError, match="inventory changed"):
        _plan(store, docker, disappeared, _reference(99))

    reference_absent = _completed(
        [],
        returncode=1,
        stdout="[]\n",
        stderr=f"Error response from daemon: No such image: {_reference(1)}\n",
    )
    missing_reference = DockerBoundary(
        {_image_id(1): _managed_document(1)},
        scripted={("image", "inspect", _reference(1)): [reference_absent]},
    )
    with pytest.raises(ImageRetentionError, match="reference disappeared"):
        _plan(store, docker, missing_reference, _reference(99))

    changed = _managed_document(1, image_id=_image_id(2))
    drift = DockerBoundary(
        {_image_id(1): _managed_document(1)},
        scripted={
            ("image", "inspect", _reference(1)): [
                _completed([], stdout=json.dumps([changed]))
            ]
        },
    )
    with pytest.raises(ImageRetentionError, match="identity changed"):
        _plan(store, docker, drift, _reference(99))

    duplicate_images = {
        _image_id(1): _managed_document(1),
        _image_id(2): _managed_document(
            2,
            reference=_reference(1),
            repository_label="Other/repository",
        ),
    }
    duplicate = _plan(
        store,
        docker,
        DockerBoundary(duplicate_images),
        _reference(99),
    )
    assert duplicate.ambiguous_references == (_reference(1),)
    assert duplicate.managed_references == ()

    second = _managed_document(
        2,
        reference=_reference(1),
        revision=_revision(2),
    )
    identity_swap = DockerBoundary(
        {
            _image_id(1): _managed_document(1),
            _image_id(2): second,
        },
        scripted={
            ("image", "inspect", _reference(1)): [
                _completed([], stdout=json.dumps([_managed_document(1)])),
                _completed([], stdout=json.dumps([second])),
            ]
        },
    )
    with pytest.raises(ImageRetentionError, match="reference is ambiguous"):
        _plan(store, docker, identity_swap, _reference(99))


def test_external_disappearance_before_replan_stops_without_removal(
    tmp_path: Path,
) -> None:
    """Stop when the last full inventory already satisfies seven plus one."""
    store = DeploymentStateStore((tmp_path / "state").resolve())
    images = {_image_id(index): _managed_document(index) for index in range(1, 9)}
    boundary = DockerBoundary(
        images,
        remove_images_before_list={2: (_image_id(1),)},
    )
    run = _mock_run(boundary)
    with (
        patch("agent.deployment_retention.subprocess.run", new=run),
        store.transaction() as transaction,
    ):
        final, admitted, removed_decisions = apply_image_retention(
            transaction,
            docker=_docker(tmp_path),
            environment=ENVIRONMENT,
            repository=REPOSITORY,
            target_reference=_reference(99),
        )
    assert admitted is True
    assert final.admitted_count == 8
    assert removed_decisions == ()
    assert not any(call[1:3] == ("image", "rm") for call in boundary.calls)


def test_replan_fails_closed_when_deterministic_order_changes(
    tmp_path: Path,
) -> None:
    """Never substitute a newly ordered candidate for the reviewed sequence."""
    store = DeploymentStateStore((tmp_path / "state").resolve())
    boundary = DockerBoundary(
        {_image_id(index): _managed_document(index) for index in range(1, 10)},
        remove_images_before_list={2: (_image_id(1),)},
    )
    run = _mock_run(boundary)
    with (
        patch("agent.deployment_retention.subprocess.run", new=run),
        store.transaction() as transaction,
        pytest.raises(ImageRetentionError, match="deletion order changed"),
    ):
        apply_image_retention(
            transaction,
            docker=_docker(tmp_path),
            environment=ENVIRONMENT,
            repository=REPOSITORY,
            target_reference=_reference(99),
        )

    removed = [
        call[-1]
        for call in boundary.calls
        if call[1:4] == ("image", "rm", "--no-prune")
    ]
    assert removed == []
    assert _image_id(2) in boundary.images


def test_replan_fails_closed_when_container_use_changes_order(tmp_path: Path) -> None:
    """Never replace a reviewed candidate after a container starts using it."""
    store = DeploymentStateStore((tmp_path / "state").resolve())
    boundary = DockerBoundary(
        {_image_id(index): _managed_document(index) for index in range(1, 9)},
        late_container=(f"{123:064x}", _image_id(1)),
    )
    run = _mock_run(boundary)

    with (
        patch("agent.deployment_retention.subprocess.run", new=run),
        store.transaction() as transaction,
        pytest.raises(ImageRetentionError, match="deletion order changed"),
    ):
        apply_image_retention(
            transaction,
            docker=_docker(tmp_path),
            environment=ENVIRONMENT,
            repository=REPOSITORY,
            target_reference=_reference(99),
        )

    removed = [
        call[-1]
        for call in boundary.calls
        if call[1:4] == ("image", "rm", "--no-prune")
    ]
    assert removed == []
    assert _image_id(1) in boundary.images
    assert _image_id(2) in boundary.images


def test_exact_identity_race_after_replan_blocks_removal(tmp_path: Path) -> None:
    """Retain the current candidate if its exact proof changes after planning."""
    store = DeploymentStateStore((tmp_path / "state").resolve())
    images = {_image_id(index): _managed_document(index) for index in range(1, 9)}
    changed = _managed_document(1, revision=_revision(99))
    boundary = DockerBoundary(
        images,
        scripted={
            ("image", "inspect", _image_id(1)): [
                _completed([], stdout=json.dumps([images[_image_id(1)]])),
                _completed([], stdout=json.dumps([changed])),
            ],
            ("image", "inspect", _reference(1)): [
                _completed([], stdout=json.dumps([images[_image_id(1)]])),
                _completed([], stdout=json.dumps([changed])),
            ],
        },
    )
    run = _mock_run(boundary)

    with (
        patch("agent.deployment_retention.subprocess.run", new=run),
        store.transaction() as transaction,
        pytest.raises(ImageRetentionError, match="changed before exact removal"),
    ):
        apply_image_retention(
            transaction,
            docker=_docker(tmp_path),
            environment=ENVIRONMENT,
            repository=REPOSITORY,
            target_reference=_reference(99),
        )

    assert not any(call[1:3] == ("image", "rm") for call in boundary.calls)


@pytest.mark.parametrize(
    ("repository", "target", "image_repository", "message"),
    [
        ("invalid", _reference(1), None, "GitHub repository"),
        (REPOSITORY, _reference(1), "INVALID", "image repository"),
        (REPOSITORY, "ghcr.io/example:latest", None, "target image"),
    ],
)
def test_identity_input_validation(
    repository: str,
    target: str,
    image_repository: str | None,
    message: str,
) -> None:
    """Reject identities before touching state or Docker."""
    with pytest.raises(ImageRetentionError, match=message):
        retention._validated_identity(repository, target, image_repository)


def test_docker_path_and_command_failures_are_bounded(
    tmp_path: Path,
) -> None:
    """Map path, timeout, spawn, exit, and output failures to safe errors."""
    missing = (tmp_path / "missing").resolve()
    with pytest.raises(ImageRetentionError, match="unavailable"):
        retention._validated_docker(missing)
    with pytest.raises(ImageRetentionError, match="invalid"):
        retention._validated_docker(tmp_path.resolve())

    docker = _docker(tmp_path)
    cases: list[tuple[BaseException | subprocess.CompletedProcess[str], str]] = [
        (subprocess.TimeoutExpired(["docker"], 60), "timed out"),
        (OSError("secret"), "could not start"),
        (_completed([], returncode=2, stderr="secret"), "failed"),
        (
            _completed([], stdout="x" * (256 * 1024 + 1)),
            "output is too large",
        ),
    ]
    for response, message in cases:
        run = create_autospec(
            subprocess.run,
            spec_set=True,
            side_effect=response if isinstance(response, BaseException) else None,
            return_value=None if isinstance(response, BaseException) else response,
        )
        with (
            patch("agent.deployment_retention.subprocess.run", new=run),
            pytest.raises(ImageRetentionError, match=message),
        ):
            retention._run(docker, ["image", "ls"], environment=ENVIRONMENT)


@pytest.mark.parametrize(
    ("operation", "message"),
    [
        (
            lambda: retention._identity_lines("bad\r\n", retention._IMAGE_ID, "image"),
            "inventory",
        ),
        (
            lambda: retention._identity_lines(
                "invalid\n", retention._IMAGE_ID, "image"
            ),
            "inventory",
        ),
        (lambda: retention._decode_single_document("{", "image"), "inspection"),
        (lambda: retention._decode_single_document("{}", "image"), "inspection"),
        (
            lambda: retention._image_document(
                json.dumps(
                    [{"Id": "bad", "RepoDigests": [], "RepoTags": [], "Config": {}}]
                )
            ),
            "identity",
        ),
        (
            lambda: retention._image_document(
                json.dumps(
                    [
                        {
                            "Id": _image_id(1),
                            "RepoDigests": [],
                            "RepoTags": [],
                        }
                    ]
                )
            ),
            "configuration",
        ),
        (
            lambda: retention._image_document(
                json.dumps(
                    [
                        {
                            "Id": _image_id(1),
                            "RepoDigests": [],
                            "RepoTags": [],
                            "Config": {"Labels": {"bad": 1}},
                        }
                    ]
                )
            ),
            "labels",
        ),
        (
            lambda: retention._image_document(
                json.dumps(
                    [
                        {
                            "Id": _image_id(1),
                            "RepoDigests": "bad",
                            "RepoTags": [],
                            "Config": {"Labels": None},
                        }
                    ]
                )
            ),
            "digests",
        ),
    ],
)
def test_malformed_docker_documents_are_rejected(
    operation: Any,
    message: str,
) -> None:
    """Fail closed for every malformed Docker JSON boundary."""
    with pytest.raises(ImageRetentionError, match=message):
        operation()


def test_optional_empty_docker_fields_are_valid_but_unmanaged() -> None:
    """Accept Docker's null empty-list encoding without claiming ownership."""
    document = retention._image_document(
        json.dumps(
            [
                {
                    "Id": _image_id(1),
                    "RepoDigests": None,
                    "RepoTags": None,
                    "Config": {"Labels": None},
                }
            ]
        )
    )
    assert document.repo_digests == ()
    assert document.repo_tags == ()
    assert document.labels == {}


@pytest.mark.parametrize("stdout", ["[]\n", "[]"])
def test_exact_docker_inspect_absence_is_accepted(
    tmp_path: Path,
    stdout: str,
) -> None:
    """Accept only Docker's bounded empty-array absence serialization."""
    docker = _docker(tmp_path)
    identifier = _reference(1)
    boundary = DockerBoundary(
        {},
        scripted={
            ("image", "inspect", identifier): [
                _completed(
                    [],
                    returncode=1,
                    stdout=stdout,
                    stderr=(
                        f"Error response from daemon: No such image: {identifier}\n"
                    ),
                )
            ]
        },
    )

    with patch(
        "agent.deployment_retention.subprocess.run",
        new=_mock_run(boundary),
    ):
        assert retention._inspect_image(docker, identifier, ENVIRONMENT) is None


@pytest.mark.parametrize(
    ("returncode", "stdout", "stderr", "message"),
    [
        (
            1,
            "",
            f"Error response from daemon: No such image: {_reference(1)}\n",
            "absence could not be proven",
        ),
        (
            1,
            "[ ]\n",
            f"Error response from daemon: No such image: {_reference(1)}\n",
            "absence could not be proven",
        ),
        (
            1,
            "[]\n",
            f"Error response from daemon: No such image: {_reference(1)}",
            "absence could not be proven",
        ),
        (
            1,
            "[]\n",
            f"Error response from daemon: No such image: {_reference(2)}\n",
            "absence could not be proven",
        ),
        (0, "[]\n", "", "inspection is invalid"),
        (2, "[]\n", "daemon unavailable", "command failed"),
    ],
)
def test_inconsistent_docker_inspect_absence_is_rejected(
    tmp_path: Path,
    returncode: int,
    stdout: str,
    stderr: str,
    message: str,
) -> None:
    """Reject every return-code/stdout/stderr combination outside the contract."""
    docker = _docker(tmp_path)
    identifier = _reference(1)
    boundary = DockerBoundary(
        {},
        scripted={
            ("image", "inspect", identifier): [
                _completed(
                    [],
                    returncode=returncode,
                    stdout=stdout,
                    stderr=stderr,
                )
            ]
        },
    )

    with (
        patch(
            "agent.deployment_retention.subprocess.run",
            new=_mock_run(boundary),
        ),
        pytest.raises(ImageRetentionError, match=message),
    ):
        retention._inspect_image(docker, identifier, ENVIRONMENT)


def test_unproven_inspect_absence_is_an_error(
    tmp_path: Path,
) -> None:
    """Do not interpret an arbitrary Docker exit one as object absence."""
    docker = _docker(tmp_path)
    boundary = DockerBoundary(
        {},
        scripted={
            ("image", "inspect", _reference(1)): [
                _completed([], returncode=1, stderr="daemon unavailable")
            ]
        },
    )
    with (
        patch(
            "agent.deployment_retention.subprocess.run",
            new=_mock_run(boundary),
        ),
        pytest.raises(ImageRetentionError, match="absence could not be proven"),
    ):
        retention._inspect_image(docker, _reference(1), ENVIRONMENT)


def test_cli_admitted_busy_missing_docker_oserror_and_module_entry(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Cover stable success and host/state failure process statuses."""
    state_dir = (tmp_path / "state").resolve()
    docker = _docker(tmp_path)
    boundary = DockerBoundary({})
    arguments = [
        "enforce",
        "--state-dir",
        str(state_dir),
        "--repository",
        REPOSITORY,
        "--target-reference",
        _reference(99),
        "--apply",
    ]
    with (
        patch(
            "agent.deployment_retention.subprocess.run",
            new=_mock_run(boundary),
        ),
        patch("agent.deployment_retention.shutil.which", return_value=str(docker)),
    ):
        assert main(arguments, environment=ENVIRONMENT) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "admitted"

    store = DeploymentStateStore(state_dir)
    with (
        store.transaction(),
        patch("agent.deployment_retention.shutil.which", return_value=str(docker)),
    ):
        assert main(arguments, environment=ENVIRONMENT) == 75
    assert "another deployment transaction" in capsys.readouterr().err

    with patch("agent.deployment_retention.shutil.which", return_value=None):
        assert main(arguments, environment=ENVIRONMENT) == 1
    assert "Docker executable is unavailable" in capsys.readouterr().err

    blocking_file = tmp_path / "not-a-directory"
    blocking_file.write_text("operator-owned", encoding="utf-8")
    broken_arguments = list(arguments)
    broken_arguments[broken_arguments.index(str(state_dir))] = str(
        blocking_file / "state"
    )
    with patch("agent.deployment_retention.shutil.which", return_value=str(docker)):
        assert main(broken_arguments, environment=ENVIRONMENT) == 1
    assert "host operation failed" in capsys.readouterr().err

    run = _mock_run(DockerBoundary({}))
    with (
        patch.object(sys, "argv", ["deployment-retention", *arguments]),
        patch("subprocess.run", new=run),
        patch("shutil.which", return_value=str(docker)),
        pytest.raises(SystemExit) as exited,
    ):
        runpy.run_module("agent.deployment_retention", run_name="__main__")
    assert exited.value.code == 0
    assert json.loads(capsys.readouterr().out)["status"] == "admitted"


def test_cli_busy_lock_is_cross_process_and_never_executes_docker(
    tmp_path: Path,
) -> None:
    """Prove a real competing process exits before any Docker observation."""
    state_dir = (tmp_path / "state").resolve()
    store = DeploymentStateStore(state_dir)
    executable_dir = tmp_path / "bin"
    executable_dir.mkdir()
    docker = executable_dir / "docker"
    sentinel = executable_dir / "docker.log"
    sentinel.write_bytes(b"")
    docker.write_text(
        '#!/bin/sh\nprintf "called\\n" >> "${0}.log"\nexit 99\n',
        encoding="utf-8",
    )
    docker.chmod(0o700)
    environment = {
        "HOME": str(tmp_path),
        "LANG": "C",
        "LC_ALL": "C",
        "PATH": str(executable_dir),
        "PYTHONPATH": str(Path("src").resolve()),
    }
    command = [
        sys.executable,
        "-m",
        "agent.deployment_retention",
        "enforce",
        "--state-dir",
        str(state_dir),
        "--repository",
        REPOSITORY,
        "--target-reference",
        _reference(99),
        "--apply",
    ]

    with store.transaction():
        result = subprocess.run(  # noqa: S603 - fixed interpreter and arguments
            command,
            env=environment,
            text=True,
            capture_output=True,
            check=False,
            timeout=10,
        )

    assert result.returncode == 75
    assert result.stdout == ""
    assert "another deployment transaction" in result.stderr
    assert sentinel.read_bytes() == b""


def test_pyproject_exposes_retention_entrypoint() -> None:
    """Keep the operator CLI available without constructing module paths."""
    document = Path("pyproject.toml").read_text(encoding="utf-8")
    assert 'deployment-retention = "agent.deployment_retention:main"' in document
