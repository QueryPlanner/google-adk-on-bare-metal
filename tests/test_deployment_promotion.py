"""Transactional promotion controller tests with real state and filesystem code."""

from __future__ import annotations

import errno
import fcntl
import io
import json
import os
import runpy
import shutil
import subprocess
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path
from unittest.mock import create_autospec

import pytest

from agent import deployment_promotion as promotion_module
from agent.compose_env import serialize_compose_environment
from agent.deployment_promotion import (
    PRODUCTION_ENVIRONMENT_NAMES,
    PromotionConfig,
    main,
    promote,
)
from agent.deployment_state import (
    CandidateReceipt,
    DeploymentStateStore,
    PersistentVolumeIdentity,
)

REAL_SUBPROCESS_RUN = subprocess.run
REAL_SHUTIL_WHICH = shutil.which

OLD_REVISION = "a" * 40
TARGET_REVISION = "b" * 40
OLD_IMAGE_ID = f"sha256:{'c' * 64}"
TARGET_IMAGE_ID = f"sha256:{'d' * 64}"
MANUAL_IMAGE_ID = f"sha256:{'9' * 64}"
OLD_IMAGE = f"ghcr.io/queryplanner/agent@sha256:{'e' * 64}"
TARGET_IMAGE = f"ghcr.io/queryplanner/agent@sha256:{'f' * 64}"
MANUAL_IMAGE = f"ghcr.io/queryplanner/agent@sha256:{'8' * 64}"
OLD_CONTAINER = "1" * 64
TARGET_CONTAINER = "2" * 64
CANDIDATE_CONTAINER = "3" * 64
FRESH_TRANSACTION = f"{'0' * 23}-{OLD_REVISION}"
PROJECT = "adk-agent"
SERVICE = "agent"
ORIGIN = "https://github.com/QueryPlanner/google-adk-on-bare-metal"
VOLUME = PersistentVolumeIdentity(
    name="adk-agent_agent_artifacts",
    driver="local",
    mountpoint="/var/lib/docker/volumes/adk-agent/_data",
    destination="/app/src/.adk",
    created_at="2026-07-28T12:00:00.000000Z",
)
REPLACED_VOLUME = PersistentVolumeIdentity(
    name=VOLUME.name,
    driver=VOLUME.driver,
    mountpoint=VOLUME.mountpoint,
    destination=VOLUME.destination,
    created_at="2026-07-28T12:00:01.000000Z",
)
SECRETS = {
    "AGENT_NAME": "production-agent",
    "DATABASE_URL": "postgresql://secret-$VALUE",
    "OPENROUTER_API_KEY": "openrouter-$(private)",
    "GOOGLE_API_KEY": "google-`private`",
    "ROOT_AGENT_MODEL": "openrouter/provider/model",
    "LANGFUSE_PUBLIC_KEY": "public",
    "LANGFUSE_SECRET_KEY": "langfuse-secret",
    "LANGFUSE_BASE_URL": "https://langfuse.example",
    "LOG_LEVEL": "INFO",
    "PORT": "8080",
    "HOST": "0.0." + "0.0",
}


def _private(path: Path, payload: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    path.chmod(0o600)
    return path


@dataclass
class Host:
    """Stateful fake for only the Git and Docker process boundaries."""

    checkout: Path
    release: Path
    state_dir: Path
    bin_dir: Path
    production_exists: bool = True
    production_revision: str = OLD_REVISION
    production_image: str = OLD_IMAGE
    production_image_id: str = OLD_IMAGE_ID
    production_healthy: bool = True
    volume_exists: bool = True
    candidate_exists: bool = False
    fail_candidate: bool = False
    fail_target_health: bool = False
    fail_rollback_up: bool = False
    fail_target_checkout: bool = False
    make_current_publish_fail: bool = False
    make_rollback_cleanup_fail: bool = False
    target_production_seen: bool = False
    log: list[tuple[str, ...]] = field(default_factory=list)
    pending_seen_before_mutation: bool = False
    lease_probe: Path | None = None

    def executable(self, name: str) -> str:
        return str(self.bin_dir / name)

    def which(self, name: str, **_kwargs: object) -> str | None:
        if name in {"git", "docker"}:
            return self.executable(name)
        return None

    def completed(
        self,
        args: list[str],
        *,
        returncode: int = 0,
        stdout: str = "",
        stderr: str = "",
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args, returncode, stdout, stderr)

    def run(
        self,
        args: list[str],
        *,
        cwd: Path | None = None,
        env: dict[str, str],
        text: bool,
        capture_output: bool,
        check: bool,
        timeout: int,
    ) -> subprocess.CompletedProcess[str]:
        del cwd, text, capture_output, check, timeout
        if self.lease_probe is not None:
            descriptor = os.open(self.lease_probe, os.O_RDWR)
            try:
                with pytest.raises(OSError) as blocked:
                    fcntl.flock(
                        descriptor,
                        fcntl.LOCK_EX | fcntl.LOCK_NB,
                    )
                assert blocked.value.errno in {errno.EACCES, errno.EAGAIN}
            finally:
                os.close(descriptor)
        command = tuple(args)
        self.log.append(command)
        surfaced = "\0".join((*args, *env.values()))
        for name in (
            "DATABASE_URL",
            "OPENROUTER_API_KEY",
            "GOOGLE_API_KEY",
            "LANGFUSE_SECRET_KEY",
        ):
            assert SECRETS[name] not in surfaced
        if args[0].endswith("/git"):
            return self._git(args)
        if args[0].endswith("/docker"):
            return self._docker(args, env)
        raise AssertionError(args)

    def _git(self, args: list[str]) -> subprocess.CompletedProcess[str]:
        selected = args[1:]
        path = Path(selected[1]) if selected[:1] == ["-C"] else self.checkout
        tail = selected[2:] if selected[:1] == ["-C"] else selected
        if tail == ["rev-parse", "--show-toplevel"]:
            return self.completed(args, stdout=f"{path}\n")
        if tail == ["remote", "get-url", "origin"]:
            return self.completed(args, stdout=f"{ORIGIN}\n")
        if tail == ["rev-parse", "--verify", "HEAD"]:
            revision = (
                TARGET_REVISION if path == self.release else self.production_revision
            )
            return self.completed(args, stdout=f"{revision}\n")
        if tail[:1] == ["diff"]:
            return self.completed(args)
        if tail[:1] == ["status"]:
            return self.completed(args)
        if tail[:3] == ["ls-files", "--error-unmatch", "--"]:
            return self.completed(args)
        if "checkout" in tail:
            assert self.state_dir.joinpath("pending.json").exists()
            self.pending_seen_before_mutation = True
            if self.fail_target_checkout and tail[-1] == TARGET_REVISION:
                return self.completed(args, returncode=41)
            self.production_revision = tail[-1]
            return self.completed(args)
        raise AssertionError(args)

    def _image_document(self, selector: str) -> dict[str, object]:
        if selector in {TARGET_IMAGE, TARGET_IMAGE_ID}:
            return {
                "Id": TARGET_IMAGE_ID,
                "RepoDigests": [TARGET_IMAGE],
                "Config": {
                    "Labels": {"org.opencontainers.image.revision": TARGET_REVISION}
                },
            }
        if selector in {OLD_IMAGE, OLD_IMAGE_ID}:
            return {
                "Id": OLD_IMAGE_ID,
                "RepoDigests": [OLD_IMAGE],
                "Config": {
                    "Labels": {"org.opencontainers.image.revision": OLD_REVISION}
                },
            }
        raise AssertionError(selector)

    def _container_document(
        self,
        *,
        candidate: bool,
    ) -> dict[str, object]:
        if candidate:
            project = next(
                item.rsplit("=", 1)[1]
                for command in reversed(self.log)
                for item in command
                if item.startswith("label=com.docker.compose.project=candidate-")
            )
            return {
                "Id": CANDIDATE_CONTAINER,
                "Image": TARGET_IMAGE_ID,
                "State": {"Status": "running", "Health": {"Status": "healthy"}},
                "Config": {
                    "Image": TARGET_IMAGE,
                    "User": "1000:1000",
                    "Labels": {
                        "com.docker.compose.project": project,
                        "com.docker.compose.service": SERVICE,
                        "com.docker.compose.project.working_dir": str(self.release),
                    },
                },
                "HostConfig": {
                    "NetworkMode": "none",
                    "PortBindings": {},
                    "RestartPolicy": {"Name": "no"},
                    "CapDrop": ["ALL"],
                    "SecurityOpt": ["no-new-privileges:true"],
                },
                "Mounts": [],
            }
        container = (
            TARGET_CONTAINER if self.production_image == TARGET_IMAGE else OLD_CONTAINER
        )
        return {
            "Id": container,
            "Image": self.production_image_id,
            "State": {
                "Status": "running",
                "Health": {
                    "Status": "healthy" if self.production_healthy else "unhealthy"
                },
            },
            "Config": {
                "Image": self.production_image,
                "Labels": {
                    "com.docker.compose.project": PROJECT,
                    "com.docker.compose.service": SERVICE,
                    "com.docker.compose.project.working_dir": str(self.checkout),
                },
            },
            "Mounts": (
                [
                    {
                        "Type": "volume",
                        "Name": VOLUME.name,
                        "Driver": VOLUME.driver,
                        "Destination": VOLUME.destination,
                    }
                ]
                if self.volume_exists
                else []
            ),
        }

    def _docker(
        self,
        args: list[str],
        environment: dict[str, str],
    ) -> subprocess.CompletedProcess[str]:
        tail = args[1:]
        if tail[:2] == ["image", "pull"]:
            return self.completed(args, stdout=f"{tail[2]}\n")
        if tail[:2] == ["image", "inspect"]:
            return self.completed(
                args,
                stdout=json.dumps([self._image_document(tail[2])]),
            )
        if tail[:2] == ["container", "ls"]:
            project_filter = next(
                value
                for value in tail
                if value.startswith("label=com.docker.compose.project=")
            )
            project = project_filter.rsplit("=", 1)[1]
            if project.startswith("candidate-"):
                value = CANDIDATE_CONTAINER[:12] if self.candidate_exists else ""
            else:
                value = (
                    (
                        TARGET_CONTAINER
                        if self.production_image == TARGET_IMAGE
                        else OLD_CONTAINER
                    )[:12]
                    if self.production_exists
                    else ""
                )
            return self.completed(args, stdout=f"{value}\n" if value else "")
        if tail[:2] == ["container", "inspect"]:
            candidate = tail[2].startswith(CANDIDATE_CONTAINER[:12])
            return self.completed(
                args,
                stdout=json.dumps([self._container_document(candidate=candidate)]),
            )
        if tail[:2] == ["container", "rm"]:
            self.production_exists = False
            return self.completed(args, stdout=f"{tail[-1]}\n")
        if tail[:2] == ["volume", "inspect"]:
            document = {
                "Name": VOLUME.name,
                "Driver": VOLUME.driver,
                "Mountpoint": VOLUME.mountpoint,
                "CreatedAt": VOLUME.created_at,
            }
            if self.make_current_publish_fail and self.production_image == TARGET_IMAGE:
                self.state_dir.chmod(0o500)
                self.make_current_publish_fail = False
            if (
                self.make_rollback_cleanup_fail
                and self.target_production_seen
                and self.production_image == OLD_IMAGE
            ):
                self.state_dir.chmod(0o500)
                self.make_rollback_cleanup_fail = False
            return self.completed(args, stdout=json.dumps([document]))
        if tail[:1] == ["compose"]:
            project = tail[tail.index("--project-name") + 1]
            if "config" in tail:
                output = f"{environment['IMAGE']}\n" if "--images" in tail else ""
                return self.completed(args, stdout=output)
            if "up" in tail:
                if project.startswith("candidate-"):
                    self.candidate_exists = True
                    if self.fail_candidate:
                        return self.completed(args, returncode=42)
                else:
                    selected = environment["IMAGE"]
                    if selected == OLD_IMAGE and self.fail_rollback_up:
                        return self.completed(args, returncode=43)
                    self.production_exists = True
                    self.production_image = selected
                    self.production_image_id = (
                        TARGET_IMAGE_ID if selected == TARGET_IMAGE else OLD_IMAGE_ID
                    )
                    self.production_revision = (
                        TARGET_REVISION if selected == TARGET_IMAGE else OLD_REVISION
                    )
                    if selected == TARGET_IMAGE:
                        self.target_production_seen = True
                    self.production_healthy = not (
                        selected == TARGET_IMAGE and self.fail_target_health
                    )
                    self.volume_exists = True
                return self.completed(args)
            if "down" in tail:
                assert project.startswith("candidate-")
                self.candidate_exists = False
                return self.completed(args)
        raise AssertionError(args)


@pytest.fixture
def setup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Host, PromotionConfig, dict[str, str]]:
    checkout = tmp_path / "checkout"
    release = tmp_path / "release"
    state_dir = tmp_path / "state"
    bin_dir = tmp_path / "bin"
    checkout.mkdir()
    release.mkdir()
    bin_dir.mkdir()
    for name in ("git", "docker"):
        selected = bin_dir / name
        selected.write_text("#!/bin/sh\n", encoding="utf-8")
        selected.chmod(0o700)
    for root in (checkout, release):
        (root / "src/agent").mkdir(parents=True)
        for relative in (
            "compose.yaml",
            "compose.candidate.yaml",
            "src/agent/deployment_promotion.py",
        ):
            (root / relative).write_text("tracked\n", encoding="utf-8")
    _private(checkout / ".env", b'AGENT_NAME="old"\n')
    host = Host(checkout, release, state_dir, bin_dir)
    run_mock = create_autospec(
        subprocess.run,
        spec_set=True,
        side_effect=host.run,
    )
    which_mock = create_autospec(
        shutil.which,
        spec_set=True,
        side_effect=host.which,
    )
    monkeypatch.setattr(subprocess, "run", run_mock)
    monkeypatch.setattr(shutil, "which", which_mock)
    environment = dict(SECRETS) | {
        "HOME": str(tmp_path),
        "PATH": str(bin_dir),
    }
    config = PromotionConfig(
        state_dir=state_dir,
        checkout=checkout,
        release_checkout=release,
        expected_origin=ORIGIN,
        compose_project=PROJECT,
        compose_service=SERVICE,
        source_revision=TARGET_REVISION,
        image_reference=TARGET_IMAGE,
        adopt_existing=True,
    )
    return host, config, environment


def _initialize(host: Host, config: PromotionConfig) -> None:
    with DeploymentStateStore(config.state_dir).transaction() as transaction:
        transaction.adopt(
            compose_project=PROJECT,
            compose_service=SERVICE,
            source_revision=OLD_REVISION,
            image_reference=OLD_IMAGE,
            image_id=OLD_IMAGE_ID,
            oci_revision=OLD_REVISION,
            environment_source=host.checkout / ".env",
            deployment_id="baseline",
            recorded_at="2026-07-28T11:00:00.000000Z",
        )


def _argv(config: PromotionConfig) -> list[str]:
    return [
        "promote",
        "--state-dir",
        str(config.state_dir),
        "--checkout",
        str(config.checkout),
        "--release-checkout",
        str(config.release_checkout),
        "--expected-origin",
        config.expected_origin,
        "--compose-project",
        config.compose_project,
        "--compose-service",
        config.compose_service,
        "--source-revision",
        config.source_revision,
        "--image-reference",
        config.image_reference,
        "--adopt-existing",
    ]


def _controller_argv(config: PromotionConfig) -> list[str]:
    lease_directory = config.release_checkout.with_name(
        f"{config.release_checkout.name}.lease"
    )
    lease_directory.mkdir(mode=0o700)
    lease = lease_directory / "lock"
    lease.touch(mode=0o600)
    lease.chmod(0o600)
    return [
        *_argv(config),
        "--release-lease",
        str(lease),
        "--environment-stdin",
    ]


def _pending(
    host: Host,
    config: PromotionConfig,
    *,
    cutover_started: bool = True,
) -> None:
    _initialize(host, config)
    target_env = _private(
        config.state_dir.parent / "target-input.env",
        b'TARGET="yes"\n',
    )
    with DeploymentStateStore(config.state_dir).transaction() as transaction:
        tail = transaction.journal()[-1]
        receipt = CandidateReceipt(
            observed_at="2026-07-28T11:30:00.000000Z",
            compose_project="candidate-pending",
            compose_service=SERVICE,
            container_id=CANDIDATE_CONTAINER,
            image_reference=TARGET_IMAGE,
            image_id=TARGET_IMAGE_ID,
            oci_revision=TARGET_REVISION,
            baseline_journal_sequence=tail.sequence,
            baseline_journal_sha256=tail.sha256,
        )
        pending = transaction.begin_promotion(
            compose_project=PROJECT,
            compose_service=SERVICE,
            source_revision=TARGET_REVISION,
            image_reference=TARGET_IMAGE,
            image_id=TARGET_IMAGE_ID,
            oci_revision=TARGET_REVISION,
            environment_source=target_env,
            candidate=receipt,
            persistent_volumes=(VOLUME,),
            transaction_id="pending",
            recorded_at="2026-07-28T11:30:00.000000Z",
        )
        if cutover_started:
            transaction.install_environment(
                pending.intent.target,
                host.checkout / ".env",
            )
    if cutover_started:
        host.production_revision = TARGET_REVISION
        host.production_image = TARGET_IMAGE
        host.production_image_id = TARGET_IMAGE_ID


def _fresh_pending(
    host: Host,
    config: PromotionConfig,
    *,
    install_environment: bool,
    production_exists: bool,
    transaction_id: str = FRESH_TRANSACTION,
) -> None:
    host.production_exists = False
    host.volume_exists = False
    (host.checkout / ".env").unlink()
    target_env = _private(
        config.state_dir.parent / "fresh-target-input.env",
        b'TARGET="yes"\n',
    )
    with DeploymentStateStore(config.state_dir).transaction() as transaction:
        pending = transaction.begin_promotion(
            compose_project=PROJECT,
            compose_service=SERVICE,
            source_revision=TARGET_REVISION,
            image_reference=TARGET_IMAGE,
            image_id=TARGET_IMAGE_ID,
            oci_revision=TARGET_REVISION,
            environment_source=target_env,
            candidate=CandidateReceipt(
                observed_at="2026-07-28T11:30:00.000000Z",
                compose_project="candidate-fresh",
                compose_service=SERVICE,
                container_id=CANDIDATE_CONTAINER,
                image_reference=TARGET_IMAGE,
                image_id=TARGET_IMAGE_ID,
                oci_revision=TARGET_REVISION,
                baseline_journal_sequence=None,
                baseline_journal_sha256=None,
            ),
            persistent_volumes=(),
            transaction_id=transaction_id,
            recorded_at="2026-07-28T11:30:00.000000Z",
        )
        if install_environment:
            transaction.install_environment(
                pending.intent.target,
                host.checkout / ".env",
            )
    host.production_exists = production_exists
    if production_exists:
        host.production_revision = TARGET_REVISION
        host.production_image = TARGET_IMAGE
        host.production_image_id = TARGET_IMAGE_ID
        host.volume_exists = True


def test_existing_deployment_promotes_after_exact_candidate_and_intent(
    setup: tuple[Host, PromotionConfig, dict[str, str]],
) -> None:
    host, config, environment = setup
    _initialize(host, config)

    current = promote(config, environment=environment)

    assert current.state.source_revision == TARGET_REVISION
    assert host.production_image == TARGET_IMAGE
    assert host.pending_seen_before_mutation
    assert not host.candidate_exists
    store = DeploymentStateStore(config.state_dir)
    assert [entry.event for entry in store.read_journal()] == [
        "adopted",
        "promoted",
    ]
    production_up = next(
        command
        for command in host.log
        if "compose" in command
        and "up" in command
        and "--project-name" in command
        and command[command.index("--project-name") + 1] == PROJECT
    )
    assert production_up.index("--no-build") < production_up.index("--pull")
    assert production_up.index("--pull") < production_up.index("--wait")
    assert "down" not in production_up


def test_candidate_failure_leaves_every_production_identity_untouched(
    setup: tuple[Host, PromotionConfig, dict[str, str]],
) -> None:
    host, config, environment = setup
    _initialize(host, config)
    before_env = (host.checkout / ".env").read_bytes()
    before_journal = DeploymentStateStore(config.state_dir).read_journal()
    host.fail_candidate = True

    assert main(_argv(config), environment) == 1

    assert host.production_revision == OLD_REVISION
    assert host.production_image == OLD_IMAGE
    assert (host.checkout / ".env").read_bytes() == before_env
    assert DeploymentStateStore(config.state_dir).read_journal() == before_journal
    assert not config.state_dir.joinpath("pending.json").exists()
    assert not host.candidate_exists


def test_post_cutover_failure_restores_and_records_rollback(
    setup: tuple[Host, PromotionConfig, dict[str, str]],
) -> None:
    host, config, environment = setup
    _initialize(host, config)
    host.fail_target_health = True

    assert main(_argv(config), environment) == 1

    assert host.production_revision == OLD_REVISION
    assert host.production_image == OLD_IMAGE
    assert host.production_healthy
    store = DeploymentStateStore(config.state_dir)
    assert [entry.event for entry in store.read_journal()] == [
        "adopted",
        "rolled_back",
    ]
    current = store.read_current()
    assert current is not None
    assert current.state.source_revision == OLD_REVISION
    assert not config.state_dir.joinpath("pending.json").exists()


def test_rollback_failure_preserves_pending_without_false_terminal(
    setup: tuple[Host, PromotionConfig, dict[str, str]],
    capsys: pytest.CaptureFixture[str],
) -> None:
    host, config, environment = setup
    _initialize(host, config)
    host.fail_target_health = True
    host.fail_rollback_up = True

    assert main(_argv(config), environment) == 1

    assert config.state_dir.joinpath("pending.json").is_file()
    assert [
        entry.event for entry in DeploymentStateStore(config.state_dir).read_journal()
    ] == ["adopted"]
    surfaced = capsys.readouterr().out + capsys.readouterr().err
    assert all(value not in surfaced for value in SECRETS.values())


def test_pending_entry_recovers_baseline_and_requires_rerun(
    setup: tuple[Host, PromotionConfig, dict[str, str]],
) -> None:
    host, config, environment = setup
    _pending(host, config)

    recovery_environment = {
        "HOME": environment["HOME"],
        "PATH": environment["PATH"],
    }
    assert main(_argv(config), recovery_environment) == 3

    assert host.production_revision == OLD_REVISION
    assert host.production_image == OLD_IMAGE
    assert [
        entry.event for entry in DeploymentStateStore(config.state_dir).read_journal()
    ] == [
        "adopted",
        "rolled_back",
    ]
    assert not config.state_dir.joinpath("pending.json").exists()
    assert not any("candidate-" in " ".join(command) for command in host.log)


def test_pending_pre_cutover_baseline_records_outcome_without_recreate(
    setup: tuple[Host, PromotionConfig, dict[str, str]],
) -> None:
    host, config, environment = setup
    _pending(host, config, cutover_started=False)
    commands_before = len(host.log)

    assert main(_argv(config), environment) == 3

    assert host.production_revision == OLD_REVISION
    assert host.production_image == OLD_IMAGE
    assert (host.checkout / ".env").read_bytes() == b'AGENT_NAME="old"\n'
    assert [
        entry.event for entry in DeploymentStateStore(config.state_dir).read_journal()
    ] == ["adopted", "aborted"]
    recovery_commands = host.log[commands_before:]
    assert not any("checkout" in command for command in recovery_commands)
    assert not any(
        "compose" in command or command[1:3] == ("container", "rm")
        for command in recovery_commands
    )


def test_pending_pre_cutover_image_mismatch_fails_without_mutation(
    setup: tuple[Host, PromotionConfig, dict[str, str]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    host, config, environment = setup
    _pending(host, config, cutover_started=False)
    commands_before = len(host.log)

    def transform(
        args: list[str],
        result: subprocess.CompletedProcess[str],
    ) -> subprocess.CompletedProcess[str]:
        if args[1:3] == ["image", "inspect"] and args[3] == OLD_IMAGE:
            document = json.loads(result.stdout)[0]
            document["Id"] = MANUAL_IMAGE_ID
            return _completed_json(args, document)
        return result

    _strict_host_run(monkeypatch, host, transform)

    assert main(_argv(config), environment) == 1

    assert host.production_revision == OLD_REVISION
    assert host.production_image == OLD_IMAGE
    assert config.state_dir.joinpath("pending.json").is_file()
    assert not any(
        "checkout" in command or "compose" in command
        for command in host.log[commands_before:]
    )


def test_pending_pre_cutover_unhealthy_runtime_is_recreated_and_verified(
    setup: tuple[Host, PromotionConfig, dict[str, str]],
) -> None:
    host, config, environment = setup
    _pending(host, config, cutover_started=False)
    host.production_healthy = False

    assert main(_argv(config), environment) == 3

    assert host.production_healthy
    assert host.production_revision == OLD_REVISION
    assert host.production_image == OLD_IMAGE
    assert [
        entry.event for entry in DeploymentStateStore(config.state_dir).read_journal()
    ] == ["adopted", "rolled_back"]
    assert any("compose" in command and "up" in command for command in host.log)


def test_pending_pre_cutover_volume_replacement_fails_without_recreate(
    setup: tuple[Host, PromotionConfig, dict[str, str]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    host, config, environment = setup
    _pending(host, config, cutover_started=False)
    commands_before = len(host.log)

    def transform(
        args: list[str],
        result: subprocess.CompletedProcess[str],
    ) -> subprocess.CompletedProcess[str]:
        if args[1:3] == ["volume", "inspect"]:
            document = json.loads(result.stdout)[0]
            document["CreatedAt"] = REPLACED_VOLUME.created_at
            return _completed_json(args, document)
        return result

    _strict_host_run(monkeypatch, host, transform)

    assert main(_argv(config), environment) == 1

    assert host.production_revision == OLD_REVISION
    assert host.production_image == OLD_IMAGE
    assert config.state_dir.joinpath("pending.json").is_file()
    assert not any(
        "checkout" in command or "compose" in command
        for command in host.log[commands_before:]
    )


def test_pending_config_mismatch_fails_before_recovery_mutation(
    setup: tuple[Host, PromotionConfig, dict[str, str]],
) -> None:
    host, config, environment = setup
    _pending(host, config)
    commands_before = len(host.log)

    assert (
        main(
            _argv(replace(config, compose_project="different-project")),
            environment,
        )
        == 1
    )

    assert host.production_revision == TARGET_REVISION
    assert host.production_image == TARGET_IMAGE
    assert config.state_dir.joinpath("pending.json").is_file()
    later = host.log[commands_before:]
    assert later
    assert all(
        "checkout" not in command and "compose" not in command for command in later
    )


def test_pending_manual_image_fails_closed_before_recovery_mutation(
    setup: tuple[Host, PromotionConfig, dict[str, str]],
) -> None:
    host, config, environment = setup
    _pending(host, config)
    host.production_image = MANUAL_IMAGE
    host.production_image_id = MANUAL_IMAGE_ID
    commands_before = len(host.log)

    assert main(_argv(config), environment) == 1

    assert host.production_image == MANUAL_IMAGE
    assert host.production_image_id == MANUAL_IMAGE_ID
    assert config.state_dir.joinpath("pending.json").is_file()
    later = host.log[commands_before:]
    assert later
    assert all(
        "checkout" not in command and "compose" not in command for command in later
    )


@pytest.mark.parametrize("revision", ["7" * 40, "not-a-revision"])
def test_pending_checkout_drift_fails_before_recovery_mutation(
    setup: tuple[Host, PromotionConfig, dict[str, str]],
    revision: str,
) -> None:
    host, config, environment = setup
    _pending(host, config)
    host.production_revision = revision
    commands_before = len(host.log)

    assert main(_argv(config), environment) == 1

    assert host.production_revision == revision
    assert host.production_image == TARGET_IMAGE
    assert config.state_dir.joinpath("pending.json").is_file()
    later = host.log[commands_before:]
    assert later
    assert all(
        "checkout" not in command and "compose" not in command for command in later
    )


def test_pending_rejects_mismatched_recorded_baseline_before_docker(
    setup: tuple[Host, PromotionConfig, dict[str, str]],
) -> None:
    host, config, environment = setup
    _pending(host, config)
    with DeploymentStateStore(config.state_dir).transaction() as transaction:
        current = transaction.current()
        pending = transaction.pending()
        assert current is not None
        assert pending is not None
        mismatched = replace(
            current,
            state=replace(current.state, compose_project="different-project"),
        )
        command_count = len(host.log)

        with pytest.raises(
            promotion_module.PromotionError,
            match="baseline Compose identity",
        ):
            promotion_module._validate_pending_ownership(
                pending=pending,
                current=mismatched,
                config=config,
                executables=_executables(host),
                environment=promotion_module._command_environment(environment),
            )

    assert all(
        command[1:3] != ("container", "ls") for command in host.log[command_count:]
    )


def test_pending_fresh_install_before_cutover_aborts_without_checkout_mutation(
    setup: tuple[Host, PromotionConfig, dict[str, str]],
) -> None:
    host, config, environment = setup
    _fresh_pending(
        host,
        config,
        install_environment=False,
        production_exists=False,
    )
    assert host.production_revision == OLD_REVISION

    assert main(_argv(config), environment) == 3

    assert host.production_revision == OLD_REVISION
    assert not host.production_exists
    assert not (host.checkout / ".env").exists()
    assert [
        entry.event for entry in DeploymentStateStore(config.state_dir).read_journal()
    ] == ["aborted"]


def test_fresh_pending_without_recorded_checkout_baseline_fails_closed(
    setup: tuple[Host, PromotionConfig, dict[str, str]],
) -> None:
    host, config, environment = setup
    _fresh_pending(
        host,
        config,
        install_environment=False,
        production_exists=False,
        transaction_id="legacy-fresh",
    )
    command_count = len(host.log)

    assert main(_argv(config), environment) == 1

    assert host.production_revision == OLD_REVISION
    assert not host.production_exists
    assert config.state_dir.joinpath("pending.json").is_file()
    assert not any(
        "checkout" in command or "compose" in command
        for command in host.log[command_count:]
    )


def test_pending_fresh_install_is_aborted_and_requires_rerun(
    setup: tuple[Host, PromotionConfig, dict[str, str]],
) -> None:
    host, config, environment = setup
    host.production_exists = False
    host.volume_exists = False
    (host.checkout / ".env").unlink()
    target_env = _private(
        config.state_dir.parent / "fresh-target-input.env",
        b'TARGET="yes"\n',
    )
    with DeploymentStateStore(config.state_dir).transaction() as transaction:
        pending = transaction.begin_promotion(
            compose_project=PROJECT,
            compose_service=SERVICE,
            source_revision=TARGET_REVISION,
            image_reference=TARGET_IMAGE,
            image_id=TARGET_IMAGE_ID,
            oci_revision=TARGET_REVISION,
            environment_source=target_env,
            candidate=CandidateReceipt(
                observed_at="2026-07-28T11:30:00.000000Z",
                compose_project="candidate-fresh",
                compose_service=SERVICE,
                container_id=CANDIDATE_CONTAINER,
                image_reference=TARGET_IMAGE,
                image_id=TARGET_IMAGE_ID,
                oci_revision=TARGET_REVISION,
                baseline_journal_sequence=None,
                baseline_journal_sha256=None,
            ),
            persistent_volumes=(),
            transaction_id=FRESH_TRANSACTION,
            recorded_at="2026-07-28T11:30:00.000000Z",
        )
        transaction.install_environment(pending.intent.target, host.checkout / ".env")
    host.production_exists = True
    host.production_revision = TARGET_REVISION
    host.production_image = TARGET_IMAGE
    host.production_image_id = TARGET_IMAGE_ID
    host.volume_exists = True

    assert main(_argv(config), environment) == 3

    assert not host.production_exists
    assert host.volume_exists
    assert host.production_revision == OLD_REVISION
    assert not (host.checkout / ".env").exists()
    assert [
        entry.event for entry in DeploymentStateStore(config.state_dir).read_journal()
    ] == ["aborted"]


def test_first_install_commits_new_volume_identity(
    setup: tuple[Host, PromotionConfig, dict[str, str]],
) -> None:
    host, config, environment = setup
    host.production_exists = False
    host.volume_exists = False
    (host.checkout / ".env").unlink()

    current = promote(config, environment=environment)

    assert current.state.source_revision == TARGET_REVISION
    assert host.production_exists
    assert host.volume_exists
    assert [
        entry.event for entry in DeploymentStateStore(config.state_dir).read_journal()
    ] == ["promoted"]


def test_fresh_install_failure_removes_only_owned_container_and_aborts(
    setup: tuple[Host, PromotionConfig, dict[str, str]],
) -> None:
    host, config, environment = setup
    host.production_exists = False
    host.volume_exists = False
    host.fail_target_health = True
    (host.checkout / ".env").unlink()

    assert main(_argv(config), environment) == 1

    assert not host.production_exists
    assert host.volume_exists
    assert host.production_revision == OLD_REVISION
    assert not (host.checkout / ".env").exists()
    assert [
        entry.event for entry in DeploymentStateStore(config.state_dir).read_journal()
    ] == ["aborted"]
    flattened = [" ".join(command) for command in host.log]
    assert any("container rm --force" in command for command in flattened)
    assert not any(
        f"compose --project-name {PROJECT}" in command and " down " in command
        for command in flattened
    )


@pytest.mark.parametrize("fault", ["absent", "ambiguous", "wrong-image"])
def test_fresh_cleanup_proves_exact_service_ownership(
    setup: tuple[Host, PromotionConfig, dict[str, str]],
    monkeypatch: pytest.MonkeyPatch,
    fault: str,
) -> None:
    host, config, environment = setup
    _fresh_pending(
        host,
        config,
        install_environment=True,
        production_exists=fault != "absent",
    )

    def transform(
        args: list[str],
        result: subprocess.CompletedProcess[str],
    ) -> subprocess.CompletedProcess[str]:
        if fault == "ambiguous" and args[1:3] == ["container", "ls"]:
            return host.completed(
                args,
                stdout=f"{TARGET_CONTAINER[:12]}\n{'4' * 12}\n",
            )
        if fault == "wrong-image" and args[1:3] == ["container", "inspect"]:
            document = json.loads(result.stdout)[0]
            document["Image"] = MANUAL_IMAGE_ID
            return _completed_json(args, document)
        return result

    _strict_host_run(monkeypatch, host, transform)
    with DeploymentStateStore(config.state_dir).transaction() as transaction:
        pending = transaction.pending()
        assert pending is not None
        if fault == "absent":
            promotion_module._remove_owned_service_container(
                target=pending.intent.target,
                config=config,
                executables=_executables(host),
                environment=promotion_module._command_environment(environment),
                target_environment_proven=True,
                verify_target_environment=lambda: True,
            )
        else:
            with pytest.raises(
                promotion_module.PromotionRecoveryFailedError,
                match="ambiguous|ownership",
            ):
                promotion_module._remove_owned_service_container(
                    target=pending.intent.target,
                    config=config,
                    executables=_executables(host),
                    environment=promotion_module._command_environment(environment),
                    target_environment_proven=True,
                    verify_target_environment=lambda: True,
                )

    assert config.state_dir.joinpath("pending.json").is_file()


def test_fresh_abort_rejects_container_reappearing_after_removal(
    setup: tuple[Host, PromotionConfig, dict[str, str]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    host, config, environment = setup
    _fresh_pending(
        host,
        config,
        install_environment=True,
        production_exists=True,
    )
    removed = False

    def transform(
        args: list[str],
        result: subprocess.CompletedProcess[str],
    ) -> subprocess.CompletedProcess[str]:
        nonlocal removed
        if args[1:3] == ["container", "rm"]:
            removed = True
        if removed and args[1:3] == ["container", "ls"]:
            return host.completed(args, stdout=f"{TARGET_CONTAINER[:12]}\n")
        return result

    _strict_host_run(monkeypatch, host, transform)
    with DeploymentStateStore(config.state_dir).transaction() as transaction:
        pending = transaction.pending()
        assert pending is not None
        with pytest.raises(
            promotion_module.PromotionRecoveryFailedError,
            match="baseline could not be verified",
        ):
            promotion_module._abort_fresh_install(
                transaction=transaction,
                pending=pending,
                config=config,
                executables=_executables(host),
                environment=promotion_module._command_environment(environment),
                baseline_revision=OLD_REVISION,
            )

    assert removed
    assert config.state_dir.joinpath("pending.json").is_file()
    assert [
        entry.event for entry in DeploymentStateStore(config.state_dir).read_journal()
    ] == []


@pytest.mark.parametrize("fault", ["missing", "tampered"])
def test_fresh_recovery_proves_target_environment_before_container_removal(
    setup: tuple[Host, PromotionConfig, dict[str, str]],
    fault: str,
) -> None:
    host, config, environment = setup
    _fresh_pending(
        host,
        config,
        install_environment=True,
        production_exists=True,
    )
    environment_path = config.checkout / ".env"
    if fault == "missing":
        environment_path.unlink()
    else:
        _private(environment_path, b"TAMPERED=secret\n")

    assert main(_argv(config), environment) == 1

    assert host.production_exists
    assert host.production_image == TARGET_IMAGE
    assert not any(command[1:3] == ("container", "rm") for command in host.log)
    assert config.state_dir.joinpath("pending.json").is_file()
    assert DeploymentStateStore(config.state_dir).read_journal() == ()


def test_fresh_recovery_revalidates_environment_after_container_inspection(
    setup: tuple[Host, PromotionConfig, dict[str, str]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    host, config, environment = setup
    _fresh_pending(
        host,
        config,
        install_environment=True,
        production_exists=True,
    )
    production_lists = 0
    environment_path = config.checkout / ".env"

    def transform(
        args: list[str],
        result: subprocess.CompletedProcess[str],
    ) -> subprocess.CompletedProcess[str]:
        nonlocal production_lists
        if args[1:3] == ["container", "ls"]:
            production_lists += 1
            if production_lists == 2:
                _private(environment_path, b"RACED=secret\n")
        return result

    _strict_host_run(monkeypatch, host, transform)

    assert main(_argv(config), environment) == 1

    assert production_lists == 2
    assert host.production_exists
    assert environment_path.read_bytes() == b"RACED=secret\n"
    assert not any(command[1:3] == ("container", "rm") for command in host.log)
    assert config.state_dir.joinpath("pending.json").is_file()


def test_terminal_pointer_failure_never_compensates(
    setup: tuple[Host, PromotionConfig, dict[str, str]],
) -> None:
    host, config, environment = setup
    _initialize(host, config)
    host.make_current_publish_fail = True

    try:
        assert main(_argv(config), environment) == 4
    finally:
        config.state_dir.chmod(0o700)

    assert host.production_revision == TARGET_REVISION
    assert host.production_image == TARGET_IMAGE
    assert not any(OLD_IMAGE in command and "up" in command for command in host.log)
    assert [
        entry.event for entry in DeploymentStateStore(config.state_dir).read_journal()
    ] == [
        "adopted",
        "promoted",
    ]


def test_reconciled_terminal_stops_invocation_before_host_commands(
    setup: tuple[Host, PromotionConfig, dict[str, str]],
) -> None:
    host, config, environment = setup
    _pending(host, config)
    store = DeploymentStateStore(config.state_dir)
    with store.transaction() as transaction:
        pending = transaction.pending()
        assert pending is not None
        stale_pointer = store.pending_path.read_bytes()
        transaction.record_rollback(
            pending.transaction_id,
            persistent_volumes=(VOLUME,),
        )
    _private(store.pending_path, stale_pointer)
    command_count = len(host.log)

    assert main(_argv(config), environment) == 3

    assert len(host.log) == command_count
    assert not store.pending_path.exists()
    assert [entry.event for entry in store.read_journal()] == [
        "adopted",
        "rolled_back",
    ]


def test_cli_surface_and_fixed_environment_allowlist(
    setup: tuple[Host, PromotionConfig, dict[str, str]],
) -> None:
    host, config, environment = setup
    _initialize(host, config)

    assert tuple(SECRETS) == PRODUCTION_ENVIRONMENT_NAMES
    assert main(_argv(config), environment) == 0


def test_cli_reads_exact_production_environment_from_bounded_stdin(
    setup: tuple[Host, PromotionConfig, dict[str, str]],
) -> None:
    """Merge canonical stdin values with only approved host transport settings."""
    host, config, environment = setup
    _initialize(host, config)
    payload = serialize_compose_environment(
        PRODUCTION_ENVIRONMENT_NAMES,
        environment,
    ).encode()
    host_environment = {
        "HOME": environment["HOME"],
        "PATH": environment["PATH"],
        "DOCKER_CONFIG": "/private/docker-config",
        "DOCKER_HOST": "unix:///private/docker.sock",
        "XDG_RUNTIME_DIR": "/private/runtime",
        "UNAPPROVED_CANARY": "must-not-reach-subprocesses",
    }

    assert (
        main(
            _controller_argv(config),
            host_environment,
            io.BytesIO(payload),
        )
        == 0
    )


@pytest.mark.parametrize(
    ("stream", "expected"),
    [
        (
            io.BytesIO(b"x" * (64 * 1024 + 1)),
            "serialized environment is too large",
        ),
        (
            io.BytesIO(b"\xff"),
            "serialized environment is not UTF-8",
        ),
        (
            io.BytesIO(b'AGENT_NAME="incomplete"\n'),
            "serialized environment is invalid",
        ),
    ],
)
def test_cli_rejects_invalid_stdin_before_host_operations(
    setup: tuple[Host, PromotionConfig, dict[str, str]],
    capsys: pytest.CaptureFixture[str],
    stream: io.BytesIO,
    expected: str,
) -> None:
    """Reject malformed secret transport without invoking Git or Docker."""
    host, config, environment = setup

    assert (
        main(
            _controller_argv(config),
            {"HOME": environment["HOME"], "PATH": environment["PATH"]},
            stream,
        )
        == 1
    )

    assert host.log == []
    captured = capsys.readouterr()
    assert expected in captured.err
    assert "incomplete" not in captured.err


def test_cli_redacts_stdin_read_failure(
    setup: tuple[Host, PromotionConfig, dict[str, str]],
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Convert an input transport failure to one secret-free diagnostic."""
    host, config, environment = setup

    class FailingStream(io.BytesIO):
        def read(self, size: int | None = -1) -> bytes:
            del size
            raise OSError("private transport detail")

    assert (
        main(
            _controller_argv(config),
            {"HOME": environment["HOME"], "PATH": environment["PATH"]},
            FailingStream(),
        )
        == 1
    )

    assert host.log == []
    captured = capsys.readouterr()
    assert captured.err == "ERROR: serialized environment could not be read\n"


@pytest.mark.parametrize(
    "extra_arguments",
    [
        ["--environment-stdin"],
        ["--release-lease", "/private/missing-release-lease"],
    ],
)
def test_cli_requires_release_lease_and_stdin_as_a_pair(
    setup: tuple[Host, PromotionConfig, dict[str, str]],
    capsys: pytest.CaptureFixture[str],
    extra_arguments: list[str],
) -> None:
    """Reject either controller transport half before reading or host commands."""
    host, config, environment = setup

    assert (
        main(
            [*_argv(config), *extra_arguments],
            {"HOME": environment["HOME"], "PATH": environment["PATH"]},
            io.BytesIO(b"private-canary"),
        )
        == 1
    )

    assert host.log == []
    captured = capsys.readouterr()
    assert captured.err == (
        "ERROR: release lease and serialized environment must be used together\n"
    )
    assert "private-canary" not in captured.err


def _new_release_lease(tmp_path: Path) -> Path:
    directory = tmp_path / "release.lease"
    directory.mkdir(mode=0o700)
    lease = directory / "lock"
    lease.touch(mode=0o600)
    lease.chmod(0o600)
    return lease


@pytest.mark.parametrize(
    "fault",
    [
        "relative",
        "non-normalized",
        "missing-parent",
        "unsafe-parent-mode",
        "parent-symlink",
    ],
)
def test_release_lease_rejects_unsafe_path_or_directory(
    tmp_path: Path,
    fault: str,
) -> None:
    """Require a canonical private directory before opening the lease."""
    lease = _new_release_lease(tmp_path)
    if fault == "relative":
        selected = Path("release.lease/lock")
        message = "absolute normalized"
    elif fault == "non-normalized":
        selected = lease.parent / ".." / lease.parent.name / lease.name
        message = "absolute normalized"
    elif fault == "missing-parent":
        selected = tmp_path / "missing" / "lock"
        message = "directory is unavailable"
    elif fault == "unsafe-parent-mode":
        lease.parent.chmod(0o755)
        selected = lease
        message = "directory is unsafe"
    else:
        target = tmp_path / "target.lease"
        target.mkdir(mode=0o700)
        target.joinpath("lock").write_bytes(b"")
        target.joinpath("lock").chmod(0o600)
        link = tmp_path / "linked.lease"
        link.symlink_to(target, target_is_directory=True)
        selected = link / "lock"
        message = "absolute normalized"

    with (
        pytest.raises(promotion_module.PromotionError, match=message),
        promotion_module._release_lease(selected),
    ):
        pytest.fail("unsafe lease was acquired")


@pytest.mark.parametrize("fault", ["missing", "symlink", "mode", "hardlink"])
def test_release_lease_rejects_unsafe_lock_file(
    tmp_path: Path,
    fault: str,
) -> None:
    """Reject missing, redirected, public, and multiply linked lock files."""
    lease = _new_release_lease(tmp_path)
    if fault == "missing":
        lease.unlink()
        message = "is unavailable"
    elif fault == "symlink":
        lease.unlink()
        target = tmp_path / "target-lock"
        target.write_bytes(b"")
        target.chmod(0o600)
        lease.symlink_to(target)
        message = "absolute normalized"
    elif fault == "mode":
        lease.chmod(0o644)
        message = "is unsafe"
    else:
        os.link(lease, tmp_path / "second-lock-link")
        message = "is unsafe"

    with (
        pytest.raises(promotion_module.PromotionError, match=message),
        promotion_module._release_lease(lease),
    ):
        pytest.fail("unsafe lease was acquired")


def test_release_lease_rejects_contention_and_releases_after_failure(
    tmp_path: Path,
) -> None:
    """Fail closed on contention and always unlock after the controlled body."""
    lease = _new_release_lease(tmp_path)
    competing = os.open(lease, os.O_RDWR)
    try:
        fcntl.flock(competing, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with (
            pytest.raises(
                promotion_module.DeploymentLockBusyError,
                match="another release controller",
            ),
            promotion_module._release_lease(lease),
        ):
            pytest.fail("contended lease was acquired")
    finally:
        fcntl.flock(competing, fcntl.LOCK_UN)
        os.close(competing)

    with (
        pytest.raises(RuntimeError, match="body failure"),
        promotion_module._release_lease(lease),
    ):
        raise RuntimeError("body failure")

    descriptor = os.open(lease, os.O_RDWR)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(descriptor, fcntl.LOCK_UN)
    finally:
        os.close(descriptor)


def test_release_lease_redacts_unexpected_lock_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Convert an unexpected host lock failure to a stable public diagnostic."""
    lease = _new_release_lease(tmp_path)

    def fail_lock(_descriptor: int, operation: int) -> None:
        assert operation == fcntl.LOCK_EX | fcntl.LOCK_NB
        raise OSError(errno.EIO, "private lock detail")

    monkeypatch.setattr(promotion_module.fcntl, "flock", fail_lock)

    with (
        pytest.raises(
            promotion_module.PromotionError,
            match="release lease could not be acquired",
        ),
        promotion_module._release_lease(lease),
    ):
        pytest.fail("failed lease was acquired")


def test_controller_holds_lease_before_stdin_and_through_promotion(
    setup: tuple[Host, PromotionConfig, dict[str, str]],
) -> None:
    """Prove the lease spans secret input parsing and every host command."""
    host, config, environment = setup
    _initialize(host, config)
    arguments = _controller_argv(config)
    lease = (
        config.release_checkout.with_name(f"{config.release_checkout.name}.lease")
        / "lock"
    )
    host.lease_probe = lease
    payload = serialize_compose_environment(
        PRODUCTION_ENVIRONMENT_NAMES,
        environment,
    ).encode()

    class LockProbingStream(io.BytesIO):
        def read(self, size: int | None = -1) -> bytes:
            descriptor = os.open(lease, os.O_RDWR)
            try:
                with pytest.raises(OSError) as blocked:
                    fcntl.flock(
                        descriptor,
                        fcntl.LOCK_EX | fcntl.LOCK_NB,
                    )
                assert blocked.value.errno in {errno.EACCES, errno.EAGAIN}
            finally:
                os.close(descriptor)
            return super().read(size)

    assert main(arguments, environment, LockProbingStream(payload)) == 0

    descriptor = os.open(lease, os.O_RDWR)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(descriptor, fcntl.LOCK_UN)
    finally:
        os.close(descriptor)


def test_explicit_adoption_happens_before_promotion(
    setup: tuple[Host, PromotionConfig, dict[str, str]],
) -> None:
    _host, config, environment = setup

    promote(config, environment=environment)

    assert [
        entry.event for entry in DeploymentStateStore(config.state_dir).read_journal()
    ] == ["adopted", "promoted"]


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("state_dir", Path("relative"), "absolute normalized"),
        ("expected_origin", f"{ORIGIN}.git", "expected origin"),
        ("compose_project", "UPPER", "Compose project"),
        ("compose_service", "bad.name", "Compose service"),
        ("source_revision", "a" * 39, "source revision"),
        ("image_reference", "agent:latest", "immutable digest"),
    ],
)
def test_invalid_controller_inputs_fail_before_external_mutation(
    setup: tuple[Host, PromotionConfig, dict[str, str]],
    field: str,
    value: object,
    message: str,
) -> None:
    host, config, environment = setup
    if field == "state_dir":
        assert isinstance(value, Path)
        changed = replace(config, state_dir=value)
    else:
        assert isinstance(value, str)
        if field == "expected_origin":
            changed = replace(config, expected_origin=value)
        elif field == "compose_project":
            changed = replace(config, compose_project=value)
        elif field == "compose_service":
            changed = replace(config, compose_service=value)
        elif field == "source_revision":
            changed = replace(config, source_revision=value)
        elif field == "image_reference":
            changed = replace(config, image_reference=value)
        else:
            raise AssertionError(field)

    with pytest.raises(promotion_module.PromotionError, match=message):
        promote(changed, environment=environment)

    assert not host.log


def test_transaction_identity_binds_the_pre_cutover_revision() -> None:
    selected = promotion_module._new_transaction_id(OLD_REVISION)

    assert len(selected) == 64
    assert selected.endswith(f"-{OLD_REVISION}")
    with pytest.raises(
        promotion_module.PromotionError,
        match="baseline revision is invalid",
    ):
        promotion_module._new_transaction_id("not-a-revision")


@pytest.mark.parametrize("value", ["", "bad\nline", "bad\rline", "bad\0line"])
def test_one_line_rejects_ambiguous_command_output(value: str) -> None:
    with pytest.raises(promotion_module.PromotionError):
        promotion_module._one_line(value, "test")


@pytest.mark.parametrize("value", ["not-json", "{}", "[]", "[1]", "[{},{}]"])
def test_json_object_rejects_noncanonical_shape(value: str) -> None:
    with pytest.raises(promotion_module.PromotionError):
        promotion_module._json_object(value, "test")


def test_typed_output_helpers_reject_wrong_types() -> None:
    with pytest.raises(promotion_module.PromotionError):
        promotion_module._mapping({"value": "wrong"}, "value", "test")
    with pytest.raises(promotion_module.PromotionError):
        promotion_module._string({"value": 1}, "value", "test")


def test_environment_hash_rejects_missing_and_empty(tmp_path: Path) -> None:
    with pytest.raises(promotion_module.PromotionError):
        promotion_module._environment_sha256(tmp_path / "missing")
    empty = tmp_path / "empty"
    empty.touch()
    with pytest.raises(promotion_module.PromotionError):
        promotion_module._environment_sha256(empty)


def test_run_redacts_process_failures_and_bounds_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executable = tmp_path / "tool"
    executable.touch()
    secret = f"private-{tmp_path.name}"
    real_run = subprocess.run
    timeout_mock = create_autospec(
        real_run,
        spec_set=True,
        side_effect=subprocess.TimeoutExpired(
            ["tool", secret],
            1,
            output=secret,
            stderr=secret,
        ),
    )
    monkeypatch.setattr(subprocess, "run", timeout_mock)
    with pytest.raises(promotion_module.PromotionError) as timeout_error:
        promotion_module._run(
            executable,
            (secret,),
            operation=promotion_module.CommandOperation.CANDIDATE_START,
            environment={"PRIVATE": secret},
        )
    assert str(timeout_error.value) == (
        "deployment command timed out during candidate Compose start"
    )
    assert secret not in str(timeout_error.value)

    failed_mock = create_autospec(
        real_run,
        spec_set=True,
        return_value=subprocess.CompletedProcess(
            ["tool", secret],
            9,
            secret,
            secret,
        ),
    )
    monkeypatch.setattr(subprocess, "run", failed_mock)
    with pytest.raises(promotion_module.PromotionError) as failed_error:
        promotion_module._run(
            executable,
            (secret,),
            operation=promotion_module.CommandOperation.CANDIDATE_START,
            environment={"PRIVATE": secret},
        )
    assert str(failed_error.value) == (
        "deployment command failed during candidate Compose start (exit 9)"
    )
    assert secret not in str(failed_error.value)

    os_error_mock = create_autospec(
        real_run,
        spec_set=True,
        side_effect=OSError(secret),
    )
    monkeypatch.setattr(subprocess, "run", os_error_mock)
    with pytest.raises(promotion_module.PromotionError) as os_error:
        promotion_module._run(
            executable,
            (secret,),
            operation=promotion_module.CommandOperation.CANDIDATE_START,
            environment={"PRIVATE": secret},
        )
    assert str(os_error.value) == (
        "deployment command could not start during candidate Compose start"
    )
    assert secret not in str(os_error.value)

    large_mock = create_autospec(
        real_run,
        spec_set=True,
        return_value=subprocess.CompletedProcess(
            ["tool"],
            0,
            "x" * (promotion_module._MAX_COMMAND_OUTPUT_BYTES + 1),
            "",
        ),
    )
    monkeypatch.setattr(subprocess, "run", large_mock)
    with pytest.raises(promotion_module.PromotionError, match="too large"):
        promotion_module._run(
            executable,
            (),
            operation=promotion_module.CommandOperation.CANDIDATE_START,
            environment={},
        )


def test_missing_environment_and_line_injection_are_redacted(
    setup: tuple[Host, PromotionConfig, dict[str, str]],
    capsys: pytest.CaptureFixture[str],
) -> None:
    _host, config, environment = setup
    missing = dict(environment)
    secret = missing.pop("DATABASE_URL")
    assert main(_argv(config), missing) == 1
    assert secret not in capsys.readouterr().err

    hostile = dict(environment)
    hostile["DATABASE_URL"] = "before\nprivate-after"
    assert main(_argv(config), hostile) == 1
    assert "private-after" not in capsys.readouterr().err


def _strict_run(
    monkeypatch: pytest.MonkeyPatch,
    side_effect: object,
) -> None:
    run_mock = create_autospec(
        REAL_SUBPROCESS_RUN,
        spec_set=True,
        side_effect=side_effect,
    )
    monkeypatch.setattr(subprocess, "run", run_mock)


BoundaryTransform = Callable[
    [list[str], subprocess.CompletedProcess[str]],
    subprocess.CompletedProcess[str],
]


def _strict_host_run(
    monkeypatch: pytest.MonkeyPatch,
    host: Host,
    transform: BoundaryTransform,
) -> None:
    def boundary(
        args: list[str],
        *,
        cwd: Path | None = None,
        env: dict[str, str],
        text: bool,
        capture_output: bool,
        check: bool,
        timeout: int,
    ) -> subprocess.CompletedProcess[str]:
        result = host.run(
            args,
            cwd=cwd,
            env=env,
            text=text,
            capture_output=capture_output,
            check=check,
            timeout=timeout,
        )
        return transform(args, result)

    _strict_run(monkeypatch, boundary)


def _executables(host: Host) -> promotion_module.Executables:
    return promotion_module.Executables(
        git=Path(host.executable("git")),
        docker=Path(host.executable("docker")),
    )


def _completed_json(
    args: list[str],
    document: object,
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(args, 0, json.dumps([document]), "")


def test_config_rejects_colliding_and_unavailable_checkouts(
    setup: tuple[Host, PromotionConfig, dict[str, str]],
    tmp_path: Path,
) -> None:
    host, config, environment = setup
    with pytest.raises(
        promotion_module.PromotionError,
        match="must be distinct",
    ):
        promote(
            replace(config, release_checkout=config.checkout), environment=environment
        )

    missing = tmp_path / "missing"
    with pytest.raises(promotion_module.PromotionError, match="is unavailable"):
        promote(replace(config, release_checkout=missing), environment=environment)

    regular_file = tmp_path / "not-a-checkout"
    regular_file.write_text("not a directory", encoding="utf-8")
    with pytest.raises(promotion_module.PromotionError, match="is not a directory"):
        promote(
            replace(config, release_checkout=regular_file),
            environment=environment,
        )

    assert not host.log


@pytest.mark.parametrize("kind", ["missing", "dangling", "directory"])
def test_executable_resolution_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
) -> None:
    selected: str | None
    if kind == "missing":
        selected = None
    elif kind == "dangling":
        selected = str(tmp_path / "does-not-exist")
    else:
        selected = str(tmp_path)
    which_mock = create_autospec(
        REAL_SHUTIL_WHICH,
        spec_set=True,
        return_value=selected,
    )
    monkeypatch.setattr(shutil, "which", which_mock)

    with pytest.raises(promotion_module.PromotionError, match="executable"):
        promotion_module._resolve_executable("docker", {"PATH": str(tmp_path)})


@pytest.mark.parametrize(
    ("checkout_kind", "fault", "message"),
    [
        ("release", "root-unavailable", "release checkout root is unavailable"),
        ("release", "root-mismatch", "release checkout root does not match"),
        ("release", "origin", "release checkout origin does not match"),
        ("release", "revision", "release checkout revision does not match"),
        ("release", "dirty", "release checkout has tracked changes"),
        ("production", "root-unavailable", "production checkout root is unavailable"),
        ("production", "root-mismatch", "production checkout root does not match"),
        ("production", "origin", "production checkout origin does not match"),
        ("production", "dirty", "production checkout has tracked changes"),
    ],
)
def test_checkout_proofs_reject_drift_before_promotion(
    setup: tuple[Host, PromotionConfig, dict[str, str]],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    checkout_kind: str,
    fault: str,
    message: str,
) -> None:
    host, config, environment = setup
    selected_checkout = (
        config.release_checkout if checkout_kind == "release" else config.checkout
    )

    def altered_run(
        args: list[str],
        *,
        cwd: Path | None = None,
        env: dict[str, str],
        text: bool,
        capture_output: bool,
        check: bool,
        timeout: int,
    ) -> subprocess.CompletedProcess[str]:
        result = host.run(
            args,
            cwd=cwd,
            env=env,
            text=text,
            capture_output=capture_output,
            check=check,
            timeout=timeout,
        )
        tail = args[3:] if args[1:2] == ["-C"] else args[1:]
        path = Path(args[2]) if args[1:2] == ["-C"] else None
        if path != selected_checkout:
            return result
        if fault == "root-unavailable" and tail == [
            "rev-parse",
            "--show-toplevel",
        ]:
            return host.completed(args, stdout=f"{tmp_path / 'gone'}\n")
        if fault == "root-mismatch" and tail == ["rev-parse", "--show-toplevel"]:
            other = (
                config.checkout
                if selected_checkout == config.release_checkout
                else config.release_checkout
            )
            return host.completed(args, stdout=f"{other}\n")
        if fault == "origin" and tail == ["remote", "get-url", "origin"]:
            return host.completed(
                args,
                stdout="https://github.com/QueryPlanner/other\n",
            )
        if fault == "revision" and tail == ["rev-parse", "--verify", "HEAD"]:
            return host.completed(args, stdout=f"{OLD_REVISION}\n")
        if fault == "dirty" and tail == [
            "diff",
            "--no-ext-diff",
            "--no-textconv",
            "--quiet",
            "--",
        ]:
            return host.completed(args, returncode=1)
        return result

    _strict_run(monkeypatch, altered_run)

    with pytest.raises(promotion_module.PromotionError, match=message):
        promote(config, environment=environment)


def test_release_checkout_rejects_ignored_or_untracked_bytes(
    setup: tuple[Host, PromotionConfig, dict[str, str]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject bytes outside the exact commit before deployment mutation."""
    host, config, environment = setup

    def transform(
        args: list[str],
        result: subprocess.CompletedProcess[str],
    ) -> subprocess.CompletedProcess[str]:
        if args[1:3] == ["-C", str(config.release_checkout)] and args[3:] == [
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
            "--ignored=matching",
        ]:
            return host.completed(
                args,
                stdout="!! src/agent/__pycache__/\n",
            )
        return result

    _strict_host_run(monkeypatch, host, transform)

    with pytest.raises(
        promotion_module.PromotionError,
        match="untracked or ignored",
    ):
        promote(config, environment=environment)


@pytest.mark.parametrize(
    ("document", "message"),
    [
        (
            {
                "Id": "not-an-image-id",
                "RepoDigests": [TARGET_IMAGE],
                "Config": {
                    "Labels": {"org.opencontainers.image.revision": TARGET_REVISION}
                },
            },
            "local image ID is invalid",
        ),
        (
            {
                "Id": TARGET_IMAGE_ID,
                "RepoDigests": "not-a-list",
                "Config": {
                    "Labels": {"org.opencontainers.image.revision": TARGET_REVISION}
                },
            },
            "local image digest does not match",
        ),
        (
            {
                "Id": TARGET_IMAGE_ID,
                "RepoDigests": [TARGET_IMAGE],
                "Config": {
                    "Labels": {"org.opencontainers.image.revision": OLD_REVISION}
                },
            },
            "image OCI revision does not match",
        ),
    ],
)
def test_image_proof_rejects_unproven_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    document: dict[str, object],
    message: str,
) -> None:
    def boundary(
        args: list[str],
        **_kwargs: object,
    ) -> subprocess.CompletedProcess[str]:
        return _completed_json(args, document)

    _strict_run(monkeypatch, boundary)
    executables = promotion_module.Executables(
        git=tmp_path / "git",
        docker=tmp_path / "docker",
    )

    with pytest.raises(promotion_module.PromotionError, match=message):
        promotion_module._inspect_image(
            proof_reference=TARGET_IMAGE,
            expected_revision=TARGET_REVISION,
            executables=executables,
            environment={},
        )


def test_container_list_rejects_non_hex_ids(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def boundary(
        args: list[str],
        **_kwargs: object,
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args, 0, "not-a-container\n", "")

    _strict_run(monkeypatch, boundary)
    with pytest.raises(promotion_module.PromotionError, match="container list"):
        promotion_module._container_ids(
            project=PROJECT,
            service=SERVICE,
            executables=promotion_module.Executables(
                git=tmp_path / "git",
                docker=tmp_path / "docker",
            ),
            environment={},
        )


def _volume_mount(
    *,
    name: str = VOLUME.name,
    destination: str = VOLUME.destination,
    driver: str = VOLUME.driver,
) -> dict[str, object]:
    return {
        "Type": "volume",
        "Name": name,
        "Driver": driver,
        "Destination": destination,
    }


def _volume_document(
    *,
    name: str = VOLUME.name,
    driver: str = VOLUME.driver,
    mountpoint: str = VOLUME.mountpoint,
    created_at: str = VOLUME.created_at,
) -> dict[str, object]:
    return {
        "Name": name,
        "Driver": driver,
        "Mountpoint": mountpoint,
        "CreatedAt": created_at,
    }


@pytest.mark.parametrize(
    ("mounts", "documents", "message"),
    [
        ("wrong", [], "mount observation"),
        ([1], [], "mount observation"),
        ([_volume_mount(destination="/")], [], "mount identity"),
        (
            [_volume_mount()],
            [_volume_document(driver="different")],
            "daemon identity does not match",
        ),
        (
            [_volume_mount()],
            [_volume_document(mountpoint="/")],
            "daemon identity is invalid",
        ),
        (
            [_volume_mount()],
            [_volume_document(created_at="not-a-time")],
            "creation time is invalid",
        ),
        (
            [_volume_mount()],
            [_volume_document(created_at="2026-07-28T12:00:00")],
            "creation time is invalid",
        ),
        (
            [_volume_mount()],
            [_volume_document(created_at="2026-07-28 12:00:00+00:00")],
            "creation time is invalid",
        ),
        (
            [_volume_mount(), _volume_mount()],
            [_volume_document(), _volume_document()],
            "identities are ambiguous",
        ),
    ],
)
def test_volume_proofs_reject_unsafe_or_ambiguous_mounts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mounts: object,
    documents: list[dict[str, object]],
    message: str,
) -> None:
    remaining = iter(documents)

    def boundary(
        args: list[str],
        **_kwargs: object,
    ) -> subprocess.CompletedProcess[str]:
        return _completed_json(args, next(remaining))

    _strict_run(monkeypatch, boundary)
    with pytest.raises(promotion_module.PromotionError, match=message):
        promotion_module._volume_proofs(
            mounts=mounts,
            executables=promotion_module.Executables(
                git=tmp_path / "git",
                docker=tmp_path / "docker",
            ),
            environment={},
        )


@pytest.mark.parametrize(
    "created_at",
    [
        "2026-07-28T12:00:00Z",
        "2026-07-28T12:00:00.123456789Z",
        "2026-07-28T17:30:00.123+05:30",
    ],
)
def test_volume_proofs_preserve_exact_daemon_creation_time(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    created_at: str,
) -> None:
    def boundary(
        args: list[str],
        **_kwargs: object,
    ) -> subprocess.CompletedProcess[str]:
        return _completed_json(args, _volume_document(created_at=created_at))

    _strict_run(monkeypatch, boundary)

    proofs = promotion_module._volume_proofs(
        mounts=[_volume_mount()],
        executables=promotion_module.Executables(
            git=tmp_path / "git",
            docker=tmp_path / "docker",
        ),
        environment={},
    )

    assert len(proofs) == 1
    assert proofs[0].created_at == created_at
    assert proofs[0].recorded_identity().created_at == created_at


def test_volume_proofs_ignore_non_volume_mounts(tmp_path: Path) -> None:
    assert (
        promotion_module._volume_proofs(
            mounts=[{"Type": "bind"}],
            executables=promotion_module.Executables(
                git=tmp_path / "git",
                docker=tmp_path / "docker",
            ),
            environment={},
        )
        == ()
    )


def _container_document(
    *,
    container_id: str = TARGET_CONTAINER,
    status: str = "running",
    health: str = "healthy",
    image_reference: str = TARGET_IMAGE,
    image_id: str = TARGET_IMAGE_ID,
    project: str = PROJECT,
) -> dict[str, object]:
    return {
        "Id": container_id,
        "Image": image_id,
        "State": {"Status": status, "Health": {"Status": health}},
        "Config": {
            "Image": image_reference,
            "Labels": {
                "com.docker.compose.project": project,
                "com.docker.compose.service": SERVICE,
                "com.docker.compose.project.working_dir": "/checkout",
            },
        },
        "Mounts": [],
    }


@pytest.mark.parametrize(
    ("document", "message"),
    [
        (_container_document(container_id="bad"), "container identity"),
        (_container_document(health="unhealthy"), "running and healthy"),
        (_container_document(image_id=OLD_IMAGE_ID), "image identity"),
        (_container_document(project="other"), "Compose labels"),
    ],
)
def test_container_inspection_rejects_identity_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    document: dict[str, object],
    message: str,
) -> None:
    def boundary(
        args: list[str],
        **_kwargs: object,
    ) -> subprocess.CompletedProcess[str]:
        return _completed_json(args, document)

    _strict_run(monkeypatch, boundary)
    with pytest.raises(promotion_module.PromotionError, match=message):
        promotion_module._inspect_container(
            container_id=TARGET_CONTAINER[:12],
            expected_checkout=Path("/checkout"),
            expected_project=PROJECT,
            expected_service=SERVICE,
            image=promotion_module.ImageProof(
                TARGET_IMAGE,
                TARGET_IMAGE_ID,
                TARGET_REVISION,
            ),
            executables=promotion_module.Executables(
                git=tmp_path / "git",
                docker=tmp_path / "docker",
            ),
            environment={},
        )


def test_run_bounds_stderr_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executable = tmp_path / "tool"
    executable.touch()
    run_mock = create_autospec(
        REAL_SUBPROCESS_RUN,
        spec_set=True,
        return_value=subprocess.CompletedProcess(
            ["tool"],
            0,
            "",
            "x" * (promotion_module._MAX_COMMAND_OUTPUT_BYTES + 1),
        ),
    )
    monkeypatch.setattr(subprocess, "run", run_mock)
    with pytest.raises(promotion_module.PromotionError, match="too large"):
        promotion_module._run(
            executable,
            (),
            operation=promotion_module.CommandOperation.CANDIDATE_START,
            environment={},
        )


def test_main_redacts_real_host_oserror(
    setup: tuple[Host, PromotionConfig, dict[str, str]],
    capsys: pytest.CaptureFixture[str],
) -> None:
    _host, config, environment = setup
    blocked = replace(config, state_dir=Path("/dev/null/state"))

    assert main(_argv(blocked), environment) == 1
    assert "deployment host operation failed" in capsys.readouterr().err


def test_module_entrypoint_returns_process_status(
    setup: tuple[Host, PromotionConfig, dict[str, str]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _host, config, _environment = setup
    monkeypatch.setattr("sys.argv", ["deployment_promotion", *_argv(config)])
    with pytest.raises(SystemExit) as raised:
        runpy.run_module("agent.deployment_promotion", run_name="__main__")
    assert raised.value.code == 1


def test_pending_recovery_rejects_ambiguous_production(
    setup: tuple[Host, PromotionConfig, dict[str, str]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    host, config, environment = setup
    _pending(host, config)

    def transform(
        args: list[str],
        result: subprocess.CompletedProcess[str],
    ) -> subprocess.CompletedProcess[str]:
        if args[1:3] == ["container", "ls"] and any(
            value == f"label=com.docker.compose.project={PROJECT}" for value in args
        ):
            return host.completed(
                args,
                stdout=f"{TARGET_CONTAINER[:12]}\n{'4' * 12}\n",
            )
        return result

    _strict_host_run(monkeypatch, host, transform)

    assert main(_argv(config), environment) == 1
    assert config.state_dir.joinpath("pending.json").is_file()
    assert host.production_revision == TARGET_REVISION


def test_pending_recovery_accepts_absent_service_then_restores(
    setup: tuple[Host, PromotionConfig, dict[str, str]],
) -> None:
    host, config, environment = setup
    _pending(host, config)
    host.production_exists = False

    assert main(_argv(config), environment) == 3

    assert host.production_exists
    assert host.production_revision == OLD_REVISION
    assert [
        entry.event for entry in DeploymentStateStore(config.state_dir).read_journal()
    ] == ["adopted", "rolled_back"]


def test_runtime_observation_returns_none_for_absent_service(
    setup: tuple[Host, PromotionConfig, dict[str, str]],
) -> None:
    host, config, environment = setup
    host.production_exists = False

    assert (
        promotion_module._observe_runtime(
            config=config,
            image=promotion_module.ImageProof(
                OLD_IMAGE,
                OLD_IMAGE_ID,
                OLD_REVISION,
            ),
            executables=_executables(host),
            environment=promotion_module._command_environment(environment),
        )
        is None
    )


def test_runtime_observation_rejects_second_read_ambiguity(
    setup: tuple[Host, PromotionConfig, dict[str, str]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    host, config, environment = setup
    production_lists = 0

    def transform(
        args: list[str],
        result: subprocess.CompletedProcess[str],
    ) -> subprocess.CompletedProcess[str]:
        nonlocal production_lists
        if args[1:3] == ["container", "ls"] and any(
            value == f"label=com.docker.compose.project={PROJECT}" for value in args
        ):
            production_lists += 1
            if production_lists == 2:
                return host.completed(
                    args,
                    stdout=f"{OLD_CONTAINER[:12]}\n{'4' * 12}\n",
                )
        return result

    _strict_host_run(monkeypatch, host, transform)

    with pytest.raises(promotion_module.PromotionError, match="ambiguous"):
        promotion_module._observe_runtime(
            config=config,
            image=promotion_module.ImageProof(
                OLD_IMAGE,
                OLD_IMAGE_ID,
                OLD_REVISION,
            ),
            executables=_executables(host),
            environment=promotion_module._command_environment(environment),
        )


def test_runtime_observation_rejects_oci_revision_race(
    setup: tuple[Host, PromotionConfig, dict[str, str]],
) -> None:
    host, config, environment = setup
    with pytest.raises(promotion_module.PromotionError, match="image does not match"):
        promotion_module._observe_runtime(
            config=config,
            image=promotion_module.ImageProof(
                OLD_IMAGE,
                OLD_IMAGE_ID,
                TARGET_REVISION,
            ),
            executables=_executables(host),
            environment=promotion_module._command_environment(environment),
        )


def test_baseline_rejects_invalid_checkout_revision(
    setup: tuple[Host, PromotionConfig, dict[str, str]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    host, config, environment = setup

    def transform(
        args: list[str],
        result: subprocess.CompletedProcess[str],
    ) -> subprocess.CompletedProcess[str]:
        if args[1:3] == ["-C", str(config.checkout)] and args[3:] == [
            "rev-parse",
            "--verify",
            "HEAD",
        ]:
            return host.completed(args, stdout="not-a-revision\n")
        return result

    _strict_host_run(monkeypatch, host, transform)
    with pytest.raises(promotion_module.PromotionError, match="revision is invalid"):
        promote(config, environment=environment)


def test_baseline_redacts_legacy_observation_failure(
    setup: tuple[Host, PromotionConfig, dict[str, str]],
) -> None:
    _host, config, environment = setup
    (config.checkout / ".env").chmod(0o644)

    with pytest.raises(
        promotion_module.PromotionError,
        match="observation failed",
    ):
        promote(config, environment=environment)


def test_baseline_rejects_missing_recorded_runtime(
    setup: tuple[Host, PromotionConfig, dict[str, str]],
) -> None:
    host, config, environment = setup
    _initialize(host, config)
    host.production_exists = False

    with pytest.raises(
        promotion_module.PromotionError,
        match="recorded production deployment is unavailable",
    ):
        promote(config, environment=environment)


def test_first_install_rejects_unrecorded_environment(
    setup: tuple[Host, PromotionConfig, dict[str, str]],
) -> None:
    host, config, environment = setup
    host.production_exists = False

    with pytest.raises(
        promotion_module.PromotionError,
        match="unrecorded production environment",
    ):
        promote(config, environment=environment)


def test_baseline_rejects_runtime_disappearing_between_proofs(
    setup: tuple[Host, PromotionConfig, dict[str, str]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    host, config, environment = setup
    production_lists = 0

    def transform(
        args: list[str],
        result: subprocess.CompletedProcess[str],
    ) -> subprocess.CompletedProcess[str]:
        nonlocal production_lists
        if args[1:3] == ["container", "ls"] and any(
            value == f"label=com.docker.compose.project={PROJECT}" for value in args
        ):
            production_lists += 1
            if production_lists == 2:
                return host.completed(args)
        return result

    _strict_host_run(monkeypatch, host, transform)
    with pytest.raises(promotion_module.PromotionError, match="disappeared"):
        promote(config, environment=environment)


def test_baseline_requires_explicit_legacy_adoption(
    setup: tuple[Host, PromotionConfig, dict[str, str]],
) -> None:
    _host, config, environment = setup
    with pytest.raises(
        promotion_module.PromotionError,
        match="is not recorded",
    ):
        promote(replace(config, adopt_existing=False), environment=environment)


def test_baseline_rejects_recorded_environment_drift(
    setup: tuple[Host, PromotionConfig, dict[str, str]],
) -> None:
    host, config, environment = setup
    _initialize(host, config)
    _private(config.checkout / ".env", b'AGENT_NAME="changed"\n')

    with pytest.raises(
        promotion_module.PromotionError,
        match="recorded and running",
    ):
        promote(config, environment=environment)


def test_baseline_rejects_replaced_volume_before_candidate(
    setup: tuple[Host, PromotionConfig, dict[str, str]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    host, config, environment = setup
    _initialize(host, config)
    promote(config, environment=environment)
    command_count = len(host.log)

    def transform(
        args: list[str],
        result: subprocess.CompletedProcess[str],
    ) -> subprocess.CompletedProcess[str]:
        if args[1:3] == ["volume", "inspect"]:
            document = json.loads(result.stdout)[0]
            document["CreatedAt"] = REPLACED_VOLUME.created_at
            return _completed_json(args, document)
        return result

    _strict_host_run(monkeypatch, host, transform)

    with pytest.raises(
        promotion_module.PromotionError,
        match="persistent volume identity changed",
    ):
        promote(config, environment=environment)

    assert not any(
        "candidate-" in " ".join(command) for command in host.log[command_count:]
    )


@pytest.mark.parametrize("fault", ["ambiguous", "volume", "cleanup", "config"])
def test_candidate_defenses_leave_production_untouched(
    setup: tuple[Host, PromotionConfig, dict[str, str]],
    monkeypatch: pytest.MonkeyPatch,
    fault: str,
) -> None:
    host, config, environment = setup
    _initialize(host, config)

    def transform(
        args: list[str],
        result: subprocess.CompletedProcess[str],
    ) -> subprocess.CompletedProcess[str]:
        candidate_project = next(
            (
                args[index + 1]
                for index, value in enumerate(args)
                if value == "--project-name" and index + 1 < len(args)
            ),
            "",
        )
        is_candidate = candidate_project.startswith("candidate-")
        if (
            fault == "ambiguous"
            and args[1:3] == ["container", "ls"]
            and any(
                value.startswith(
                    "label=com.docker.compose.project=candidate-",
                )
                for value in args
            )
        ):
            return host.completed(
                args,
                stdout=f"{CANDIDATE_CONTAINER[:12]}\n{'4' * 12}\n",
            )
        if (
            fault == "volume"
            and args[1:3] == ["container", "inspect"]
            and args[3].startswith(CANDIDATE_CONTAINER[:12])
        ):
            document = json.loads(result.stdout)[0]
            document["Mounts"] = [_volume_mount()]
            return _completed_json(args, document)
        if fault == "cleanup" and is_candidate and "down" in args:
            return host.completed(args, returncode=44)
        if fault == "config" and is_candidate and "config" in args:
            return host.completed(args, returncode=45)
        return result

    _strict_host_run(monkeypatch, host, transform)

    assert main(_argv(config), environment) == 1
    assert host.production_revision == OLD_REVISION
    assert host.production_image == OLD_IMAGE
    assert not config.state_dir.joinpath("pending.json").exists()


@pytest.mark.parametrize("fault", ["revision", "dirty"])
def test_post_intent_checkout_proof_failure_rolls_back(
    setup: tuple[Host, PromotionConfig, dict[str, str]],
    monkeypatch: pytest.MonkeyPatch,
    fault: str,
) -> None:
    host, config, environment = setup
    _initialize(host, config)
    target_checkout_seen = False
    injected = False

    def transform(
        args: list[str],
        result: subprocess.CompletedProcess[str],
    ) -> subprocess.CompletedProcess[str]:
        nonlocal target_checkout_seen, injected
        if (
            args[0].endswith("/git")
            and "checkout" in args
            and args[-1] == TARGET_REVISION
        ):
            target_checkout_seen = True
            return result
        if target_checkout_seen and not injected:
            if fault == "revision" and args[3:] == [
                "rev-parse",
                "--verify",
                "HEAD",
            ]:
                injected = True
                return host.completed(args, stdout=f"{OLD_REVISION}\n")
            if fault == "dirty" and args[3:] == [
                "diff",
                "--no-ext-diff",
                "--no-textconv",
                "--quiet",
                "--",
            ]:
                injected = True
                return host.completed(args, returncode=1)
        return result

    _strict_host_run(monkeypatch, host, transform)

    assert main(_argv(config), environment) == 1
    assert injected
    assert host.production_revision == OLD_REVISION
    assert host.production_image == OLD_IMAGE


def test_production_compose_image_drift_rolls_back(
    setup: tuple[Host, PromotionConfig, dict[str, str]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    host, config, environment = setup
    _initialize(host, config)
    injected = False

    def transform(
        args: list[str],
        result: subprocess.CompletedProcess[str],
    ) -> subprocess.CompletedProcess[str]:
        nonlocal injected
        if (
            not injected
            and args[1:2] == ["compose"]
            and "--project-name" in args
            and args[args.index("--project-name") + 1] == PROJECT
            and "config" in args
            and "--images" in args
        ):
            injected = True
            return host.completed(args, stdout=f"{OLD_IMAGE}\n")
        return result

    _strict_host_run(monkeypatch, host, transform)

    assert main(_argv(config), environment) == 1
    assert injected
    assert host.production_revision == OLD_REVISION
    assert host.production_image == OLD_IMAGE


@pytest.mark.parametrize(
    "fault",
    ["unavailable", "revision", "volume", "first-install-volume"],
)
def test_target_verification_failures_never_commit(
    setup: tuple[Host, PromotionConfig, dict[str, str]],
    monkeypatch: pytest.MonkeyPatch,
    fault: str,
) -> None:
    host, config, environment = setup
    if fault == "first-install-volume":
        host.production_exists = False
        host.volume_exists = False
        (host.checkout / ".env").unlink()
    else:
        _initialize(host, config)
    target_up_seen = False

    def transform(
        args: list[str],
        result: subprocess.CompletedProcess[str],
    ) -> subprocess.CompletedProcess[str]:
        nonlocal target_up_seen
        if (
            args[1:2] == ["compose"]
            and "--project-name" in args
            and args[args.index("--project-name") + 1] == PROJECT
            and "up" in args
            and host.production_image == TARGET_IMAGE
        ):
            target_up_seen = True
        if not target_up_seen:
            return result
        if (
            fault == "unavailable"
            and args[1:3] == ["container", "ls"]
            and host.production_image == TARGET_IMAGE
        ):
            return host.completed(args)
        if (
            fault == "revision"
            and args[0].endswith("/git")
            and args[1:3] == ["-C", str(config.checkout)]
            and args[3:] == ["rev-parse", "--verify", "HEAD"]
            and host.production_image == TARGET_IMAGE
        ):
            return host.completed(args, stdout=f"{OLD_REVISION}\n")
        if (
            fault == "volume"
            and args[1:3] == ["volume", "inspect"]
            and host.production_image == TARGET_IMAGE
        ):
            document = json.loads(result.stdout)[0]
            document["CreatedAt"] = REPLACED_VOLUME.created_at
            return _completed_json(args, document)
        if (
            fault == "first-install-volume"
            and args[1:3] == ["container", "inspect"]
            and args[3].startswith(TARGET_CONTAINER[:12])
        ):
            document = json.loads(result.stdout)[0]
            document["Mounts"] = []
            return _completed_json(args, document)
        return result

    _strict_host_run(monkeypatch, host, transform)

    assert main(_argv(config), environment) == 1

    store = DeploymentStateStore(config.state_dir)
    if fault != "first-install-volume":
        assert store.read_current() is not None
    assert store.read_journal()[-1].event in {"rolled_back", "aborted"}
    assert not config.state_dir.joinpath("pending.json").exists()
    if fault == "first-install-volume":
        assert not host.production_exists
    else:
        assert host.production_image == OLD_IMAGE
        assert host.production_revision == OLD_REVISION


def test_target_verification_rejects_intent_volume_disagreement(
    setup: tuple[Host, PromotionConfig, dict[str, str]],
) -> None:
    host, config, environment = setup
    _pending(host, config)
    command_environment = promotion_module._command_environment(environment)
    executables = _executables(host)
    with DeploymentStateStore(config.state_dir).transaction() as transaction:
        pending = transaction.pending()
        assert pending is not None
        actual = promotion_module._observe_runtime(
            config=config,
            image=promotion_module.ImageProof(
                TARGET_IMAGE,
                TARGET_IMAGE_ID,
                TARGET_REVISION,
            ),
            executables=executables,
            environment=command_environment,
        )
        assert actual is not None
        mismatched = replace(
            pending,
            intent=replace(
                pending.intent,
                persistent_volumes=(REPLACED_VOLUME,),
            ),
        )

        with pytest.raises(
            promotion_module.PromotionError,
            match="volume intent does not match",
        ):
            promotion_module._verify_target(
                transaction=transaction,
                pending=mismatched,
                baseline_runtime=actual,
                config=config,
                executables=executables,
                environment=command_environment,
            )


def test_restore_baseline_requires_a_recorded_current_state(
    setup: tuple[Host, PromotionConfig, dict[str, str]],
) -> None:
    host, config, environment = setup
    _fresh_pending(
        host,
        config,
        install_environment=False,
        production_exists=False,
    )
    with DeploymentStateStore(config.state_dir).transaction() as transaction:
        pending = transaction.pending()
        assert pending is not None
        command_count = len(host.log)
        with pytest.raises(
            promotion_module.PromotionRecoveryFailedError,
            match="no recorded production baseline",
        ):
            promotion_module._restore_baseline(
                transaction=transaction,
                pending=pending,
                current=None,
                baseline_runtime=None,
                config=config,
                executables=_executables(host),
                environment=promotion_module._command_environment(environment),
            )

    assert len(host.log) == command_count


def test_restore_preflights_recorded_image_before_any_mutation(
    setup: tuple[Host, PromotionConfig, dict[str, str]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    host, config, environment = setup
    _pending(host, config)
    command_count = len(host.log)

    def transform(
        args: list[str],
        result: subprocess.CompletedProcess[str],
    ) -> subprocess.CompletedProcess[str]:
        if args[1:3] == ["image", "inspect"] and args[3] == OLD_IMAGE:
            document = json.loads(result.stdout)[0]
            document["Id"] = MANUAL_IMAGE_ID
            return _completed_json(args, document)
        return result

    _strict_host_run(monkeypatch, host, transform)
    with DeploymentStateStore(config.state_dir).transaction() as transaction:
        pending = transaction.pending()
        current = transaction.current()
        assert pending is not None
        assert current is not None
        with pytest.raises(
            promotion_module.PromotionRecoveryFailedError,
            match="baseline image identity",
        ):
            promotion_module._restore_baseline(
                transaction=transaction,
                pending=pending,
                current=current,
                baseline_runtime=None,
                config=config,
                executables=_executables(host),
                environment=promotion_module._command_environment(environment),
            )

    assert host.production_revision == TARGET_REVISION
    assert host.production_image == TARGET_IMAGE
    assert all(
        "checkout" not in command and "compose" not in command
        for command in host.log[command_count:]
    )


def test_restore_rejects_baseline_disappearing_after_compose_up(
    setup: tuple[Host, PromotionConfig, dict[str, str]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    host, config, environment = setup
    _pending(host, config)
    old_up_seen = False

    def transform(
        args: list[str],
        result: subprocess.CompletedProcess[str],
    ) -> subprocess.CompletedProcess[str]:
        nonlocal old_up_seen
        if (
            args[1:2] == ["compose"]
            and "up" in args
            and host.production_image == OLD_IMAGE
        ):
            old_up_seen = True
        if old_up_seen and args[1:3] == ["container", "ls"]:
            return host.completed(args)
        return result

    _strict_host_run(monkeypatch, host, transform)
    with DeploymentStateStore(config.state_dir).transaction() as transaction:
        pending = transaction.pending()
        current = transaction.current()
        assert pending is not None
        assert current is not None
        with pytest.raises(
            promotion_module.PromotionRecoveryFailedError,
            match="did not restore",
        ):
            promotion_module._restore_baseline(
                transaction=transaction,
                pending=pending,
                current=current,
                baseline_runtime=None,
                config=config,
                executables=_executables(host),
                environment=promotion_module._command_environment(environment),
            )

    assert old_up_seen
    assert config.state_dir.joinpath("pending.json").is_file()


@pytest.mark.parametrize("fault", ["intent", "prior-observation"])
def test_restore_rejects_any_persistent_volume_identity_change(
    setup: tuple[Host, PromotionConfig, dict[str, str]],
    fault: str,
) -> None:
    host, config, environment = setup
    _pending(host, config)
    command_environment = promotion_module._command_environment(environment)
    executables = _executables(host)
    observed_target = promotion_module._observe_runtime(
        config=config,
        image=promotion_module.ImageProof(
            TARGET_IMAGE,
            TARGET_IMAGE_ID,
            TARGET_REVISION,
        ),
        executables=executables,
        environment=command_environment,
    )
    assert observed_target is not None
    replaced_proof = promotion_module.VolumeProof(
        name=REPLACED_VOLUME.name,
        driver=REPLACED_VOLUME.driver,
        destination=REPLACED_VOLUME.destination,
        mountpoint=REPLACED_VOLUME.mountpoint,
        created_at=REPLACED_VOLUME.created_at,
    )
    with DeploymentStateStore(config.state_dir).transaction() as transaction:
        pending = transaction.pending()
        current = transaction.current()
        assert pending is not None
        assert current is not None
        selected_pending = (
            replace(
                pending,
                intent=replace(
                    pending.intent,
                    persistent_volumes=(REPLACED_VOLUME,),
                ),
            )
            if fault == "intent"
            else pending
        )
        baseline_runtime = (
            None
            if fault == "intent"
            else replace(observed_target, volumes=(replaced_proof,))
        )

        with pytest.raises(
            promotion_module.PromotionRecoveryFailedError,
            match="persistent volume identity",
        ):
            promotion_module._restore_baseline(
                transaction=transaction,
                pending=selected_pending,
                current=current,
                baseline_runtime=baseline_runtime,
                config=config,
                executables=executables,
                environment=command_environment,
            )

    assert config.state_dir.joinpath("pending.json").is_file()


def test_rollback_terminal_commit_is_authoritative_and_not_compensated(
    setup: tuple[Host, PromotionConfig, dict[str, str]],
) -> None:
    host, config, environment = setup
    _initialize(host, config)
    host.fail_target_health = True
    host.make_rollback_cleanup_fail = True

    try:
        assert main(_argv(config), environment) == 4
    finally:
        config.state_dir.chmod(0o700)

    store = DeploymentStateStore(config.state_dir)
    assert [entry.event for entry in store.read_journal()] == [
        "adopted",
        "rolled_back",
    ]
    assert host.production_image == OLD_IMAGE
    assert host.production_revision == OLD_REVISION
    rollback_up_index = max(
        index
        for index, command in enumerate(host.log)
        if "up" in command
        and "--project-name" in command
        and command[command.index("--project-name") + 1] == PROJECT
    )
    assert not any(
        "compose" in command for command in host.log[rollback_up_index + 1 :]
    )
