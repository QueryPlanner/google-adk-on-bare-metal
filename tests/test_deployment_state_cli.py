"""Operator CLI tests for VM deployment state inspection and adoption."""

from __future__ import annotations

import errno
import fcntl
import json
import runpy
import shutil
import subprocess
import sys
import tomllib
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from unittest.mock import create_autospec, patch

import pytest

from agent.deployment_state import CandidateReceipt, DeploymentStateStore
from agent.deployment_state_cli import main

ORIGIN = "https://github.com/QueryPlanner/google-adk-on-bare-metal"
REVISION = "a" * 40
OCI_REVISION = "b" * 40
CONTAINER_ID = "e" * 64
SHORT_CONTAINER_ID = CONTAINER_ID[:12]
IMAGE_ID = f"sha256:{'c' * 64}"
IMAGE_REFERENCE = f"ghcr.io/queryplanner/agent@sha256:{'d' * 64}"
SECRET_BYTES = b'API_KEY="secret-cli-canary"\n'


def _checkout(tmp_path: Path) -> tuple[Path, Path]:
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    environment = checkout / ".env"
    environment.write_bytes(SECRET_BYTES)
    environment.chmod(0o600)
    return checkout.resolve(), environment


@dataclass(slots=True)
class CliExternalBoundary:
    """Deterministic Git and Docker subprocess boundary for the real probe."""

    checkout: Path
    git: Path
    docker: Path
    container_list: str = f"{SHORT_CONTAINER_ID}\n"
    health: str = "healthy"
    calls: list[tuple[str, ...]] = field(default_factory=list)

    def run(
        self,
        command: list[str],
        *,
        cwd: Path | None,
        env: dict[str, str],
        text: bool,
        capture_output: bool,
        check: bool,
        timeout: int,
    ) -> subprocess.CompletedProcess[str]:
        """Return exact external responses without replacing application logic."""
        assert cwd is None
        assert text is True
        assert capture_output is True
        assert check is False
        assert timeout == 30
        assert env["GIT_OPTIONAL_LOCKS"] == "0"
        assert env["GIT_TERMINAL_PROMPT"] == "0"
        selected = tuple(command)
        self.calls.append(selected)

        if command[0] == str(self.git):
            arguments = command[1:]
            if arguments == [
                "-C",
                str(self.checkout),
                "rev-parse",
                "--show-toplevel",
            ]:
                return _completed(command, stdout=f"{self.checkout}\n")
            if arguments == [
                "-C",
                str(self.checkout),
                "remote",
                "get-url",
                "origin",
            ]:
                return _completed(command, stdout=f"{ORIGIN}\n")
            if arguments in (
                [
                    "-C",
                    str(self.checkout),
                    "diff",
                    "--no-ext-diff",
                    "--no-textconv",
                    "--quiet",
                    "--",
                ],
                [
                    "-C",
                    str(self.checkout),
                    "diff",
                    "--cached",
                    "--no-ext-diff",
                    "--no-textconv",
                    "--quiet",
                    "--",
                ],
            ):
                return _completed(command)
            if arguments == [
                "-C",
                str(self.checkout),
                "rev-parse",
                "--verify",
                "HEAD",
            ]:
                return _completed(command, stdout=f"{REVISION}\n")

        if command[0] == str(self.docker):
            arguments = command[1:]
            if arguments[:2] == ["container", "ls"]:
                return _completed(command, stdout=self.container_list)
            if arguments == ["container", "inspect", SHORT_CONTAINER_ID]:
                return _completed(
                    command,
                    stdout=json.dumps(
                        [
                            {
                                "Id": CONTAINER_ID,
                                "Image": IMAGE_ID,
                                "State": {
                                    "Status": "running",
                                    "Health": {"Status": self.health},
                                },
                                "Config": {
                                    "Image": IMAGE_REFERENCE,
                                    "Labels": {
                                        "com.docker.compose.project": ("adk-template"),
                                        "com.docker.compose.service": "agent",
                                        "com.docker.compose.project.working_dir": (
                                            str(self.checkout)
                                        ),
                                    },
                                },
                            }
                        ]
                    ),
                )
            if arguments == ["image", "inspect", IMAGE_ID]:
                return _completed(
                    command,
                    stdout=json.dumps(
                        [
                            {
                                "Id": IMAGE_ID,
                                "RepoDigests": [IMAGE_REFERENCE],
                                "Config": {
                                    "Labels": {
                                        "org.opencontainers.image.revision": (
                                            OCI_REVISION
                                        )
                                    }
                                },
                            }
                        ]
                    ),
                )

        return _completed(command, returncode=99, stderr="unexpected command")


def _completed(
    command: list[str],
    *,
    stdout: str = "",
    returncode: int = 0,
    stderr: str = "",
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(
        command,
        returncode,
        stdout=stdout,
        stderr=stderr,
    )


def _adopt_arguments(
    state_dir: Path,
    checkout: Path,
    *,
    environment_file: Path | None = None,
) -> list[str]:
    arguments = [
        "adopt",
        "--state-dir",
        str(state_dir),
        "--checkout",
        str(checkout),
        "--expected-origin",
        ORIGIN,
        "--compose-project",
        "adk-template",
    ]
    if environment_file is not None:
        arguments.extend(["--environment-file", str(environment_file)])
    return arguments


def _external_boundary(
    tmp_path: Path,
    checkout: Path,
) -> CliExternalBoundary:
    bin_directory = tmp_path / "bin"
    bin_directory.mkdir()
    git = bin_directory / "git"
    docker = bin_directory / "docker"
    for executable in (git, docker):
        executable.write_text("#!/bin/sh\nexit 99\n", encoding="utf-8")
        executable.chmod(0o755)
    return CliExternalBoundary(
        checkout=checkout,
        git=git.resolve(),
        docker=docker.resolve(),
    )


def _resolved_which_mock(boundary: CliExternalBoundary) -> object:
    def selected(name: str) -> str:
        assert name in {"git", "docker"}
        return str(boundary.git if name == "git" else boundary.docker)

    return create_autospec(
        shutil.which,
        spec_set=True,
        side_effect=selected,
    )


def _run_adoption(
    arguments: list[str],
    boundary: CliExternalBoundary,
) -> int:
    run = create_autospec(
        subprocess.run,
        spec_set=True,
        side_effect=boundary.run,
    )
    with (
        patch("agent.deployment_adoption.subprocess.run", new=run),
        patch(
            "agent.deployment_state_cli.shutil.which",
            new=_resolved_which_mock(boundary),
        ),
    ):
        return main(arguments)


def test_inspect_empty_state_is_secret_free_json(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Create and verify an empty private store without claiming rollback state."""
    state_dir = tmp_path / "state"

    assert main(["inspect", "--state-dir", str(state_dir)]) == 0

    captured = capsys.readouterr()
    assert json.loads(captured.out) == {
        "status": "empty",
        "current": None,
        "journal": [],
        "pending": None,
    }
    assert captured.err == ""
    assert stat_mode(state_dir) == 0o700


def stat_mode(path: Path) -> int:
    """Return only portable permission bits for one test path."""
    return path.stat().st_mode & 0o777


def test_adopt_then_inspect_exact_state(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Combine the observation and durable state APIs through the real CLI."""
    checkout, environment = _checkout(tmp_path)
    state_dir = tmp_path / "state"
    boundary = _external_boundary(tmp_path, checkout)

    assert _run_adoption(_adopt_arguments(state_dir, checkout), boundary) == 0

    adopted_output = capsys.readouterr()
    adopted = json.loads(adopted_output.out)
    assert adopted["status"] == "adopted"
    assert adopted["current"]["state"]["image_reference"] == IMAGE_REFERENCE
    assert "secret-cli-canary" not in adopted_output.out
    assert any(call[0] == str(boundary.git) for call in boundary.calls)
    assert any(call[0] == str(boundary.docker) for call in boundary.calls)
    snapshot = state_dir / adopted["current"]["state"]["environment_snapshot"]
    assert snapshot.read_bytes() == environment.read_bytes()

    assert main(["inspect", "--state-dir", str(state_dir)]) == 0
    inspected_output = capsys.readouterr()
    inspected = json.loads(inspected_output.out)
    assert inspected["status"] == "recorded"
    assert inspected["current"] == adopted["current"]
    assert inspected["pending"] is None
    assert len(inspected["journal"]) == 1
    assert inspected["journal"][0]["sha256"] == adopted["current"]["journal_sha256"]
    assert "secret-cli-canary" not in inspected_output.out


def test_explicit_environment_file_is_forwarded(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Allow an explicit absolute path while the probe enforces checkout .env."""
    checkout, environment = _checkout(tmp_path)
    boundary = _external_boundary(tmp_path, checkout)
    state_dir = tmp_path / "state"

    assert (
        _run_adoption(
            _adopt_arguments(
                state_dir,
                checkout,
                environment_file=environment,
            ),
            boundary,
        )
        == 0
    )

    adopted = json.loads(capsys.readouterr().out)
    snapshot = state_dir / adopted["current"]["state"]["environment_snapshot"]
    assert snapshot.read_bytes() == environment.read_bytes()


def test_zero_container_observation_records_nothing(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Report a fresh install without manufacturing a rollback target."""
    checkout, _ = _checkout(tmp_path)
    state_dir = tmp_path / "state"
    boundary = _external_boundary(tmp_path, checkout)
    boundary.container_list = ""

    assert _run_adoption(_adopt_arguments(state_dir, checkout), boundary) == 0

    assert json.loads(capsys.readouterr().out) == {
        "status": "fresh",
        "current": None,
        "journal": [],
    }
    assert DeploymentStateStore(state_dir).read_current() is None
    assert DeploymentStateStore(state_dir).read_journal() == ()


def test_existing_state_fails_before_reobservation(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Never reinterpret or overwrite an initialized recovery boundary."""
    checkout, _ = _checkout(tmp_path)
    state_dir = tmp_path / "state"
    boundary = _external_boundary(tmp_path, checkout)
    assert _run_adoption(_adopt_arguments(state_dir, checkout), boundary) == 0
    capsys.readouterr()

    forbidden_which = create_autospec(
        shutil.which,
        spec_set=True,
        side_effect=AssertionError("external observation should not run"),
    )
    with patch(
        "agent.deployment_state_cli.shutil.which",
        new=forbidden_which,
    ):
        assert main(_adopt_arguments(state_dir, checkout)) == 1

    captured = capsys.readouterr()
    assert "already been initialized" in captured.err
    assert captured.out == ""
    forbidden_which.assert_not_called()


def test_inspect_pending_state_and_reject_adoption_before_observation(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Expose recovery status without secrets or reinterpreting pending state."""
    checkout, _ = _checkout(tmp_path)
    state_dir = tmp_path / "state"
    target_environment = tmp_path / "target.env"
    target_environment.write_bytes(b'PRIVATE_KEY="pending-secret-canary"\n')
    target_environment.chmod(0o600)
    receipt = CandidateReceipt(
        observed_at="2026-07-28T10:00:00.000000Z",
        compose_project="adk-template-candidate",
        compose_service="agent",
        container_id=CONTAINER_ID,
        image_reference=IMAGE_REFERENCE,
        image_id=IMAGE_ID,
        oci_revision=OCI_REVISION,
        baseline_journal_sequence=None,
        baseline_journal_sha256=None,
    )
    store = DeploymentStateStore(state_dir)
    with store.transaction() as transaction:
        pending = transaction.begin_promotion(
            compose_project="adk-template",
            compose_service="agent",
            source_revision=OCI_REVISION,
            image_reference=IMAGE_REFERENCE,
            image_id=IMAGE_ID,
            oci_revision=OCI_REVISION,
            environment_source=target_environment,
            candidate=receipt,
            persistent_volumes=(),
            transaction_id="pending-cli-1234567890",
            recorded_at="2026-07-28T10:00:01.000000Z",
        )

    assert main(["inspect", "--state-dir", str(state_dir)]) == 0

    captured = capsys.readouterr()
    inspected = json.loads(captured.out)
    assert inspected == {
        "status": "pending",
        "current": None,
        "journal": [],
        "pending": pending.as_document(),
    }
    assert "pending-secret-canary" not in captured.out
    assert captured.err == ""

    forbidden_which = create_autospec(
        shutil.which,
        spec_set=True,
        side_effect=AssertionError("external observation should not run"),
    )
    with patch(
        "agent.deployment_state_cli.shutil.which",
        new=forbidden_which,
    ):
        assert main(_adopt_arguments(state_dir, checkout)) == 1

    rejected = capsys.readouterr()
    assert "already been initialized" in rejected.err
    assert rejected.out == ""
    forbidden_which.assert_not_called()


def test_missing_executable_returns_safe_error(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Fail before observation when the fixed Git/Docker boundary is absent."""
    checkout, _ = _checkout(tmp_path)
    which = create_autospec(shutil.which, spec_set=True, return_value=None)

    with patch("agent.deployment_state_cli.shutil.which", new=which):
        assert main(_adopt_arguments(tmp_path / "state", checkout)) == 1

    captured = capsys.readouterr()
    assert "executable is unavailable" in captured.err
    assert captured.out == ""


def test_observation_contract_error_returns_one(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Surface the probe's deterministic safe reason without a traceback."""
    checkout, _ = _checkout(tmp_path)
    boundary = _external_boundary(tmp_path, checkout)
    boundary.health = "unhealthy"

    assert _run_adoption(_adopt_arguments(tmp_path / "state", checkout), boundary) == 1

    captured = capsys.readouterr()
    assert captured.err == "ERROR: legacy container is not healthy\n"
    assert captured.out == ""


def test_unexpected_os_error_is_redacted(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Do not expose filesystem or external process details from the CLI."""
    checkout, _ = _checkout(tmp_path)
    missing = create_autospec(
        shutil.which,
        spec_set=True,
        return_value="/missing/secret-os-detail",
    )

    with patch("agent.deployment_state_cli.shutil.which", new=missing):
        assert main(_adopt_arguments(tmp_path / "state", checkout)) == 1

    captured = capsys.readouterr()
    assert captured.err == "ERROR: deployment state operation failed\n"
    assert "secret-os-detail" not in captured.err


def test_lock_contention_returns_temporary_failure(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Give automation a distinct retryable exit status for lock contention."""
    lock_failure = create_autospec(
        fcntl.flock,
        spec_set=True,
        side_effect=OSError(errno.EAGAIN, "busy"),
    )

    with patch("agent.deployment_state.fcntl.flock", new=lock_failure):
        assert main(["inspect", "--state-dir", str(tmp_path / "state")]) == 75

    captured = capsys.readouterr()
    assert "another deployment transaction" in captured.err
    assert captured.out == ""


def test_usage_error_is_distinct(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Let argparse return its conventional usage status before touching state."""
    with pytest.raises(SystemExit) as error:
        main(["unknown"])

    assert error.value.code == 2
    captured = capsys.readouterr()
    assert "usage:" in captured.err


def test_module_entrypoint_exits_with_main_status(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Keep `python -m agent.deployment_state_cli` executable."""
    with (
        patch.object(
            sys,
            "argv",
            [
                "deployment-state",
                "inspect",
                "--state-dir",
                str(tmp_path / "state"),
            ],
        ),
        warnings.catch_warnings(),
        pytest.raises(SystemExit) as error,
    ):
        warnings.simplefilter("ignore", RuntimeWarning)
        runpy.run_module("agent.deployment_state_cli", run_name="__main__")

    assert error.value.code == 0
    assert json.loads(capsys.readouterr().out)["status"] == "empty"


def test_project_registers_operator_cli() -> None:
    """Expose both reviewed deployment modules through installed commands."""
    pyproject = tomllib.loads(
        (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text(
            encoding="utf-8"
        )
    )

    assert pyproject["project"]["scripts"]["deployment-state"] == (
        "agent.deployment_state_cli:main"
    )
    assert pyproject["project"]["scripts"]["deployment-promote"] == (
        "agent.deployment_promotion:main"
    )
