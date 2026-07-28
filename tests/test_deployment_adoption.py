"""Read-only legacy Compose deployment observation tests."""

from __future__ import annotations

import copy
import json
import os
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from unittest.mock import create_autospec, patch

import pytest

from agent import deployment_adoption as adoption_module
from agent.deployment_adoption import (
    DeploymentAdoptionError,
    DeploymentObservation,
    observe_legacy_deployment,
)

ORIGIN = "https://github.com/QueryPlanner/google-adk-on-bare-metal"
REVISION = "a" * 40
OCI_REVISION = "b" * 40
CONTAINER_ID = "c" * 64
SHORT_CONTAINER_ID = CONTAINER_ID[:12]
IMAGE_ID = f"sha256:{'d' * 64}"
IMAGE_REFERENCE = f"ghcr.io/queryplanner/agent@sha256:{'e' * 64}"
PROJECT = "adk-template"
SERVICE = "agent"
SECRET_ENVIRONMENT = b'API_KEY="secret-observation-canary"\n'


def _completed(
    command: Sequence[str],
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


@dataclass(slots=True)
class ObservationBoundary:
    """Deterministic external Git and Docker command boundary."""

    checkout: Path
    git: Path
    docker: Path
    top_level: Path | None = None
    origin: str = ORIGIN
    revision: str = REVISION
    unstaged_returncode: int = 0
    staged_returncode: int = 0
    container_list: str = f"{SHORT_CONTAINER_ID}\n"
    container_document: object = field(default_factory=dict)
    image_document: object = field(default_factory=dict)
    fail_command: tuple[str, ...] | None = None
    calls: list[tuple[str, ...]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.top_level is None:
            self.top_level = self.checkout
        if self.container_document == {}:
            self.container_document = {
                "Id": CONTAINER_ID,
                "Image": IMAGE_ID,
                "State": {
                    "Status": "running",
                    "Health": {"Status": "healthy"},
                },
                "Config": {
                    "Image": IMAGE_REFERENCE,
                    "Labels": {
                        "com.docker.compose.project": PROJECT,
                        "com.docker.compose.service": SERVICE,
                        "com.docker.compose.project.working_dir": str(self.checkout),
                    },
                },
            }
        if self.image_document == {}:
            self.image_document = {
                "Id": IMAGE_ID,
                "RepoDigests": [IMAGE_REFERENCE],
                "Config": {
                    "Labels": {
                        "org.opencontainers.image.revision": OCI_REVISION,
                    }
                },
            }

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
        """Return one exact synthetic external response."""
        assert cwd is None
        assert text is True
        assert capture_output is True
        assert check is False
        assert timeout == 30
        assert env["GIT_OPTIONAL_LOCKS"] == "0"
        assert env["GIT_TERMINAL_PROMPT"] == "0"
        assert env["GIT_CONFIG_COUNT"] == "2"
        assert env["GIT_CONFIG_GLOBAL"] == "/dev/null"
        assert env["GIT_CONFIG_SYSTEM"] == "/dev/null"
        assert env["GIT_CONFIG_NOSYSTEM"] == "1"
        assert env["GIT_NO_REPLACE_OBJECTS"] == "1"
        assert env["GIT_CONFIG_KEY_0"] == "core.hooksPath"
        assert env["GIT_CONFIG_VALUE_0"] == "/dev/null"
        assert env["GIT_CONFIG_KEY_1"] == "core.fsmonitor"
        assert env["GIT_CONFIG_VALUE_1"] == "false"
        assert env["LC_ALL"] == "C"
        assert env["LANG"] == "C"
        assert "GIT_DIR" not in env
        assert "OPENROUTER_API_KEY" not in env
        assert env["DOCKER_CONFIG"] == "/private/docker-config"
        if "DOCKER_HOST" in os.environ:
            assert env["DOCKER_HOST"] == os.environ["DOCKER_HOST"]
        if "XDG_RUNTIME_DIR" in os.environ:
            assert env["XDG_RUNTIME_DIR"] == os.environ["XDG_RUNTIME_DIR"]

        selected = tuple(command)
        self.calls.append(selected)
        if self.fail_command is not None and selected[1:] == self.fail_command:
            return _completed(command, returncode=29, stderr="secret-boundary-error")

        if command[0] == str(self.git):
            arguments = command[1:]
            if arguments == [
                "-C",
                str(self.checkout),
                "rev-parse",
                "--show-toplevel",
            ]:
                return _completed(command, stdout=f"{self.top_level}\n")
            if arguments == [
                "-C",
                str(self.checkout),
                "remote",
                "get-url",
                "origin",
            ]:
                return _completed(command, stdout=f"{self.origin}\n")
            if arguments == [
                "-C",
                str(self.checkout),
                "diff",
                "--no-ext-diff",
                "--no-textconv",
                "--quiet",
                "--",
            ]:
                return _completed(command, returncode=self.unstaged_returncode)
            if arguments == [
                "-C",
                str(self.checkout),
                "diff",
                "--cached",
                "--no-ext-diff",
                "--no-textconv",
                "--quiet",
                "--",
            ]:
                return _completed(command, returncode=self.staged_returncode)
            if arguments == [
                "-C",
                str(self.checkout),
                "rev-parse",
                "--verify",
                "HEAD",
            ]:
                return _completed(command, stdout=f"{self.revision}\n")

        if command[0] == str(self.docker):
            arguments = command[1:]
            if arguments[:2] == ["container", "ls"]:
                return _completed(command, stdout=self.container_list)
            if arguments == ["container", "inspect", SHORT_CONTAINER_ID]:
                return _completed(
                    command,
                    stdout=json.dumps([self.container_document]),
                )
            if arguments == ["image", "inspect", IMAGE_ID]:
                return _completed(
                    command,
                    stdout=json.dumps([self.image_document]),
                )

        return _completed(command, returncode=99, stderr="unexpected command")


@dataclass(frozen=True, slots=True)
class ObservationHarness:
    """Private checkout, executables, inputs, and external boundary."""

    checkout: Path
    environment_path: Path
    git: Path
    docker: Path
    boundary: ObservationBoundary

    def arguments(self) -> dict[str, object]:
        return {
            "checkout_path": self.checkout,
            "expected_origin": ORIGIN,
            "compose_project": PROJECT,
            "compose_service": SERVICE,
            "environment_path": self.environment_path,
            "git_executable": self.git,
            "docker_executable": self.docker,
            "environment": {
                "PATH": "/usr/bin:/bin",
                "DOCKER_CONFIG": "/private/docker-config",
                "GIT_DIR": "secret-git-override",
                "LANG": "host-locale",
            },
        }


@pytest.fixture
def observation_harness(tmp_path: Path) -> ObservationHarness:
    """Build one safe legacy checkout with inert executable boundaries."""
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    environment_path = checkout / ".env"
    environment_path.write_bytes(SECRET_ENVIRONMENT)
    environment_path.chmod(0o600)
    bin_directory = tmp_path / "bin"
    bin_directory.mkdir()
    git = bin_directory / "git"
    docker = bin_directory / "docker"
    for executable in (git, docker):
        executable.write_text("#!/bin/sh\nexit 99\n", encoding="utf-8")
        executable.chmod(0o755)
    boundary = ObservationBoundary(checkout.resolve(), git.resolve(), docker.resolve())
    return ObservationHarness(
        checkout=checkout.resolve(),
        environment_path=environment_path,
        git=git,
        docker=docker,
        boundary=boundary,
    )


def _observe(
    harness: ObservationHarness,
    *,
    arguments: dict[str, object] | None = None,
) -> DeploymentObservation | None:
    run = create_autospec(
        subprocess.run,
        spec_set=True,
        side_effect=harness.boundary.run,
    )
    with patch("agent.deployment_adoption.subprocess.run", new=run):
        return observe_legacy_deployment(
            **(harness.arguments() if arguments is None else arguments)  # type: ignore[arg-type]
        )


def test_observe_exact_healthy_legacy_deployment(
    observation_harness: ObservationHarness,
) -> None:
    """Return only proven metadata while never reading private env bytes."""
    before = observation_harness.environment_path.read_bytes()

    observation = _observe(observation_harness)

    assert observation == DeploymentObservation(
        checkout_path=observation_harness.checkout,
        origin=ORIGIN,
        revision=REVISION,
        compose_project=PROJECT,
        compose_service=SERVICE,
        environment_path=observation_harness.environment_path,
        image_reference=IMAGE_REFERENCE,
        image_id=IMAGE_ID,
        oci_revision=OCI_REVISION,
    )
    assert observation_harness.environment_path.read_bytes() == before
    assert [
        call[1:4]
        for call in observation_harness.boundary.calls
        if call[0].endswith("git")
    ] == [
        ("-C", str(observation_harness.checkout), "rev-parse"),
        ("-C", str(observation_harness.checkout), "remote"),
        ("-C", str(observation_harness.checkout), "diff"),
        ("-C", str(observation_harness.checkout), "diff"),
        ("-C", str(observation_harness.checkout), "rev-parse"),
    ]
    docker_commands = [
        call[1:3]
        for call in observation_harness.boundary.calls
        if call[0].endswith("docker")
    ]
    assert docker_commands == [
        ("container", "ls"),
        ("container", "inspect"),
        ("image", "inspect"),
    ]
    forbidden = {"start", "stop", "run", "create", "rm", "pull", "push", "tag"}
    assert not forbidden.intersection(
        argument for call in observation_harness.boundary.calls for argument in call
    )


def test_optional_dot_git_origin_is_normalized(
    observation_harness: ObservationHarness,
) -> None:
    """Accept the one equivalent legacy HTTPS remote spelling."""
    observation_harness.boundary.origin = f"{ORIGIN}.git"

    observation = _observe(observation_harness)

    assert observation is not None
    assert observation.origin == ORIGIN


def test_nested_directory_cannot_impersonate_checkout_root(
    observation_harness: ObservationHarness,
) -> None:
    """Require the claimed checkout to be Git's exact work-tree root."""
    nested = observation_harness.checkout / "nested"
    nested.mkdir()
    observation_harness.boundary.checkout = nested
    observation_harness.boundary.top_level = observation_harness.checkout
    arguments = observation_harness.arguments()
    arguments["checkout_path"] = nested
    arguments["environment_path"] = nested / ".env"

    with pytest.raises(DeploymentAdoptionError, match="root does not match"):
        _observe(observation_harness, arguments=arguments)

    assert len(observation_harness.boundary.calls) == 1


def test_reported_checkout_root_must_exist(
    observation_harness: ObservationHarness,
) -> None:
    """Reject a Git root response that cannot be resolved to a directory."""
    observation_harness.boundary.top_level = (
        observation_harness.checkout / "missing-root"
    )

    with pytest.raises(DeploymentAdoptionError, match="root is unavailable"):
        _observe(observation_harness)

    assert len(observation_harness.boundary.calls) == 1


def test_zero_containers_is_clean_first_install_without_env(
    observation_harness: ObservationHarness,
) -> None:
    """Return no rollback target before requiring a legacy environment file."""
    observation_harness.boundary.container_list = ""
    observation_harness.environment_path.unlink()

    assert _observe(observation_harness) is None
    assert all(
        call[1:3] != ("container", "inspect")
        for call in observation_harness.boundary.calls
    )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("expected_origin", "git@github.com:owner/repo.git", "origin"),
        ("expected_origin", f"{ORIGIN}.git", "origin"),
        ("compose_project", "Invalid", "compose_project"),
        ("compose_service", "-agent", "compose_service"),
    ],
)
def test_invalid_public_inputs_fail_before_external_calls(
    observation_harness: ObservationHarness,
    field: str,
    value: str,
    message: str,
) -> None:
    """Reject ambiguous identifiers before consulting Git or Docker."""
    arguments = observation_harness.arguments()
    arguments[field] = value

    with pytest.raises(DeploymentAdoptionError, match=message):
        _observe(observation_harness, arguments=arguments)

    assert observation_harness.boundary.calls == []


@pytest.mark.parametrize("unsafe_kind", ["relative", "missing", "file"])
def test_invalid_checkout_fails_before_external_calls(
    observation_harness: ObservationHarness,
    tmp_path: Path,
    unsafe_kind: str,
) -> None:
    """Require one absolute existing checkout directory."""
    arguments = observation_harness.arguments()
    if unsafe_kind == "relative":
        arguments["checkout_path"] = Path("relative")
    elif unsafe_kind == "missing":
        arguments["checkout_path"] = tmp_path / "missing"
    else:
        selected = tmp_path / "not-directory"
        selected.write_text("file", encoding="utf-8")
        arguments["checkout_path"] = selected

    with pytest.raises(DeploymentAdoptionError, match="checkout"):
        _observe(observation_harness, arguments=arguments)

    assert observation_harness.boundary.calls == []


@pytest.mark.parametrize(
    ("field", "unsafe_kind"),
    [
        ("git_executable", "relative"),
        ("git_executable", "missing"),
        ("git_executable", "directory"),
        ("git_executable", "not-executable"),
        ("docker_executable", "relative"),
    ],
)
def test_invalid_executable_fails_closed(
    observation_harness: ObservationHarness,
    tmp_path: Path,
    field: str,
    unsafe_kind: str,
) -> None:
    """Resolve fixed external boundaries instead of searching an ambient PATH."""
    arguments = observation_harness.arguments()
    if unsafe_kind == "relative":
        selected = Path("git")
    elif unsafe_kind == "missing":
        selected = tmp_path / "missing"
    elif unsafe_kind == "directory":
        selected = tmp_path / "directory"
        selected.mkdir()
    else:
        selected = tmp_path / "not-executable"
        selected.write_text("no", encoding="utf-8")
        selected.chmod(0o600)
    arguments[field] = selected

    with pytest.raises(DeploymentAdoptionError, match="executable"):
        _observe(observation_harness, arguments=arguments)

    assert observation_harness.boundary.calls == []


@pytest.mark.parametrize(
    ("origin", "message"),
    [
        ("https://github.com/other/repository", "does not match"),
        ("", "origin"),
        (f"{ORIGIN}\nsecond", "origin"),
        (f"{ORIGIN}\0suffix", "origin"),
    ],
)
def test_git_origin_must_be_one_exact_safe_line(
    observation_harness: ObservationHarness,
    origin: str,
    message: str,
) -> None:
    """Do not infer identity from a different or malformed remote."""
    observation_harness.boundary.origin = origin

    with pytest.raises(DeploymentAdoptionError, match=message):
        _observe(observation_harness)


@pytest.mark.parametrize("dirty_field", ["unstaged_returncode", "staged_returncode"])
def test_dirty_checkout_is_rejected(
    observation_harness: ObservationHarness,
    dirty_field: str,
) -> None:
    """Require the recorded source revision to describe tracked checkout bytes."""
    setattr(observation_harness.boundary, dirty_field, 1)

    with pytest.raises(DeploymentAdoptionError, match="tracked changes"):
        _observe(observation_harness)


def test_git_command_failure_is_secret_free(
    observation_harness: ObservationHarness,
) -> None:
    """Do not reflect external stderr in adoption diagnostics."""
    observation_harness.boundary.fail_command = (
        "-C",
        str(observation_harness.checkout),
        "remote",
        "get-url",
        "origin",
    )

    with pytest.raises(DeploymentAdoptionError, match="command failed") as error:
        _observe(observation_harness)

    assert "secret-boundary-error" not in str(error.value)


@pytest.mark.parametrize("revision", ["A" * 40, "a" * 39, "a" * 40 + "\nextra"])
def test_invalid_git_revision_is_rejected(
    observation_harness: ObservationHarness,
    revision: str,
) -> None:
    """Require one exact lowercase commit identity."""
    observation_harness.boundary.revision = revision

    with pytest.raises(DeploymentAdoptionError, match="revision"):
        _observe(observation_harness)


@pytest.mark.parametrize(
    "container_list",
    [
        "not-a-container\n",
        f"{SHORT_CONTAINER_ID}\n{'f' * 12}\n",
    ],
)
def test_container_selection_rejects_invalid_or_ambiguous_results(
    observation_harness: ObservationHarness,
    container_list: str,
) -> None:
    """Never guess between malformed or multiple matching containers."""
    observation_harness.boundary.container_list = container_list

    with pytest.raises(DeploymentAdoptionError, match="invalid|ambiguous"):
        _observe(observation_harness)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda document: document.update(Id="short"), "identity"),
        (lambda document: document.update(Id="f" * 64), "identity"),
        (lambda document: document.pop("State"), "container state"),
        (
            lambda document: document["State"].update(Status="exited"),
            "not running",
        ),
        (
            lambda document: document["State"].update(Status=7),
            "container status",
        ),
        (
            lambda document: document["State"].pop("Health"),
            "container health",
        ),
        (
            lambda document: document["State"]["Health"].update(Status="starting"),
            "not healthy",
        ),
        (lambda document: document.pop("Config"), "configuration"),
        (
            lambda document: document["Config"].update(Image="agent:latest"),
            "not immutable",
        ),
        (
            lambda document: document["Config"].update(
                Image=(f"ghcr.io/queryplanner/agent:latest@sha256:{'e' * 64}")
            ),
            "not immutable",
        ),
        (
            lambda document: document["Config"].update(
                Image=f"ghcr.io/queryplanner//agent@sha256:{'e' * 64}"
            ),
            "not immutable",
        ),
        (
            lambda document: document["Config"].pop("Labels"),
            "container labels",
        ),
        (
            lambda document: document["Config"]["Labels"].update(
                {"com.docker.compose.project": "other"}
            ),
            "labels do not match",
        ),
        (lambda document: document.update(Image="sha256:short"), "image ID"),
    ],
)
def test_container_contract_fails_closed(
    observation_harness: ObservationHarness,
    mutation: object,
    message: str,
) -> None:
    """Require exact running, health, label, reference, and image identity."""
    document = copy.deepcopy(observation_harness.boundary.container_document)
    mutation(document)  # type: ignore[operator]
    observation_harness.boundary.container_document = document

    with pytest.raises(DeploymentAdoptionError, match=message):
        _observe(observation_harness)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda document: document.update(Id=f"sha256:{'f' * 64}"), "identity"),
        (lambda document: document.update(RepoDigests=None), "digest"),
        (lambda document: document.update(RepoDigests=[7]), "digest"),
        (
            lambda document: document.update(
                RepoDigests=[f"ghcr.io/other/image@sha256:{'e' * 64}"]
            ),
            "digest",
        ),
        (lambda document: document.pop("Config"), "image configuration"),
        (lambda document: document["Config"].pop("Labels"), "image labels"),
        (
            lambda document: document["Config"]["Labels"].update(
                {"org.opencontainers.image.revision": "invalid"}
            ),
            "OCI revision",
        ),
    ],
)
def test_image_contract_fails_closed(
    observation_harness: ObservationHarness,
    mutation: object,
    message: str,
) -> None:
    """Bind the running container to a local digest and valid OCI revision."""
    document = copy.deepcopy(observation_harness.boundary.image_document)
    mutation(document)  # type: ignore[operator]
    observation_harness.boundary.image_document = document

    with pytest.raises(DeploymentAdoptionError, match=message):
        _observe(observation_harness)


@pytest.mark.parametrize(
    "unsafe_kind",
    [
        "wrong-path",
        "missing",
        "directory",
        "symlink",
        "mode",
        "hardlink",
        "empty",
        "owner",
    ],
)
def test_environment_metadata_must_be_private_and_stable(
    observation_harness: ObservationHarness,
    tmp_path: Path,
    unsafe_kind: str,
) -> None:
    """Validate recovery-file metadata without reading or logging its contents."""
    arguments = observation_harness.arguments()
    environment_path = observation_harness.environment_path
    owner_patch = None
    if unsafe_kind == "wrong-path":
        selected = tmp_path / ".env"
        selected.write_bytes(SECRET_ENVIRONMENT)
        selected.chmod(0o600)
        arguments["environment_path"] = selected
    elif unsafe_kind == "missing":
        environment_path.unlink()
    elif unsafe_kind == "directory":
        environment_path.unlink()
        environment_path.mkdir(mode=0o700)
    elif unsafe_kind == "symlink":
        environment_path.unlink()
        target = tmp_path / "target.env"
        target.write_bytes(SECRET_ENVIRONMENT)
        target.chmod(0o600)
        environment_path.symlink_to(target)
    elif unsafe_kind == "mode":
        environment_path.chmod(0o640)
    elif unsafe_kind == "hardlink":
        os.link(environment_path, tmp_path / "second-name")
    elif unsafe_kind == "empty":
        environment_path.write_bytes(b"")
        environment_path.chmod(0o600)
    else:
        owner_patch = patch(
            "agent.deployment_adoption.os.geteuid",
            new=create_autospec(os.geteuid, spec_set=True, return_value=1),
        )

    context = owner_patch if owner_patch is not None else patch.dict({}, {})
    with context, pytest.raises(DeploymentAdoptionError, match="environment"):
        _observe(observation_harness, arguments=arguments)


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ("not-json", "invalid"),
        ("[]", "invalid"),
        ("[{},{}]", "invalid"),
        ("[7]", "invalid"),
    ],
)
def test_json_boundary_requires_exactly_one_object(
    payload: str,
    message: str,
) -> None:
    """Reject malformed or ambiguous Docker inspect responses."""
    with pytest.raises(DeploymentAdoptionError, match=message):
        adoption_module._json_document(payload, "synthetic")


def test_oversized_external_output_is_rejected(
    observation_harness: ObservationHarness,
) -> None:
    """Bound memory used for external command responses."""
    observation_harness.boundary.origin = "x" * (adoption_module._MAX_OUTPUT_BYTES + 1)

    with pytest.raises(DeploymentAdoptionError, match="too large"):
        _observe(observation_harness)


@pytest.mark.parametrize(
    "failure",
    [
        OSError("secret-os-error"),
        subprocess.TimeoutExpired(["git"], 30, output="secret-timeout-output"),
    ],
)
def test_external_execution_failures_are_redacted(
    observation_harness: ObservationHarness,
    failure: BaseException,
) -> None:
    """Convert launch and timeout details into one secret-free diagnostic."""
    run = create_autospec(subprocess.run, spec_set=True, side_effect=failure)
    with (
        patch("agent.deployment_adoption.subprocess.run", new=run),
        pytest.raises(DeploymentAdoptionError, match="command failed") as error,
    ):
        observe_legacy_deployment(
            **observation_harness.arguments()  # type: ignore[arg-type]
        )

    assert "secret" not in str(error.value)


def test_default_process_environment_is_sanitized(
    observation_harness: ObservationHarness,
) -> None:
    """Exercise the production environment path without preserving Git overrides."""
    arguments = observation_harness.arguments()
    arguments["environment"] = None

    with patch.dict(
        os.environ,
        {
            "DOCKER_CONFIG": "/private/docker-config",
            "DOCKER_HOST": "unix:///private/docker.sock",
            "GIT_WORK_TREE": "secret",
            "OPENROUTER_API_KEY": "secret-canary",
            "XDG_RUNTIME_DIR": "/private/runtime",
        },
        clear=True,
    ):
        observation = _observe(observation_harness, arguments=arguments)

    assert observation is not None


def test_nonzero_dirty_check_error_is_not_misreported_as_dirty(
    observation_harness: ObservationHarness,
) -> None:
    """Distinguish an unreadable checkout from a clean/dirty result."""
    observation_harness.boundary.unstaged_returncode = 2

    with pytest.raises(DeploymentAdoptionError, match="command failed"):
        _observe(observation_harness)
