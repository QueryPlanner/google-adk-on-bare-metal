"""Tests for fail-closed filesystem artifact storage preparation."""

import os
import stat
import tempfile
from pathlib import Path
from typing import Any, Protocol, cast
from unittest.mock import MagicMock, create_autospec, patch

import pytest
from google.adk.artifacts import FileArtifactService
from google.adk.cli.utils.service_factory import (
    create_artifact_service_from_options,
)
from google.genai import types

from agent.artifact_storage import (
    ArtifactStorageError,
    prepare_artifact_storage,
)

_ERROR_MESSAGE = "Artifact storage is unavailable."


class _ProbeFile(Protocol):
    """Strict surface used by the write-failure boundary double."""

    @property
    def name(self) -> str: ...

    def write(self, data: bytes) -> int: ...

    def flush(self) -> None: ...

    def fileno(self) -> int: ...

    def close(self) -> None: ...


async def test_prepare_uses_restrictive_probe_and_real_adk_file_service(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Verify permissions, cleanup, URI wiring, and persistence with real code."""
    monkeypatch.setenv("ADK_DISABLE_LOCAL_STORAGE", "true")
    agents_dir = tmp_path / "agents with spaces"
    agents_dir.mkdir()
    observed_modes: dict[str, int] = {}
    real_fsync = os.fsync

    def inspect_probe_permissions(file_descriptor: int) -> None:
        artifact_dir = agents_dir / ".adk" / "artifacts"
        [probe_dir] = artifact_dir.iterdir()
        [probe_file] = probe_dir.iterdir()
        observed_modes["dir"] = stat.S_IMODE(probe_dir.stat().st_mode)
        observed_modes["file"] = stat.S_IMODE(probe_file.stat().st_mode)
        real_fsync(file_descriptor)

    fsync_spy = create_autospec(
        os.fsync,
        spec_set=True,
        side_effect=inspect_probe_permissions,
    )
    with patch("agent.artifact_storage.os.fsync", new=fsync_spy):
        prepared = prepare_artifact_storage(agents_dir)

    resolved_agents_dir = agents_dir.resolve()
    expected_artifact_dir = resolved_agents_dir / ".adk" / "artifacts"
    assert prepared.agents_dir == resolved_agents_dir
    assert prepared.artifact_dir == expected_artifact_dir
    assert prepared.artifact_service_uri == expected_artifact_dir.as_uri()
    assert observed_modes == {"dir": 0o700, "file": 0o600}
    assert list(expected_artifact_dir.iterdir()) == []
    fsync_spy.assert_called_once()

    service = create_artifact_service_from_options(
        base_dir=prepared.agents_dir,
        artifact_service_uri=prepared.artifact_service_uri,
        strict_uri=True,
    )
    assert isinstance(service, FileArtifactService)
    version = await service.save_artifact(
        app_name="agent",
        user_id="user",
        session_id="session",
        filename="proof.txt",
        artifact=types.Part(text="durable"),
    )
    loaded = await service.load_artifact(
        app_name="agent",
        user_id="user",
        session_id="session",
        filename="proof.txt",
        version=version,
    )
    assert loaded is not None
    assert loaded.text == "durable"


@pytest.mark.parametrize("agent_dir_kind", ["missing", "file"])
def test_prepare_requires_an_existing_directory(
    agent_dir_kind: str,
    tmp_path: Path,
) -> None:
    """Reject missing paths and files with one stable public error."""
    agent_dir = tmp_path / agent_dir_kind
    expected_cause: type[OSError]
    if agent_dir_kind == "file":
        agent_dir.touch()
        expected_cause = NotADirectoryError
    else:
        expected_cause = FileNotFoundError

    with pytest.raises(ArtifactStorageError) as error:
        prepare_artifact_storage(agent_dir)

    assert str(error.value) == _ERROR_MESSAGE
    assert isinstance(error.value.__cause__, expected_cause)
    assert not (tmp_path / ".adk").exists()


def test_prepare_rejects_symlink_escape(tmp_path: Path) -> None:
    """Prevent a pre-existing symlink from moving artifacts outside agent_dir."""
    agents_dir = tmp_path / "agents"
    outside_dir = tmp_path / "outside"
    agents_dir.mkdir()
    outside_dir.mkdir()
    (agents_dir / ".adk").symlink_to(outside_dir, target_is_directory=True)

    with pytest.raises(ArtifactStorageError) as error:
        prepare_artifact_storage(agents_dir)

    assert str(error.value) == _ERROR_MESSAGE
    assert isinstance(error.value.__cause__, ValueError)
    assert list(outside_dir.iterdir()) == []


def test_prepare_cleans_probe_without_masking_fsync_failure(tmp_path: Path) -> None:
    """Keep the fsync error as the sole cause while removing probe resources."""
    agents_dir = tmp_path / "agents"
    agents_dir.mkdir()
    fsync_error = OSError("filesystem rejected fsync")
    fsync_failure = create_autospec(
        os.fsync,
        spec_set=True,
        side_effect=fsync_error,
    )

    with (
        patch("agent.artifact_storage.os.fsync", new=fsync_failure),
        pytest.raises(ArtifactStorageError) as error,
    ):
        prepare_artifact_storage(agents_dir)

    artifact_dir = agents_dir / ".adk" / "artifacts"
    assert str(error.value) == _ERROR_MESSAGE
    assert error.value.__cause__ is fsync_error
    assert list(artifact_dir.iterdir()) == []
    fsync_failure.assert_called_once()


def test_prepare_cleans_probe_without_masking_write_failure(tmp_path: Path) -> None:
    """Translate a real probe-file write boundary failure and clean up."""
    agents_dir = tmp_path / "agents"
    agents_dir.mkdir()
    write_error = OSError("filesystem rejected write")

    def failing_named_temporary_file(
        *args: Any,
        **kwargs: Any,
    ) -> MagicMock:
        assert args == ()
        probe_path = Path(kwargs["dir"]) / ".probe-write-failure"
        probe_path.touch(mode=0o600)
        probe_file = cast(
            MagicMock,
            create_autospec(_ProbeFile, instance=True, spec_set=True),
        )
        probe_file.name = str(probe_path)
        probe_file.write.side_effect = write_error
        return probe_file

    named_temporary_file = cast(
        MagicMock,
        create_autospec(
            tempfile.NamedTemporaryFile,
            spec_set=True,
            side_effect=failing_named_temporary_file,
        ),
    )

    with (
        patch(
            "agent.artifact_storage.tempfile.NamedTemporaryFile",
            new=named_temporary_file,
        ),
        pytest.raises(ArtifactStorageError) as error,
    ):
        prepare_artifact_storage(agents_dir)

    artifact_dir = agents_dir / ".adk" / "artifacts"
    assert str(error.value) == _ERROR_MESSAGE
    assert error.value.__cause__ is write_error
    assert list(artifact_dir.iterdir()) == []
    named_temporary_file.assert_called_once()


def test_prepare_preserves_write_failure_when_probe_close_also_fails(
    tmp_path: Path,
) -> None:
    """Keep the body failure as cause when best-effort close also fails."""
    agents_dir = tmp_path / "agents"
    agents_dir.mkdir()
    write_error = OSError("filesystem rejected write")
    close_error = OSError("filesystem rejected close")
    probe_files: list[MagicMock] = []

    def failing_named_temporary_file(
        *args: Any,
        **kwargs: Any,
    ) -> MagicMock:
        assert args == ()
        probe_path = Path(kwargs["dir"]) / ".probe-write-close-failure"
        probe_path.touch(mode=0o600)
        probe_file = cast(
            MagicMock,
            create_autospec(_ProbeFile, instance=True, spec_set=True),
        )
        probe_file.name = str(probe_path)
        probe_file.write.side_effect = write_error
        probe_file.close.side_effect = close_error
        probe_files.append(probe_file)
        return probe_file

    named_temporary_file = cast(
        MagicMock,
        create_autospec(
            tempfile.NamedTemporaryFile,
            spec_set=True,
            side_effect=failing_named_temporary_file,
        ),
    )

    with (
        patch(
            "agent.artifact_storage.tempfile.NamedTemporaryFile",
            new=named_temporary_file,
        ),
        pytest.raises(ArtifactStorageError) as error,
    ):
        prepare_artifact_storage(agents_dir)

    artifact_dir = agents_dir / ".adk" / "artifacts"
    assert str(error.value) == _ERROR_MESSAGE
    assert error.value.__cause__ is write_error
    assert list(artifact_dir.iterdir()) == []
    assert len(probe_files) == 1
    probe_files[0].close.assert_called_once()


@pytest.mark.parametrize("failure_stage", ["mkdir", "unlink", "rmdir"])
def test_prepare_translates_create_and_cleanup_failures(
    failure_stage: str,
    tmp_path: Path,
) -> None:
    """Fail closed for artifact creation and both probe-cleanup boundaries."""
    agents_dir = tmp_path / "agents"
    agents_dir.mkdir()
    boundary_error = OSError(f"filesystem rejected {failure_stage}")

    if failure_stage == "mkdir":
        boundary = cast(
            MagicMock,
            create_autospec(
                Path.mkdir,
                spec_set=True,
                side_effect=boundary_error,
            ),
        )
        boundary_patch = patch.object(Path, "mkdir", new=boundary)
    elif failure_stage == "unlink":
        real_unlink = Path.unlink

        def fail_probe_unlink(path: Path, *args: Any, **kwargs: Any) -> None:
            if path.name.startswith(".probe-"):
                raise boundary_error
            real_unlink(path, *args, **kwargs)

        boundary = cast(
            MagicMock,
            create_autospec(
                Path.unlink,
                spec_set=True,
                side_effect=fail_probe_unlink,
            ),
        )
        boundary_patch = patch.object(Path, "unlink", new=boundary)
    else:
        real_rmdir = Path.rmdir

        def fail_probe_rmdir(path: Path) -> None:
            if path.name.startswith(".artifact-storage-probe-"):
                raise boundary_error
            real_rmdir(path)

        boundary = cast(
            MagicMock,
            create_autospec(
                Path.rmdir,
                spec_set=True,
                side_effect=fail_probe_rmdir,
            ),
        )
        boundary_patch = patch.object(Path, "rmdir", new=boundary)

    with (
        boundary_patch,
        pytest.raises(ArtifactStorageError) as error,
    ):
        prepare_artifact_storage(agents_dir)

    assert str(error.value) == _ERROR_MESSAGE
    assert error.value.__cause__ is boundary_error
    boundary.assert_called()
