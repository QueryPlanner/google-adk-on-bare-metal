"""Fail-closed preparation for ADK filesystem artifact storage."""

import os
import tempfile
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import IO

ARTIFACT_STORAGE_ERROR_MESSAGE = "Artifact storage is unavailable."


class ArtifactStorageError(RuntimeError):
    """Indicate that durable artifact storage cannot be used safely."""


@dataclass(frozen=True, slots=True)
class PreparedArtifactStorage:
    """Resolved paths and URI for one prepared artifact storage root."""

    agents_dir: Path
    artifact_dir: Path
    artifact_service_uri: str


def _require_contained(path: Path, agents_dir: Path) -> None:
    """Reject an artifact path that resolves outside its agent directory."""
    if not path.is_relative_to(agents_dir):
        raise ValueError("Artifact directory resolves outside the agent directory.")


def _cleanup_probe(
    probe_file: IO[bytes] | None,
    probe_file_path: Path | None,
    probe_dir: Path | None,
) -> None:
    """Best-effort cleanup that cannot replace the probe's original failure."""
    if probe_file is not None:
        with suppress(Exception):
            probe_file.close()
    if probe_file_path is not None:
        with suppress(Exception):
            probe_file_path.unlink(missing_ok=True)
    if probe_dir is not None:
        with suppress(Exception):
            probe_dir.rmdir()


def prepare_artifact_storage(agent_dir: str | Path) -> PreparedArtifactStorage:
    """Resolve and verify durable artifact storage beneath an existing agent dir."""
    probe_dir: Path | None = None
    probe_file_path: Path | None = None
    probe_file: IO[bytes] | None = None

    try:
        agents_dir = Path(agent_dir).resolve(strict=True)
        if not agents_dir.is_dir():
            raise NotADirectoryError("Agent directory is not a directory.")

        artifact_candidate = agents_dir / ".adk" / "artifacts"
        _require_contained(
            artifact_candidate.resolve(strict=False),
            agents_dir,
        )
        artifact_candidate.mkdir(parents=True, exist_ok=True)
        artifact_dir = artifact_candidate.resolve(strict=True)
        _require_contained(artifact_dir, agents_dir)

        probe_dir = Path(
            tempfile.mkdtemp(prefix=".artifact-storage-probe-", dir=artifact_dir)
        )
        probe_dir.chmod(0o700)
        # Explicit lifetime preserves a write/sync failure if close also fails.
        probe_file = tempfile.NamedTemporaryFile(  # noqa: SIM115
            mode="w+b",
            prefix=".probe-",
            dir=probe_dir,
            delete=False,
        )
        probe_file_path = Path(probe_file.name)
        probe_file_path.chmod(0o600)
        probe_file.write(b"\0")
        probe_file.flush()
        os.fsync(probe_file.fileno())
        probe_file.close()
        probe_file = None

        probe_file_path.unlink()
        probe_file_path = None
        probe_dir.rmdir()
        probe_dir = None

        return PreparedArtifactStorage(
            agents_dir=agents_dir,
            artifact_dir=artifact_dir,
            artifact_service_uri=artifact_dir.as_uri(),
        )
    except Exception as exc:
        _cleanup_probe(probe_file, probe_file_path, probe_dir)
        raise ArtifactStorageError(ARTIFACT_STORAGE_ERROR_MESSAGE) from exc
