"""Docker build-context isolation contract tests."""

import shlex
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DOCKERFILE_PATH = REPOSITORY_ROOT / "Dockerfile"
DOCKERIGNORE_PATH = REPOSITORY_ROOT / ".dockerignore"

EXPECTED_INCLUDES = {
    "Dockerfile",
    ".dockerignore",
    "entrypoint.sh",
    "pyproject.toml",
    "uv.lock",
    "src/",
    "src/agent/",
    "src/agent/**",
}
EXPECTED_NESTED_EXCLUSIONS = {
    "**/.adk/",
    "**/.adk/**",
    "**/.env",
    "**/.env.*",
    "**/.git/",
    "**/.git/**",
    "**/.venv/",
    "**/.venv/**",
    "**/__pycache__/",
    "**/__pycache__/**",
    "**/*.py[cod]",
    "**/.DS_Store",
}


def _effective_patterns() -> list[str]:
    """Return non-comment Docker ignore patterns in evaluation order."""
    return [
        line.strip()
        for line in DOCKERIGNORE_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]


def _local_copy_sources() -> set[str]:
    """Extract sources copied from the local build context."""
    sources: set[str] = set()

    for raw_line in DOCKERFILE_PATH.read_text(encoding="utf-8").splitlines():
        if not raw_line.startswith("COPY "):
            continue

        tokens = shlex.split(raw_line)
        arguments = tokens[1:]
        if any(argument.startswith("--from=") for argument in arguments):
            continue

        paths = [argument for argument in arguments if not argument.startswith("--")]
        assert len(paths) >= 2, f"Unsupported COPY instruction: {raw_line}"
        sources.update(path.rstrip("/") for path in paths[:-1])

    return sources


def _source_is_included(source: str, includes: set[str]) -> bool:
    """Return whether an allowlist entry makes a COPY source available."""
    normalized_source = source.removeprefix("./").rstrip("/")
    normalized_includes = {pattern.rstrip("/") for pattern in includes}
    return normalized_source in normalized_includes or any(
        pattern.startswith(f"{normalized_source}/") for pattern in normalized_includes
    )


def _path_is_explicitly_included(path: str, includes: set[str]) -> bool:
    """Return whether a root path is covered by an allowlist exception."""
    for pattern in includes:
        if pattern.endswith("/**"):
            prefix = pattern.removesuffix("/**")
            if path == prefix or path.startswith(f"{prefix}/"):
                return True
        elif path == pattern.rstrip("/"):
            return True

    return False


def test_docker_context_uses_exact_default_deny_allowlist() -> None:
    patterns = _effective_patterns()
    includes = {pattern.removeprefix("!") for pattern in patterns if pattern[0] == "!"}

    assert patterns[0] == "**"
    assert includes == EXPECTED_INCLUDES
    assert set(patterns[1 : 1 + len(EXPECTED_INCLUDES)]) == {
        f"!{path}" for path in EXPECTED_INCLUDES
    }


def test_local_dockerfile_copy_sources_are_allowlisted() -> None:
    patterns = _effective_patterns()
    includes = {pattern.removeprefix("!") for pattern in patterns if pattern[0] == "!"}
    sources = _local_copy_sources()

    assert sources == {"pyproject.toml", "uv.lock", "src", "entrypoint.sh"}
    for source in sources:
        assert (REPOSITORY_ROOT / source).exists()
        assert _source_is_included(source, includes)


def test_nested_runtime_state_is_reexcluded_after_source() -> None:
    patterns = _effective_patterns()
    last_include = max(
        index for index, pattern in enumerate(patterns) if pattern.startswith("!")
    )

    assert set(patterns[last_include + 1 :]) == EXPECTED_NESTED_EXCLUSIONS


@pytest.mark.parametrize(
    "path",
    [
        ".env",
        ".env.production",
        ".git/config",
        ".venv/bin/python",
        ".coverage",
        "coverage.xml",
        "data/qdrant",
        "dist/package.whl",
        "tests/test_agent.py",
        "CLAUDE.md",
    ],
)
def test_local_only_root_paths_are_not_allowlisted(path: str) -> None:
    patterns = _effective_patterns()
    includes = {pattern.removeprefix("!") for pattern in patterns if pattern[0] == "!"}

    assert not _path_is_explicitly_included(path, includes)


def test_dockerfile_specific_ignore_cannot_override_policy() -> None:
    assert not (REPOSITORY_ROOT / "Dockerfile.dockerignore").exists()


def test_runtime_image_has_no_netcat_database_readiness_dependency() -> None:
    """Keep database readiness independent of the netcat package."""
    dockerfile = DOCKERFILE_PATH.read_text(encoding="utf-8").casefold()

    assert "netcat" not in dockerfile
