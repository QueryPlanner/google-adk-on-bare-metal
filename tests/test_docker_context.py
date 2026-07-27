"""Docker build-context isolation contract tests."""

import shlex
from pathlib import Path
from typing import NamedTuple

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


class _DockerStage(NamedTuple):
    """A named Docker build stage and its logical instructions."""

    source: str
    name: str
    instructions: tuple[tuple[str, str], ...]


def _logical_dockerfile_instructions(contents: str) -> list[tuple[str, str]]:
    """Parse logical Dockerfile instructions without matching comments or values."""
    instructions: list[tuple[str, str]] = []
    continuation: list[str] = []

    for raw_line in contents.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        continues = line.endswith("\\")
        continuation.append(line[:-1].rstrip() if continues else line)
        if continues:
            continue

        logical_line = " ".join(continuation)
        keyword, separator, arguments = logical_line.partition(" ")
        assert separator, f"Instruction has no arguments: {logical_line}"
        instructions.append((keyword.upper(), arguments.strip()))
        continuation = []

    assert not continuation, "Dockerfile ends with an incomplete instruction"
    return instructions


def _parse_from(arguments: str) -> tuple[str, str]:
    """Parse the supported named FROM form, including an optional platform flag."""
    tokens = shlex.split(arguments)
    while tokens and tokens[0].startswith("--"):
        option = tokens.pop(0)
        assert option.startswith("--platform="), f"Unsupported FROM option: {option}"

    assert len(tokens) == 3 and tokens[1].casefold() == "as", (
        f"FROM stages must use a named source: {arguments}"
    )
    return tokens[0], tokens[2].casefold()


def _dockerfile_stages(contents: str) -> list[_DockerStage]:
    """Build a stage-aware representation of the Dockerfile."""
    stages: list[_DockerStage] = []
    current_source: str | None = None
    current_name: str | None = None
    current_instructions: list[tuple[str, str]] = []

    for keyword, arguments in _logical_dockerfile_instructions(contents):
        if keyword != "FROM":
            if current_name is not None:
                current_instructions.append((keyword, arguments))
            continue

        if current_source is not None and current_name is not None:
            stages.append(
                _DockerStage(
                    current_source,
                    current_name,
                    tuple(current_instructions),
                )
            )

        current_source, current_name = _parse_from(arguments)
        current_instructions = []

    assert current_source is not None and current_name is not None, (
        "Dockerfile must contain a named FROM stage"
    )
    stages.append(
        _DockerStage(current_source, current_name, tuple(current_instructions))
    )
    return stages


def _is_official_python_image(source: str) -> bool:
    """Return whether a literal image source is the official Python repository."""
    repository = source.partition("@")[0].rsplit(":", maxsplit=1)[0].casefold()
    return repository in {"python", "docker.io/library/python"}


def _assert_shared_python_base_contract(contents: str) -> None:
    """Assert Python stage inheritance and final runtime security invariants."""
    stages = _dockerfile_stages(contents)
    stages_by_name: dict[str, _DockerStage] = {}
    external_stages: list[_DockerStage] = []

    for stage in stages:
        assert stage.name not in stages_by_name, (
            f"Duplicate Docker stage alias: {stage.name}"
        )
        if stage.source.casefold() not in stages_by_name:
            external_stages.append(stage)
        stages_by_name[stage.name] = stage

    external_python_stages = [
        stage for stage in external_stages if _is_official_python_image(stage.source)
    ]
    assert len(external_python_stages) == 1, (
        "Dockerfile must contain exactly one external official Python image"
    )

    python_base = stages_by_name.get("python-base")
    assert python_base is external_python_stages[0]
    assert stages[0] is python_base, "The shared Python base must be the first stage"

    for stage_name in ("builder", "runtime"):
        required_stage = stages_by_name.get(stage_name)
        assert required_stage is not None, f"Missing {stage_name} stage"
        assert required_stage.source.casefold() == python_base.name, (
            f"{stage_name} must derive directly from {python_base.name}"
        )

    runtime = stages_by_name["runtime"]
    assert stages[-1] is runtime, "Runtime must remain the final Docker stage"

    runtime_users = [
        arguments for keyword, arguments in runtime.instructions if keyword == "USER"
    ]
    assert runtime_users and shlex.split(runtime_users[-1]) == ["app"], (
        "Runtime must end as the non-root app user"
    )

    assert any(
        keyword == "COPY"
        and any(
            token.casefold() == "--from=builder" for token in shlex.split(arguments)
        )
        for keyword, arguments in runtime.instructions
    ), "Runtime must copy the built application from builder"


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


def test_docker_stages_share_one_external_python_base() -> None:
    _assert_shared_python_base_contract(DOCKERFILE_PATH.read_text(encoding="utf-8"))


def test_docker_stage_parser_ignores_from_text_outside_instructions() -> None:
    contents = """
    # FROM python:3.12-slim AS stale-example
    FROM --platform=linux/amd64 python:3.13-slim AS python-base
    RUN echo "FROM python:3.11-slim AS not-a-stage"
    FROM python-base AS builder
    FROM python-base AS runtime
    COPY --from=builder /app /app
    USER app
    """

    _assert_shared_python_base_contract(contents)


@pytest.mark.parametrize(
    ("contents", "message"),
    [
        (
            """
            FROM python:3.13-slim AS python-base
            FROM python:3.12-slim AS builder
            FROM python-base AS runtime
            COPY --from=builder /app /app
            USER app
            """,
            "exactly one external official Python image",
        ),
        (
            """
            FROM python:3.13-slim AS python-base
            FROM python-base AS builder
            FROM builder AS runtime
            COPY --from=builder /app /app
            USER app
            """,
            "runtime must derive directly from python-base",
        ),
        (
            """
            FROM python:3.13-slim AS python-base
            FROM python-base AS builder
            FROM python-base AS runtime
            COPY --from=builder /app /app
            USER app
            USER root
            """,
            "Runtime must end as the non-root app user",
        ),
    ],
)
def test_docker_stage_contract_rejects_unsafe_inheritance(
    contents: str,
    message: str,
) -> None:
    with pytest.raises(AssertionError, match=message):
        _assert_shared_python_base_contract(contents)
