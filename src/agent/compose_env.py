"""Serialize selected process settings into a private Docker Compose env file."""

from __future__ import annotations

import os
import re
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path

_ENVIRONMENT_KEY = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
_FORBIDDEN_VALUE_CHARACTERS = frozenset({"\0", "\r", "\n"})


class ComposeEnvironmentError(ValueError):
    """Report a safe, deterministic Compose environment validation failure."""


def quote_compose_value(value: str) -> str:
    """Quote one value without line injection or Compose interpolation."""
    if any(character in value for character in _FORBIDDEN_VALUE_CHARACTERS):
        raise ComposeEnvironmentError(
            "environment values cannot contain NUL or line breaks"
        )

    escaped = value.replace("\\", "\\\\").replace('"', '\\"').replace("$", "$$")
    return f'"{escaped}"'


def serialize_compose_environment(
    names: Sequence[str],
    environment: Mapping[str, str],
) -> str:
    """Serialize an ordered allowlist of environment variables."""
    if not names:
        raise ComposeEnvironmentError("at least one environment key is required")

    seen_names: set[str] = set()
    lines: list[str] = []
    for name in names:
        if _ENVIRONMENT_KEY.fullmatch(name) is None:
            raise ComposeEnvironmentError("environment key is invalid")
        if name in seen_names:
            raise ComposeEnvironmentError(f"environment key is duplicated: {name}")
        if name not in environment:
            raise ComposeEnvironmentError(f"environment key is missing: {name}")

        seen_names.add(name)
        lines.append(f"{name}={quote_compose_value(environment[name])}")
    return "\n".join(lines) + "\n"


def parse_compose_environment(
    payload: str,
    names: Sequence[str],
) -> dict[str, str]:
    """Invert the canonical serializer for one exact ordered allowlist."""
    if not names:
        raise ComposeEnvironmentError("at least one environment key is required")

    seen_names: set[str] = set()
    for name in names:
        if _ENVIRONMENT_KEY.fullmatch(name) is None:
            raise ComposeEnvironmentError("environment key is invalid")
        if name in seen_names:
            raise ComposeEnvironmentError(f"environment key is duplicated: {name}")
        seen_names.add(name)

    if not payload.endswith("\n") or "\0" in payload or "\r" in payload:
        raise ComposeEnvironmentError("serialized environment is invalid")
    lines = payload.split("\n")
    if len(lines) != len(names) + 1:
        raise ComposeEnvironmentError("serialized environment is invalid")

    environment: dict[str, str] = {}
    for name, line in zip(names, lines[:-1], strict=True):
        prefix = f'{name}="'
        if not line.startswith(prefix) or not line.endswith('"'):
            raise ComposeEnvironmentError("serialized environment is invalid")

        encoded = line[len(prefix) : -1]
        decoded: list[str] = []
        index = 0
        while index < len(encoded):
            character = encoded[index]
            if character == "\\":
                index += 1
                if index >= len(encoded):
                    raise ComposeEnvironmentError("serialized environment is invalid")
                decoded.append(encoded[index])
            elif character == "$":
                index += 1
                if index >= len(encoded) or encoded[index] != "$":
                    raise ComposeEnvironmentError("serialized environment is invalid")
                decoded.append("$")
            elif character == '"':
                raise ComposeEnvironmentError("serialized environment is invalid")
            else:
                decoded.append(character)
            index += 1
        environment[name] = "".join(decoded)

    if serialize_compose_environment(names, environment) != payload:
        raise ComposeEnvironmentError("serialized environment is invalid")
    return environment


def write_compose_environment(
    path: Path,
    names: Sequence[str],
    environment: Mapping[str, str],
) -> None:
    """Atomically publish a complete mode-0600 Compose environment file."""
    payload = serialize_compose_environment(names, environment).encode()
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".compose-env-",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary_path = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        remaining = memoryview(payload)
        while remaining:
            written = os.write(descriptor, remaining)
            if written <= 0:
                raise OSError("environment file write made no progress")
            remaining = remaining[written:]
        os.fsync(descriptor)
        descriptor_to_close = descriptor
        descriptor = -1
        os.close(descriptor_to_close)
        os.link(
            temporary_path,
            path,
            follow_symlinks=False,
        )
    finally:
        try:
            if descriptor >= 0:
                os.close(descriptor)
        finally:
            temporary_path.unlink(missing_ok=True)


def main(
    argv: Sequence[str] | None = None,
    environment: Mapping[str, str] | None = None,
) -> int:
    """Write requested variables and return a process exit status."""
    arguments = list(sys.argv[1:] if argv is None else argv)
    if len(arguments) < 2:
        print(
            "usage: python -m agent.compose_env OUTPUT VARIABLE [VARIABLE ...]",
            file=sys.stderr,
        )
        return 2

    selected_environment = os.environ if environment is None else environment
    try:
        write_compose_environment(
            Path(arguments[0]),
            arguments[1:],
            selected_environment,
        )
    except ComposeEnvironmentError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    except OSError:
        print("ERROR: environment file could not be created", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
