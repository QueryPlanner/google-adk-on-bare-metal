"""Private Docker Compose environment serializer tests."""

from __future__ import annotations

import json
import os
import runpy
import shutil
import subprocess
import sys
import warnings
from pathlib import Path
from unittest.mock import create_autospec, patch

import pytest

from agent.compose_env import (
    ComposeEnvironmentError,
    main,
    parse_compose_environment,
    quote_compose_value,
    serialize_compose_environment,
    write_compose_environment,
)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("", '""'),
        ("simple", '"simple"'),
        (" leading and trailing ", '" leading and trailing "'),
        ('double"quote', '"double\\"quote"'),
        ("single'quote", '"single\'quote"'),
        ("back\\slash", '"back\\\\slash"'),
        ("equals=value # literal", '"equals=value # literal"'),
        ("$VAR ${OTHER} $$", '"$$VAR $${OTHER} $$$$"'),
        ("\tunicode-हॅलो", '"\tunicode-हॅलो"'),
    ],
)
def test_quote_compose_value_preserves_supported_bytes(
    value: str,
    expected: str,
) -> None:
    """Preserve supported values while neutralizing Compose interpolation."""
    assert quote_compose_value(value) == expected


@pytest.mark.parametrize("unsafe_character", ["\0", "\r", "\n"])
def test_quote_compose_value_rejects_line_injection(
    unsafe_character: str,
) -> None:
    """Reject control bytes without including the supplied value in the error."""
    canary = f"secret-before{unsafe_character}secret-after"

    with pytest.raises(
        ComposeEnvironmentError,
        match="environment values cannot contain NUL or line breaks",
    ) as error:
        quote_compose_value(canary)

    assert "secret-before" not in str(error.value)
    assert "secret-after" not in str(error.value)


def test_serialize_compose_environment_preserves_allowlist_order() -> None:
    """Write only selected variables in the caller's explicit order."""
    environment = {
        "FIRST": "one",
        "SECOND": "$two",
        "UNSELECTED": "three",
    }

    assert (
        serialize_compose_environment(
            ["SECOND", "FIRST"],
            environment,
        )
        == 'SECOND="$$two"\nFIRST="one"\n'
    )


def test_serialize_compose_environment_requires_at_least_one_key() -> None:
    """Reject an empty allowlist instead of creating an ambiguous file."""
    with pytest.raises(
        ComposeEnvironmentError,
        match="at least one environment key is required",
    ):
        serialize_compose_environment([], {})


@pytest.mark.parametrize(
    "name",
    ["", "1INVALID", "INVALID-NAME", "INVALID.NAME", "INVALID\nINJECT"],
)
def test_serialize_compose_environment_rejects_invalid_keys(name: str) -> None:
    """Reject unsafe key syntax without reflecting it into the error."""
    with pytest.raises(
        ComposeEnvironmentError,
        match="environment key is invalid",
    ) as error:
        serialize_compose_environment([name], {name: "secret-canary"})

    assert str(error.value) == "environment key is invalid"
    assert "secret-canary" not in str(error.value)


def test_serialize_compose_environment_rejects_duplicate_keys() -> None:
    """Prevent ambiguous duplicate assignments."""
    with pytest.raises(
        ComposeEnvironmentError,
        match="environment key is duplicated: DUPLICATE",
    ):
        serialize_compose_environment(
            ["DUPLICATE", "DUPLICATE"],
            {"DUPLICATE": "secret-canary"},
        )


def test_serialize_compose_environment_rejects_missing_keys() -> None:
    """Require every allowlisted name without exposing another value."""
    with pytest.raises(
        ComposeEnvironmentError,
        match="environment key is missing: MISSING",
    ) as error:
        serialize_compose_environment(
            ["PRESENT", "MISSING"],
            {"PRESENT": "secret-canary"},
        )

    assert "secret-canary" not in str(error.value)


def test_parse_compose_environment_inverts_the_exact_serializer() -> None:
    """Recover every supported escape without evaluating shell syntax."""
    environment = {
        "FIRST": "",
        "SECOND": (
            "dollar$VAR${OTHER} quote\" single' backslash\\ tab\t unicode-हॅलो # equals="
        ),
    }
    payload = serialize_compose_environment(
        ["SECOND", "FIRST"],
        environment,
    )

    assert parse_compose_environment(payload, ["SECOND", "FIRST"]) == {
        "SECOND": environment["SECOND"],
        "FIRST": "",
    }


@pytest.mark.parametrize(
    ("payload", "names"),
    [
        ('KEY="value"\n', []),
        ('INVALID-NAME="value"\n', ["INVALID-NAME"]),
        ('KEY="value"\nKEY="value"\n', ["KEY", "KEY"]),
        ('KEY="value"', ["KEY"]),
        ('KEY="before\0after"\n', ["KEY"]),
        ('KEY="before\rafter"\n', ["KEY"]),
        ('KEY="value"\nEXTRA="value"\n', ["KEY"]),
        ('OTHER="value"\n', ["KEY"]),
        ("KEY=value\n", ["KEY"]),
        ('KEY="value\n', ["KEY"]),
        ('KEY="trailing\\"\n', ["KEY"]),
        ('KEY="$value"\n', ["KEY"]),
        ('KEY="trailing$"\n', ["KEY"]),
        ('KEY="raw"quote"\n', ["KEY"]),
        ('KEY="noncanonical\\q"\n', ["KEY"]),
    ],
)
def test_parse_compose_environment_rejects_noncanonical_payloads(
    payload: str,
    names: list[str],
) -> None:
    """Reject malformed framing and escapes without reflecting payload bytes."""
    with pytest.raises(
        ComposeEnvironmentError,
        match=(
            "at least one environment key is required"
            "|environment key is invalid"
            "|environment key is duplicated"
            "|serialized environment is invalid"
        ),
    ) as error:
        parse_compose_environment(payload, names)

    assert "value" not in str(error.value)
    assert "before" not in str(error.value)


def test_write_compose_environment_creates_complete_private_file(
    tmp_path: Path,
) -> None:
    """Persist exact serialized bytes with owner-only permissions."""
    output = tmp_path / "deploy.env"

    write_compose_environment(
        output,
        ["SECRET", "EMPTY"],
        {"SECRET": "$value", "EMPTY": ""},
    )

    assert output.read_bytes() == b'SECRET="$$value"\nEMPTY=""\n'
    assert output.stat().st_mode & 0o777 == 0o600
    assert list(tmp_path.iterdir()) == [output]


@pytest.mark.parametrize(
    ("names", "environment"),
    [
        ([], {}),
        (["INVALID-NAME"], {"INVALID-NAME": "secret-canary"}),
        (["DUPLICATE", "DUPLICATE"], {"DUPLICATE": "secret-canary"}),
        (["MISSING"], {}),
        (["SECRET"], {"SECRET": "before\0after"}),
        (["SECRET"], {"SECRET": "before\rafter"}),
        (["SECRET"], {"SECRET": "before\nafter"}),
    ],
)
def test_write_validation_fails_before_destination_creation(
    names: list[str],
    environment: dict[str, str],
    tmp_path: Path,
) -> None:
    """Complete validation before opening an output path."""
    output = tmp_path / "deploy.env"

    with pytest.raises(ComposeEnvironmentError):
        write_compose_environment(output, names, environment)

    assert not output.exists()


def test_write_compose_environment_never_replaces_existing_file(
    tmp_path: Path,
) -> None:
    """Fail closed when the requested output path already exists."""
    output = tmp_path / "deploy.env"
    output.write_text("operator-owned\n", encoding="utf-8")

    with pytest.raises(FileExistsError):
        write_compose_environment(output, ["SECRET"], {"SECRET": "new-value"})

    assert output.read_text(encoding="utf-8") == "operator-owned\n"
    assert list(tmp_path.iterdir()) == [output]


def test_write_compose_environment_never_replaces_symlink(
    tmp_path: Path,
) -> None:
    """Treat an existing symlink as a collision without changing its target."""
    target = tmp_path / "operator-owned"
    output = tmp_path / "deploy.env"
    target.write_text("preserve\n", encoding="utf-8")
    output.symlink_to(target)

    with pytest.raises(FileExistsError):
        write_compose_environment(output, ["SECRET"], {"SECRET": "new-value"})

    assert output.is_symlink()
    assert target.read_text(encoding="utf-8") == "preserve\n"
    assert set(tmp_path.iterdir()) == {output, target}


def test_write_compose_environment_removes_partial_write(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Remove the private temporary file if writing stops making progress."""
    output = tmp_path / "deploy.env"
    real_write = os.write
    calls = 0

    def partial_then_stalled(descriptor: int, payload: bytes | memoryview) -> int:
        nonlocal calls
        assert not output.exists()
        calls += 1
        if calls == 1:
            return real_write(descriptor, bytes(payload[:1]))
        return 0

    monkeypatch.setattr(os, "write", partial_then_stalled)

    with pytest.raises(OSError, match="write made no progress"):
        write_compose_environment(
            output,
            ["SECRET"],
            {"SECRET": "secret-canary"},
        )

    assert calls == 2
    assert not output.exists()
    assert list(tmp_path.iterdir()) == []


def test_write_compose_environment_handles_successful_short_writes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Publish only after every successful short write is complete."""
    output = tmp_path / "deploy.env"
    real_write = os.write
    calls = 0

    def one_byte_at_a_time(descriptor: int, payload: bytes | memoryview) -> int:
        nonlocal calls
        assert not output.exists()
        calls += 1
        return real_write(descriptor, bytes(payload[:1]))

    monkeypatch.setattr(os, "write", one_byte_at_a_time)
    write_compose_environment(
        output,
        ["SECRET"],
        {"SECRET": "secret-canary"},
    )

    assert calls == len(b'SECRET="secret-canary"\n')
    assert output.read_bytes() == b'SECRET="secret-canary"\n'
    assert list(tmp_path.iterdir()) == [output]


def test_write_compose_environment_close_failure_is_not_published(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Keep the final pathname absent when closing the complete temp file fails."""
    output = tmp_path / "deploy.env"
    real_close = os.close

    def close_then_fail(descriptor: int) -> None:
        real_close(descriptor)
        raise OSError("synthetic close failure")

    monkeypatch.setattr(os, "close", close_then_fail)

    with pytest.raises(OSError, match="synthetic close failure"):
        write_compose_environment(
            output,
            ["SECRET"],
            {"SECRET": "secret-canary"},
        )

    assert not output.exists()
    assert list(tmp_path.iterdir()) == []


def test_write_compose_environment_removes_unsynced_file(
    tmp_path: Path,
) -> None:
    """Remove the private temporary file when synchronization fails."""
    output = tmp_path / "deploy.env"
    sync_failure = create_autospec(
        os.fsync,
        spec_set=True,
        side_effect=OSError("synthetic sync failure"),
    )

    with (
        patch("agent.compose_env.os.fsync", new=sync_failure),
        pytest.raises(OSError, match="synthetic sync failure"),
    ):
        write_compose_environment(
            output,
            ["SECRET"],
            {"SECRET": "secret-canary"},
        )

    sync_failure.assert_called_once()
    assert not output.exists()


def test_cli_reports_usage_without_creating_output(
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    """Return a distinct usage status before opening a destination."""
    output = tmp_path / "deploy.env"

    assert main([str(output)], {}) == 2

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == (
        "usage: python -m agent.compose_env OUTPUT VARIABLE [VARIABLE ...]\n"
    )
    assert not output.exists()


def test_cli_writes_selected_environment(
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    """Return success without printing serialized environment bytes."""
    output = tmp_path / "deploy.env"

    assert (
        main(
            [str(output), "SECRET"],
            {"SECRET": "secret-canary"},
        )
        == 0
    )

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""
    assert output.read_text(encoding="utf-8") == 'SECRET="secret-canary"\n'


@pytest.mark.parametrize(
    ("environment", "expected_error"),
    [
        ({}, "ERROR: environment key is missing: SECRET\n"),
        (
            {"SECRET": "secret-canary\ninjected"},
            "ERROR: environment values cannot contain NUL or line breaks\n",
        ),
    ],
)
def test_cli_reports_safe_validation_failures(
    capsys: pytest.CaptureFixture[str],
    environment: dict[str, str],
    expected_error: str,
    tmp_path: Path,
) -> None:
    """Report deterministic validation errors without echoing values."""
    output = tmp_path / "deploy.env"

    assert main([str(output), "SECRET"], environment) == 1

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == expected_error
    assert "secret-canary" not in captured.err
    assert not output.exists()


def test_cli_reports_output_collision_without_path_or_value(
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    """Map filesystem failures to one secret-free public error."""
    output = tmp_path / "deploy.env"
    output.write_text("operator-owned\n", encoding="utf-8")

    assert (
        main(
            [str(output), "SECRET"],
            {"SECRET": "secret-canary"},
        )
        == 1
    )

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == "ERROR: environment file could not be created\n"
    assert str(output) not in captured.err
    assert "secret-canary" not in captured.err
    assert output.read_text(encoding="utf-8") == "operator-owned\n"


def test_module_entrypoint_uses_process_environment(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Exercise the supported module CLI without exposing its selected value."""
    output = tmp_path / "deploy.env"
    monkeypatch.setattr(
        sys,
        "argv",
        ["agent.compose_env", str(output), "SECRET"],
    )
    monkeypatch.setenv("SECRET", "process-secret-canary")

    with (
        warnings.catch_warnings(),
        pytest.raises(SystemExit) as exit_error,
    ):
        warnings.filterwarnings(
            "ignore",
            message=".*found in sys.modules.*",
            category=RuntimeWarning,
        )
        runpy.run_module("agent.compose_env", run_name="__main__")

    assert exit_error.value.code == 0
    assert output.read_text(encoding="utf-8") == ('SECRET="process-secret-canary"\n')


def test_real_compose_round_trip_preserves_shell_sensitive_value(
    tmp_path: Path,
) -> None:
    """Use Docker Compose's parser to verify the serialized byte contract."""
    docker = shutil.which("docker")
    if docker is None:
        pytest.skip("Docker CLI is unavailable")

    sample_value = (
        "dollar$VAR${OTHER} quote\" single' backslash\\ space # hash & equals="
    )
    env_file = tmp_path / "deploy.env"
    compose_file = tmp_path / "compose.yaml"
    write_compose_environment(
        env_file,
        ["SECRET"],
        {"SECRET": sample_value},
    )
    compose_file.write_text(
        "services:\n"
        "  agent:\n"
        "    image: synthetic.invalid/agent:compose-env-contract\n"
        "    env_file:\n"
        "      - ${ENV_FILE:?Set ENV_FILE to the private environment file}\n",
        encoding="utf-8",
    )
    environment = {
        "COMPOSE_DISABLE_ENV_FILE": "1",
        "ENV_FILE": str(env_file),
        "HOME": os.environ.get("HOME", str(tmp_path)),
        "LANG": "C",
        "PATH": os.environ["PATH"],
    }

    result = subprocess.run(  # noqa: S603 - resolved Docker, fixed arguments
        [
            docker,
            "compose",
            "--project-name",
            f"compose-env-{os.getpid()}",
            "--env-file",
            str(env_file),
            "-f",
            str(compose_file),
            "config",
            "--format",
            "json",
        ],
        cwd=tmp_path,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    service = json.loads(result.stdout)["services"]["agent"]
    # Canonical Compose output re-escapes literal dollars for safe re-parsing.
    assert service["environment"]["SECRET"] == sample_value.replace("$", "$$")
    assert service["image"] == "synthetic.invalid/agent:compose-env-contract"
    assert env_file.stat().st_mode & 0o777 == 0o600
