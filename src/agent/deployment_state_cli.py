"""Operator CLI for inspecting or explicitly adopting VM deployment state."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections.abc import Sequence
from pathlib import Path

from agent.deployment_adoption import (
    DeploymentAdoptionError,
    observe_legacy_deployment,
)
from agent.deployment_state import (
    DeploymentLockBusyError,
    DeploymentStateError,
    DeploymentStateStore,
)


class DeploymentStateCliError(RuntimeError):
    """Report a safe command-line boundary failure."""


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="deployment-state",
        description="Inspect or adopt one private VM deployment state store.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser(
        "inspect",
        help="verify and print secret-free current state and journal metadata",
    )
    inspect_parser.add_argument("--state-dir", type=Path, required=True)

    adopt_parser = subparsers.add_parser(
        "adopt",
        help="record one exact healthy legacy Compose deployment",
    )
    adopt_parser.add_argument("--state-dir", type=Path, required=True)
    adopt_parser.add_argument("--checkout", type=Path, required=True)
    adopt_parser.add_argument("--expected-origin", required=True)
    adopt_parser.add_argument("--compose-project", required=True)
    adopt_parser.add_argument("--compose-service", default="agent")
    adopt_parser.add_argument("--environment-file", type=Path)
    return parser


def _resolved_executable(name: str) -> Path:
    selected = shutil.which(name)
    if selected is None:
        raise DeploymentStateCliError(
            "required deployment observation executable is unavailable"
        )
    return Path(selected).resolve(strict=True)


def _emit(document: dict[str, object]) -> None:
    print(
        json.dumps(
            document,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    )


def _inspect(store: DeploymentStateStore) -> int:
    with store.transaction() as transaction:
        current = transaction.current()
        journal = transaction.journal()
        _emit(
            {
                "status": "empty" if current is None else "recorded",
                "current": None if current is None else current.as_document(),
                "journal": [
                    {
                        "sha256": entry.sha256,
                        **entry.as_document(),
                    }
                    for entry in journal
                ],
            }
        )
    return 0


def _adopt(arguments: argparse.Namespace, store: DeploymentStateStore) -> int:
    checkout = arguments.checkout
    environment_file = (
        checkout / ".env"
        if arguments.environment_file is None
        else arguments.environment_file
    )
    with store.transaction() as transaction:
        if transaction.current() is not None:
            raise DeploymentStateCliError(
                "deployment state has already been initialized"
            )
        observation = observe_legacy_deployment(
            checkout_path=checkout,
            expected_origin=arguments.expected_origin,
            compose_project=arguments.compose_project,
            compose_service=arguments.compose_service,
            environment_path=environment_file,
            git_executable=_resolved_executable("git"),
            docker_executable=_resolved_executable("docker"),
        )
        if observation is None:
            _emit(
                {
                    "status": "fresh",
                    "current": None,
                    "journal": [],
                }
            )
            return 0
        current = transaction.adopt(
            compose_project=observation.compose_project,
            compose_service=observation.compose_service,
            source_revision=observation.revision,
            image_reference=observation.image_reference,
            image_id=observation.image_id,
            oci_revision=observation.oci_revision,
            environment_source=observation.environment_path,
        )
        _emit(
            {
                "status": "adopted",
                "current": current.as_document(),
            }
        )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run one state command and return a secret-free process status."""
    arguments = _parser().parse_args(argv)
    try:
        store = DeploymentStateStore(arguments.state_dir)
        if arguments.command == "inspect":
            return _inspect(store)
        return _adopt(arguments, store)
    except DeploymentLockBusyError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 75
    except (
        DeploymentAdoptionError,
        DeploymentStateCliError,
        DeploymentStateError,
    ) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    except OSError:
        print("ERROR: deployment state operation failed", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
