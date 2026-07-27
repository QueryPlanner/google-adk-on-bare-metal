"""Deployment workflow safety contract tests."""

from pathlib import Path

import pytest

WORKFLOW_PATH = (
    Path(__file__).resolve().parents[1] / ".github" / "workflows" / "docker-publish.yml"
)
EXPECTED_DEPLOY_GUARD = (
    "github.event_name == 'workflow_dispatch' && "
    "github.ref == 'refs/heads/main' && "
    "inputs.deploy"
)
EXPECTED_DEPLOY_CLAUSES = {
    "github.event_name == 'workflow_dispatch'",
    "github.ref == 'refs/heads/main'",
    "inputs.deploy",
}


def _indented_block(document: str, heading: str, indentation: int) -> str:
    """Return the lines nested beneath an exact YAML heading."""
    lines = document.splitlines()
    start = lines.index(f"{' ' * indentation}{heading}") + 1
    block: list[str] = []

    for line in lines[start:]:
        if line and not line.startswith(" " * (indentation + 1)):
            break
        block.append(line)

    return "\n".join(block)


def _deploy_guard(document: str) -> str:
    """Extract and normalize the deploy job condition."""
    deploy_block = _indented_block(document, "deploy:", 2)
    lines = deploy_block.splitlines()
    condition_start = lines.index("    if: >-") + 1
    condition_lines: list[str] = []

    for line in lines[condition_start:]:
        if not line.startswith("      "):
            break
        condition_lines.append(line.strip())

    return " ".join(condition_lines)


def _evaluate_guard(
    expression: str,
    *,
    event_name: str,
    git_ref: str,
    deploy: bool | None,
) -> bool:
    """Evaluate the supported clauses extracted from the workflow guard."""
    clauses = tuple(clause.strip() for clause in expression.split("&&"))
    outcomes = {
        "github.event_name == 'workflow_dispatch'": (event_name == "workflow_dispatch"),
        "github.ref == 'refs/heads/main'": git_ref == "refs/heads/main",
        "inputs.deploy": deploy is True,
    }

    assert len(clauses) == len(EXPECTED_DEPLOY_CLAUSES)
    assert set(clauses) == EXPECTED_DEPLOY_CLAUSES
    return all(outcomes[clause] for clause in clauses)


def test_workflow_exposes_explicit_deploy_confirmation() -> None:
    document = WORKFLOW_PATH.read_text(encoding="utf-8")
    trigger_block = _indented_block(document, "on:", 0)
    dispatch_block = _indented_block(trigger_block, "workflow_dispatch:", 2)
    deploy_block = _indented_block(document, "deploy:", 2)

    assert "  push:" in trigger_block
    assert "  workflow_dispatch:" in trigger_block
    assert "    inputs:" in dispatch_block
    assert "      deploy:" in dispatch_block
    assert "        required: true" in dispatch_block
    assert "        type: boolean" in dispatch_block
    assert "        default: false" in dispatch_block
    assert "    needs: build" in deploy_block


@pytest.mark.parametrize(
    ("event_name", "git_ref", "deploy", "expected"),
    [
        ("push", "refs/heads/main", True, False),
        ("push", "refs/tags/v1.0.0", True, False),
        ("workflow_dispatch", "refs/heads/feature", True, False),
        ("workflow_dispatch", "refs/tags/main", True, False),
        ("workflow_dispatch", "refs/heads/main", False, False),
        ("workflow_dispatch", "refs/heads/main", None, False),
        ("workflow_dispatch", "refs/heads/main", True, True),
        ("schedule", "refs/heads/main", True, False),
    ],
)
def test_deploy_guard_requires_manual_main_confirmation(
    event_name: str,
    git_ref: str,
    deploy: bool | None,
    expected: bool,
) -> None:
    document = WORKFLOW_PATH.read_text(encoding="utf-8")
    guard = _deploy_guard(document)

    assert guard == EXPECTED_DEPLOY_GUARD
    assert (
        _evaluate_guard(
            guard,
            event_name=event_name,
            git_ref=git_ref,
            deploy=deploy,
        )
        is expected
    )
