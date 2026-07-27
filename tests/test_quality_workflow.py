"""Code-quality workflow contract tests."""

from pathlib import Path

WORKFLOW_PATH = (
    Path(__file__).resolve().parents[1] / ".github" / "workflows" / "code-quality.yml"
)


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


def test_quality_workflow_runs_for_every_main_change() -> None:
    document = WORKFLOW_PATH.read_text(encoding="utf-8")
    trigger_block = _indented_block(document, "on:", 0)
    push_block = _indented_block(trigger_block, "push:", 2)
    pull_request_block = _indented_block(trigger_block, "pull_request:", 2)

    assert '    branches: [ "main" ]' in push_block
    assert '    branches: [ "main" ]' in pull_request_block
    assert "  workflow_call:" in trigger_block
    assert "paths:" not in trigger_block
    assert "paths-ignore:" not in trigger_block
    assert "migration/port-agent-foundation" not in trigger_block


def test_quality_workflow_disables_ruff_mutation() -> None:
    document = WORKFLOW_PATH.read_text(encoding="utf-8")

    assert "run: uv run ruff check --no-fix --output-format=github" in document
