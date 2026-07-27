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


def test_quality_workflow_provisions_real_postgres() -> None:
    document = WORKFLOW_PATH.read_text(encoding="utf-8")
    test_job = _indented_block(document, "test:", 2)
    job_environment = _indented_block(test_job, "env:", 4)
    services = _indented_block(test_job, "services:", 4)
    postgres = _indented_block(services, "postgres:", 6)
    postgres_environment = _indented_block(postgres, "env:", 8)
    ports = _indented_block(postgres, "ports:", 8)

    assert "    timeout-minutes: 15" in test_job
    assert (
        "      TEST_POSTGRES_ADMIN_URL: "
        "postgresql://postgres:postgres-test-password@127.0.0.1:5432/postgres"
        in job_environment
    )
    assert "${{ secrets" not in job_environment
    assert "DATABASE_URL:" not in job_environment
    assert (
        "        image: "
        "postgres:17.9-alpine3.23@sha256:"
        "c7526c0f6c3f30260a563d7bcf8ad778effac59a44f8ffa86678c35418338609" in postgres
    )
    assert "          POSTGRES_USER: postgres" in postgres_environment
    assert "          POSTGRES_PASSWORD: postgres-test-password" in postgres_environment
    assert "          POSTGRES_DB: postgres" in postgres_environment
    assert "          - 5432:5432" in ports
    assert '--health-cmd "pg_isready -U postgres -d postgres"' in postgres
    assert "--health-interval 5s" in postgres
    assert "--health-timeout 5s" in postgres
    assert "--health-retries 10" in postgres
    assert (
        "run: uv run pytest --cov=src --cov-report=xml "
        "--cov-report=term-missing" in test_job
    )
