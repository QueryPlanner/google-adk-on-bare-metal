"""Docker Compose healthcheck contract tests."""

from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
COMPOSE_PATH = REPOSITORY_ROOT / "compose.yaml"
DEPLOYMENT_GUIDE_PATH = REPOSITORY_ROOT / "docs" / "DEPLOYMENT.md"
COMPOSE_GUIDE_PATH = (
    REPOSITORY_ROOT / "docs" / "base-infra" / "docker-compose-workflow.md"
)
README_PATH = REPOSITORY_ROOT / "README.md"
EXPECTED_HEALTHCHECK_BLOCK = """\
      test:
        - CMD
        - python
        - -c
        - >-
          import urllib.request;
          urllib.request.build_opener(
          urllib.request.ProxyHandler({})).open(
          "http://127.0.0.1:8080/ready", timeout=3).read()
      interval: 10s
      timeout: 5s
      retries: 5
      start_period: 60s"""


def _indented_block(document: str, heading: str, indentation: int) -> str:
    """Return lines nested beneath an exact YAML heading."""
    lines = document.splitlines()
    start = lines.index(f"{' ' * indentation}{heading}") + 1
    block: list[str] = []

    for line in lines[start:]:
        if line and not line.startswith(" " * (indentation + 1)):
            break
        block.append(line)

    return "\n".join(block)


def test_agent_healthcheck_has_exact_database_readiness_contract() -> None:
    """Keep one bounded exec-form readiness probe for the small-VM runtime."""
    document = COMPOSE_PATH.read_text(encoding="utf-8")
    agent_block = _indented_block(document, "agent:", 2)
    healthcheck_block = _indented_block(agent_block, "healthcheck:", 4)

    assert agent_block.splitlines().count("    healthcheck:") == 1
    assert healthcheck_block == EXPECTED_HEALTHCHECK_BLOCK


def test_operator_guides_wait_for_container_health() -> None:
    """Document a bounded startup command wherever operators start Compose."""
    guide_paths = [DEPLOYMENT_GUIDE_PATH, COMPOSE_GUIDE_PATH, README_PATH]

    for guide_path in guide_paths:
        guide = guide_path.read_text(encoding="utf-8")
        assert "docker compose up --build --wait --wait-timeout 180" in guide
        assert "`/ready`" in guide
        assert "`/live`" in guide
        assert "not_configured" in guide
