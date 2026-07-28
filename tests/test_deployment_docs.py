"""Manual GHCR deployment documentation contract tests."""

import re
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
README_PATH = REPOSITORY_ROOT / "README.md"
DEPLOYMENT_GUIDE_PATH = REPOSITORY_ROOT / "docs" / "DEPLOYMENT.md"
IMAGE_EXPORT = 'export IMAGE="ghcr.io/<your-org-or-username>/<your-repository>:main"'
PREBUILT_DEPLOY_COMMAND = (
    "docker compose pull && docker compose up --no-build --wait --wait-timeout 180"
)
LOCAL_BUILD_COMMAND = "docker compose up --build --wait --wait-timeout 180"


def _bash_blocks(document: str) -> list[str]:
    """Return the contents of each fenced Bash block."""
    return re.findall(r"(?ms)^```bash[ \t]*\n(.*?)^```[ \t]*$", document)


def _commands(block: str) -> list[str]:
    """Return non-comment commands from one Bash block."""
    return [
        line.strip()
        for line in block.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


@pytest.mark.parametrize("guide_path", (README_PATH, DEPLOYMENT_GUIDE_PATH))
def test_prebuilt_image_workflow_uses_configurable_compose_contract(
    guide_path: Path,
) -> None:
    """Select, pull, and start one fork-owned image in a fail-closed sequence."""
    document = guide_path.read_text(encoding="utf-8")
    deployment_blocks = [
        block for block in _bash_blocks(document) if "docker compose pull" in block
    ]

    assert len(deployment_blocks) == 1
    assert _commands(deployment_blocks[0]) == [
        IMAGE_EXPORT,
        PREBUILT_DEPLOY_COMMAND,
    ]
    assert "cloned repository" in document
    assert "`.env`" in document
    assert "lowercase" in document
    assert "session-scoped" in document
    assert LOCAL_BUILD_COMMAND in document


def test_guides_remove_stale_registry_and_compose_edit_instructions() -> None:
    """Keep forks away from the template image and disconnected manual pulls."""
    readme = README_PATH.read_text(encoding="utf-8")
    deployment_guide = DEPLOYMENT_GUIDE_PATH.read_text(encoding="utf-8")
    combined = f"{readme}\n{deployment_guide}".casefold()

    assert "ghcr.io/queryplanner/google-adk-on-bare-metal" not in combined
    assert "docker pull " not in combined
    assert "update your `compose.yaml`" not in deployment_guide.casefold()
    assert "image: ghcr.io/" not in deployment_guide.casefold()


def test_private_registry_login_uses_least_privilege_stdin_contract() -> None:
    """Document optional private-package auth without embedding a credential."""
    document = DEPLOYMENT_GUIDE_PATH.read_text(encoding="utf-8")
    login_blocks = [
        block for block in _bash_blocks(document) if "docker login ghcr.io" in block
    ]

    assert len(login_blocks) == 1
    assert _commands(login_blocks[0]) == [
        (
            """printf '%s' "$GHCR_TOKEN" | docker login ghcr.io """
            """-u YOUR_GITHUB_USERNAME --password-stdin"""
        )
    ]
    assert "Public GHCR packages can be pulled anonymously." in document
    assert "classic personal access token with only `read:packages`" in document
    assert "$GITHUB_TOKEN" not in document
    assert "echo " not in login_blocks[0]
