"""Static contracts for bounded VM image-retention integration."""

from __future__ import annotations

import re
from pathlib import Path
from typing import cast

import yaml  # type: ignore[import-untyped]

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_PATH = ROOT / ".github" / "workflows" / "docker-publish.yml"
BOOTSTRAP_PATH = ROOT / "scripts" / "deployment_bootstrap.sh"
DEPLOYMENT_GUIDE_PATH = ROOT / "docs" / "DEPLOYMENT.md"
RETENTION_MODULE = "src/agent/deployment_retention.py"


def _workflow() -> dict[str, object]:
    document = yaml.safe_load(WORKFLOW_PATH.read_text(encoding="utf-8"))
    assert isinstance(document, dict)
    return document


def _steps(job: str) -> list[dict[str, object]]:
    jobs = _workflow()["jobs"]
    assert isinstance(jobs, dict)
    selected = jobs[job]
    assert isinstance(selected, dict)
    steps = selected["steps"]
    assert isinstance(steps, list)
    return cast(list[dict[str, object]], steps)


def _step(job: str, name: str) -> dict[str, object]:
    matches = [step for step in _steps(job) if step.get("name") == name]
    assert len(matches) == 1
    return matches[0]


def _run_text(job: str, name: str) -> str:
    command = _step(job, name)["run"]
    assert isinstance(command, str)
    return command


def test_published_image_has_stable_repository_ownership_label() -> None:
    """Bind retention ownership to the repository that built the image."""
    build = _step("build", "Build and push Docker image")
    configuration = build["with"]
    assert isinstance(configuration, dict)
    labels = configuration["labels"]
    assert isinstance(labels, str)

    assert (
        "io.queryplanner.adk.repository=${{ github.repository }}" in labels.splitlines()
    )


def test_bootstrap_binds_retention_module_to_the_exact_release() -> None:
    """Verify the module before either local source use or remote execution."""
    prepare = _run_text("deploy", "Prepare private deployment transport")
    bootstrap = BOOTSTRAP_PATH.read_text(encoding="utf-8")

    assert RETENTION_MODULE in prepare
    assert 'git ls-tree "$DEPLOY_SHA" -- "$SOURCE_PATH"' in prepare
    assert bootstrap.count(RETENTION_MODULE) == 2
    assert 'ls-tree "$DEPLOY_SHA" -- "$TARGET_PATH"' in bootstrap
    assert '[ ! -f "$RELEASE_DIR/$TARGET_PATH" ]' in bootstrap
    assert '[ -L "$RELEASE_DIR/$TARGET_PATH" ]' in bootstrap


def test_retention_blocks_before_promotion_with_only_safe_inputs() -> None:
    """Run retention first and keep production secrets out of its boundary."""
    script = _run_text("deploy", "Run locked transactional deployment")
    retention_start = script.index('RETENTION_COMMAND="')
    retention_result = script.index("RETENTION_STATUS=$?")
    retention_gate = script.index('if [ "$RETENTION_STATUS" -ne 0 ]')
    promotion_start = script.index('CONTROLLER_COMMAND="')
    promotion_result = script.index("CONTROLLER_STATUS=$?")
    retention = script[retention_start:promotion_start]
    promotion = script[promotion_start:promotion_result]

    assert script.index("BOOTSTRAP_STATUS=${PIPESTATUS[0]}") < retention_start
    assert retention_start < retention_result < retention_gate < promotion_start
    assert 'exit "$RETENTION_STATUS"' in script[retention_gate:promotion_start]
    assert (
        'runpy.run_module(\\"agent.deployment_retention\\", '
        'run_name=\\"__main__\\", alter_sys=True)'
    ) in retention
    assert "python3 -I -S -B" in retention
    assert "exec flock -n" in retention
    assert "env -i" in retention
    assert "--state-dir" in retention
    assert '--repository \\"${{ github.repository }}\\"' in retention
    assert '--target-reference \\"$IMAGE_REPOSITORY@$IMAGE_DIGEST\\"' in retention
    assert "--apply" in retention
    assert '"$RETENTION_COMMAND" < /dev/null' in retention
    assert "10m" in retention
    assert "incomplete" not in promotion

    for forbidden in (
        "AGENT_NAME",
        "DATABASE_URL",
        "OPENROUTER_API_KEY",
        "GOOGLE_API_KEY",
        "LANGFUSE_SECRET_KEY",
        "SSH_PRIVATE_KEY",
        "PAYLOAD_FILE",
        "--environment-stdin",
        "docker image",
        " prune",
        "--force",
    ):
        assert forbidden not in retention

    assert (
        script.count(
            'runpy.run_module(\\"agent.deployment_promotion\\", '
            'run_name=\\"__main__\\", alter_sys=True)'
        )
        == 1
    )
    assert '--repository \\"${{ github.repository }}\\"' in promotion
    assert "--environment-stdin" in promotion


def test_deployment_guide_documents_the_bounded_policy_and_dry_run() -> None:
    """Keep cache limits and pre-pull failure semantics operator-visible."""
    document = DEPLOYMENT_GUIDE_PATH.read_text(encoding="utf-8")
    normalized = " ".join(document.split())
    blocks = re.findall(r"(?ms)^```bash[ \t]*\n(.*?)^```[ \t]*$", document)
    dry_runs = [
        block for block in blocks if "-m agent.deployment_retention enforce" in block
    ]

    assert len(dry_runs) == 1
    assert "--apply" not in dry_runs[0]
    for fragment in (
        "hard maximum of eight managed digest references",
        "reduces the cache to seven",
        "reserves the eighth slot",
        "two newest distinct prior",
        "three-generation local rollback cache",
        "at most five exact",
        "incomplete nonzero result",
        "workflow stops before the target pull",
        "An unreachable plan fails without deleting anything",
        "exact digest with `--no-prune`",
        "retention failure does not roll back",
    ):
        assert fragment in normalized
