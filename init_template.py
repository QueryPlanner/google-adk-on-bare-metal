"""Initialize repository from template.

⚠️ ONE-TIME USE SCRIPT - Delete after running!

This script customizes your repository after creating from the template.
It auto-detects the new repository info from git and performs all necessary renames.

USAGE:
    # Preview changes (no modifications)
    uv run init_template.py --dry-run

    # Apply changes
    uv run init_template.py

REQUIREMENTS:
    - Git repository with configured remote (clone from GitHub, not local init)
    - Repository name must be kebab-case (lowercase, hyphens only)
    - Examples: my-agent, cool-app-v2, data-processor

WHAT IT DOES:
    1. Auto-detects YOUR new repository owner/name from git remote URL
    2. Validates repository name is kebab-case (enforces Python naming)
    3. Derives package name (my-agent → my_agent)
    4. Replaces template names with yours:
       - agent_foundation → your_package_name
       - agent-foundation → your-repo-name
       - template-owner/agent-foundation → your-owner/your-repo
    5. Renames src/agent_foundation/ → src/{package_name}/
    6. Updates imports, config, and docs in all files
    7. Updates GitHub Actions badge URLs
    8. Resets CODEOWNERS file (remove template owner)
    9. Resets version to 0.1.0 in pyproject.toml
    10. Resets CHANGELOG.md with clean template
    11. Regenerates UV lockfile

OUTPUT:
    Creates init_template_results.md (or init_template_dry_run.md) with
    detailed log of all changes. Review this file to verify changes.

AFTER RUNNING:
    1. Review: git status
    2. Delete this script: rm init_template.py
    3. Delete log file: rm init_template_results.md
    4. Update README.md and CLAUDE.md to remove template initialization section
    5. Configure .env: cp .env.example .env (add your GCP details)
    6. Commit: git add -A && git commit -m "chore: initialize from template"

REUSING IN OTHER TEMPLATES:
    To adapt this script for a different template repository, update:
    - ORIGINAL_PACKAGE_NAME: Original Python package (snake_case)
    - ORIGINAL_REPO_NAME: Original repository name (kebab-case)
    - ORIGINAL_GITHUB_OWNER: Original GitHub owner/organization

TECHNICAL NOTES:
    - Excluded from test coverage (not in src/, one-time use)
    - Excluded from ruff/mypy checks (see pyproject.toml)
    - No automated tests (manual validation via --dry-run)
"""

import re
import subprocess
import sys
from collections.abc import Generator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import NoReturn

from pydantic import BaseModel, Field, ValidationError, computed_field

# Original template names - update these when reusing in other template projects
ORIGINAL_PACKAGE_NAME = "agent_foundation"
ORIGINAL_REPO_NAME = "agent-foundation"
ORIGINAL_GITHUB_OWNER = "doughayden"

# Output file names for logging results
DRY_RUN_OUTPUT_FILE = "init_template_dry_run.md"
ACTUAL_RUN_OUTPUT_FILE = "init_template_results.md"


class TemplateConfig(BaseModel):
    """Configuration model for template initialization with validation.

    Used by init_template.py to validate repository names and derive package names.
    Enforces kebab-case repository naming for proper Python package compatibility.

    Attributes:
        repo_name: GitHub repository name in kebab-case format.
        github_owner: GitHub repository owner (username or organization).
        package_name: Python package name (computed from repo_name).
    """

    repo_name: str = Field(
        ...,
        pattern=r"^[a-z0-9]([a-z0-9-]*[a-z0-9])?$",
        description="GitHub repository name (kebab-case, e.g., 'my-agent')",
    )
    github_owner: str = Field(
        ...,
        description="GitHub repository owner (username or organization)",
    )

    @computed_field
    @property
    def package_name(self) -> str:
        """Python package name derived from repo_name (kebab-case → snake_case)."""
        return self.repo_name.replace("-", "_")


class DualOutput:
    """Write to both stdout and a file simultaneously.

    This class wraps stdout to capture all print statements and write them
    to both the terminal and a markdown file.
    """

    def __init__(self, file_path: Path) -> None:
        """Initialize dual output handler.

        Args:
            file_path: Path to markdown file for logging output.
        """
        self.terminal = sys.stdout
        self.log_file = file_path.open("w")
        self._write_header()

    def _write_header(self) -> None:
        """Write markdown header to log file."""
        timestamp = datetime.now(tz=UTC).strftime("%Y-%m-%d %H:%M:%S UTC")
        self.log_file.write("# Template Initialization Log\n\n")
        self.log_file.write(f"**Timestamp:** {timestamp}\n\n")
        self.log_file.write("---\n\n")
        self.log_file.flush()

    def write(self, message: str) -> None:
        """Write message to both terminal and file.

        Args:
            message: Text to write.
        """
        self.terminal.write(message)
        self.log_file.write(message)

    def flush(self) -> None:
        """Flush both output streams."""
        self.terminal.flush()
        self.log_file.flush()

    def close(self) -> None:
        """Close the log file."""
        self.log_file.close()


@contextmanager
def dual_output_context(dry_run: bool = False) -> Generator[None]:
    """Context manager for dual output (terminal + file).

    Args:
        dry_run: If True, use dry-run output file, otherwise use actual output file.

    Yields:
        None. Redirects sys.stdout to DualOutput during context.
    """
    output_file = DRY_RUN_OUTPUT_FILE if dry_run else ACTUAL_RUN_OUTPUT_FILE
    output_path = Path(output_file)

    dual_out = DualOutput(output_path)
    original_stdout = sys.stdout
    sys.stdout = dual_out

    try:
        yield
    finally:
        sys.stdout = original_stdout
        dual_out.close()
        print(f"\n📄 Output saved to: {output_file}")  # This prints to terminal only


def parse_github_remote_url(url: str) -> tuple[str, str] | None:
    """Parse GitHub owner and repo from remote URL.

    Supports both SSH and HTTPS formats:
    - SSH: git@github.com:owner/repo.git
    - HTTPS: https://github.com/owner/repo.git

    Args:
        url: Git remote URL to parse.

    Returns:
        Tuple of (owner, repo), or None if not a GitHub URL.
    """
    # SSH format: git@github.com:owner/repo.git
    ssh_match = re.match(r"^git@github\.com:([^/]+)/(.+?)(?:\.git)?$", url)
    if ssh_match:
        return (ssh_match.group(1), ssh_match.group(2))

    # HTTPS format: https://github.com/owner/repo.git
    https_match = re.match(r"^https://github\.com/([^/]+)/(.+?)(?:\.git)?$", url)
    if https_match:
        return (https_match.group(1), https_match.group(2))

    return None


def get_github_info_from_git() -> tuple[str, str] | None:
    """Get GitHub owner and repository name from git remote URL.

    Returns:
        Tuple of (owner, repo) from origin remote, or None if unavailable.
    """
    try:
        result = subprocess.run(
            ["git", "remote", "get-url", "origin"],  # noqa: S603, S607
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        url = result.stdout.strip()
        return parse_github_remote_url(url)
    except (
        subprocess.CalledProcessError,
        FileNotFoundError,
        subprocess.TimeoutExpired,
    ):
        return None


def get_validated_config(dry_run: bool = False) -> TemplateConfig:
    """Auto-detect and validate repository configuration.

    This function enforces Python package naming conventions by validating
    the repository name is kebab-case. If the repository name doesn't conform,
    the script fails with instructions to create a new repository with proper
    naming.

    Args:
        dry_run: If True, skip detection and use example values.

    Returns:
        Validated TemplateConfig instance with package_name derived from repo_name.

    Raises:
        SystemExit: If repository name is not detected or invalid.
    """
    if dry_run:
        print("🔍 DRY RUN MODE - Using example values\n")
        return TemplateConfig(repo_name="my-agent", github_owner="example-owner")

    print("🚀 Initializing repository from template\n")
    print("This script will:")
    print("  1. Validate repository name (must be kebab-case)")
    print(f"  2. Rename src/{ORIGINAL_PACKAGE_NAME}/ to your package name")
    print("  3. Update configuration files")
    print("  4. Update documentation")
    print("  5. Update GitHub Actions badge URLs")
    print("  6. Reset CODEOWNERS file")
    print("  7. Reset version to 0.1.0")
    print("  8. Reset CHANGELOG.md")
    print("  9. Regenerate UV lockfile\n")

    # Auto-detect repository name and owner from git
    github_info = get_github_info_from_git()

    if not github_info:
        print("❌ Failed to detect repository info from git remote.\n")
        print("This script requires a git repository with a configured remote.")
        print("\nPlease ensure:")
        print("  1. You created this repository from the template on GitHub")
        print("  2. You cloned it locally (git clone)")
        print("  3. The remote is configured (git remote -v)\n")
        sys.exit(1)

    detected_owner, detected_repo = github_info
    print(f"✨ Detected GitHub owner: {detected_owner}")
    print(f"✨ Detected repository name: {detected_repo}\n")

    # Validate repository name conforms to kebab-case
    try:
        config = TemplateConfig(repo_name=detected_repo, github_owner=detected_owner)
        print("✅ Repository name is valid kebab-case")
        print(f"✨ Package name (auto-derived): {config.package_name}\n")
        return config
    except ValidationError:
        print(f"❌ Invalid repository name: '{detected_repo}'\n")
        print("Repository names must follow kebab-case naming:")
        print("  • Use lowercase letters, numbers, and hyphens only")
        print("  • Cannot start or end with a hyphen")
        print("  • Examples: my-agent, agent-v2, cool-app\n")
        print("To fix this:")
        print("  1. Delete this repository on GitHub")
        print("  2. Create a new repository from the template with a kebab-case name")
        print("  3. Clone the new repository")
        print("  4. Run this init script again\n")
        sys.exit(1)


def replace_in_file(
    file_path: Path, replacements: dict[str, str], dry_run: bool = False
) -> None:
    """Perform text replacements in a file.

    Args:
        file_path: Path to file to modify.
        replacements: Dictionary mapping old strings to new strings.
        dry_run: If True, only print what would be changed.
    """
    if not file_path.exists():
        print(f"  ⚠️  Skipping {file_path} (not found)")
        return

    content = file_path.read_text()
    modified = content

    for old, new in replacements.items():
        modified = modified.replace(old, new)

    if content != modified:
        if dry_run:
            print(f"  📝 Would update {file_path}")
        else:
            file_path.write_text(modified)
            print(f"  ✅ Updated {file_path}")
    else:
        if dry_run:
            print(f"  ⏭️  Would skip {file_path} (no changes needed)")


def remove_authors_from_pyproject(dry_run: bool = False) -> None:
    """Remove authors field from pyproject.toml.

    Args:
        dry_run: If True, only print what would be changed.
    """
    pyproject_path = Path("pyproject.toml")

    if not pyproject_path.exists():
        print("  ⚠️  Skipping pyproject.toml (not found)")
        return

    content = pyproject_path.read_text()

    # Remove the authors array (supports single or multiple authors)
    # Matches: authors = [...] with any content between brackets
    modified = re.sub(
        r"authors\s*=\s*\[[^\]]*\]\s*\n?",
        "",
        content,
        flags=re.MULTILINE,
    )

    if content != modified:
        if dry_run:
            print("  📝 Would remove authors field from pyproject.toml")
        else:
            pyproject_path.write_text(modified)
            print("  ✅ Removed authors field from pyproject.toml")
    else:
        if dry_run:
            print("  ⏭️  Would skip pyproject.toml authors removal (not found)")


def reset_version_in_pyproject(dry_run: bool = False) -> None:
    """Reset version to 0.1.0 in pyproject.toml.

    Args:
        dry_run: If True, only print what would be changed.
    """
    pyproject_path = Path("pyproject.toml")

    if not pyproject_path.exists():
        print("  ⚠️  Skipping pyproject.toml (not found)")
        return

    content = pyproject_path.read_text()

    # Reset version to 0.1.0
    # Matches: version = "X.Y.Z" at start of line (after optional whitespace)
    # This avoids matching "version" in the middle of strings elsewhere
    modified = re.sub(
        r'^(\s*)version\s*=\s*"[^"]*"',
        r'\1version = "0.1.0"',
        content,
        flags=re.MULTILINE,
    )

    if dry_run:
        print("  📝 Would reset version to 0.1.0 in pyproject.toml")
    else:
        pyproject_path.write_text(modified)
        print("  ✅ Reset version to 0.1.0 in pyproject.toml")


def replace_changelog(dry_run: bool = False) -> None:
    """Replace CHANGELOG.md with fresh template.

    Args:
        dry_run: If True, only print what would be changed.
    """
    changelog_path = Path("CHANGELOG.md")

    fresh_changelog = """# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial project setup from template
"""

    if dry_run:
        print("  📝 Would replace CHANGELOG.md with fresh template")
    else:
        changelog_path.write_text(fresh_changelog)
        print("  ✅ Replaced CHANGELOG.md")


def replace_codeowners(dry_run: bool = False) -> None:
    """Replace CODEOWNERS with fresh template.

    Args:
        dry_run: If True, only print what would be changed.
    """
    codeowners_path = Path(".github/CODEOWNERS")

    fresh_codeowners = """# CODEOWNERS - Automatically request reviews from code owners
# See: https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/about-code-owners

# Default owner for everything in the repo
# * @your-github-username
"""

    if dry_run:
        print("  📝 Would replace .github/CODEOWNERS with fresh template")
    else:
        codeowners_path.write_text(fresh_codeowners)
        print("  ✅ Replaced .github/CODEOWNERS")


def run_uv_sync(dry_run: bool = False) -> None:
    """Regenerate UV lockfile.

    Args:
        dry_run: If True, only print what would be done.
    """
    if dry_run:
        print("  📝 Would run: uv sync")
        return

    print("  🔄 Running uv sync...")
    try:
        subprocess.run(
            ["uv", "sync"],  # noqa: S603, S607
            check=True,
            capture_output=True,
            timeout=60,
        )
        print("  ✅ UV lockfile regenerated")
    except subprocess.CalledProcessError as e:
        print(f"  ❌ Failed to run uv sync: {e}")
        print(f"     stderr: {e.stderr.decode()}")
        sys.exit(1)
    except subprocess.TimeoutExpired:
        print("  ❌ UV sync timed out after 60 seconds")
        sys.exit(1)


def print_summary(config: TemplateConfig, dry_run: bool = False) -> None:
    """Print summary of changes.

    Args:
        config: Validated template configuration.
        dry_run: If True, prefix all messages with "Would".
    """
    verb = "Would make" if dry_run else "Made"
    print(f"\n✅ {verb} the following changes:")
    print(f"  • Package name: {ORIGINAL_PACKAGE_NAME} → {config.package_name}")
    print(f"  • Repo name: {ORIGINAL_REPO_NAME} → {config.repo_name}")
    print(f"  • GitHub owner: {ORIGINAL_GITHUB_OWNER} → {config.github_owner}")
    print(f"  • Directory: src/{ORIGINAL_PACKAGE_NAME}/ → src/{config.package_name}/")
    print("  • Updated configuration and test files")
    print("  • Updated GitHub Actions badge URLs")
    print("  • Removed template author from pyproject.toml")
    print("  • Reset version to 0.1.0 in pyproject.toml")
    print("  • Replaced CHANGELOG.md with fresh template")
    print("  • Replaced CODEOWNERS with fresh template")
    print("  • Regenerated UV lockfile")

    if not dry_run:
        print("\n🎉 Template initialization complete!")
        print("\nNext steps:")
        print("  1. Review changes: git status")
        print("  2. Create .env file: cp .env.example .env")
        print("  3. Configure .env with your GCP project details")
        print("  4. Test locally: uv run local-agent")
        print(
            "  5. Commit: git add -A && git commit -m 'chore: initialize from template'"
        )
    else:
        print("\n💡 Run without --dry-run to apply these changes")


def main() -> NoReturn:
    """Main initialization function."""
    # Check for dry-run flag
    dry_run = "--dry-run" in sys.argv

    # Run with dual output (terminal + markdown file)
    with dual_output_context(dry_run):
        # Get and validate configuration
        config = get_validated_config(dry_run)

        # Define replacements
        replacements = {
            f"https://github.com/{ORIGINAL_GITHUB_OWNER}/{ORIGINAL_REPO_NAME}/": f"https://github.com/{config.github_owner}/{config.repo_name}/",
            ORIGINAL_PACKAGE_NAME: config.package_name,
            ORIGINAL_REPO_NAME: config.repo_name,
        }

        # Files to update (paths relative to repo root)
        files_to_update = [
            "CLAUDE.md",
            "Dockerfile",
            "pyproject.toml",
            "README.md",
            "tests/conftest.py",
        ]

        # Glob docker-compose files
        compose_files = Path().glob("docker-compose*.yml")
        files_to_update.extend(str(path) for path in compose_files)

        # Glob the docs directory
        doc_files = Path("docs").rglob("*.md")
        files_to_update.extend(str(path) for path in doc_files)

        # Glob the test suite
        test_files = Path("tests").rglob("test_*.py")
        files_to_update.extend(str(path) for path in test_files)

        # Rename directory
        old_dir = Path(f"src/{ORIGINAL_PACKAGE_NAME}")
        new_dir = Path(f"src/{config.package_name}")

        if old_dir.exists():
            print("\n📁 Renaming directory:")
            if dry_run:
                print(f"  📝 Would rename {old_dir} → {new_dir}")
            else:
                old_dir.rename(new_dir)
                print(f"  ✅ Renamed {old_dir} → {new_dir}")
        else:
            print(f"\n⚠️  Directory {old_dir} not found - already renamed?")

        # Update files
        print("\n📝 Updating files:")
        for file_path_str in files_to_update:
            file_path = Path(file_path_str)
            replace_in_file(file_path, replacements, dry_run)

        # Remove authors from pyproject.toml
        print("\n👤 Removing template author:")
        remove_authors_from_pyproject(dry_run)

        # Reset version in pyproject.toml
        print("\n🔢 Resetting version:")
        reset_version_in_pyproject(dry_run)

        # Replace CHANGELOG
        print("\n📄 Replacing CHANGELOG:")
        replace_changelog(dry_run)

        # Replace CODEOWNERS
        print("\n👥 Replacing CODEOWNERS:")
        replace_codeowners(dry_run)

        # Regenerate lockfile
        print("\n🔒 Regenerating lockfile:")
        run_uv_sync(dry_run)

        # Print summary
        print_summary(config, dry_run)

    sys.exit(0)


if __name__ == "__main__":
    main()
