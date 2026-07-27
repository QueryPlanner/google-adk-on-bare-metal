"""Third-party provenance contract tests."""

from hashlib import sha256
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
NOTICE_PATH = REPOSITORY_ROOT / "THIRD_PARTY_NOTICES.md"
UPSTREAM_LICENSE_PATH = REPOSITORY_ROOT / "licenses" / "agent-foundation-MIT.txt"
UPSTREAM_COMMIT = "c4f1db9218f265ff10c7212363b8d72e4b59e2d7"
UPSTREAM_LICENSE_SHA256 = (
    "747962afcb950b95594bca26cb893dd071f9ea3b7c72b1f8e1e8805091156573"
)


def test_agent_foundation_license_matches_pinned_upstream() -> None:
    license_digest = sha256(UPSTREAM_LICENSE_PATH.read_bytes()).hexdigest()

    assert license_digest == UPSTREAM_LICENSE_SHA256


def test_third_party_notice_records_source_and_license_scope() -> None:
    notice = NOTICE_PATH.read_text(encoding="utf-8")

    assert "doughayden/agent-foundation" in notice
    assert UPSTREAM_COMMIT in notice
    assert "licenses/agent-foundation-MIT.txt" in notice
    assert "does not select or grant a license" in notice


def test_readme_links_third_party_notice() -> None:
    readme = (REPOSITORY_ROOT / "README.md").read_text(encoding="utf-8")

    assert "[Third-Party Notices](THIRD_PARTY_NOTICES.md)" in readme
