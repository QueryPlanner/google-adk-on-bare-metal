# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.1] - 2026-02-20

### Fixed
- Prevent overlapping template replacements in `init_template.py`.
- Mandated explicit sequence for CI checks in `AGENTS.md`.

### Added
- Added `.gemini/skills/google-adk/SKILL.md`.

## [0.1.0] - 2026-02-08

### Added
- **Automated Server Setup:** Introduced `setup.sh` to automate system updates, Docker installation, firewall (UFW) configuration, and Fail2Ban setup.
- **Production Deployment Workflow:** Enhanced GitHub Actions (`docker-publish.yml`) to securely inject environment variables (DB credentials, API keys) into the production server.
- **Configurable Connection Pooling:** Exposed PostgreSQL connection pool settings (`DB_POOL_SIZE`, `DB_MAX_OVERFLOW`, etc.) via environment variables.
- **Observability Configuration:** Added support for configuring `ROOT_AGENT_MODEL` and Langfuse keys (`LANGFUSE_PUBLIC_KEY`, etc.) via deployment secrets.
- **Testing Standards:** Added `AGENTS.md` with strict guidelines for AI assistants, enforcing real-code testing over mocking internal logic.
- **Documentation:** Updated `README.md` and `docs/DEPLOYMENT.md` with comprehensive deployment guides.

### Changed
- Refactored `agent.py` to dynamically load `LiteLlm` for OpenRouter models.
- Standardized CI checks (`ruff`, `mypy`, `pytest`) to run before every build.

### Fixed
- Resolved `ValueError: Missing key inputs argument` by ensuring API keys are properly injected into the container environment.
- Addressed interactive prompt issues in `setup.sh` by setting `DEBIAN_FRONTEND=noninteractive`.
\n[Unreleased]: https://github.com/QueryPlanner/google-adk-wo-gcp/compare/v0.1.1...HEAD\n[0.1.1]: https://github.com/QueryPlanner/google-adk-wo-gcp/compare/v0.1.0...v0.1.1\n[0.1.0]: https://github.com/QueryPlanner/google-adk-wo-gcp/releases/tag/v0.1.0
