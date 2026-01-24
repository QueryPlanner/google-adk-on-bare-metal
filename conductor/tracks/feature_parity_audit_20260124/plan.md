# Implementation Plan - Feature Parity Audit & Documentation Update

## Phase 1: Setup & Analysis [checkpoint: eea6d4f]
- [x] Task: Create a temporary directory and clone the upstream `doughayden/agent-foundation` repository. [checkpoint: setup]
    - [x] Create temp dir using `mktemp -d`. -> `/Users/lordpatil-air/.gemini/tmp/6363541dffbb68d50e3fe7de33019bec044de9b2731c3487e47864efc74f342d/tmp.YLHzwpeWge`
    - [x] Clone repo: `git clone git@github.com-personal:doughayden/agent-foundation.git <temp_dir>`.
- [x] Task: Generate a file listing for both repositories to compare structures. [checkpoint: file-listing]
    - [x] Run `tree` or `find` on `src/` for both repos.
    - [x] Save output to `audit_file_structure.txt`.
- [x] Task: Conductor - User Manual Verification 'Setup & Analysis' (Protocol in workflow.md)

## Phase 2: Feature Parity Audit [checkpoint: c187651]
- [x] Task: Compare `src/agent_foundation/agent.py` logic. [checkpoint: agent-parity]
    - [x] Diff the files.
    - [x] Note any missing logic in `audit_report.md`.
- [x] Task: Compare `src/agent_foundation/tools.py` available tools. [checkpoint: tools-parity]
    - [x] List tools in upstream.
    - [x] Verify existence in local.
    - [x] Note gaps in `audit_report.md`.
- [x] Task: Compare `src/agent_foundation/server.py` and API endpoints. [checkpoint: server-parity]
    - [x] Verify all routes exist (except GCP-specific health checks if irrelevant).
- [x] Task: Compare `src/agent_foundation/utils` and configuration handling. [checkpoint: utils-parity]
    - [x] Ensure `config.py` handles env vars correctly for self-hosted (e.g., `DATABASE_URL` vs GCP Secret Manager).
- [x] Task: Conductor - User Manual Verification 'Feature Parity Audit' (Protocol in workflow.md)

## Phase 3: Gap Resolution (Iterative) [checkpoint: adc29aa]
- [x] Task: Port any identified missing non-GCP features. [checkpoint: no-missing-features]
    - [x] **Sub-task:** Write tests for missing feature.
    - [x] **Sub-task:** Implement feature.
    - [x] **Sub-task:** Verify parity.
- [x] Task: Implement Multi-LLM Configuration. [checkpoint: multillm-implemented]
    - [x] **Sub-task:** Analyze `.venv/lib/python3.13/site-packages/google/adk/models` to understand LiteLLM integration in `google-adk`. [checkpoint: adk-models-analysis]
    - [x] **Sub-task:** Add `OPENROUTER_API_KEY` to `ServerEnv` in `config.py`.
    - [x] **Sub-task:** Verify `litellm` dependency or `google-adk` support.
    - [x] **Sub-task:** Test with OpenRouter key (mocked).
- [x] Task: Clean up any residual GCP-only code that causes errors in self-hosted mode. [checkpoint: no-residual-gcp-errors]
- [x] Task: Conductor - User Manual Verification 'Gap Resolution' (Protocol in workflow.md)

## Phase 4: GCP Removal (Bare Metal Adaptation)
- [x] Task: Remove GCP-specific parameters from `ServerEnv` in `config.py`. [checkpoint: config-degcp]
    - [x] Remove `GOOGLE_CLOUD_PROJECT`, `GOOGLE_CLOUD_LOCATION`, `ARTIFACT_SERVICE_URI`, `OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT`.
- [x] Task: Update `server.py` to remove GCP-specific initialization logic. [checkpoint: server-degcp]
    - [x] Remove conditional OpenTelemetry setup based on GCP project.
- [x] Task: Verify removal does not break core functionality. [checkpoint: gcp-removal-verified]
    - [x] Run tests.
- [x] Task: Conductor - User Manual Verification 'GCP Removal' (Protocol in workflow.md)

## Phase 5: Documentation Overhaul
- [ ] Task: Audit `README.md`.
    - [ ] Remove GCP deployment badges/links.
    - [ ] Ensure "Quickstart" is purely local/docker.
- [ ] Task: Audit `docs/architecture.md`.
    - [ ] Update diagrams/text to reflect Docker/Postgres vs Cloud Run/Cloud SQL.
- [ ] Task: Audit `docs/development.md`.
    - [ ] Ensure dev workflow matches `uv` and local server.
- [ ] Task: Create/Update `DEPLOYMENT.md`.
    - [ ] Ensure it focuses on `docker compose` and `systemd`.
- [ ] Task: Conductor - User Manual Verification 'Documentation Overhaul' (Protocol in workflow.md)

## Phase 6: Final Cleanup
- [ ] Task: Remove temporary clone of upstream repo.
- [ ] Task: Final pass on `audit_report.md` to summarize findings.
- [ ] Task: Conductor - User Manual Verification 'Final Cleanup' (Protocol in workflow.md)
