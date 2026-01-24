# Implementation Plan - Feature Parity Audit & Documentation Update

## Phase 1: Setup & Analysis
- [x] Task: Create a temporary directory and clone the upstream `doughayden/agent-foundation` repository. [checkpoint: setup]
    - [x] Create temp dir using `mktemp -d`. -> `/Users/lordpatil-air/.gemini/tmp/6363541dffbb68d50e3fe7de33019bec044de9b2731c3487e47864efc74f342d/tmp.YLHzwpeWge`
    - [x] Clone repo: `git clone git@github.com-personal:doughayden/agent-foundation.git <temp_dir>`.
- [ ] Task: Generate a file listing for both repositories to compare structures.
    - [ ] Run `tree` or `find` on `src/` for both repos.
    - [ ] Save output to `audit_file_structure.txt`.
- [ ] Task: Conductor - User Manual Verification 'Setup & Analysis' (Protocol in workflow.md)

## Phase 2: Feature Parity Audit
- [ ] Task: Compare `src/agent_foundation/agent.py` logic.
    - [ ] Diff the files.
    - [ ] Note any missing logic in `audit_report.md`.
- [ ] Task: Compare `src/agent_foundation/tools.py` available tools.
    - [ ] List tools in upstream.
    - [ ] Verify existence in local.
    - [ ] Note gaps in `audit_report.md`.
- [ ] Task: Compare `src/agent_foundation/server.py` and API endpoints.
    - [ ] Verify all routes exist (except GCP-specific health checks if irrelevant).
- [ ] Task: Compare `src/agent_foundation/utils` and configuration handling.
    - [ ] Ensure `config.py` handles env vars correctly for self-hosted (e.g., `DATABASE_URL` vs GCP Secret Manager).
- [ ] Task: Conductor - User Manual Verification 'Feature Parity Audit' (Protocol in workflow.md)

## Phase 3: Gap Resolution (Iterative)
- [ ] Task: Port any identified missing non-GCP features.
    - [ ] **Sub-task:** Write tests for missing feature.
    - [ ] **Sub-task:** Implement feature.
    - [ ] **Sub-task:** Verify parity.
- [ ] Task: Clean up any residual GCP-only code that causes errors in self-hosted mode.
    - [ ] **Sub-task:** Write test case reproducing error (if applicable).
    - [ ] **Sub-task:** Refactor to be platform-agnostic or remove.
- [ ] Task: Conductor - User Manual Verification 'Gap Resolution' (Protocol in workflow.md)

## Phase 4: Documentation Overhaul
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

## Phase 5: Final Cleanup
- [ ] Task: Remove temporary clone of upstream repo.
- [ ] Task: Final pass on `audit_report.md` to summarize findings.
- [ ] Task: Conductor - User Manual Verification 'Final Cleanup' (Protocol in workflow.md)
