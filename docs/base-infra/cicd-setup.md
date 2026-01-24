# CI/CD Workflow Guide

This guide explains the GitHub Actions workflows that automate building and deploying the LLM agent to Cloud Run.

## Overview

The project uses **automated CI/CD** via GitHub Actions with four workflows:

1. **`ci-cd.yml`** - Orchestrator (meta → build → deploy)
2. **`metadata-extract.yml`** - Reusable metadata extraction
3. **`docker-build.yml`** - Reusable Docker image build
4. **`terraform-plan-apply.yml`** - Reusable Terraform deployment

**Key principle:** Zero manual intervention after initial setup. Merge to main = automatic deployment.

## Prerequisites

Complete one-time bootstrap setup before using CI/CD workflows:

1. ✅ Run the Terraform bootstrap module (see [Bootstrap Setup](./bootstrap-setup.md))
2. ✅ Verify GitHub Variables created: `gh variable list`

Bootstrap creates all CI/CD infrastructure: WIF, Artifact Registry, GitHub Variables, Terraform state bucket.

## GitHub Variables (Auto-Created)

The bootstrap module creates these Variables in your repository:

| Variable Name | Description | Created By |
|---------------|-------------|------------|
| `GCP_PROJECT_ID` | GCP project ID | Bootstrap |
| `GCP_LOCATION` | GCP region | Bootstrap |
| `IMAGE_NAME` | Docker image name (also used as agent_name) | Bootstrap |
| `GCP_WORKLOAD_IDENTITY_PROVIDER` | WIF provider resource name | Bootstrap |
| `ARTIFACT_REGISTRY_URI` | Registry URI | Bootstrap |
| `ARTIFACT_REGISTRY_LOCATION` | Registry location | Bootstrap |
| `TERRAFORM_STATE_BUCKET` | GCS bucket for main module state | Bootstrap |

**Note:** These are Variables (not Secrets) because they're resource identifiers, not credentials. Security comes from WIF IAM policies, not obscurity.

## Workflow Architecture

### Four-Workflow Pattern

```
ci-cd.yml (orchestrator)
├── meta (calls metadata-extract.yml) - Extract metadata and determine tags
├── build (calls docker-build.yml) - Build multi-platform image
└── deploy (calls terraform-plan-apply.yml) - Deploy to Cloud Run
```

**Benefits:**
- Clean separation of concerns
- Reusable workflows called by orchestrator (no manual triggers)
- Direct output passing (no tag reconstruction)
- Single orchestrator controls the full pipeline

### Workflow Flow

**On Pull Request:**
```
1. metadata-extract.yml extracts: pr-{number}-{sha}
2. docker-build.yml pushes: registry/image:pr-123-abc123
3. terraform-plan-apply.yml runs: plan (no apply)
4. Plan output posted as PR comment
```

**On Push to Main or Version Tag:**
```
1. metadata-extract.yml extracts: {sha}, latest, {version} (if tagged)
2. docker-build.yml pushes: registry/image:abc123 + latest (+ v1.0.0 if tagged)
3. terraform-plan-apply.yml runs: apply (auto-approved)
4. Cloud Run service updated with image digest
```

**On Manual Dispatch:**
```
1. metadata-extract.yml runs (uses current commit SHA)
2. docker-build.yml builds image
3. terraform-plan-apply.yml runs plan or apply (user choice)
4. Workspace selection: default/dev/stage/prod
```

## Image Tagging Strategy

### Pull Request Builds

**Format:** `pr-{number}-{sha}` (e.g., `pr-123-abc1234`)
- Unique per commit, isolated from main builds

### Main Branch Builds

**Tags:** `{sha}` (primary), `latest`, `{version}` (if tagged)

**Example:** Commit `abc1234` tagged `v0.9.0` produces:
```
registry/image:abc1234  registry/image:latest  registry/image:v0.9.0
```

**Deployment uses image digest** (not tags) to ensure every rebuild triggers a new Cloud Run revision.

### Version Tag Builds

**Trigger:** Push git tag matching `v*` (e.g., `git push origin v0.4.0`)

Automatically builds version-tagged image after release PR is merged. Safe because tags point to already-reviewed code and only authorized users can push tags.

## Workflow Behavior

**Concurrency:** PRs cancel in-progress builds; main runs sequentially. Per-workspace Terraform locking prevents state corruption.

**Path filtering:** Triggers on code/config changes, ignores docs. See `.github/workflows/ci-cd.yml` for complete path list. Tag triggers (`v*`) always run.

**Multi-platform builds:** Images built for multiple platforms (see `docker-build.yml`). Manifest auto-selects architecture.

**Build cache:** Registry cache with protected `buildcache` tag provides significant speedup on cache hits.

## Shell Configuration

**`bash {0}` pattern:** Terraform plan step uses `shell: bash {0}` to override default `bash -e` behavior. This allows capturing output on failure for job summaries and PR comments. Default behavior exits immediately on error, preventing output capture. See [GitHub Actions workflow syntax](https://docs.github.com/en/actions/writing-workflows/workflow-syntax-for-github-actions#exit-codes-and-error-action-preference).

## Pull Request Comments

terraform-plan-apply.yml posts formatted comments showing format/init/validation/plan results with collapsible sections for detailed output.

## Job Summaries

Both the metadata extraction and Terraform deployment jobs generate formatted job summaries visible in the GitHub Actions UI. These summaries provide quick insight into build and deployment status without requiring log analysis.

**Metadata Extraction Summary** includes:
- Build context (PR, main branch, tag push, manual)
- Branch/tag name and commit SHA
- All image tags (formatted as bulleted list)

**Terraform Deployment Summary** includes:
- Workspace and action (plan/apply)
- Docker image being deployed
- Step-by-step outcomes (format, init, validate, plan, apply)
- Collapsible plan output (first 50 lines)
- Deployed resources (Cloud Run URLs, service account, Agent Engine ID, artifact bucket) for successful applies

Job summaries appear at the top of each workflow run, providing a quick overview without navigating through individual job logs.

## Workload Identity Federation

Keyless authentication via WIF: GitHub Actions requests OIDC token, GCP validates against WIF provider, grants temporary credentials scoped to repository.

**IAM roles:** See `terraform/bootstrap/main.tf` for complete role list.

## Testing Workflows

Manual trigger via `ci-cd.yml` orchestrator (also available via GitHub UI: Actions > CI/CD Pipeline > Run workflow):

```bash
# Trigger full pipeline (build + deploy)
gh workflow run ci-cd.yml \
  -f workspace=default \
  -f terraform_action=plan

# Verify deployment
gh run list --workflow=ci-cd.yml --limit 5
gh run view RUN_ID
gcloud run services describe IMAGE_NAME --region=us-central1 --format="value(status.url)"
```

**Note:** `docker-build.yml` and `terraform-plan-apply.yml` are reusable workflows without manual triggers. Use `ci-cd.yml` to trigger the full pipeline.

## Troubleshooting

### Trace Deployed Image to Git Commit

```bash
# Get commit SHA from deployed image digest
gcloud artifacts docker images describe \
  "$(gcloud run services describe SERVICE_NAME --region REGION \
     --format='value(spec.template.spec.containers[0].image)')" \
  --format="value(tags)" | cut -d',' -f1
```

### Missing Variables

```bash
gh variable list  # Verify Variables exist
terraform -chdir=terraform/bootstrap apply  # Re-run if missing
```

### WIF Authentication Failed

```bash
# Check WIF provider
terraform -chdir=terraform/bootstrap output -raw workload_identity_provider

# Verify IAM bindings
gcloud projects get-iam-policy PROJECT_ID \
  --flatten="bindings[].members" \
  --filter="bindings.members:principalSet*"
```

### Image Push Denied

```bash
# Verify artifactregistry.writer role
gcloud projects get-iam-policy PROJECT_ID \
  --flatten="bindings[].members" \
  --filter="bindings.role:roles/artifactregistry.writer"
```

### PR Comment Not Posted

Ensure workflow has `pull-requests: write` permission.

### Build Cache Miss

Builds taking longer than expected? Verify `keep-buildcache` policy exists in bootstrap module.

### Terraform State Lock

```bash
gh run list --workflow=ci-cd.yml --limit 10  # Find stuck runs
gh run cancel RUN_ID  # Cancel if needed
terraform -chdir=terraform/main force-unlock LOCK_ID  # Last resort
```

## Workflow Timeouts

Workflows include timeouts to prevent runaway processes. See workflow files for specific values. If timeouts occur frequently, investigate network issues, large build context, or Terraform state locking.

## Security Best Practices

- **Keyless auth via WIF** - No service account keys, repository-scoped IAM bindings
- **Minimal permissions** - Only required IAM roles, workspace isolation for environments
- **Encrypted state** - Remote GCS state with versioning and locking
- **Immutable images** - SHA-tagged with cleanup policies, multi-platform manifests

## Next Steps

1. ✅ Complete bootstrap setup (see [Terraform Infrastructure Guide](./terraform-infrastructure.md))
2. ✅ Verify GitHub Variables: `gh variable list`
3. ✅ Test workflow: Create PR, verify plan posted as comment
4. ✅ Merge PR, verify automatic deployment to Cloud Run
5. 📖 Monitor deployments: `gh run list --workflow=ci-cd.yml`

## Related Documentation

- [Terraform Infrastructure Guide](./terraform-infrastructure.md) - Bootstrap and main module setup
- [Validating Multi-Platform Builds](./validating-multiplatform-builds.md) - Verify deployed image provenance
- [Development Guide](./development.md) - Local development workflow
