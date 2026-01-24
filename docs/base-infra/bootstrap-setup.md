# Bootstrap Setup

One-time infrastructure provisioning for CI/CD automation.

> [!NOTE]
> This is a one-time setup. After successful bootstrap, you won't need to run these commands again unless changing GCP projects or GitHub repositories.

## Prerequisites

- [Terraform](https://developer.hashicorp.com/terraform/downloads) >= 1.14.0
- [Google Cloud SDK](https://cloud.google.com/sdk/docs/install) (gcloud CLI)
- [GitHub CLI](https://cli.github.com/) (gh)

## What Bootstrap Creates

1. **Workload Identity Federation** - Keyless GitHub Actions authentication
2. **Artifact Registry** - Docker image storage with cleanup policies
3. **Terraform State Bucket** - Remote state for main module (GCS)
4. **GitHub Variables** - Auto-configured repository variables for CI/CD

See [Terraform Infrastructure Guide](./terraform-infrastructure.md) for detailed resource descriptions and IAM roles.

## Commands

**1. Authenticate:**
```bash
gcloud auth application-default login
gcloud config set project YOUR_PROJECT_ID
gh auth login
```

**2. Configure app runtime environment:**
```bash
cp .env.example .env
# Edit .env with your values (see inline comments)
```

Required variables in `.env`:
- `GOOGLE_CLOUD_PROJECT`
- `GOOGLE_CLOUD_LOCATION`
- `AGENT_NAME`
- `OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT`

**3. Configure bootstrap for GitHub:**
```bash
cp terraform/bootstrap/terraform.tfvars.example terraform/bootstrap/terraform.tfvars
# Edit terraform.tfvars with your GitHub repository info
```

Required variables in `terraform/bootstrap/terraform.tfvars`:
- `repository_owner` - GitHub username or organization
- `repository_name` - GitHub repository name

**4. Initialize Terraform:**
```bash
terraform -chdir=terraform/bootstrap init
```

**5. Preview changes:**
```bash
terraform -chdir=terraform/bootstrap plan
```

**6. Apply:**
```bash
terraform -chdir=terraform/bootstrap apply
```

Bootstrap typically completes in 2-3 minutes.

## Verification

**Check GitHub Variables:**
```bash
gh variable list
```

Expected output:
- `ARTIFACT_REGISTRY_LOCATION`
- `ARTIFACT_REGISTRY_URI`
- `GCP_LOCATION`
- `GCP_PROJECT_ID`
- `GCP_WORKLOAD_IDENTITY_PROVIDER`
- `IMAGE_NAME`
- `OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT`
- `TERRAFORM_STATE_BUCKET`

**View Terraform outputs:**
```bash
terraform -chdir=terraform/bootstrap output
```

## State Management

Bootstrap uses **local state** by default:
- State file: `terraform/bootstrap/terraform.tfstate` (gitignored)
- One-time operation, no need for remote state
- Team collaboration can optionally configure remote state

See [Terraform Infrastructure Guide](./terraform-infrastructure.md#terraform-infrastructure) for details.

## Next Steps

1. ✅ Verify GitHub Variables created
2. 📖 Continue to [Development Guide](./development.md) for feature branch workflow
3. 📖 See [CI/CD Workflow Guide](./cicd-setup.md) for deployment automation
