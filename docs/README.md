# Documentation

## Organization

This directory separates **base infrastructure documentation** from **your custom agent documentation**.

### `base-infra/` - Base Infrastructure

Base infrastructure documentation. Reference these for deployment, CI/CD, and infrastructure patterns. Do not modify unless contributing back to the template.

**Contents:**
- Bootstrap and deployment setup
- CI/CD workflows and automation
- Docker and development environment
- Terraform infrastructure patterns
- Observability and production features

See [base-infra/](./base-infra/) for complete list.

### Root - Your Custom Documentation

Add your agent-specific documentation here:
- Custom tools and capabilities
- Domain-specific logic and patterns
- Agent instructions and prompts
- Integration guides
- API documentation

**Examples:**
```
docs/
├── base-infra/              # Base infrastructure (don't modify)
├── custom-tools.md          # Your custom tool documentation
├── domain-guide.md          # Your domain-specific patterns
├── api-integration.md       # Your API integrations
└── agent-instructions.md    # Your agent instruction docs
```

## Quick Links

### Getting Started
- [Bootstrap Setup](base-infra/bootstrap-setup.md) - One-time CI/CD provisioning
- [Development Guide](base-infra/development.md) - Local workflow and code quality
- [Environment Variables](base-infra/environment-variables.md) - Complete configuration reference

### Infrastructure
- [CI/CD Setup](base-infra/cicd-setup.md) - GitHub Actions automation
- [Terraform Infrastructure](base-infra/terraform-infrastructure.md) - IaC setup and patterns
- [Docker Compose Workflow](base-infra/docker-compose-workflow.md) - Local development

### Production
- [Observability](base-infra/observability.md) - Traces and logs
- [Dockerfile Strategy](base-infra/dockerfile-strategy.md) - Build optimization
- [Validating Multi-Platform Builds](base-infra/validating-multiplatform-builds.md) - Image verification
