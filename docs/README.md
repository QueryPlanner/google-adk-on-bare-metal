# Documentation

## Organization

This directory separates **base infrastructure documentation** from **your custom agent documentation**.

### `base-infra/` - Base Infrastructure

Base infrastructure documentation. Reference these for deployment, Docker workflows, and infrastructure patterns.

**Contents:**
- Docker and development environment
- Environment Variables
- Observability features

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
- [Development Guide](development.md) - Local workflow and code quality
- [Deployment Guide](DEPLOYMENT.md) - Bare metal and Docker deployment
- [Environment Variables](base-infra/environment-variables.md) - Complete configuration reference
- [Docker Compose Workflow](base-infra/docker-compose-workflow.md) - Local development

### Production
- [Observability](base-infra/observability.md) - Traces and logs
- [Dockerfile Strategy](base-infra/dockerfile-strategy.md) - Build optimization
