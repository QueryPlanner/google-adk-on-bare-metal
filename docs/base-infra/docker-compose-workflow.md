# Docker Compose Local Development Workflow

This guide covers the recommended workflow for local development using Docker Compose.

## Quick Start

### Daily Development (Recommended)

```bash
docker compose up --build --watch
```

**Why both flags?**
- `--build`: Ensures you have the latest code and dependencies
- `--watch`: Enables hot reloading for instant feedback

**What happens:**
- Container starts with your latest code
- Watch mode monitors your files for changes
- Edits to `src/` files are **synced instantly** (no rebuild needed)
- Changes to `pyproject.toml` or `uv.lock` **trigger automatic rebuild**

**Leave it running** while you develop - changes are applied automatically!

---

## Common Commands

### Start with hot reloading (default workflow)
```bash
docker compose up --build --watch
```

### Stop the service
```bash
# Press Ctrl+C to gracefully stop
# Or in another terminal:
docker compose down
```

### View logs
```bash
# If running in detached mode
docker compose logs -f

# View just the app logs
docker compose logs -f app
```

### Rebuild without starting
```bash
docker compose build
```

### Run without watch mode
```bash
docker compose up --build
```

---

## How Watch Mode Works

Watch mode uses the configuration in `docker-compose.yml`:

```yaml
develop:
  watch:
    # Sync: Instant file copy, no rebuild
    - action: sync
      path: ./src
      target: /app/src

    # Rebuild: Triggers full image rebuild
    - action: rebuild
      path: ./pyproject.toml

    - action: rebuild
      path: ./uv.lock
```

### Sync Action
- **Triggers when:** You edit files in `src/`
- **What happens:** Files are copied into running container instantly
- **Speed:** Immediate (no rebuild)
- **Use case:** Code changes during development

### Rebuild Action
- **Triggers when:** You edit `pyproject.toml` or `uv.lock`
- **What happens:** Full image rebuild, container recreated
- **Speed:** ~5-10 seconds (with cache)
- **Use case:** Dependency changes

---

## File Locations

### Source Code
- **Host:** `./src/`
- **Container:** `/app/src`
- **Sync:** Automatic via watch mode

### Data Directory
- **Host:** `./data/`
- **Container:** `/app/data` (read-only)
- **Purpose:** Optional data files for agent

---

## Environment Variables

Docker Compose loads `.env` automatically. Key variables:

```bash
# Google Cloud Vertex AI model authentication
GOOGLE_GENAI_USE_VERTEXAI=TRUE
GOOGLE_CLOUD_PROJECT=your-project-id
GOOGLE_CLOUD_LOCATION=us-central1

# Logging verbosity
LOG_LEVEL=DEBUG

# Server configuration
# **RECOMMENDED**: DO NOT OVERRIDE PORT here when running with docker-compose
# Dockerfile overrides HOST=0.0.0.0 in container
# docker-compose maps container port 8000 to 8000 on the host machine

# Enable web UI
SERVE_WEB_INTERFACE=true

# Additional variables as needed...
```

**Note:** The container uses `HOST=0.0.0.0` to allow connections from the host machine.

---

## Troubleshooting

### Container keeps restarting
- Check logs: `docker compose logs -f`
- Verify `.env` file exists and has required variables
- Ensure Application Default Credentials are configured: `gcloud auth application-default login`

### Changes not appearing
- **For code changes:** Should sync instantly via watch mode
- **For dependency changes:** Watch should auto-rebuild
- **If stuck:** Stop and restart with `docker compose up --build --watch`

### Permission errors
- Data directory: Mounted read-only, should not need write access
- Credentials: Ensure `~/.config/gcloud/application_default_credentials.json` exists and is readable

### Port already in use
```bash
# Check what's using port 8000
lsof -i :8000

# Stop the conflicting process or change PORT in .env
PORT=8001
```

### Windows path compatibility
- The `docker-compose.yml` uses `${HOME}` which is Unix/Mac specific
- Windows users need to update the volume path in `docker-compose.yml`:
  - Replace `${HOME}/.config/gcloud/application_default_credentials.json`
  - With your Windows path: `C:\Users\YourUsername\AppData\Roaming\gcloud\application_default_credentials.json`
- Alternative: Use `%USERPROFILE%` environment variable in PowerShell
- See the comment in `docker-compose.yml` for the exact syntax

---

## Testing Registry Images Locally

Test the exact container image built by CI/CD without rebuilding locally.

**One-time setup:**
```bash
# Authenticate to your container registry
# For GCP Artifact Registry:
gcloud auth configure-docker <registry-location>-docker.pkg.dev

# For other registries:
# AWS ECR: aws ecr get-login-password --region <region> | docker login --username AWS --password-stdin <account-id>.dkr.ecr.<region>.amazonaws.com
# Docker Hub: docker login
```

**Pull and run:**
```bash
# Set your full image name
export REGISTRY_IMAGE="<location>-docker.pkg.dev/<project-id>/<repository>/<image-name>:latest"

# Pull the image (explicit pull required for multi-platform image behavior with docker-compose)
docker pull $REGISTRY_IMAGE

# Run the image using docker-compose with registry override
docker compose -f docker-compose.yml -f docker-compose.registry.yml up
```

This tests the exact artifact from your CI/CD pipeline. The container runs with a `-registry` suffix (e.g., `app-registry`) to make it easy to distinguish registry-pulled images from those locally-built.

**Note on multi-platform images:** The `docker images` command displays [manifest list](https://docs.docker.com/reference/cli/docker/manifest/) metadata rather than platform-specific image metadata for multi-platform images pulled from registries. You may see an epoch timestamp (1970-01-01) and manifest size (~43MB) instead of the actual image details. This is expected Docker behavior - the real platform-specific image (~171MB) is fully pulled and functional. Use `docker inspect <image>` to view actual image metadata, or simply run the container to verify it works correctly.

---

## Direct Docker Commands (Without Compose)

If you need to build and run without docker-compose:

```bash
# Build the image with BuildKit
DOCKER_BUILDKIT=1 docker build -t your-agent-name:latest .

# Run directly
docker run \
  -v ./data:/app/data:ro \
  -p 127.0.0.1:8000:8000 \
  --env-file .env \
  your-agent-name:latest
```

**Note:** Docker Compose is recommended - it handles volumes, environment, and networking automatically.

---

## References

- [Docker Compose Documentation](https://docs.docker.com/compose/)
- [Docker Compose Watch Mode](https://docs.docker.com/compose/file-watch/)
- [Dockerfile Strategy Guide](./dockerfile-strategy.md) - Architecture decisions and design rationale
