# Dockerfile Strategy Explained

This document explains our multi-stage Docker build strategy and why we chose this approach for containerizing the agent.

## Table of Contents

- [Overview](#overview)
- [Dockerfile Breakdown](#dockerfile-breakdown)
- [Why Not Use UV Image Directly](#why-not-use-uv-image-directly)
- [Performance Comparison](#performance-comparison)
- [Summary](#summary)

## Overview

Our Dockerfile uses a **multi-stage build** with the following architecture:

1. **Builder Stage**: `python:3.13-slim` + uv binary (copied from Astral's distroless image)
2. **Runtime Stage**: Clean `python:3.13-slim` + only the virtual environment

This approach gives us:
- ✅ Official uv binary (always latest)
- ✅ Full build capabilities (shell, Python, package manager)
- ✅ Minimal runtime image (~200MB vs ~500MB)
- ✅ Fast rebuilds (5-10s for code changes)
- ✅ Maximum security (non-root, minimal attack surface)

---

## Dockerfile Breakdown

### BuildKit Directive
```dockerfile
# syntax=docker/dockerfile:1
```
**What:** Tells Docker to use BuildKit parser (modern Docker build engine)
**Why:** Enables advanced features like `--mount=type=cache` and parallel builds

---

### Builder Stage Base Image
```dockerfile
FROM python:3.13-slim AS builder
```
**What:** Start builder stage with official Python 3.13 slim image (Debian-based)
**Why:** We need:
- Python runtime for `uv sync` to work
- Shell and basic utilities (cp, mkdir, etc.) for build commands

**Why not `ghcr.io/astral-sh/uv:latest` as base?**
- Astral's uv image is **distroless** (no shell, no package manager)
- You can't run `RUN` commands in distroless images
- It's designed to copy the binary FROM, not build FROM

---

### Copy UV Binary
```dockerfile
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
```
**What:** Extract just the `uv` and `uvx` binaries from Astral's image
**Why:**
- Gets latest uv version without manually tracking releases
- Copies only ~10MB of binaries (not a whole base image)
- Puts them in `/bin/` so they're in PATH

**Key insight:** We get official uv without the distroless constraints.

---

### Working Directory (Builder Stage)
```dockerfile
WORKDIR /app
```
**What:** Set working directory to `/app`
**Why:**
- All subsequent commands run from this directory
- Creates the directory if it doesn't exist
- Standard convention for application code

---

### UV Environment Variables
```dockerfile
ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PYTHON_DOWNLOADS=never
```
**What:** Configure uv behavior
**Why:**

| Variable | Value | Reason |
|----------|-------|--------|
| `UV_LINK_MODE=copy` | Copy files instead of hardlinking | Safer for Docker layers, works across filesystems |
| `UV_COMPILE_BYTECODE=1` | Pre-compile .py → .pyc files | Faster startup (no compilation at runtime) |
| `UV_PYTHON_DOWNLOADS=never` | Don't download Python | Use system Python from base image |

---

### Install Dependencies
```dockerfile
# Copy dependency files - explicit cache invalidation when either file changes
COPY pyproject.toml uv.lock ./

# Install dependencies (cache mount provides the performance optimization)
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-install-project --no-dev
```
**What:** Install dependencies WITHOUT installing the project itself
**Why:** Key optimization strategy

| Flag | Meaning | Benefit |
|------|---------|---------|
| `--mount=type=cache` | Persist `/root/.cache/uv` across builds | Don't re-download packages if already cached (~80% speedup) |
| `--locked` | Validate lockfile matches pyproject.toml | Catches mistakes, prevents silent failures (our standard) |
| `--no-install-project` | Skip installing `src/may_package` | Separates dependencies (slow) from code (fast) |
| `--no-dev` | Skip dev dependencies | Smaller image |

**Why COPY both files?**
- ✅ **Explicit cache invalidation** - When dependencies change, this layer rebuilds (as it should)
- ✅ **Predictable** - Standard Docker caching behavior, no surprises
- ✅ **Simple** - Easy to understand and maintain
- ✅ **Fast anyway** - Cache mount makes rebuilds quick (~2-5s even on version bumps)

**Why `--locked` (not `--frozen`)?**

We **standardize on `--locked`** for all builds. Here's why:

| Flag | Behavior | When to Use |
|------|----------|-------------|
| `--locked` | **Validates lockfile matches pyproject.toml** | ✅ **Always** (our standard) |
| `--frozen` | Skips validation, silently uses stale lockfile | ❌ **Avoid** |

**Why avoid `--frozen`:**
- ❌ Silently installs wrong dependencies if lockfile is stale
- ❌ Even UV's official CI/CD examples use `--locked`
- ❌ Hides developer mistakes instead of catching them

**Why use `--locked` everywhere:**
- ✅ Catches developer mistakes (forgot to run `uv lock`)
- ✅ Ensures lockfile and pyproject.toml stay synchronized
- ✅ Fails fast with clear error message
- ✅ Validation is negligible cost (~milliseconds)
- ✅ Enforces correct workflow: change deps → `uv lock` → commit both files

**Example of `--locked` preventing bugs:**
```bash
# Developer adds pandas to pyproject.toml but forgets to run uv lock
$ docker build .
ERROR: The lockfile is out of sync with pyproject.toml
# Developer: "Oh right, I need to run uv lock first!"
$ uv lock
$ docker build .  # Now succeeds with correct dependencies
```

**Bottom line:** We use `--locked` in development, CI/CD, and production. We'll only consider `--frozen` if we encounter a very specific use case that requires skipping validation (none identified yet).

**Cache mount is the real optimization** - It persists across builds, so even "unnecessary" rebuilds are fast.

---

### Copy Source Code
```dockerfile
# Copy only source code (documentation changes won't invalidate this layer)
COPY src ./src
```
**What:** Copy only the application source code
**Why:**
- Done AFTER dependencies to maximize cache hits
- **Optimization**: Only copies source code, not documentation (README.md)
- Documentation updates won't invalidate this layer or trigger project reinstall
- Code changes trigger rebuild as expected
- This layer rebuilds only on source code changes (~5-10s)
- More targeted than `COPY . /app` - avoids redundant pyproject.toml/uv.lock copy

---

### Install Project
```dockerfile
# Install project itself (create empty README to satisfy package metadata requirements)
RUN --mount=type=cache,target=/root/.cache/uv \
    touch README.md && \
    uv sync --locked --no-editable --no-dev
```
**What:** Install the project itself (now that source code exists)
**Why:**
- `touch README.md`: Creates empty README to satisfy package metadata requirements
  - **Performance optimization**: README changes won't trigger this layer rebuild
  - Only source code changes (`src/`) invalidate this layer
  - Intentional trade-off: runtime container won't have real README (not needed for execution)
- `--locked`: Validates lockfile matches pyproject.toml (catches mistakes)
- `--no-editable`: Install as a regular package (not editable/develop mode)
- Reuses dependencies from previous step (already in .venv)
- Fast because dependencies already installed (~5-10s)

---

### Runtime Stage Base
```dockerfile
FROM python:3.13-slim AS runtime
```
**What:** Start fresh with clean Python image for runtime
**Why:** Multi-stage build benefits:
- Builder stage has uv, build tools, cache → ~500MB
- Runtime stage only has Python and your .venv → ~200MB
- 50%+ size reduction by discarding build tools

---

### Non-Root User
```dockerfile
RUN groupadd -r app && useradd -r -g app app
```
**What:** Create a non-root system user named `app`
**Why:** Security best practice
- Containers shouldn't run as root
- Limits damage if container is compromised
- Standard container orchestration pattern

**Command breakdown:**
- `groupadd -r app`: Create system group (`-r` assigns GID < 1000 automatically)
- `useradd -r -g app app`: Create system user (`-r` assigns UID < 1000, `-g app` assigns to group)
- No home directory created, shell defaults to `/usr/sbin/nologin` (prevents login)

**Alternative approach:** Some Dockerfiles use explicit UIDs (e.g., `useradd -u 1001 -g appgroup -m -d /app`) for traditional volume sharing (NFS, Docker volumes). We use system users (`-r`) because:
- **GCS fuse authentication** uses service account IAM, not file UIDs
- **Simpler** - fewer flags, system assigns non-conflicting IDs
- **No home directory needed** - app doesn't write to `~/`

For traditional shared volumes with POSIX permissions, explicit UIDs are needed. For object storage (GCS), system users are sufficient.

**Reference:** [Depot.dev Python UV Dockerfile](https://depot.dev/docs/container-builds/how-to-guides/optimal-dockerfiles/python-uv-dockerfile) shows explicit UID approach

---

### Working Directory (Runtime Stage)
```dockerfile
WORKDIR /app
```
**What:** Set working directory again (new stage = new filesystem)
**Why:** Container starts in `/app` when it runs

---

### Copy Application from Builder
```dockerfile
COPY --from=builder --chown=app:app /app .
```
**What:** Copy entire application directory from builder stage
**Why:**
- `--from=builder`: Get everything from builder's `/app`
- `--chown=app:app`: Make the `app` user own it
- Destination `.` uses WORKDIR context (copies to `/app`)
- Includes `.venv/` (all dependencies + installed package)
- Includes `src/` (application code)
- Includes metadata (pyproject.toml, uv.lock, README.md)
- **Simple, conventional pattern** - one COPY instead of multiple
- Metadata files are tiny (~30 KB) and harmless
- **uv is NOT copied** (only in builder stage, not needed at runtime)

---

### Runtime Environment
```dockerfile
ENV VIRTUAL_ENV=/app/.venv \
    PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    AGENT_DIR=/app/src \
    HOST=0.0.0.0 \
    PORT=8000
```
**What:** Configure runtime environment
**Why:**

| Variable | Purpose |
|----------|---------|
| `VIRTUAL_ENV=/app/.venv` | Tell Python which venv to use |
| `PATH="/app/.venv/bin:$PATH"` | Make venv binaries available (python, uvicorn) |
| `PYTHONUNBUFFERED=1` | Don't buffer stdout/stderr (better logs in Docker) |
| `HOST=0.0.0.0` | Explicitly bind all interfaces for containers (server.py defaults to 127.0.0.1) |
| `PORT=8000` | Explicitly set default port (matches EXPOSE and server.py default) |
| `AGENT_DIR=/app/src` | Override agent directory path (see AGENT_DIR section below) |

---

### AGENT_DIR Configuration

**The Problem:**

When the package is installed in **non-editable mode** (Docker), the source code is copied to the virtual environment's site-packages:
- Local (editable): `Path(__file__)` → `/path/to/project/src/your_agent_name/server.py`
- Docker (non-editable): `Path(__file__)` → `/app/.venv/lib/python3.13/site-packages/your_agent_name/server.py`

Using `Path(__file__).parent.parent` for `AGENT_DIR`:
- Local: Resolves to `/path/to/project/src/` ✅ Correct (contains only your_agent_name/)
- Docker: Resolves to `/app/.venv/lib/python3.13/site-packages/` ❌ Wrong (contains all packages)

This causes the ADK web UI to show all installed packages (.dist-info directories) instead of just our agent.

**The Solution:**

```python
# In server.py - configurable with smart default
AGENT_DIR = os.getenv("AGENT_DIR", str(Path(__file__).parent.parent))
```

```dockerfile
# In Dockerfile - override for Docker environment
ENV AGENT_DIR=/app/src
```

**Why this works:**
- Local dev: No `AGENT_DIR` env var → uses `Path(__file__).parent.parent` → `/path/to/project/src/` ✅
- Docker: `AGENT_DIR=/app/src` env var set → overrides default → `/app/src/` ✅
- Both point to directory containing only the agent source code
- Configurable via environment variable for other deployment scenarios

---

### Switch to Non-Root
```dockerfile
USER app
```
**What:** All subsequent commands run as `app` user
**Why:**
- Container starts as `app` user at runtime
- Can't escalate to root
- Security best practice

---

### Expose Port
```dockerfile
EXPOSE 8000
```
**What:** Document that container listens on port 8000
**Why:**
- Documentation only (doesn't actually publish port)
- Tools like docker-compose read this for defaults
- Good practice for clarity

---

### Startup Command
```dockerfile
# Run the FastAPI server via main() for unified startup logic (logging, etc.)
CMD ["python", "-m", "your_agent_name.server"]
```
**What:** Default command when container starts
**Why:**
- Calls `server.main()` for unified startup logic
  - Sets up OpenTelemetry observability (traces and logs to Google Cloud)
  - Consistent entry point for both local dev (`uv run server`) and Docker
- `main()` calls `uvicorn.run(app, host=os.getenv("HOST", "127.0.0.1"), port=...)`
  - Secure default: 127.0.0.1 (only local connections)
  - Dockerfile sets `HOST=0.0.0.0` to explicitly bind all interfaces for containers
  - Respects HOST and PORT environment variables for flexibility
- JSON array format (exec form, not shell form) → more efficient
- Can be overridden at runtime

---

## Why Not Use UV Image Directly?

Let's compare the approaches:

### ❌ Using UV Image Directly (Won't Work)
```dockerfile
FROM ghcr.io/astral-sh/uv:latest

# ERROR: No shell to run these commands!
RUN uv sync  # FAILS - distroless has no /bin/sh
COPY src ./src  # Works, but then what?
```

**Problems:**
- Distroless = no shell → can't run `RUN` commands
- No package manager → can't install system dependencies if needed
- Image is ~100MB but you can't build anything with it
- Designed for copying FROM, not building FROM

### ✅ Our Approach (Multi-Stage)
```dockerfile
# Builder: Use python:3.13-slim + uv binary
FROM python:3.13-slim AS builder
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/
# ... build with full shell/utilities ...

# Runtime: Clean python:3.13-slim + only .venv
FROM python:3.13-slim AS runtime
COPY --from=builder /app/.venv /app/.venv
```

**Benefits:**
- Builder has shell + Python + uv → can build anything
- Runtime is minimal → small image size
- Gets official uv binary → always latest
- Best of both worlds

---

## Why We COPY Instead of Using Bind Mounts

### The Question: Should We Optimize Further?

You might wonder: "Why not use bind mounts to avoid copying dependency files into layers?"

UV's documentation shows bind mount examples, but we deliberately chose the simpler COPY approach. Here's why:

### COPY Approach (What We Use)
```dockerfile
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-install-project
```

**What happens:**
- Both files copied into builder layer
- Docker tracks both files for cache invalidation
- When either file changes, layer rebuilds
- Cache mount makes rebuilds fast (~2-5s)

### Bind Mount Alternative (What We Don't Use)
```dockerfile
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    uv sync --locked --no-install-project
```

**Potential issues:**
- Docker doesn't track bind-mounted files
- Layer might stay cached even when dependencies change
- Requires careful hybrid strategies (COPY some, bind others)
- More complex to understand and debug

### Our Philosophy: Explicit Over Implicit

**Dependency changes SHOULD trigger rebuilds:**
- ✅ Clear signal that something material changed
- ✅ Validates that dependencies actually update
- ✅ Predictable Docker caching behavior
- ✅ Easy to understand and debug

**The "penalty" for this explicitness is negligible:**
- Version bump in pyproject.toml → rebuilds layer → cache mount makes it ~2-5s
- The cache mount provides 90% of the performance benefit
- Simplicity is worth more than saving 3 seconds

### Benefits of COPY Approach
- ✅ **Simple** - Standard Docker pattern, easy to understand
- ✅ **Explicit** - Cache invalidation happens when it should
- ✅ **Reliable** - Dependencies always update correctly
- ✅ **Fast** - Cache mount handles performance optimization
- ✅ **Maintainable** - Less explanation needed, easier debugging

## Performance Comparison

| Approach | Builder Size | Runtime Size | Rebuild Time (code change) | Simplicity | Notes |
|----------|--------------|--------------|----------------------------|------------|-------|
| Single-stage | 500MB | 500MB | 2-5 minutes | ✅ Simple | No separation, slow rebuilds |
| **Multi-stage + COPY (ours)** | **~500MB** | **~200MB** | **5-10 seconds** | **✅ Simple** | **Best balance of performance and clarity** |
| Multi-stage + bind mounts | ~480MB | ~200MB | 5-10 seconds | ❌ Complex | Marginal savings, complex caching |
| UV distroless base | Won't work | N/A | N/A | N/A | No shell for build commands |

**Key insight:** The multi-stage build and cache mount provide 95% of the optimization. Bind mounts add complexity for minimal additional benefit (~20MB, ~2-3s).

---

## Summary

**Why our multi-stage COPY approach?**

1. **We can't use distroless uv image as base** → no shell to run build commands
2. **We still want official uv binary** → copy it from distroless image into python:slim
3. **We separate build from runtime** → smaller final image, faster rebuilds
4. **Layer caching optimization** → dependencies cached separately from code
5. **Simple COPY over bind mounts** → explicit, reliable, maintainable

**Key architectural decisions:**

| Component | Approach | Why |
|-----------|----------|-----|
| `uv` binary | Copy from distroless | Get official binary without distroless constraints |
| Base image | `python:3.13-slim` | Need shell + Python for build, minimal for runtime |
| Build pattern | Multi-stage | 50% size reduction (discard build tools) |
| Dependency caching | Cache mount | Persist packages across builds (~80% speedup) |
| Dependency files | **COPY both** | Explicit cache invalidation, predictable, simple |
| Code separation | `--no-install-project` | Dependencies (slow) separate from code (fast) |
| Source copy | **Copy src/ only** | Documentation changes don't trigger code layer rebuild |
| README file | **touch in RUN** | Optimization: README updates don't invalidate install layer |

This gives us:
- ✅ Official uv binary (always latest)
- ✅ Full build capabilities (shell, Python, package manager)
- ✅ Minimal runtime image (~200MB vs ~500MB)
- ✅ Fast rebuilds (5-10s for code changes, ~2-5s for dependency changes with cache)
- ✅ Documentation updates don't trigger code/install layer rebuilds
- ✅ Reliable dependency updates (explicit cache invalidation)
- ✅ Simple and maintainable (standard Docker patterns)
- ✅ Maximum security (non-root, minimal attack surface)

## Local Development

For local development workflow using Docker Compose (recommended), see [Docker Compose Workflow Guide](./docker-compose-workflow.md).

For direct Docker builds without Compose:
```bash
DOCKER_BUILDKIT=1 docker build -t your-agent-name:latest .
```

---

## References

### UV Documentation
- [UV Docker Integration Guide](https://docs.astral.sh/uv/guides/integration/docker/)
- [UV Docker Intermediate Layers (Bind Mounts)](https://docs.astral.sh/uv/guides/integration/docker/#intermediate-layers)
- [UV Documentation](https://docs.astral.sh/uv/)

### Docker Documentation
- [Docker Best Practices](https://docs.docker.com/build/building/best-practices/)
- [Docker Multi-Stage Builds](https://docs.docker.com/build/building/multi-stage/)
- [Docker RUN Command Reference](https://docs.docker.com/reference/dockerfile/#run)
- [BuildKit Cache Mounts](https://docs.docker.com/build/cache/optimize/)
- [Docker Bind Mounts](https://docs.docker.com/build/building/best-practices/#use-bind-mounts)

### Docker Compose
- [Docker Compose Workflow Guide](./docker-compose-workflow.md) - Local development workflow
- [Docker Compose Watch Mode](https://docs.docker.com/compose/file-watch/)
