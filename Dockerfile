# Use a Python image with uv pre-installed
FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim

# Install system dependencies
# - netcat-openbsd: for checking DB readiness
RUN apt-get update && apt-get install -y --no-install-recommends \
    netcat-openbsd \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user and group
RUN groupadd --gid 1001 appgroup && \
    useradd --uid 1001 --gid 1001 --create-home appuser

# Set working directory
WORKDIR /app

# Enable bytecode compilation
ENV UV_COMPILE_BYTECODE=1

# syntax=docker/dockerfile:1

# ============================================================================
# Builder Stage: Install dependencies with optimal caching
# ============================================================================
FROM python:3.13-slim AS builder

# Install uv from official distroless image
COPY --from=ghcr.io/astral-sh/uv:0.9.6 /uv /uvx /bin/

# Set working directory
WORKDIR /app

# Environment variables for optimal uv behavior
ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PYTHON_DOWNLOADS=never

# Copy dependency files - explicit cache invalidation when either file changes
COPY pyproject.toml uv.lock ./

# Install dependencies (cache mount provides the performance optimization)
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-install-project --no-dev

# Copy only source code (documentation changes won't invalidate this layer)
COPY src ./src

# Install project (create empty README to satisfy package metadata requirements)
RUN --mount=type=cache,target=/root/.cache/uv \
    touch README.md && \
    uv sync --locked --no-editable --no-dev

# ============================================================================
# Runtime Stage: Minimal production image
# ============================================================================
FROM python:3.13-slim AS runtime

# Install system dependencies
# - netcat-openbsd: for checking DB readiness
RUN apt-get update && apt-get install -y --no-install-recommends \
    netcat-openbsd \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN groupadd -r app && useradd -r -g app app

# Set working directory
WORKDIR /app

# Copy application and virtual environment from builder
COPY --from=builder --chown=app:app /app .

# Copy entrypoint script and set ownership/permissions
COPY --chown=app:app entrypoint.sh .
RUN chmod +x entrypoint.sh

# Set environment to use virtual environment
# AGENT_DIR=/app/src: Use source code instead of installed package (.venv)
# HOST=0.0.0.0: Bind to all interfaces for container networking
# PORT=8000: Explicitly set for consistent deployment
ENV VIRTUAL_ENV=/app/.venv \
    PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    AGENT_DIR=/app/src \
    HOST=0.0.0.0 \
    PORT=8000

# Switch to non-root user
USER app

# Expose port (configurable via PORT env var, default 8000)
EXPOSE 8000

# Set the entrypoint
ENTRYPOINT ["/app/entrypoint.sh"]

# Run the FastAPI server via main() for unified startup logic (logging, etc.)
CMD ["python", "-m", "agent_foundation.server"]