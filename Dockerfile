# syntax=docker/dockerfile:1

# ============================================================================
# Builder Stage: Install dependencies with optimal caching
# ============================================================================
FROM python:3.13-slim AS builder

# Install uv
RUN pip install uv==0.9.26

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

# Copy only source code
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
# - netcat-openbsd: for checking DB readiness (used in entrypoint.sh)
RUN apt-get update && apt-get install -y --no-install-recommends \
    netcat-openbsd \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security (matching common host UID 1000)
RUN groupadd -g 1000 app && \
    useradd -u 1000 -g app -s /bin/sh -m app

# Set working directory
WORKDIR /app

# Pre-create the artifacts directory and set ownership
RUN mkdir -p /app/src/.adk/artifacts && chown -R app:app /app

# Copy application and virtual environment from builder
COPY --from=builder --chown=app:app /app .

# Copy entrypoint script and set ownership/permissions
COPY --chown=app:app entrypoint.sh .
RUN chmod +x entrypoint.sh

# Set environment to use virtual environment
ENV VIRTUAL_ENV=/app/.venv \
    PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    AGENT_DIR=/app/src \
    HOST=0.0.0.0 \
    PORT=8080

# Switch to non-root user
USER app

# Expose port (default 8080)
EXPOSE 8080

# Set the entrypoint
ENTRYPOINT ["/app/entrypoint.sh"]

# Run the FastAPI server
CMD ["python", "-m", "agent.server"]
