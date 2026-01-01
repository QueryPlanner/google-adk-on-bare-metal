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

# Copy dependency files first and set ownership
COPY --chown=appuser:appgroup pyproject.toml uv.lock ./

# Install dependencies
# Use --frozen to ensure exact versions from uv.lock
# Use --no-dev to exclude dev dependencies
RUN uv sync --frozen --no-dev

# Copy the application code and set ownership
COPY --chown=appuser:appgroup agents/ agents/
COPY --chown=appuser:appgroup platform_utils/ platform_utils/
COPY --chown=appuser:appgroup server.py .
COPY --chown=appuser:appgroup entrypoint.sh .

# Set environment variables
ENV PATH="/app/.venv/bin:$PATH"
ENV AGENTS_DIR="/app/agents"
ENV HOST="0.0.0.0"
ENV PORT="8080"

# Expose the port
EXPOSE 8080

# Switch to the non-root user
USER appuser

# Set the entrypoint
ENTRYPOINT ["/app/entrypoint.sh"]

# Run the server
CMD ["python", "-m", "server"]