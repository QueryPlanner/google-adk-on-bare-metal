# Google ADK on Bare Metal

A **production-ready template** for building and deploying Google ADK agents on your own infrastructure (bare metal, VPS, or private cloud) without the complexity or lock-in of heavy cloud providers.

**Philosophy**
We believe you should own your agents. This template is designed to strip away the "cloud magic" and give you a clean, performant, and observable foundation that runs anywhere—from a $5/mo VPS to a Raspberry Pi cluster.

## Key Features

- 🐳 **Deploy Anywhere**: Pre-configured Docker & Compose setup. Runs on Hetzner, DigitalOcean, or your basement server.
- 🛠️ **Automated Setup**: Includes a `setup.sh` script to harden your server (UFW, Fail2Ban) and install dependencies in minutes.
- 🔄 **CI/CD Included**: GitHub Actions workflow builds multi-arch images (AMD64/ARM64) and pushes to GHCR automatically.
- 🔭 **Open Observability**: Built-in OpenTelemetry (OTel) instrumentation. Pre-configured for **Langfuse**, but easily adaptable to Jaeger, Prometheus, or any OTel-compatible backend.
- 🚀 **Modern Stack**: Python 3.13, `uv`, `fastapi`, `asyncpg`.
- 💾 **VM Persistence**: Postgres-backed sessions and local artifacts retained
  across normal process and container recreation.

## Quickstart

### Prerequisites
- Python **3.13+**
- [`uv`](https://github.com/astral-sh/uv)
- A Postgres connection string
- An LLM API Key (OpenRouter or Google)

### 1) Configure Environment

Copy `.env.example` to `.env`:

- **`AGENT_NAME`**: Unique ID for your agent.
- **`DATABASE_URL`**: Postgres connection string.
- **`OPENROUTER_API_KEY`**: Recommended for accessing varied models.
- **`GOOGLE_API_KEY`**: Optional. Required only if using Gemini models directly.

### 2) Install Dependencies

```bash
uv sync
```

### 3) Run Locally

```bash
uv run python -m agent.server
```
Visit `http://127.0.0.1:8080/docs` for the API documentation. The ADK
development UI is disabled by default; enable it only for local or SSH-tunneled
development:

```bash
SERVE_WEB_INTERFACE=true uv run python -m agent.server
```

Then visit `http://127.0.0.1:8080`.

## Deployment: It's Just One Command

We've simplified deployment to the absolute basics. No Kubernetes required.

### Option 1: Using the Pre-built Image (Recommended)

Every successful Docker Publish workflow on `main` publishes an image for that
repository. From the cloned repository on your server, create `.env`, then
replace both placeholders below with the lowercase owner and repository that
published the image, and replace the digest placeholder with the lowercase
OCI digest from the Docker Publish workflow summary. Tags such as `main` can
move; the digest is the immutable deployment identity. The export is
session-scoped, so repeat it in each new deployment shell.

```bash
export IMAGE="ghcr.io/<your-org-or-username>/<your-repository>@sha256:<64-lowercase-hex>"
docker compose pull && docker compose up --no-build --wait --wait-timeout 180
```

Public GHCR packages can be pulled anonymously. Private packages require the
least-privilege login described in the full deployment guide.

When `Docker Publish` is manually dispatched from `main` with deployment
enabled, it binds the rollout to the workflow commit SHA and that run's build
digest. The VM checks out the commit in detached mode, uses only the tracked
`compose.yaml`, and verifies the running image reference and OCI revision before
the deployment succeeds.

### Option 2: Build Yourself

```bash
git pull
docker compose up --build --wait --wait-timeout 180
```

The bounded `--wait` command returns only after the agent's container
healthcheck passes. Compose calls `/ready`, which runs one bounded PostgreSQL
`SELECT 1` whenever `DATABASE_URL` is configured. Without PostgreSQL it reports
`database: not_configured` and remains ready for local, in-memory, or Agent
Engine-only development.

`/live` and ADK's existing `/health` endpoint are process-only. Use `/ready` for
traffic admission and deployment waits. It proves fresh PostgreSQL connectivity
only—not ADK pool capacity, Agent Engine, model/provider, tool, or post-start
artifact health.

Keep `/ready` internal or behind authenticated, rate-limited ingress because
each call opens one short-lived database connection.

## Artifact persistence

The filesystem artifact root is fixed at `<AGENT_DIR>/.adk/artifacts`. In the
container, `AGENT_DIR` is `/app/src`; Compose mounts the `agent_artifacts` named
volume at `/app/src/.adk`, so artifacts survive normal agent-container
recreation. PostgreSQL session persistence is a separate storage contract.

Dedicate each artifact root to one ADK application. Run only one writer at a
time for a given `user_id`/`session_id`/`filename` key because the local backend
does not coordinate concurrent version allocation for the same key.

This is persistence across normal process or container recreation, not
power-loss safety or crash consistency. The template does not add application
authentication, quotas, retention, encryption, or backup automation. ADK
creates artifact directories and files according to the process umask, so
ownership, permissions, umask, and backups remain operator responsibilities.
Treat the volume as trusted application data and do not grant untrusted users
host or volume write access.

`docker compose down` retains the named volume. In contrast,
`docker compose down --volumes` permanently deletes the artifact volume and all
artifacts in it.

## Secure access

Compose publishes the agent only on `127.0.0.1:8080` by default, and the ADK
development web interface and agent reload are disabled. Google ADK's HTTP,
streaming, WebSocket, session, and artifact routes do not gain application
authentication from those settings. Do not open or publicly publish port 8080.

Use an SSH tunnel for temporary development access, or place an authenticated
HTTPS reverse proxy in front of the entire upstream on port 443. The
[deployment guide](docs/DEPLOYMENT.md#network-security-boundary) documents the
requirements and the migration checks for existing VMs.

👉 **[Read the Full Deployment Guide](docs/DEPLOYMENT.md)**

## Observability

The template comes pre-wired with **OpenTelemetry**. By default, it's set up to export traces to **Langfuse** for beautiful, actionable insights into your agent's performance and costs.

To change the backend, simply update the OTel exporter configuration in your `.env`. You are not locked into any specific observability vendor.

## Documentation

- [Development Guide](docs/development.md)
- [Architecture](docs/architecture.md)
- [Observability Setup](docs/base-infra/observability.md)

## Provenance

This template includes material derived from
[Agent Foundation](https://github.com/doughayden/agent-foundation). See
[Third-Party Notices](THIRD_PARTY_NOTICES.md) for the pinned source and
preserved upstream license.
