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
- 💾 **Production Persistence**: Postgres-backed sessions out of the box.

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
published the image. The export is session-scoped, so repeat it in each new
deployment shell.

```bash
export IMAGE="ghcr.io/<your-org-or-username>/<your-repository>:main"
docker compose pull && docker compose up --no-build --wait --wait-timeout 180
```

Public GHCR packages can be pulled anonymously. Private packages require the
least-privilege login described in the full deployment guide.

### Option 2: Build Yourself

```bash
git pull
docker compose up --build --wait --wait-timeout 180
```

The bounded `--wait` command returns only after the agent's container
healthcheck passes, so scripts can distinguish a healthy startup from a
container that merely started.

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
