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
Visit `http://127.0.0.1:8080`.

## Deployment: It's Just One Command

We've simplified deployment to the absolute basics. No Kubernetes required.

### Option 1: Using the Pre-built Image (Recommended)

Since we include CI/CD, every push to `main` builds a fresh image. On your server:

```bash
# 1. Pull the latest image
docker pull ghcr.io/queryplanner/google-adk-on-bare-metal:main

# 2. Start the service
docker compose up -d
```

### Option 2: Build Yourself

```bash
git pull
docker compose up --build -d
```

👉 **[Read the Full Deployment Guide](docs/DEPLOYMENT.md)**

## Observability

The template comes pre-wired with **OpenTelemetry**. By default, it's set up to export traces to **Langfuse** for beautiful, actionable insights into your agent's performance and costs.

To change the backend, simply update the OTel exporter configuration in your `.env`. You are not locked into any specific observability vendor.

## Documentation

- [Development Guide](docs/development.md)
- [Architecture](docs/architecture.md)
- [Observability Setup](docs/base-infra/observability.md)