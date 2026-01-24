# Environment Variables

Complete reference for all environment variables used in this project.

## Configuration

### Database

**DATABASE_URL**
- **When:** Always required
- **Value:** Postgres connection string (e.g., `postgresql://user:pass@localhost:5432/dbname`)
- **Purpose:** Persistent storage for agent sessions and memory

### API Keys

**GOOGLE_API_KEY**
- **When:** Required if using Google models
- **Value:** AI Studio API Key

**OPENROUTER_API_KEY**
- **When:** Required if using OpenRouter models
- **Value:** OpenRouter API Key

### Google Cloud (Optional)

**GOOGLE_CLOUD_PROJECT**
- **When:** Optional (required for Vertex AI or Cloud Observability)
- **Value:** Your GCP project ID
- **Purpose:** Identifies the Google Cloud project

**GOOGLE_CLOUD_LOCATION**
- **When:** Optional
- **Value:** GCP region (e.g., `us-central1`)
- **Purpose:** Region for Vertex AI calls

### Agent Runtime Configuration

**AGENT_NAME**
- **When:** Optional
- **Value:** Unique identifier (e.g., `my-agent`)
- **Default:** `agent`
- **Purpose:** Identifies logs and traces

**LOG_LEVEL**
- **Options:** `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`
- **Default:** `INFO`
- **Purpose:** Controls logging verbosity

**HOST**
- **Default:** `127.0.0.1`
- **Purpose:** Server bind address (use `0.0.0.0` for Docker)

**PORT**
- **Default:** `8080`
- **Purpose:** Server listening port

### Feature Flags

**SERVE_WEB_INTERFACE**
- **Default:** `FALSE`
- **Purpose:** Enables ADK web UI at http://127.0.0.1:8080
- **Options:** `TRUE` / `FALSE`

**RELOAD_AGENTS**
- **Default:** `FALSE`
- **Purpose:** Enable agent hot-reloading on file changes (development only)

**OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT**
- **Default:** `FALSE`
- **Purpose:** Capture full prompts/responses in traces. Set to `TRUE` to see conversation content in Langfuse or Jaeger.

### Observability (Langfuse)

**LANGFUSE_PUBLIC_KEY**
- **Value:** `pk-lf-...`
- **Purpose:** Automatically configures OTel to export to Langfuse.

**LANGFUSE_SECRET_KEY**
- **Value:** `sk-lf-...`
- **Purpose:** Authentication for Langfuse OTLP.

**LANGFUSE_BASE_URL**
- **Default:** `https://cloud.langfuse.com`
- **Options:** `https://us.cloud.langfuse.com` or your self-hosted URL.

## Environment Variable Precedence

1. **Environment variables** (highest priority)
2. **.env file** (loaded via `python-dotenv`)
3. **Default values** (defined in code)

## Security Best Practices

- **Never commit `.env` files** - Already gitignored
- **Rotate credentials** - If `.env` is accidentally committed, rotate all credentials

