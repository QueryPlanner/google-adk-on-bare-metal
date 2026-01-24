# Environment Variables

Complete reference for all environment variables used in this project.

## Required Configuration

### Google Cloud Vertex AI Model Authentication

**GOOGLE_GENAI_USE_VERTEXAI**
- **When:** Set before running locally or bootstrap
- **Value:** `TRUE`
- **Purpose:** Enables Vertex AI authentication for Gemini models

**GOOGLE_CLOUD_PROJECT**
- **When:** Set before running locally or bootstrap
- **Value:** Your GCP project ID (e.g., `my-project-123`)
- **Purpose:** Identifies the Google Cloud project for Vertex AI and other GCP services

**GOOGLE_CLOUD_LOCATION**
- **When:** Set before running locally or bootstrap
- **Value:** GCP region (e.g., `us-central1`)
- **Purpose:** Sets the region for Vertex AI model calls and resource deployment

### Agent Identification

**AGENT_NAME**
- **When:** Set before bootstrap (used for naming GCP resources)
- **Value:** Unique identifier (e.g., `my-agent`)
- **Purpose:** Identifies cloud resources, logs, and traces
- **Note:** Used as base name for Terraform resources (`{agent_name}-{workspace}`)

### OpenTelemetry Configuration

**OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT**
- **When:** Set before bootstrap
- **Options:**
  - `TRUE` - Capture full prompts and responses in traces
  - `FALSE` - Capture metadata only (no message content)
- **Purpose:** Controls LLM message content capture in OpenTelemetry traces
- **Reference:** [OpenTelemetry GenAI Instrumentation](https://opentelemetry.io/blog/2024/otel-generative-ai/#example-usage)
- **Consideration:** Set to `FALSE` if handling sensitive data

---

## Cloud Resources (Optional)

**Optional:** Add these for production-consistent testing with durable persistence. Defaults to in-memory services if unset.

### Session and Memory Persistence

**AGENT_ENGINE**
- **When:** Set AFTER first deployment for local development with persistent sessions
- **Value:** Agent Engine resource name (e.g., `projects/123/locations/us-central1/reasoningEngines/456`)
- **Default:** Unset (in-memory ephemeral sessions)
- **How to get:** GitHub Actions job summary (`gh run view <run-id>` or Actions tab UI) OR GCP Console (Vertex AI → Agent Builder → Reasoning Engines)
- **Why:** Enables session persistence across server restarts

### Artifact Storage

**ARTIFACT_SERVICE_URI**
- **When:** Set AFTER first deployment for local development with artifact persistence
- **Value:** GCS bucket URI (e.g., `gs://my-artifact-bucket`)
- **Default:** Unset (in-memory ephemeral storage)
- **How to get:** GitHub Actions job summary (`gh run view <run-id>` or Actions tab UI) OR GCP Console (Cloud Storage → Buckets)
- **Why:** Enables artifact storage persistence

---

## Optional: Agent Runtime Configuration

### Logging

**LOG_LEVEL**
- **Options:** `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`
- **Default:** `INFO`
- **Purpose:** Controls logging verbosity
- **Usage:**
  ```bash
  LOG_LEVEL=DEBUG uv run server
  ```

### OpenTelemetry

**TELEMETRY_NAMESPACE**
- **Default:** `local`
- **Purpose:** Groups traces and logs by developer or environment in Cloud Trace
- **Usage:** Filter traces in Cloud Trace by namespace to isolate your local development traces
- **Example:** `TELEMETRY_NAMESPACE=alice-local`
- **Note:** Cloud Run deployments automatically set this to workspace name (`default`, `dev`, `stage`, `prod`)

### Server Configuration

**HOST**
- **Default:** `127.0.0.1`
- **Purpose:** Server bind address
- **Note:** `127.0.0.1` binds to localhost only (recommended for local development)

**PORT**
- **Default:** `8000`
- **Purpose:** Server listening port
- **Note:** Cloud Run always uses port 8000, Docker Compose maps to host port 8000

### Agent Features

**SERVE_WEB_INTERFACE**
- **Default:** `FALSE`
- **Purpose:** Enables ADK web UI at http://127.0.0.1:8000
- **Options:**
  - `FALSE` - API-only mode
  - `TRUE` - Enable web interface

**RELOAD_AGENTS**
- **Default:** `FALSE`
- **Purpose:** Enable agent hot-reloading on file changes (development only)
- **WARNING:** Set to `FALSE` in production (Cloud Run forces `FALSE`)
- **Usage:** Automatically reload agent configuration when code changes during development

### CORS Configuration

**ALLOW_ORIGINS**
- **Default:** `["http://127.0.0.1", "http://127.0.0.1:8000"]`
- **Format:** JSON array string
- **Purpose:** Configure CORS allowed origins
- **Example:** `ALLOW_ORIGINS='["https://your-domain.com", "http://127.0.0.1:3000"]'`

### Model Configuration

**ROOT_AGENT_MODEL**
- **Default:** `gemini-2.5-flash`
- **Options:** Any Gemini model (e.g., `gemini-2.5-pro`, `gemini-2.0-flash-exp`)
- **Purpose:** Override default root agent model
- **Usage:** Test different Gemini models without code changes

---

## Optional: Advanced Features

**ADK_SUPPRESS_EXPERIMENTAL_FEATURE_WARNINGS**
- **Default:** `FALSE`
- **Purpose:** Suppress ADK experimental feature warnings
- **Options:**
  - `FALSE` - Show warnings
  - `TRUE` - Suppress warnings

---

## Environment Variable Precedence

1. **Environment variables** (highest priority)
2. **.env file** (loaded via `python-dotenv`)
3. **Default values** (defined in code)

## Security Best Practices

- **Never commit `.env` files** - Already gitignored
- **Use Workload Identity Federation** - No service account keys needed for CI/CD
- **Rotate credentials** - If `.env` is accidentally committed, rotate all credentials
- **Limit OTEL content capture** - Set `OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=FALSE` for sensitive data

## Reference

See `.env.example` for a template configuration with inline comments.
