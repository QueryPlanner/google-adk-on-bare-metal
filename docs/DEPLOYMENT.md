# Deployment Guide

You can deploy this Agent Platform using **Docker** (easiest compatibility) or **Bare Metal** (lowest resource usage).

## Option 0: Automated Server Setup (Infrastructure as Code)

To prepare a fresh Ubuntu/Debian server for production, run the included `setup.sh` script. This script automates:
1.  **System Updates**: Ensures the OS is patched.
2.  **Dependencies**: Installs Docker, Docker Compose, Git, UFW, and Fail2Ban.
3.  **Security**: Configures a basic firewall (UFW) allowing SSH and ports
    80/443 for an operator-supplied authenticated reverse proxy. It does not
    configure that proxy or open the agent's unauthenticated port 8080.
4.  **Log Rotation**: Prevents Docker logs from filling up the disk.
5.  **Dedicated User**: Creates an `agent-runner` user for secure operation.

**Run on your server (as root):**

> [!WARNING]
> Piping scripts directly from the internet to `bash` can be dangerous. Please review the script's contents before executing it to understand the actions it will perform on your server.

```bash
curl -fsSL https://raw.githubusercontent.com/<your-username>/google-adk-on-bare-metal/main/setup.sh | bash
# OR if you have cloned the repo:
sudo ./setup.sh
```

---

## Prerequisites (Both Methods)

1.  **Managed Postgres Database**: You need a connection string (e.g., from Neon, AWS RDS, Supabase).
2.  **OpenRouter or Google API Key**.
3.  **AGENT_NAME**: A unique identifier for your agent service.
4.  **Server**: A Linux server (Ubuntu/Debian recommended).
5.  **Docker Engine 28+ (Docker method only)**: Required when loopback port
    publishing is the isolation boundary. Older releases had a documented
    local-network reachability issue for localhost-published ports.

---

## CI/CD with GitHub Actions

This repository includes a GitHub Actions workflow that automatically:
1.  **Builds** a multi-platform Docker image (**AMD64 & ARM64**) on every push.
2.  **Validates** code quality via `ruff`, `mypy`, and `pytest` before building.
3.  **Caches** build layers using GitHub Actions cache (`type=gha`) for ultra-fast rebuilds.
4.  **Pushes** the image to **GitHub Container Registry (GHCR)**.
5.  **Deploys only on explicit manual confirmation** from `main`, using the
    exact workflow commit SHA and OCI digest produced by that run.

The automated VM deployment treats `(workflow commit SHA, build digest)` as one
provenance pair. Before changing the VM it validates both values, then checks
out the commit in detached mode without resetting or cleaning operator files.
It invokes only the tracked `compose.yaml` under an explicit project name and
verifies both the running digest-qualified image reference and the image's OCI
revision label. The immutable image reference is also written to the Docker
Publish workflow summary.

### Using GHCR Images

Instead of building locally, you can run the image published by your
repository's successful Docker Publish workflow. Run these commands from the
cloned repository after creating `.env`.

Public GHCR packages can be pulled anonymously. For a private package, create a
classic personal access token with only `read:packages`, load it into
`GHCR_TOKEN` from your secret manager, and authenticate without placing the
token in this repository or the command itself:

```bash
printf '%s' "$GHCR_TOKEN" | docker login ghcr.io -u YOUR_GITHUB_USERNAME --password-stdin
```

Replace both placeholders below with the lowercase GitHub owner and repository
whose workflow published the image. Replace the digest placeholder with the
lowercase digest from the Docker Publish workflow summary. `IMAGE` selects the
existing `${IMAGE:-agent}` Compose image contract; do not edit `compose.yaml`.
Tags such as `main` can move, while a digest selects one immutable OCI image.
The export is session-scoped, so repeat it in each new deployment shell.

```bash
export IMAGE="ghcr.io/<your-org-or-username>/<your-repository>@sha256:<64-lowercase-hex>"
docker compose pull && docker compose up --no-build --wait --wait-timeout 180
```

---

## Option 1: Docker (Recommended for Ease)

Best if you don't want to manage Python versions on the host.

1.  **Clone & Config**
    ```bash
    git clone <your-repo-url>
    cd google-adk-on-bare-metal
    cp .env.example .env
    # Edit .env with your DATABASE_URL and API Keys
    ```

2.  **Run**
    ```bash
    docker compose up --build --wait --wait-timeout 180
    ```

3.  **Update**
    ```bash
    git pull
    docker compose up --build --wait --wait-timeout 180
    ```

The Compose healthcheck calls `/ready` from inside the container. When
`DATABASE_URL` is configured, every call opens one bounded connection and runs
`SELECT 1`; a failed connection, authentication, query, or close returns HTTP
503. When PostgreSQL is not configured, the response explicitly reports
`database: not_configured` and remains ready for local, in-memory, or Agent
Engine-only development. `--wait` exits successfully only after this contract
passes, while `--wait-timeout` prevents an unhealthy startup from hanging
automation.

Use `/live` for process-only liveness. ADK's existing `/health` endpoint remains
a process-only compatibility route. Neither process endpoint checks PostgreSQL.
The database readiness attempt defaults to two seconds, which is shorter than
the healthcheck client's three-second HTTP timeout.

This is deliberately a PostgreSQL connectivity check, not a claim that every
dependency is healthy. It does not prove spare capacity in ADK's internal
SQLAlchemy pool, schema permissions beyond `SELECT 1`, Agent Engine
reachability, model/provider availability, tool health, or artifact capacity
after startup.

Each `/ready` request creates a short-lived database connection and each failure
emits one generic warning. Keep the endpoint on the loopback/internal operations
path. If an ingress exposes it, require authentication and rate limiting so
untrusted callers cannot amplify connection or log load.

## Remote Trace Export

Remote trace export is optional and does not participate in liveness or
readiness. Google ADK owns the process-wide OpenTelemetry provider; the template
preserves ADK's internal processors and configures at most one outbound
OTLP/HTTP batch processor.

Choose one complete mode in `.env`:

- **Langfuse:** set `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY`.
  `LANGFUSE_BASE_URL` is optional and defaults to
  `https://cloud.langfuse.com`. The application derives the full
  `/api/public/otel/v1/traces` endpoint and the Basic authentication plus
  `x-langfuse-ingestion-version=4` trace headers.
- **Explicit collector:** set the complete
  `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT` ending once in `/v1/traces`, optionally
  `OTEL_EXPORTER_OTLP_TRACES_HEADERS`, and either omit
  `OTEL_EXPORTER_OTLP_TRACES_PROTOCOL` or set it to `http/protobuf`.

Either complete mode may set `OTEL_EXPORTER_OTLP_TRACES_TIMEOUT`. It defaults
effectively to two seconds and must be finite, greater than zero, and at most
two seconds. Timeout alone is incomplete and does not enable export.

Do not mix Langfuse settings with an explicit endpoint, protocol, or headers.
Remote collectors require HTTPS; plaintext HTTP is accepted only for loopback
development. Embedded URL credentials, query strings, fragments, malformed or
CRLF-bearing headers, and `grpc` fail before the server binds without echoing
the rejected values.

Structured prompt/response fields and tool arguments/results are elided by
default. Enabling
`OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=true` is a
privacy-sensitive opt-in. Disabled content is not arbitrary redaction:
exception messages, stack traces, status descriptions, static agent or tool
descriptions, service identity, model and tool names, timing, token counts, and
user, session, invocation, or conversation identifiers can still be exported.
Sanitize exception text and treat the collector as sensitive.

An unreachable collector does not fail `/live`, ADK's `/health`, or
database-independent `/ready`; exporter errors and lost spans remain possible.
On graceful termination, the application performs a best-effort flush for at
most five seconds after ADK runner cleanup. Compose gives the process a
ten-second stop grace period. `SIGKILL`, a host crash, or a shorter external
deadline cannot flush queued spans.

The template never calls `provider.shutdown()`. The OpenTelemetry SDK owns final
shutdown through `atexit` on normal interpreter exit; Uvicorn's
signal-termination path relies on the explicit outer flush instead.

The two-second per-request maximum is designed to fit the HTTP exporter's one
`ConnectionError` retry within that five-second flush and leave additional room
inside the surrounding Compose grace period.

The deployment contract is one server process using `python -m agent.server` or
`uv run server`. The base deployment does not support pre-fork workers, OTLP
metrics or logs, HTTP server tracing, Cloud Trace or Cloud Logging export, or a
Collector.

Only the documented trace endpoint, protocol, header, certificate, and timeout
variables are accepted. Metric-specific, log-specific, and other
`OTEL_EXPORTER_OTLP_*` settings fail configuration validation.

For a single-VM outbound privacy boundary, add
`compose.trace-gateway.yaml` using the
[gateway runbook](base-infra/observability.md#optional-vm-trace-redaction-gateway).
This optional overlay requires Docker Compose 2.24.4 or newer and a dedicated
host group for its bind-mounted secrets.
That optional mode keeps Langfuse credentials in the Collector, uses
private-network TLS plus bearer authentication, removes unapproved trace
content before egress, and leaves application readiness independent of
Collector health. It requires approximately 256 MiB of additional memory,
certificate and token rotation, and a separate failure/upgrade lifecycle. It
does not claim that raw trace content never enters the local telemetry pipeline.

Existing VMs must remove the legacy generic
`OTEL_EXPORTER_OTLP_ENDPOINT`, `OTEL_EXPORTER_OTLP_PROTOCOL`, and
`OTEL_EXPORTER_OTLP_HEADERS` variables before selecting either mode. Those
values are rejected because the pinned ADK runtime could otherwise enable
trace, metric, and log exporters. Recreate the container after updating
`.env`; do not leave generic and trace-specific variables together.

See [Trace Observability with OpenTelemetry](base-infra/observability.md) for
the complete privacy, endpoint, and failure contract.

## Artifact Storage Contract

The local artifact root is not independently configurable. It is always
`<AGENT_DIR>/.adk/artifacts`, where `AGENT_DIR` is resolved to an absolute
directory at startup. The image and Compose service pin `AGENT_DIR` to
`/app/src`, and Compose mounts the `agent_artifacts` named volume at
`/app/src/.adk`. As a result, the artifact payload and metadata files under
`/app/src/.adk/artifacts` survive normal process restarts and agent-container
recreation. PostgreSQL remains the separate session store.

Startup creates the artifact directory when needed and probes create, write,
flush, sync, unlink, and directory-cleanup access before starting the server.
If the directory is unusable, startup exits non-zero and reports the stable
public message `Artifact storage is unavailable.` It does not silently select
in-memory artifact storage.

The supported server entrypoints are `python -m agent.server` and
`uv run server`. Use one of those entrypoints so `main()` can enforce the
sanitized startup-failure contract. The server intentionally no longer creates
a module-global `app`, so the older `uvicorn agent.server:app` target is not
supported. Invoking `create_app` directly through an external Uvicorn factory
also bypasses the sanitized command boundary and is not a supported deployment
entrypoint.

Operate the filesystem backend within these boundaries:

- Dedicate one artifact root to one ADK application. The filesystem key does
  not include the application name.
- Use one writer at a time for each
  `user_id`/`session_id`/`filename` key. Concurrent version allocation for the
  same key is not locked.
- Expect persistence only across normal process or container recreation.
  Payload and metadata files are written directly, so the template does not
  claim power-loss safety or crash consistency.
- Treat the startup probe as a point-in-time check. It does not reserve
  capacity, lock the path against a trusted local actor, or prevent later
  permission and storage failures.
- Supply application authentication at the trusted ingress. Artifact storage
  adds no authentication, quotas, retention, encryption, or backup automation.
- Treat the artifact volume as trusted application data. Do not grant untrusted
  users host or volume write access.
- Set an appropriate process umask and filesystem ownership. ADK creates
  directories and files according to that umask; owner-only modes are not
  guaranteed by the template.

For the provided image, the runtime user and group have UID/GID 1000. Docker
initializes the named volume from the image-owned `/app/src/.adk` directory.
For a bare-metal or operator-supplied bind mount, ensure the service account can
create and remove files beneath `<AGENT_DIR>/.adk` before starting the service.
The operator is responsible for host ownership, permissions, umask, capacity,
monitoring, and backups.

`docker compose down` removes containers and networks but retains the named
artifact volume. The following variant is destructive:

```bash
docker compose down --volumes
```

It permanently removes this Compose project's `agent_artifacts` volume and all
stored artifacts. There is no automated backup or restore workflow. Quiesce the
single writer before taking an operator-managed volume snapshot or copy; a live
copy is not promised to be crash-consistent.

## Network Security Boundary

The process listens on `0.0.0.0:8080` inside its container so Docker can reach
it. Compose separately publishes that port on `127.0.0.1:8080` on the VM. The
host-side address is the intended security boundary; verify it after every
deployment:

```bash
docker compose port agent 8080
```

The expected output is `127.0.0.1:8080`. The default environment also disables
the ADK development web interface and agent reload.

This boundary requires Docker Engine 28+ with Docker's normal bridge/NAT packet
filtering intact. Do not set `DOCKER_INSECURE_NO_IPTABLES_RAW=1`, enable the
daemon's `allow-direct-routing` option, configure
`com.docker.network.bridge.trusted_host_interfaces`, or use `routed` or
`nat-unprotected` gateway modes. Those non-default settings can make the
container address remotely reachable even when the host publication reports
`127.0.0.1`. If the VM needs direct container routing, enforce the boundary with
an independently managed host or network firewall instead of relying on this
Compose port mapping.

These settings are not application authentication. Headless ADK still exposes
run, streaming, WebSocket, session, artifact, trace, documentation, and
evaluation routes. For public access, terminate TLS on port 443 and authenticate
the entire upstream—not only `/dev-ui`. The reverse proxy must support SSE
without response buffering and WebSocket upgrades. It must overwrite
client-supplied forwarding headers and the upstream must trust those headers
only from the proxy. Configuring a particular proxy, DNS, and certificates is a
separate deployment decision.

For temporary browser access, enable the development UI explicitly on the VM:

```bash
SERVE_WEB_INTERFACE=true docker compose up --build --wait --wait-timeout 180
```

Then create a tunnel from your computer rather than publishing the service:

```bash
ssh -L 8080:127.0.0.1:8080 your-user@your-vm
```

Open `http://127.0.0.1:8080` on your computer.

> [!CAUTION]
> The following escape hatch publishes the unauthenticated ADK service on every
> host interface. It is not safe for the public internet, does not provide TLS,
> and is not made safe by UFW alone because Docker manages its own packet
> filtering rules.
>
> ```bash
> AGENT_PUBLISH_HOST=0.0.0.0 docker compose up --build --wait --wait-timeout 180
> ```
>
> Prefer a specific private interface or private network when loopback cannot be
> used.

---

## Option 2: Bare Metal (Lowest Resources)

Best for small servers (e.g., 512MB RAM) since you avoid Docker overhead.

### 1. Install Dependencies
```bash
sudo apt update
sudo apt install -y python3-venv git
# Ensure Python 3.13+ is installed (e.g., via deadsnakes PPA on Ubuntu)
# sudo add-apt-repository ppa:deadsnakes/ppa
# sudo apt install python3.13 python3.13-venv

# Install uv (fast python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.cargo/env
```

### 2. Clone & Setup
```bash
git clone <your-repo-url>
cd google-adk-on-bare-metal

# Install Python dependencies
uv sync

# Configure Env
cp .env.example .env
# Edit .env with your real keys!
# Keep HOST=127.0.0.1, SERVE_WEB_INTERFACE=false, and RELOAD_AGENTS=false.
```

### 3. Setup Systemd (Keep it running)

1.  Edit `systemd/agent.service` and check the paths (User, WorkingDirectory).
2.  Install the service:
    ```bash
    sudo cp systemd/agent.service /etc/systemd/system/agent.service
    sudo systemctl daemon-reload
    sudo systemctl enable agent
    sudo systemctl start agent
    ```

### 4. Logs & Status
```bash
sudo systemctl status agent
sudo journalctl -u agent -f
```

## Existing VM Migration

Earlier versions published port 8080 on all interfaces, enabled the development
UI and reload, and added an allow rule for 8080 to UFW. Pulling this change does
not rewrite an existing `.env`, recreate a running container, or remove
operator-owned firewall policy.

1. Upgrade to Docker Engine 28 or later and verify the server version. The setup
   script fails instead of upgrading an existing Docker installation:

   ```bash
   sudo ./setup.sh --verify-docker-version
   docker version --format '{{.Server.Version}}'
   ```

2. Audit `/etc/docker/daemon.json`, Docker's systemd unit and drop-ins, and the
   project network. Do not rely on loopback publication when
   `DOCKER_INSECURE_NO_IPTABLES_RAW=1`, `allow-direct-routing`, trusted host
   interfaces, or `routed`/`nat-unprotected` gateway modes are enabled:

   ```bash
   sudo systemctl cat docker
   sudo systemctl show docker --property=Environment
   sudo test ! -f /etc/docker/daemon.json \
     || sudo jq . /etc/docker/daemon.json
   ```

3. Update the existing `.env`:

   ```dotenv
   AGENT_PUBLISH_HOST=127.0.0.1
   HOST=127.0.0.1
   SERVE_WEB_INTERFACE=false
   RELOAD_AGENTS=false
   ```

   Compose still overrides `HOST` to `0.0.0.0` inside its container. The
   loopback value protects direct bare-metal/systemd runs.

4. Pull and recreate the Compose service, then verify the resolved publication:

   ```bash
   git pull
   docker compose up --build --force-recreate --wait --wait-timeout 180
   docker compose port agent 8080
   ```

   Require the final command to report `127.0.0.1:8080`.

5. For a bare-metal systemd installation, restart the service and verify the
   listening socket:

   ```bash
   sudo systemctl restart agent
   sudo ss -ltnp | grep '127.0.0.1:8080'
   ```

6. Inspect UFW and manually remove the legacy rule only if it belongs to this
   deployment:

   ```bash
   sudo ufw status numbered
   sudo ufw delete allow 8080/tcp
   ```

7. Move every remote client to the authenticated HTTPS ingress on port 443
   before relying on the new boundary.

## Troubleshooting

### Permission Errors with Artifacts

When startup reports `Artifact storage is unavailable.`, inspect only bounded
diagnostics and the resolved volume:

```bash
docker compose logs --no-color --tail=200 agent
docker volume inspect YOUR_COMPOSE_PROJECT_agent_artifacts
```

Confirm that the selected volume or bind mount is trusted, has free space, and
allows the service UID/GID to create, sync, and remove files beneath
`<AGENT_DIR>/.adk`. Rebuilding the image does not repair ownership or a
read-only operator-supplied mount. Do not delete the named volume as a
permission fix; `docker compose down --volumes` destroys all stored artifacts.
