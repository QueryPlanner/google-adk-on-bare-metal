# Deployment Guide

You can deploy this Agent Platform using **Docker** (easiest compatibility) or **Bare Metal** (lowest resource usage).

## Option 0: Automated Server Setup (Infrastructure as Code)

To prepare a fresh Ubuntu/Debian server for production, run the included `setup.sh` script. This script automates:
1.  **System Updates**: Ensures the OS is patched.
2.  **Dependencies**: Installs Docker, Docker Compose, Git, UFW, and Fail2Ban.
3.  **Security**: Configures a basic firewall (UFW) allowing SSH (22), HTTP (80/443), and the Agent port (8080).
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

---

## CI/CD with GitHub Actions

This repository includes a GitHub Actions workflow that automatically:
1.  **Builds** a multi-platform Docker image (**AMD64 & ARM64**) on every push.
2.  **Validates** code quality via `ruff`, `mypy`, and `pytest` before building.
3.  **Caches** build layers using GitHub Actions cache (`type=gha`) for ultra-fast rebuilds.
4.  **Pushes** the image to **GitHub Container Registry (GHCR)**.

### Using GHCR Images

Instead of building locally, you can pull the pre-built image from GHCR.

1.  **Login to GHCR** (on your server):
    ```bash
    echo $GITHUB_TOKEN | docker login ghcr.io -u YOUR_GITHUB_USERNAME --password-stdin
    ```
2.  **Pull the latest image**:
    ```bash
    docker pull ghcr.io/<your-org-or-username>/google-adk-on-bare-metal:main
    ```

### Automatic Deployment

To automate deployment, update your `compose.yaml` to use the GHCR image:

```yaml
services:
  agent:
    image: ghcr.io/<your-org-or-username>/google-adk-on-bare-metal:main
    # ... rest of config
```

Then your update command becomes:
```bash
docker compose pull && docker compose up -d
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
    docker compose up --build -d
    ```

3.  **Update**
    ```bash
    git pull
    docker compose up --build -d
    ```

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

## Troubleshooting

### Permission Errors with Artifacts
If you encounter `PermissionError: [Errno 13] Permission denied: '/app/src/.adk'` when running with Docker:
1.  This usually happens because the container user (UID 1000) cannot write to the host volume mounted at `./src`.
2.  **Fix:** Ensure you have rebuilt the image to include the latest permission fixes:
    ```bash
    docker compose up -d --build
    ```
3.  If that fails, ensure your host user has UID 1000 (run `id -u`).