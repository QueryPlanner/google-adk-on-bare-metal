# Deployment Guide

You can deploy this Agent Platform using **Docker** (easiest compatibility) or **Bare Metal** (lowest resource usage).

## Prerequisites (Both Methods)

1.  **Managed Postgres Database**: You need a connection string (e.g., from Neon, AWS RDS, Supabase).
2.  **OpenRouter API Key**.
3.  **Server**: A Linux server (Ubuntu/Debian recommended).

---

## Option 1: Docker (Recommended for Ease)

Best if you don't want to manage Python versions on the host.

1.  **Clone & Config**
    ```bash
    git clone <your-repo-url>
    cd open-services-agent-starter-pack
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
# Install uv (fast python package manager)
# Note: Inspect the script before running if you have security concerns.
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.cargo/env
```

### 2. Clone & Setup
```bash
git clone <your-repo-url>
cd open-services-agent-starter-pack

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