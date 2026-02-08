#!/bin/bash
set -euo pipefail

# Prevent interactive prompts during package installation
export DEBIAN_FRONTEND=noninteractive

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%dT%H:%M:%S%z')] $1${NC}"
}

error() {
    echo -e "${RED}[ERROR] $1${NC}"
    exit 1
}

warn() {
    echo -e "${YELLOW}[WARN] $1${NC}"
}

# Check if running as root
if [[ $EUID -ne 0 ]]; then
   error "This script must be run as root"
fi

# 1. System Updates
log "Updating system packages..."
apt-get update && apt-get upgrade -y

# 2. Install Essential Tools
log "Installing essential tools..."
apt-get install -y \
    curl \
    git \
    ufw \
    fail2ban \
    unzip \
    jq \
    software-properties-common

# 3. Install Docker & Docker Compose
if ! command -v docker &> /dev/null; then
    log "Installing Docker..."
    curl -fsSL https://get.docker.com -o get-docker.sh
    sh get-docker.sh
    rm get-docker.sh
    
    # Enable Docker to start on boot
    systemctl enable docker
    systemctl start docker
else
    log "Docker already installed. Skipping..."
fi

# 4. Configure Firewall (UFW)
log "Configuring UFW firewall..."
ufw default deny incoming
ufw default allow outgoing
ufw allow ssh
ufw allow 80/tcp
ufw allow 443/tcp
# Allow custom agent port if needed (default 8080 for agent)
ufw allow 8080/tcp

# Enable UFW non-interactively
if ! ufw status | grep -q "Status: active"; then
    echo "y" | ufw enable
    log "UFW enabled."
else
    log "UFW already enabled."
fi

# 5. Configure Fail2Ban
log "Configuring Fail2Ban..."
# Create a local configuration to avoid overwriting defaults
if [ ! -f /etc/fail2ban/jail.local ]; then
    cat > /etc/fail2ban/jail.local <<EOF
[DEFAULT]
bantime = 1h
findtime = 10m
maxretry = 5

[sshd]
enabled = true
EOF
    systemctl restart fail2ban
    log "Fail2Ban configured and restarted."
else
    log "Fail2Ban configuration exists. Skipping..."
fi

# 6. Setup Log Rotation for Docker Containers
log "Configuring Docker log rotation..."
# Create daemon.json if it doesn't exist
if [ ! -f /etc/docker/daemon.json ]; then
    cat > /etc/docker/daemon.json <<EOF
{
  "log-driver": "json-file",
  "log-opts": {
    "max-size": "10m",
    "max-file": "3"
  }
}
EOF
    systemctl restart docker
    log "Docker configured with log rotation."
else
    warn "/etc/docker/daemon.json exists. Please manually verify log rotation settings."
fi

# 7. Create Dedicated User (Optional but recommended)
AGENT_USER="agent-runner"
if ! id "$AGENT_USER" &>/dev/null; then
    log "Creating dedicated user: $AGENT_USER"
    useradd -m -s /bin/bash "$AGENT_USER"
    usermod -aG docker "$AGENT_USER"
    log "User $AGENT_USER created and added to docker group."
else
    log "User $AGENT_USER already exists."
fi

log "Setup complete! Please log out and log back in (or switch to $AGENT_USER) to use Docker without sudo."
log "To deploy your agent:"
log "  1. Switch to user: su - $AGENT_USER"
log "  2. Clone repo: git clone <your-repo-url>"
log "  3. Configure .env"
log "  4. Run: docker compose up -d"
