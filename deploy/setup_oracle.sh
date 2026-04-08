#!/bin/bash
# ============================================================
# Stockbot Oracle Cloud Free Tier Setup
# Run this on a fresh Ubuntu VM after cloning the repo.
# Usage: bash deploy/setup_oracle.sh
# ============================================================
set -e

echo "=== Stockbot Oracle Cloud Setup ==="
echo ""

# ── 1. Install system dependencies ──────────────────────────
echo "[1/5] Installing system packages..."
sudo apt-get update -qq
sudo apt-get install -y -qq python3 python3-pip python3-venv git > /dev/null 2>&1
echo "  Done."

# ── 2. Set up Python virtualenv ─────────────────────────────
echo "[2/5] Creating Python virtualenv..."
cd "$(dirname "$0")/.."
PROJECT_DIR=$(pwd)
python3 -m venv venv
source venv/bin/activate
pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt
echo "  Done. Installed $(pip list 2>/dev/null | wc -l) packages."

# ── 3. Configure API keys ───────────────────────────────────
echo "[3/5] Configuring API keys..."
if [ ! -f .env ]; then
    cp .env.example .env
    echo ""
    echo "  ╔══════════════════════════════════════════════════╗"
    echo "  ║  You need to fill in your API keys in .env      ║"
    echo "  ║  Run: nano $PROJECT_DIR/.env                    ║"
    echo "  ║                                                  ║"
    echo "  ║  Required:                                       ║"
    echo "  ║    ALPACA_API_KEY=your_key                       ║"
    echo "  ║    ALPACA_SECRET_KEY=your_secret                 ║"
    echo "  ║    NEWS_API_KEY=your_newsapi_key                 ║"
    echo "  ║    PAPER_TRADING=true                            ║"
    echo "  ╚══════════════════════════════════════════════════╝"
    echo ""
    read -p "  Edit .env now? (y/n) " edit_env
    if [[ "$edit_env" == "y" ]]; then
        nano .env
    fi
else
    echo "  .env already exists, skipping."
fi

# ── 4. Create systemd service ───────────────────────────────
echo "[4/5] Installing systemd service..."
SERVICE_FILE=/etc/systemd/system/stockbot.service
sudo tee "$SERVICE_FILE" > /dev/null << EOF
[Unit]
Description=Stockbot Paper Trading Bot
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=$(whoami)
WorkingDirectory=$PROJECT_DIR
ExecStart=$PROJECT_DIR/venv/bin/python -u main.py live
Restart=always
RestartSec=60
StandardOutput=journal
StandardError=journal

# Env vars from .env file
EnvironmentFile=$PROJECT_DIR/.env

# Safety limits
MemoryMax=512M
CPUQuota=50%

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable stockbot
echo "  Service installed and enabled."

# ── 5. Start the bot ────────────────────────────────────────
echo "[5/5] Starting stockbot..."
sudo systemctl start stockbot
sleep 3

if sudo systemctl is-active --quiet stockbot; then
    echo ""
    echo "  ╔══════════════════════════════════════════════════╗"
    echo "  ║  Stockbot is RUNNING!                            ║"
    echo "  ║                                                  ║"
    echo "  ║  Useful commands:                                ║"
    echo "  ║    View logs:    journalctl -u stockbot -f       ║"
    echo "  ║    Stop bot:     sudo systemctl stop stockbot    ║"
    echo "  ║    Restart bot:  sudo systemctl restart stockbot ║"
    echo "  ║    Bot status:   sudo systemctl status stockbot  ║"
    echo "  ╚══════════════════════════════════════════════════╝"
    echo ""
    echo "  Showing last few log lines:"
    journalctl -u stockbot --no-pager -n 10
else
    echo ""
    echo "  ERROR: Stockbot failed to start. Check logs:"
    echo "    journalctl -u stockbot --no-pager -n 20"
fi
