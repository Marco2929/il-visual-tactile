#!/usr/bin/env bash
set -euo pipefail

echo "=== SenseGlove Setup ==="
echo ""

# ── Docker ────────────────────────────────────────────────────
if command -v docker &>/dev/null; then
    echo "[✓] Docker already installed"
else
    echo "[→] Installing Docker ..."
    curl -fsSL https://get.docker.com | sh
    sudo usermod -aG docker "$USER"
    echo "[✓] Docker installed"
    echo ""
    echo "NOTE: You have been added to the 'docker' group."
    echo "      This script will continue, but you may need to"
    echo "      log out and back in for Docker to work without sudo."
    echo ""
fi

# ── uv ───────────────────────────────────────────────────────
if command -v uv &>/dev/null; then
    echo "[✓] uv already installed"
else
    echo "[→] Installing uv ..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source "$HOME/.local/bin/env"
    echo "[✓] uv installed"
fi

# ── Docker Hub login ─────────────────────────────────────────
echo ""
echo "Please enter the Docker Hub token provided by SenseGlove:"
docker login -u zippeth

echo ""
echo "=== Setup complete ==="
echo ""
echo "You can now run:"
echo "  uv run image_manager.py --pull     # download the SenseGlove image"
echo "  uv run start.py r1                 # launch R1"
echo "  uv run start.py tesollo            # launch Tesollo"
