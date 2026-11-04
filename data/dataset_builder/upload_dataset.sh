#!/usr/bin/env bash
# Usage: upload_dataset.sh <dataset_name> [--replace]
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="$HERE/../configs/dataset_server.env"

# Load defaults from dataset_server.env if present
if [[ -f "$ENV_FILE" ]]; then
  # shellcheck source=/dev/null
  set -a
  source "$ENV_FILE"
  set +a
fi

NAME="${1:?usage: upload_dataset.sh <dataset_name> [--replace]}"
REPLACE="${2:-${DATASET_SERVER_REPLACE:-true}}"
SERVER="${SERVER:-${DATASET_SERVER:-192.168.50.44}}"
SERVER_USER="${SERVER_USER:-${DATASET_SERVER_USER:-dataset_server}}"
KEY="${SSH_KEY:-${DATASET_SERVER_KEY:-$HOME/.ssh/prehensio_upload}}"
ROOT="${REMOTE_ROOT:-${DATASET_SERVER_ROOT:-/data/prehensio_data}}"
PORT="${VIEWER_PORT:-${DATASET_SERVER_VIEWER_PORT:-8080}}"

# Expand ~ if key starts with it
KEY="${KEY/#\~/$HOME}"

[[ "$NAME" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]{0,99}$ ]] || { echo "Ungültiger Name: $NAME" >&2; exit 2; }

# Locate local dataset
SRC=""
for cand in "$HERE/../lerobot_datasets/$NAME" "/ros2_ws/data/lerobot_datasets/$NAME" "$NAME"; do
  if [[ -f "$cand/meta/info.json" ]]; then
    SRC="$(cd "$cand" && pwd)"
    break
  fi
done

[[ -n "$SRC" ]] || { echo "Kein LeRobot-Dataset gefunden für: $NAME" >&2; exit 2; }

# Leserechte für rsync und Viewer sicherstellen
chmod -R a+rX "$SRC" 2>/dev/null || true

SSH_OPTS=(-i "$KEY" -o BatchMode=yes -o ConnectTimeout=10)
SSH=(ssh "${SSH_OPTS[@]}")
REMOTE="$SERVER_USER@$SERVER"

echo "==> Pushing dataset '$NAME' to $SERVER..."

if ! "${SSH[@]}" "$REMOTE" "test -d '$ROOT'"; then
  echo "Fehler: Kann $REMOTE nicht erreichen oder $ROOT existiert nicht." >&2
  exit 1
fi

if "${SSH[@]}" "$REMOTE" "test -e '$ROOT/datasets/$NAME'"; then
  if [[ "$REPLACE" != "--replace" && "$REPLACE" != "true" ]]; then
    echo "datasets/$NAME existiert schon auf dem Server. Verwende --replace zum Überschreiben (Backup)." >&2
    exit 1
  fi
  STAMP="$(date +%Y%m%d_%H%M%S)"
  "${SSH[@]}" "$REMOTE" "mkdir -p '$ROOT/datasets_replaced' && mv '$ROOT/datasets/$NAME' '$ROOT/datasets_replaced/$NAME.$STAMP'"
  echo "Altes Dataset gesichert nach: datasets_replaced/$NAME.$STAMP"
fi

"${SSH[@]}" "$REMOTE" "rm -rf '$ROOT/incoming_builds/$NAME'"
rsync -avz --partial --mkpath -e "ssh -i $KEY -o BatchMode=yes" \
  "$SRC/" "$REMOTE:$ROOT/incoming_builds/$NAME/"

# Prüfsummen-Vergleich: jede Zeile ist ein Unterschied
DIFF="$(rsync -anc --itemize-changes -e "ssh -i $KEY -o BatchMode=yes" \
  "$SRC/" "$REMOTE:$ROOT/incoming_builds/$NAME/")"
[[ -z "$DIFF" ]] || { echo "Prüfsumme weicht ab:" >&2; echo "$DIFF" >&2; exit 3; }

"${SSH[@]}" "$REMOTE" "mkdir -p '$ROOT/datasets' && mv '$ROOT/incoming_builds/$NAME' '$ROOT/datasets/$NAME'"
echo "OK: $NAME liegt auf $SERVER unter $ROOT/datasets/$NAME"
echo "Viewer URL: http://$SERVER:$PORT/prehensio/$NAME/episode_0"
