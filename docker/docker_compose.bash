#!/usr/bin/env bash
set -e

XAUTH=/tmp/.docker.xauth

# Erstelle falls nicht vorhanden
if [ ! -f $XAUTH ]; then
    touch $XAUTH
    xauth nlist $DISPLAY | sed -e 's/^..../ffff/' | xauth -f $XAUTH nmerge - || echo "Warning: xauth setup failed"
    chmod 644 $XAUTH
fi

# X11 Zugriff erlauben
xhost +local:root

# Docker Compose starten
docker compose up
