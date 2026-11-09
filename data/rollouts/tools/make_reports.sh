#!/usr/bin/env bash
# One .rrd per rollout, plus the HTML dashboard.
#
# Artefacts land NEXT TO THE DATA, not in a shared reports/ directory:
#
#   <ckpt>/rollout_010/rollout.rrd    <- one per rollout
#   <ckpt>/dashboard.html             <- one per checkpoint
#
# A single reports/ folder keys files by rollout number only (r010.rrd), so a
# second checkpoint silently overwrites the first one's files. Keying by the
# directory the data already lives in cannot collide, and `rm -r <ckpt>`
# takes its reports with it.
#
# Video is ON (--video): without it the wrist camera is missing from the
# timeline, which is the point of looking at a single rollout. Costs ~180 MB
# per rollout at 200 frames.
set -euo pipefail

cd "$(dirname "$0")"
PY=${PY:-/ros2_ws/venv/bin/python}
CKPT=${CKPT:-epoch=0040-val_action_mse=9.478029}
FRAMES=${FRAMES:-200}

[ -d "$CKPT" ] || { echo "no such checkpoint dir: $CKPT" >&2; exit 1; }

echo "=== dashboard ==="
"$PY" rollout_viz.py --ckpt "$CKPT" dash --out "$CKPT/dashboard.html"

echo
echo "=== per-rollout .rrd (video on, ${FRAMES} frames) ==="
for d in "$CKPT"/rollout_*; do
    [ -d "$d" ] || continue
    n=$(basename "$d")
    short=${n#rollout_}
    f="$d/rollout.rrd"
    printf '%s ... ' "$n"
    if [ -f "$f" ] && [ -z "${FORCE:-}" ]; then
        printf 'exists, skipped (FORCE=1 to rebuild)\n'
        continue
    fi
    if "$PY" rollout_viz.py --ckpt "$CKPT" rerun "$short" \
            --video --max-frames "$FRAMES" --save "$f" >/dev/null 2>&1; then
        printf '%s\n' "$(du -h "$f" | cut -f1)"
    else
        printf 'FAILED\n'
    fi
done

echo
echo "wrote:"
du -h "$CKPT/dashboard.html" "$CKPT"/rollout_*/rollout.rrd 2>/dev/null