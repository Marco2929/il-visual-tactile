#!/usr/bin/env bash
# One-off: move the flat reports/ artefacts next to their data.
#
#   reports/r010.rrd  ->  <ckpt>/rollout_010/rollout.rrd
#   reports/<ckpt>.html -> <ckpt>/dashboard.html
#
# mv, not a rebuild: these took ~40 min of video decoding to produce.
set -euo pipefail

cd "$(dirname "$0")"
CKPT=${CKPT:-epoch=0040-val_action_mse=9.478029}

[ -d reports ] || { echo "no reports/ to relocate"; exit 0; }

for f in reports/r*.rrd; do
    [ -f "$f" ] || continue
    short=$(basename "$f" .rrd)
    short=${short#r}
    dest="$CKPT/rollout_${short}"
    if [ -d "$dest" ]; then
        mv -v "$f" "$dest/rollout.rrd"
    else
        echo "no target dir for $f (expected $dest) - left in place"
    fi
done

if [ -f "reports/$CKPT.html" ]; then
    mv -v "reports/$CKPT.html" "$CKPT/dashboard.html"
fi

rmdir reports 2>/dev/null && echo "removed empty reports/" || true