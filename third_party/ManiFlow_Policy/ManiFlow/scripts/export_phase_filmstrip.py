"""Export wrist frames with their detected phase burnt in, as PNG strips.

The ASCII tracks show WHEN the boundaries sit but not WHAT the camera sees
there, so they cannot confirm that 'search' really is the open-hand search.
This writes one image strip per episode: evenly spaced frames, each labelled
with its phase, frame index, episode fraction, grasp signal and ee z.

    python scripts/export_phase_filmstrip.py --zarr <zarr> \
        --episodes 0,83,167 --out /ros2_ws/data/outputs/phase_strips

Boundary frames (approach->search->hold) are always included and marked, since
those are the ones worth judging.
"""

import argparse
import pathlib
import sys

import numpy as np

_HERE = pathlib.Path(__file__).resolve().parent
for cand in [_HERE, *_HERE.parents]:
    if (cand / 'maniflow' / 'policy').is_dir():
        sys.path.insert(0, str(cand))
        break

from maniflow.common.phase_detection import (  # noqa: E402
    GRASP_THRESHOLD, PHASE_NAMES, detect_episode_phases, grasp_signal)
from maniflow.common.replay_buffer import ReplayBuffer  # noqa: E402

_COLOURS = {0: (80, 140, 255), 1: (60, 220, 90), 2: (240, 220, 60),
            3: (255, 120, 60)}


def _annotate(img, lines, colour):
    """Draw a colour bar plus text lines onto an RGB uint8 frame."""
    from PIL import Image, ImageDraw

    im = Image.fromarray(img).convert('RGB')
    pad = 34
    out = Image.new('RGB', (im.width, im.height + pad), (18, 18, 18))
    out.paste(im, (0, 0))
    d = ImageDraw.Draw(out)
    # Phase colour bar across the top of the frame: readable even in a
    # thumbnail where the text is too small.
    d.rectangle([0, 0, im.width - 1, 5], fill=colour)
    for i, line in enumerate(lines):
        d.text((3, im.height + 2 + 11 * i), line, fill=(235, 235, 235))
    return np.asarray(out)


def export_episode(ep_idx, state, rgb, out_dir, n_cols=10):
    from PIL import Image

    info = detect_episode_phases(state)
    n = len(state)
    sig = grasp_signal(state)
    z = state[:, 2]

    picks = set(np.linspace(0, n - 1, n_cols).astype(int).tolist())
    boundaries = {info['arrived_idx']: 'ARRIVED',
                  info['search_start']: 'DECISION WINDOW',
                  info['grasp_idx']: 'CLOSING',
                  info['closed_idx']: 'CLOSED'}
    picks |= {i for i in boundaries if 0 <= i < n}
    picks = sorted(picks)

    tiles = []
    for i in picks:
        ph = int(info['phases'][i])
        lines = [
            f"{PHASE_NAMES[ph]}  f{i} ({i / n:.2f})",
            f"grasp {sig[i]:5.1f}/{GRASP_THRESHOLD:.0f}  z {z[i]:+.3f}",
        ]
        if i in boundaries:
            lines.append(f"<< {boundaries[i]}")
        tiles.append(_annotate(np.asarray(rgb[i]), lines, _COLOURS[ph]))

    h = max(t.shape[0] for t in tiles)
    w = sum(t.shape[1] for t in tiles)
    strip = np.full((h, w, 3), 18, dtype=np.uint8)
    x = 0
    for t in tiles:
        strip[:t.shape[0], x:x + t.shape[1]] = t
        x += t.shape[1]

    path = out_dir / f"ep{ep_idx:04d}_strip.png"
    Image.fromarray(strip).save(path)

    counts = np.bincount(info['phases'], minlength=len(PHASE_NAMES))
    print(f"[ep {ep_idx}] {n} frames | "
          + ' '.join(f"{PHASE_NAMES[k]} {counts[k]}"
                     for k in range(len(PHASE_NAMES)))
          + f" | arrived {info['arrived_idx']} closing {info['grasp_idx']} "
            f"closed {info['closed_idx']}"
          + (f" | {info['n_attempts']} attempts" if info['n_attempts'] > 1
             else '')
          + f" -> {path.name}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--zarr', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--episodes', default=None,
                   help='comma-separated indices; default spreads over the set')
    p.add_argument('--n', type=int, default=4)
    p.add_argument('--n-cols', type=int, default=10)
    p.add_argument('--retries', action='store_true')
    args = p.parse_args()

    rb = ReplayBuffer.copy_from_path(args.zarr,
                                     keys=['state', 'wrist_rgb'])
    state = np.asarray(rb['state'])
    rgb = rb['wrist_rgb']
    ends = np.asarray(rb.episode_ends[:])
    starts = np.concatenate([[0], ends[:-1]])

    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.episodes:
        sel = [int(s) for s in args.episodes.split(',')]
    elif args.retries:
        sel = [i for i, (s, e) in enumerate(zip(starts, ends))
               if detect_episode_phases(state[s:e])['n_attempts'] > 1]
    else:
        sel = np.linspace(0, len(ends) - 1, args.n).astype(int).tolist()

    print(f"legend: blue=approach green=search(decision) "
          f"yellow=transition orange=hold\n")
    for i in sel:
        s, e = starts[i], ends[i]
        export_episode(i, state[s:e], rgb[s:e], out_dir,
                       n_cols=args.n_cols)
    print(f"\nwrote {len(sel)} strips to {out_dir}")


if __name__ == '__main__':
    main()