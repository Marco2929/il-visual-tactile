# dataset_builder/tests/export_trim_filmstrip.py
"""Visual check: does the trim anchor sit where it should?

The spread/segment numbers say WHERE the cut lands but not WHAT the camera
sees there, so they cannot answer the one question that matters: is the
first frame of a trimmed segment plausibly 'just above the surface, hand
still open'? Or is it still the long approach from far away?

Writes one PNG strip per episode: evenly spaced wrist frames, each labelled
with frame index, episode fraction, grasp signal and ee z. The anchor
candidates are always included and marked, since those are the frames worth
judging:

    ARRIVED    end of the descent            (anchor = "arrived")
    H+<n>cm    n cm above the grasp height   (anchor = "height")
    CLOSING    fingers start to close        (first_grasp)

Everything from the anchor to the END of the episode is kept by the trim,
including any retry - a failed pinch plus re-grasp is one recovery
behaviour, not two samples.

    python dataset_builder/tests/export_trim_filmstrip.py \\
        --zarr zarr_datasets/foil_lifting_wrist_tesollo.zarr \\
        --out outputs/trim_strips --n 6

    # the episodes that disagree with 'at most one re-grasp':
    ... --retries
"""

import argparse
import pathlib
import sys

import numpy as np
import zarr

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from processing.grasp_detection import (  # noqa: E402
    EE_Z_DIM, detect_episode_grasps, grasp_signal, iter_episodes, smooth)


# approach (before anchor) = grey, kept segment = green,
# closed pinch = orange.
C_DROP = (90, 90, 96)
C_KEEP = (60, 200, 90)
C_CLOSED = (255, 140, 60)


def height_anchor(ep, first_grasp, margin_m):
    """Last frame before `first_grasp` that is >= margin above grasp height.

    Relative to the grasp height OF THIS EPISODE, never an absolute z: the
    pallet height differs per collection and is going to vary more.
    """
    if first_grasp <= 0:
        return -1
    z = ep[:, EE_Z_DIM]
    target = z[first_grasp] + margin_m
    above = np.flatnonzero(z[:first_grasp] > target)
    return int(above[-1]) if len(above) else 0


def annotate(img, lines, colour):
    """Colour bar + text lines under an RGB uint8 frame."""
    from PIL import Image, ImageDraw

    im = Image.fromarray(np.asarray(img)).convert("RGB")
    pad = 46
    out = Image.new("RGB", (im.width, im.height + pad), (18, 18, 18))
    out.paste(im, (0, 0))
    d = ImageDraw.Draw(out)
    # Readable even in a thumbnail where the text is too small.
    d.rectangle([0, 0, im.width - 1, 5], fill=colour)
    for i, line in enumerate(lines):
        d.text((3, im.height + 2 + 11 * i), line, fill=(235, 235, 235))
    return np.asarray(out)


def export_episode(idx, ep, rgb, out_dir, n_cols, margin_m):
    from PIL import Image

    info = detect_episode_grasps(ep)
    n = len(ep)
    sig = smooth(grasp_signal(ep), 5)
    z = ep[:, EE_Z_DIM]
    thr = info["threshold"]

    first = info["first_grasp"]
    anchor = info["anchor_idx"]
    arrived = info["arrived_idx"]
    h_anchor = height_anchor(ep, first, margin_m)

    marks = {}
    if anchor >= 0:
        marks.setdefault(anchor, []).append(
            f"** CUT (lowz+{int(margin_m * 100)}cm)")
    if arrived >= 0:
        marks.setdefault(arrived, []).append("arrived")
    if h_anchor >= 0:
        marks.setdefault(h_anchor, []).append(
            f"grasp-{int(margin_m * 100)}cm")
    if first >= 0:
        marks.setdefault(first, []).append("CLOSING")
    for k, (a, b) in enumerate(info["attempts"]):
        marks.setdefault(int(a), []).append(f"pinch{k + 1}")

    picks = set(np.linspace(0, n - 1, n_cols).astype(int).tolist())
    picks |= {i for i in marks if 0 <= i < n}
    picks = sorted(picks)

    closed = np.zeros(n, dtype=bool)
    for a, b in info["attempts"]:
        closed[a:b] = True

    tiles = []
    for i in picks:
        if closed[i]:
            colour = C_CLOSED
        elif anchor >= 0 and i >= anchor:
            colour = C_KEEP
        else:
            colour = C_DROP
        lines = [
            f"f{i} ({i / max(n - 1, 1):.2f})",
            f"grasp {sig[i]:5.1f}/{thr:.0f}",
            f"z {z[i]:+.3f} (+{z[i] - z[first]:+.3f})" if first >= 0
            else f"z {z[i]:+.3f}",
        ]
        if i in marks:
            lines.append("<< " + " ".join(marks[i]))
        tiles.append(annotate(rgb[i], lines, colour))

    h = max(t.shape[0] for t in tiles)
    w = sum(t.shape[1] for t in tiles)
    strip = np.full((h, w, 3), 18, dtype=np.uint8)
    x = 0
    for t in tiles:
        strip[:t.shape[0], x:x + t.shape[1]] = t
        x += t.shape[1]

    path = out_dir / f"ep{idx:04d}_strip.png"
    Image.fromarray(strip).save(path)

    seg = n - anchor if anchor >= 0 else -1
    z_above = z[anchor] - z.min() if anchor >= 0 else float("nan")
    print(f"[ep {idx:3d}] n={n:3d}  CUT={anchor:3d} "
          f"(z {z_above:+.3f} over low)  closing={first:3d}  "
          f"keep={seg:3d} ({seg / max(n, 1):.0%})  "
          f"attempts={info['n_attempts']}  -> {path.name}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--zarr", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--episodes", default=None,
                   help="comma-separated indices")
    p.add_argument("--n", type=int, default=6,
                   help="how many episodes to spread over the set")
    p.add_argument("--n-cols", type=int, default=10)
    p.add_argument("--margin", type=float, default=0.10,
                   help="height anchor: metres above the grasp height")
    p.add_argument("--retries", action="store_true",
                   help="only episodes with more than one pinch")
    args = p.parse_args()

    root = zarr.open(args.zarr, mode="r")
    data = root["data"] if "data" in root else root
    if "wrist_rgb" not in data:
        raise SystemExit("no wrist_rgb in this zarr - nothing to look at")
    state = np.asarray(data["state"])
    rgb = data["wrist_rgb"]
    ends = np.asarray(root["meta/episode_ends"][:])
    starts = np.concatenate([[0], ends[:-1]])

    if args.episodes:
        sel = [int(s) for s in args.episodes.split(",")]
    elif args.retries:
        sel = [i for i, s, e, ep in iter_episodes(state, ends)
               if detect_episode_grasps(ep)["n_attempts"] > 1]
    else:
        sel = np.linspace(0, len(ends) - 1, args.n).astype(int).tolist()

    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("legend: grey = dropped by trim, green = kept, "
          "orange = pinch closed\n")
    for i in sel:
        s, e = int(starts[i]), int(ends[i])
        export_episode(i, state[s:e], rgb[s:e], out_dir,
                       args.n_cols, args.margin)
    print(f"\nwrote {len(sel)} strips to {out_dir}")


if __name__ == "__main__":
    main()