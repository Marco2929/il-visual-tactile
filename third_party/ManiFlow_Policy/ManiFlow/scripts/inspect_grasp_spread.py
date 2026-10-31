"""How much do the grasp points vary WITHIN one pallet position?

The recordings cover four pallet positions. An across-the-board result like
std(pred)/std(gt) ~ 1 is therefore not evidence that the policy locates the
foil: recognising which of four pallets it faces and aiming at that cluster's
mean grasp point produces the same number, because the between-cluster spread
dominates.

This settles the prior question without running any model: if the grasp points
within a single pallet position barely move, there is nothing to localise and
the confound is harmless. If they move a lot, the model test has to be redone
per cluster.

Grasp point = ee (x, y) at the frame where the fingers start closing
(phase_detection.grasp_idx), i.e. where the operator committed to a spot.

    python scripts/inspect_grasp_spread.py --zarr <zarr> --out <png>
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
    detect_episode_phases)
from maniflow.common.replay_buffer import ReplayBuffer  # noqa: E402


def kmeans(pts, k, iters=100, seed=0):
    rng = np.random.default_rng(seed)
    c = pts[rng.choice(len(pts), size=k, replace=False)].copy()
    lab = np.full(len(pts), -1)
    for _ in range(iters):
        d = ((pts[:, None, :] - c[None, :, :]) ** 2).sum(-1)
        new = d.argmin(1)
        if (new == lab).all():
            break
        lab = new
        for j in range(k):
            if (lab == j).any():
                c[j] = pts[lab == j].mean(0)
    return lab, c


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--zarr', required=True)
    p.add_argument('--out', default='/ros2_ws/data/outputs/grasp_spread.png')
    p.add_argument('-k', type=int, default=4)
    args = p.parse_args()

    rb = ReplayBuffer.copy_from_path(args.zarr, keys=['state'])
    state = np.asarray(rb['state'])
    ends = np.asarray(rb.episode_ends[:])
    starts = np.concatenate([[0], ends[:-1]])

    grasp, start_xy, eps = [], [], []
    for i, (s, e) in enumerate(zip(starts, ends)):
        info = detect_episode_phases(state[s:e])
        if info['grasp_idx'] < 0:
            continue
        grasp.append(state[s + info['grasp_idx'], :2])
        start_xy.append(state[s, :2])
        eps.append(i)
    grasp = np.array(grasp)
    start_xy = np.array(start_xy)

    print(f"\n[data] {len(grasp)} episodes with a detected grasp")
    print(f"[all]  std(x)={grasp[:, 0].std():.4f}  "
          f"std(y)={grasp[:, 1].std():.4f}")

    # Cluster on the EPISODE START pose: that is the pallet the run was
    # recorded at, independent of where on it the operator grasped. Using the
    # grasp point itself would partly cluster on the answer.
    lab, cen = kmeans(start_xy.copy(), args.k)

    print(f"\n=== grasp spread per pallet position ({args.k} clusters) ===")
    print(f"{'cl':>3} {'n':>4}  {'start centre':>17}  "
          f"{'grasp centre':>17}  {'std_x':>7} {'std_y':>7} {'rms':>7}")
    print('-' * 76)
    within = []
    for j in range(args.k):
        m = lab == j
        if not m.any():
            continue
        g = grasp[m]
        sx, sy = float(g[:, 0].std()), float(g[:, 1].std())
        rms = float(np.hypot(sx, sy))
        within.append(rms)
        print(f"{j:>3} {int(m.sum()):>4}  [{cen[j, 0]:+.3f} "
              f"{cen[j, 1]:+.3f}]  [{g[:, 0].mean():+.3f} "
              f"{g[:, 1].mean():+.3f}]  {sx:7.4f} {sy:7.4f} {rms:7.4f}")

    between = float(np.hypot(
        np.array([grasp[lab == j][:, 0].mean() for j in range(args.k)
                  if (lab == j).any()]).std(),
        np.array([grasp[lab == j][:, 1].mean() for j in range(args.k)
                  if (lab == j).any()]).std()))
    w = float(np.mean(within))
    print(f"\n  within-cluster  rms spread: {w:.4f} m")
    print(f"  between-cluster rms spread: {between:.4f} m")
    print(f"  ratio within/between      : {w / max(between, 1e-9):.2f}")
    print("\n  within << between -> grasp points are essentially fixed per")
    print("     pallet; ratio~1 in the model test is explained by position")
    print("     recognition alone and proves nothing about localisation.")
    print("  within ~ between -> there is real per-episode variation, so the")
    print("     model test must be repeated within clusters.")

    # ---- plot ----
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n[plot] matplotlib unavailable, skipping figure")
        return

    fig, ax = plt.subplots(1, 2, figsize=(13, 5.5))
    cols = plt.cm.tab10(np.linspace(0, 1, 10))

    for j in range(args.k):
        m = lab == j
        if not m.any():
            continue
        ax[0].scatter(grasp[m, 0], grasp[m, 1], s=26, alpha=.75,
                      color=cols[j], label=f"pallet {j} (n={int(m.sum())})")
        ax[0].scatter(*grasp[m].mean(0), marker='x', s=170, lw=2.5,
                      color=cols[j])
    ax[0].set_title("Grasp points, coloured by pallet position\n"
                    "x = cluster mean")
    ax[0].set_xlabel('ee x [m]')
    ax[0].set_ylabel('ee y [m]')
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=.3)
    ax[0].set_aspect('equal', adjustable='datalim')

    # Right: cluster means removed. What remains is the variation a policy
    # could only get from the image - the quantity the confound is about.
    for j in range(args.k):
        m = lab == j
        if not m.any():
            continue
        d = grasp[m] - grasp[m].mean(0)
        ax[1].scatter(d[:, 0], d[:, 1], s=26, alpha=.75, color=cols[j])
    lim = float(np.abs(np.concatenate(
        [grasp[lab == j] - grasp[lab == j].mean(0)
         for j in range(args.k) if (lab == j).any()])).max()) * 1.15
    for r in (0.01, 0.02, 0.05):
        ax[1].add_patch(plt.Circle((0, 0), r, fill=False, ls='--',
                                   color='grey', alpha=.6))
        ax[1].annotate(f"{int(r * 1000)} mm", (0, r), fontsize=7,
                       color='grey', ha='center', va='bottom')
    ax[1].set_xlim(-lim, lim)
    ax[1].set_ylim(-lim, lim)
    ax[1].set_title("Same points, cluster mean removed\n"
                    "spread a policy must read from the image")
    ax[1].set_xlabel('dx [m]')
    ax[1].set_ylabel('dy [m]')
    ax[1].grid(alpha=.3)
    ax[1].set_aspect('equal')

    fig.suptitle(f"within-cluster rms {w * 1000:.0f} mm  vs  "
                 f"between-cluster rms {between * 1000:.0f} mm")
    fig.tight_layout()
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    print(f"\n[plot] wrote {out}")


if __name__ == '__main__':
    main()
