#!/usr/bin/env python3
"""Rerun view for logged rollouts: commanded vs executed, over time.

What Rerun is good at here, and what it is not:

  GOOD  one shared timeline across heterogeneous streams - commanded chunks
        at 20 Hz, joint states at ~470 Hz and point clouds at ~14 Hz all
        scrub together, which is exactly the alignment question ("what did
        the hand see when it closed").
  GOOD  3D: the commanded end-effector path and the logged point cloud in
        one space, so a grasp aimed next to the foil is visible rather than
        inferred from numbers.
  POOR  cross-rollout comparison. Rerun's timeline is one recording; putting
        28 rollouts on it means 28 entity trees and no way to scatter
        "deficit vs commanded closure" across runs. That is what the HTML
        dashboard is for.

So the two views answer different questions and neither replaces the other.
Use this one to understand ONE rollout, the dashboard to compare ALL of them.

The wrist video IS loaded when a reader is available, because the camera is
the stream that answers "what did the hand see when it closed" and putting it
on the same timeline as the closure curves is exactly what this view is for.
It is off by default (--video) since the bags are ~1 GB each.
"""
import numpy as np
import rerun as rr
import rerun.blueprint as rrb

from . import pinch, wrist
from .data import GRASP_LOCAL, Rollout


def log_rollout(r: Rollout, ns: str = "", with_pc: bool = True,
                with_video: bool = False, max_frames: int = 200,
                pinch_model=None):
    """Log one rollout. ns prefixes every entity path, so several rollouts
    can coexist in one recording and be switched with the blueprint tabs."""
    st = r.stats()
    p = (lambda path: f"{ns}/{path}") if ns else (lambda path: path)

    # Tip-to-tip width, which separates a posture change from a force
    # change in a way the joint sum cannot - see viz/pinch.py.
    cmd_w = exe_w = None
    if pinch_model is not None:
        cmd_w = pinch_model.widths_mm(r.cmd_hand)
        exe_w = pinch_model.widths_mm(r.exec_hand)

    rr.log(p("meta"), rr.TextDocument(
        f"# {r.name}\n\n"
        f"## {r.notes or '(no notes)'}\n\n"
        f"- checkpoint: `{r.meta['checkpoint']}`\n"
        f"- inference steps: {r.meta.get('num_inference_steps')}\n"
        f"- started: {r.meta.get('started_wall', '?')[11:19]}\n"
        f"- duration: {st['dur']:.1f} s, {st['steps']} executed steps\n"
        f"- stop reason: {st['stop']}\n\n"
        f"## grasp\n\n"
        f"- commanded closure: **{st['peak_cmd']:.1f} deg**\n"
        f"- executed closure: **{st['peak_exec']:.1f} deg**\n"
        f"- deficit: **{st['deficit']:.1f} deg** "
        f"({st['deficit_pct']:.1f} %), ratio {st['ratio']:.3f}\n"
        f"- reopening after peak: {st['reopen']:.1f} deg\n"
        f"- commanded z at closure: {st['z_at_close']:+.4f} m\n"
        f"- z rise after closure: {st['z_rise']:+.3f} m\n"
        + (f"- pinch width at closure: "
           f"**{exe_w[int(np.argmax(r.exec_close))]:.1f} mm** "
           f"(open hand ~208 mm)\n" if exe_w is not None else "")
        + (f"\n## progress head ({r.prog_split or 'unified'})\n\n"
           f"- max progress: **{st['max_prog']:.1%}** "
           f"({'⚠️ EXCEEDED 80%' if st['reached_80'] else 'stayed <80%'})\n"
           f"- final progress: **{st['final_prog']:.1%}**\n"
           if st.get('max_prog') is not None else ""),
        media_type=rr.MediaType.MARKDOWN), static=True)

    # --- commanded, on the executor's clock (step / 20 Hz) ---
    for i, t in enumerate(r.cmd_t):
        rr.set_time("rollout_time", duration=float(t))

        pose = r.cmd_pose[i]
        rr.log(p("world/commanded/ee"),
               rr.Points3D([pose[:3]], radii=0.008, colors=[(255, 140, 0)]))
        # Quaternion is xyzw on the wire, which is also Rerun's order.
        rr.log(p("world/commanded/ee_frame"),
               rr.Transform3D(translation=pose[:3], quaternion=pose[3:7],
                              axis_length=0.05))

        rr.log(p("cmd/pose/x"), rr.Scalars(float(pose[0])))
        rr.log(p("cmd/pose/y"), rr.Scalars(float(pose[1])))
        rr.log(p("cmd/pose/z"), rr.Scalars(float(pose[2])))
        rr.log(p("closure/commanded"), rr.Scalars(float(r.cmd_close[i])))
        if cmd_w is not None:
            rr.log(p("pinch/commanded"), rr.Scalars(float(cmd_w[i])))
        # The gap, on the same timeline as the two curves that form it, so a
        # growing lag is visible as it happens rather than only at the peak.
        rr.log(p("closure/error"), rr.Scalars(float(r.track_err[i])))
        for k, j in enumerate(GRASP_LOCAL, start=1):
            rr.log(p(f"hand/commanded/rj_dg_2_{k}"),
                   rr.Scalars(float(r.cmd_hand[i, j])))

    # The whole commanded path as one static line, so it is visible even when
    # the time cursor sits at the start.
    rr.log(p("world/commanded/path"),
           rr.LineStrips3D([r.cmd_pose[:, :3]], colors=[(255, 140, 0)],
                           radii=0.002), static=True)

    # The grasp point, static, so it is locatable without scrubbing.
    kc = int(np.argmax(r.cmd_close))
    rr.log(p("world/commanded/closure_point"),
           rr.Points3D([r.cmd_pose[kc, :3]], radii=0.014,
                       colors=[(214, 39, 40)]), static=True)

    # --- executed ---
    # /joint_states arrives at ~470 Hz, far denser than anything else here.
    # Logging every sample would make the time series unreadable and the file
    # large for no gain, so it is decimated to ~50 Hz.
    stride = max(1, len(r.js_t) // int(max(r.js_t[-1], 1) * 50))
    for i in range(0, len(r.js_t), stride):
        rr.set_time("rollout_time", duration=float(r.js_t[i]))
        rr.log(p("closure/executed"), rr.Scalars(float(r.exec_close[i])))
        if exe_w is not None:
            rr.log(p("pinch/executed"), rr.Scalars(float(exe_w[i])))
        for k, j in enumerate(GRASP_LOCAL, start=1):
            rr.log(p(f"hand/executed/rj_dg_2_{k}"),
                   rr.Scalars(float(r.exec_hand[i, j])))

    # --- point clouds, the encoder's actual input ---
    if r.has_pc and with_pc:
        pcs = r.point_clouds()
        for i, t in enumerate(r.pc_t):
            rr.set_time("rollout_time", duration=float(t))
            pts = np.asarray(pcs[i], dtype=np.float32)
            xyz = pts[:, :3]
            if pts.shape[1] >= 6:
                # deploy.py publishes r/g/b as separate float fields; the
                # builder's convention is 0..1, so scale only if it looks
                # like it rather than assuming either way.
                rgb = pts[:, 3:6]
                scale = 255.0 if float(np.nanmax(rgb)) <= 1.001 else 1.0
                cols = np.clip(rgb * scale, 0, 255).astype(np.uint8)
                rr.log(p("world/point_cloud"),
                       rr.Points3D(xyz, colors=cols, radii=0.002))
            else:
                rr.log(p("world/point_cloud"),
                       rr.Points3D(xyz, radii=0.002))

    # --- wrist camera ---
    if with_video:
        t0 = r.meta.get("started_unix")
        if t0 is None:
            print(f"  {r.name}: no started_unix in meta, cannot align video")
        else:
            times, frames, note = wrist.load_frames(
                r.path, t0, max_frames=max_frames)
            print(f"  {r.name}: wrist video - {note}")
            for t, img in zip(times, frames):
                rr.set_time("rollout_time", duration=float(t))
                rr.log(p("wrist/image"), rr.Image(img))

    # --- progress head (DINOv3 Spatio-Temporal) ---
    if r.has_progress:
        for idx_p, (t, prog) in enumerate(zip(r.prog_t, r.prog_val)):
            rr.set_time("rollout_time", duration=float(t))
            rr.log(p("progress/predicted"), rr.Scalars(float(prog)))
            if getattr(r, "prog_gt", None) is not None and idx_p < len(r.prog_gt):
                rr.log(p("progress/ground_truth"), rr.Scalars(float(r.prog_gt[idx_p])))
            rr.log(p("progress/threshold_80"), rr.Scalars(0.80))
            rr.log(p("progress/gate_90"), rr.Scalars(0.90))


def _panels(prefix="", name=None):
    return rrb.Horizontal(
        rrb.Vertical(
            rrb.Spatial3DView(origin=f"{prefix}/world",
                              name="EE path + point cloud"),
            # The camera sits next to the 3D view rather than in the plot
            # column: both are "what the scene looked like", and the eye
            # compares them without crossing the window.
            rrb.Spatial2DView(origin=f"{prefix}/wrist",
                              name="wrist camera"),
            rrb.TextDocumentView(origin=f"{prefix}/meta", name="summary"),
            row_shares=[3, 2, 1],
        ),
        rrb.Vertical(
            # The one plot that answers the deficit question: both curves on
            # one axis, same timeline.
            rrb.TimeSeriesView(origin=f"{prefix}/closure",
                               name="closure: commanded vs executed"),
            # Progress head prediction signal with 80% and 90% threshold lines
            rrb.TimeSeriesView(origin=f"{prefix}/progress",
                               name="progress head (DINOv3)"),
            # Its own view, not merged into closure: millimetres and degrees
            # on one axis would make both unreadable.
            rrb.TimeSeriesView(origin=f"{prefix}/pinch",
                               name="pinch width (mm)"),
            rrb.TimeSeriesView(origin=f"{prefix}/cmd/pose",
                               name="commanded EE position"),
            rrb.TimeSeriesView(origin=f"{prefix}/hand",
                               name="finger 2 joints"),
        ),
        column_shares=[1, 1],
        name=name,
    )


def blueprint(entries=None):
    """One tab per rollout when several are given - the closest Rerun gets to
    clicking through episodes, since it has one timeline per recording."""
    if not entries:
        return rrb.Blueprint(_panels())
    return rrb.Blueprint(
        rrb.Tabs(*[_panels(f"/{ns}", name=label) for ns, label in entries]),
        collapse_panels=True)