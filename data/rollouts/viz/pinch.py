#!/usr/bin/env python3
"""Pinch width from the hand URDF: thumb tip to index tip, in millimetres.

WHY THIS EXISTS: the closure metric sums the four index joints, and that sum
is not a grasp width. In rollout_010 the failing first attempt sums to 137.5
deg and the successful retry to 125.9 - the retry looks WEAKER while it is the
one that holds. Forward kinematics resolves it: the same sum can be a curled
fingertip pointing away from the thumb or a flat finger pressing against it.
Measured as a distance, the retry closes 5.5 mm further (42.4 -> 36.9 mm).

So closure and pinch width are kept side by side rather than one replacing
the other: their divergence is what distinguishes a posture change from a
force change.

CAVEAT: this is tip-to-tip, not a contact distance. With a flat finger the
foil sits proximal of the fingertip, so the real gap at the contact point is
smaller than the number reported here. It is a consistent relative measure,
not an absolute gap.
"""
import os
import pathlib

import numpy as np

from .data import HAND_JOINT_NAMES

THUMB_TIP = "rl_dg_1_tip"
INDEX_TIP = "rl_dg_2_tip"

# The same file lives in three places depending on where this runs: the
# installed share directory in the container, and two different source
# layouts (tesollo_hand in the container, delto_m_ros2 on the host). Guessing
# one of them is how this broke the first time, so all are tried and
# $DG5F_URDF overrides.
URDF_CANDIDATES = [
    "/ros2_ws/install/dg5f_description/share/dg5f_description/urdf/"
    "dg5f_right.urdf",
    "/ros2_ws/src/hands/tesollo_hand/dg5f_ros2/dg5f_description/urdf/"
    "dg5f_right.urdf",
    "/ros2_ws/src/hands/delto_m_ros2/dg5f_ros2/dg5f_description/urdf/"
    "dg5f_right.urdf",
]


def find_urdf(explicit=None):
    """-> path, or None. Checked in order: argument, env var, candidates.

    An explicit path that does not exist raises instead of falling through:
    silently using a different model than the one asked for would make a
    typo in --urdf look like a successful run.
    """
    if explicit and not pathlib.Path(explicit).is_file():
        raise FileNotFoundError(f"--urdf {explicit} does not exist")
    for cand in ([explicit, os.environ.get("DG5F_URDF")]
                 + URDF_CANDIDATES):
        if cand and pathlib.Path(cand).is_file():
            return pathlib.Path(cand)
    # Also try relative to this checkout, so it works outside the container.
    root = pathlib.Path(__file__).resolve().parents[3]
    for pat in ("src/hands/*/dg5f_ros2/dg5f_description/urdf/"
                "dg5f_right.urdf",):
        hits = sorted(root.glob(pat))
        if hits:
            return hits[0]
    return None


class PinchModel:
    """Thumb-index tip distance as a function of the 20 hand angles."""

    def __init__(self, urdf=None):
        import pinocchio as pin

        path = find_urdf(urdf)
        if path is None:
            raise FileNotFoundError(
                "dg5f_right.urdf not found - pass --urdf or set $DG5F_URDF. "
                f"Looked in: {URDF_CANDIDATES}")
        self.path = path
        self._pin = pin
        self.model = pin.buildModelFromUrdf(str(path))
        self.data = self.model.createData()

        # The log names (rj_dg_*) happen to match the URDF joint names here,
        # but the tip FRAMES use the rl_ prefix. Map by the finger/joint
        # suffix so a prefix change does not silently zero a joint.
        urdf_joints = [self.model.names[i]
                       for i in range(1, self.model.njoints)]
        self.qidx = {}
        for logname in HAND_JOINT_NAMES:
            suffix = logname.replace("rj_dg_", "")
            for jn in urdf_joints:
                if jn.replace("rj_dg_", "").replace("rl_dg_", "") == suffix:
                    jid = self.model.getJointId(jn)
                    self.qidx[logname] = self.model.joints[jid].idx_q
                    break
        missing = [n for n in HAND_JOINT_NAMES if n not in self.qidx]
        if missing:
            raise ValueError(f"unmapped hand joints: {missing}")

        self.f_thumb = self.model.getFrameId(THUMB_TIP)
        self.f_index = self.model.getFrameId(INDEX_TIP)
        self._q = pin.neutral(self.model)

    def width_mm(self, angles_deg):
        """20 hand angles in degrees -> tip distance in mm."""
        for name, a in zip(HAND_JOINT_NAMES, angles_deg):
            self._q[self.qidx[name]] = np.radians(a)
        self._pin.forwardKinematics(self.model, self.data, self._q)
        self._pin.updateFramePlacements(self.model, self.data)
        d = (self.data.oMf[self.f_thumb].translation
             - self.data.oMf[self.f_index].translation)
        return float(np.linalg.norm(d)) * 1000.0

    def widths_mm(self, angles_deg):
        """(T, 20) -> (T,) widths. ~0.1 ms per frame, so a whole rollout is
        cheap enough to compute eagerly."""
        return np.array([self.width_mm(a) for a in np.asarray(angles_deg)])


def try_load(urdf=None):
    """-> (model, note). Never raises: pinocchio is optional and the rest of
    the visualisation must still work without it."""
    try:
        m = PinchModel(urdf)
        return m, f"pinch width from {m.path.name}"
    except ImportError:
        return None, "pinch width skipped: pinocchio not installed"
    except Exception as e:
        return None, f"pinch width skipped: {e}"