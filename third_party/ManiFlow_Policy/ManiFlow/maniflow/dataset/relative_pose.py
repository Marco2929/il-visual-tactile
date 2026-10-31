"""Pose maths for base-relative EE actions.

Pure functions over numpy arrays. Deliberately imports nothing from the
dataset or config layer so that deploy-side code can use the exact same
implementation to decode what the dataset encoded - two copies of this maths
that drift apart would be very hard to debug from a rollout.

CONVENTIONS
-----------
* Quaternions are (x, y, z, w) - the layout the builder writes (see
  data/dataset_builder/processing/proprioception.py).
* Rotations are passed around as 3x3 matrices internally and as the
  continuous 6D representation (first two columns, column-major) at the
  network boundary.
* All functions are batched over leading dimensions and never mutate their
  inputs.

WHY 6D RATHER THAN QUATERNIONS AT THE NETWORK BOUNDARY
------------------------------------------------------
q and -q are the same rotation, so a quaternion target is a two-valued
regression problem: the network is punished for predicting the equally
correct opposite sign. The 6D representation is a continuous, unique
encoding of SO(3) - every rotation has exactly one 6D vector, and every 6D
vector near the manifold maps back to a valid rotation after
Gram-Schmidt. Zhou et al., "On the Continuity of Rotation Representations in
Neural Networks" (CVPR 2019).

WHY BASE-RELATIVE AT ALL
------------------------
The absolute pose tells the policy where the arm is in the cell, which
correlates with the pallet position and therefore with the answer. Encoding
the chunk relative to the last OBSERVED pose removes that: the policy sees
"where do I go from here", which is the same question regardless of which
pallet it is standing at, and the spatial part has to come from the image.
"""

import numpy as np

# Layout of the 15-dim state/action vector written by the builder:
# ee xyz (3) + ee quat xyzw (4) + pinch hand joints (8).
POS_SLICE = slice(0, 3)
QUAT_SLICE = slice(3, 7)
HAND_SLICE = slice(7, 15)
# After relativisation the quaternion becomes a 6D rotation, so the vector
# grows by two: 3 + 6 + 8 = 17. Task shape_meta must match.
RELATIVE_DIM = 17

_EPS = 1e-8


def quat_to_matrix(q):
    """(..., 4) quaternion xyzw -> (..., 3, 3) rotation matrix.

    Normalises first: the builder writes quaternions straight out of FK, and
    a norm that is off by 1e-6 would otherwise leak into a matrix whose
    determinant is not quite 1 and fail the orthonormality assert.
    """
    q = np.asarray(q, dtype=np.float64)
    q = q / np.maximum(np.linalg.norm(q, axis=-1, keepdims=True), _EPS)
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]

    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z

    R = np.empty(q.shape[:-1] + (3, 3), dtype=np.float64)
    R[..., 0, 0] = 1.0 - 2.0 * (yy + zz)
    R[..., 0, 1] = 2.0 * (xy - wz)
    R[..., 0, 2] = 2.0 * (xz + wy)
    R[..., 1, 0] = 2.0 * (xy + wz)
    R[..., 1, 1] = 1.0 - 2.0 * (xx + zz)
    R[..., 1, 2] = 2.0 * (yz - wx)
    R[..., 2, 0] = 2.0 * (xz - wy)
    R[..., 2, 1] = 2.0 * (yz + wx)
    R[..., 2, 2] = 1.0 - 2.0 * (xx + yy)
    return R


def matrix_to_quat(R):
    """(..., 3, 3) -> (..., 4) quaternion xyzw.

    Branches on the largest diagonal term (Shepperd's method) instead of the
    short w-only formula, which loses precision when w approaches 0.
    Sign is canonicalised to w >= 0 so round-trips are reproducible.
    """
    R = np.asarray(R, dtype=np.float64)
    batch = R.shape[:-2]
    Rf = R.reshape(-1, 3, 3)
    n = Rf.shape[0]
    q = np.empty((n, 4), dtype=np.float64)

    m00, m11, m22 = Rf[:, 0, 0], Rf[:, 1, 1], Rf[:, 2, 2]
    trace = m00 + m11 + m22

    big_w = trace > 0.0
    if np.any(big_w):
        i = np.flatnonzero(big_w)
        s = np.sqrt(trace[i] + 1.0) * 2.0
        q[i, 3] = 0.25 * s
        q[i, 0] = (Rf[i, 2, 1] - Rf[i, 1, 2]) / s
        q[i, 1] = (Rf[i, 0, 2] - Rf[i, 2, 0]) / s
        q[i, 2] = (Rf[i, 1, 0] - Rf[i, 0, 1]) / s

    rest = ~big_w
    if np.any(rest):
        idx = np.flatnonzero(rest)
        diag = np.stack([m00[idx], m11[idx], m22[idx]], axis=-1)
        pick = np.argmax(diag, axis=-1)

        for axis in range(3):
            sel = idx[pick == axis]
            if sel.size == 0:
                continue
            a, b = (axis + 1) % 3, (axis + 2) % 3
            s = np.sqrt(
                1.0 + Rf[sel, axis, axis] - Rf[sel, a, a] - Rf[sel, b, b]
            ) * 2.0
            q[sel, axis] = 0.25 * s
            q[sel, a] = (Rf[sel, a, axis] + Rf[sel, axis, a]) / s
            q[sel, b] = (Rf[sel, b, axis] + Rf[sel, axis, b]) / s
            q[sel, 3] = (Rf[sel, b, a] - Rf[sel, a, b]) / s

    q /= np.maximum(np.linalg.norm(q, axis=-1, keepdims=True), _EPS)
    # q and -q are the same rotation; fix the sign so tests and logs are
    # comparable.
    q = np.where(q[..., 3:4] < 0.0, -q, q)
    return q.reshape(batch + (4,))


def matrix_to_rot6d(R):
    """(..., 3, 3) -> (..., 6): the first two COLUMNS, stacked.

    Columns rather than rows because rot6d_to_matrix rebuilds columns; using
    rows here and columns there is a silent transpose that still passes a
    round-trip test on symmetric rotations only.
    """
    R = np.asarray(R, dtype=np.float64)
    r1 = R[..., :, 0]
    r2 = R[..., :, 1]
    return np.concatenate([r1, r2], axis=-1)


def rot6d_to_matrix(r6):
    """(..., 6) -> (..., 3, 3) via Gram-Schmidt.

    The network output is not orthonormal, so this projects it back onto
    SO(3): normalise the first column, remove its component from the second,
    normalise that, and take the cross product for the third. The result is
    a valid rotation for ANY finite input, which is what makes the 6D
    representation safe to regress.
    """
    r6 = np.asarray(r6, dtype=np.float64)
    a1 = r6[..., 0:3]
    a2 = r6[..., 3:6]

    b1 = a1 / np.maximum(np.linalg.norm(a1, axis=-1, keepdims=True), _EPS)
    # Project a2 onto the plane orthogonal to b1.
    a2_proj = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = a2_proj / np.maximum(
        np.linalg.norm(a2_proj, axis=-1, keepdims=True), _EPS)
    b3 = np.cross(b1, b2)

    return np.stack([b1, b2, b3], axis=-1)


def to_relative(T, R, T_base, R_base):
    """Express poses in the base frame.

        T_rel = R_base^T @ (T - T_base)
        R_rel = R_base^T @ R

    Args:
        T: (..., 3) positions.
        R: (..., 3, 3) rotations.
        T_base: (3,) or broadcastable base position.
        R_base: (3, 3) or broadcastable base rotation.

    Returns:
        (T_rel, R_rel), same leading shape as the inputs.
    """
    T = np.asarray(T, dtype=np.float64)
    R = np.asarray(R, dtype=np.float64)
    T_base = np.asarray(T_base, dtype=np.float64)
    R_base = np.asarray(R_base, dtype=np.float64)

    R_base_T = np.swapaxes(R_base, -1, -2)
    T_rel = np.einsum("...ij,...j->...i", R_base_T, T - T_base)
    R_rel = np.einsum("...ij,...jk->...ik", R_base_T, R)
    return T_rel, R_rel


def from_relative(T_rel, R_rel, T_base, R_base):
    """Inverse of `to_relative`.

        T = T_base + R_base @ T_rel
        R = R_base @ R_rel
    """
    T_rel = np.asarray(T_rel, dtype=np.float64)
    R_rel = np.asarray(R_rel, dtype=np.float64)
    T_base = np.asarray(T_base, dtype=np.float64)
    R_base = np.asarray(R_base, dtype=np.float64)

    T = T_base + np.einsum("...ij,...j->...i", R_base, T_rel)
    R = np.einsum("...ij,...jk->...ik", R_base, R_rel)
    return T, R


# ---------------------------------------------------------------------------
# Invariant checks. Cheap enough to leave on during training; see
# RelativeRealDexDataset(assert_invariants=...).
# ---------------------------------------------------------------------------

def assert_identity_pose(T_rel, R_rel, atol=1e-6, what="base frame"):
    """The base pose expressed in its own frame must be (0, I).

    This is the single most useful check in the whole pipeline: it fails if
    the base pose used for encoding is not the pose it claims to be, which
    is exactly the mistake that silently degrades a relative-action policy
    (encoding against a commanded target instead of the observed pose).
    """
    T_rel = np.asarray(T_rel)
    R_rel = np.asarray(R_rel)
    if not np.allclose(T_rel, 0.0, atol=atol):
        raise AssertionError(
            f"{what}: relative position is not zero "
            f"(max |T_rel| = {np.abs(T_rel).max():.3e} > {atol:.1e})"
        )
    if not np.allclose(R_rel, np.eye(3), atol=atol):
        raise AssertionError(
            f"{what}: relative rotation is not identity "
            f"(max |R_rel - I| = {np.abs(R_rel - np.eye(3)).max():.3e} "
            f"> {atol:.1e})"
        )


def assert_valid_rotation(R, atol=1e-5, what="rotation"):
    """R must be orthonormal with det = +1 (not -1, i.e. not a reflection)."""
    R = np.asarray(R, dtype=np.float64)
    if R.shape[-2:] != (3, 3):
        raise AssertionError(f"{what}: expected (..., 3, 3), got {R.shape}")

    should_be_I = np.einsum("...ij,...kj->...ik", R, R)
    err = np.abs(should_be_I - np.eye(3)).max()
    if err > atol:
        raise AssertionError(
            f"{what}: not orthonormal (max |R R^T - I| = {err:.3e})")

    det = np.linalg.det(R)
    if np.abs(det - 1.0).max() > atol:
        raise AssertionError(
            f"{what}: det != +1 (min {det.min():.6f}, max {det.max():.6f}) "
            f"- a determinant of -1 means the basis was mirrored somewhere"
        )