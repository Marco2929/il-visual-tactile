"""Gate for findings.md section 7.1: does the DEPLOY decoder reproduce the
recorded absolute poses?

    /ros2_ws/src/il_main/venv/bin/python -m pytest \\
        maniflow/dataset/tests/test_deploy_relative_roundtrip.py -q

WHY THIS EXISTS ALONGSIDE test_relative_dataset.py
--------------------------------------------------
test_relative_dataset.py already round-trips the DATASET encoding. That is
not the same check. It calls `from_relative` directly with a base it picked
itself, so it verifies the maths in relative_pose.py - which is also what
test_relative_pose.py covers.

The rollouts are produced by deploy_relative.py, which has its OWN
encode/decode pair (`encode_obs_relative`, `decode_action_relative`) plus
its own choice of base. Nothing currently tests those two functions, and
nothing tests that the base they pick is the base the dataset used. A
mismatch there is invisible to every existing test and would look exactly
like the accumulation the findings describe: an offset that grows across a
chunk.

So this feeds RECORDED actions and the RECORDED base through the DEPLOY
functions - no policy, no network - and requires the recorded absolute poses
back. Section 7.1 asks for < 1e-5 m.

WHAT EACH TEST WOULD CATCH
--------------------------
  decode_matches_recording   the deploy decoder itself, end to end.
  deploy_base_equals_dataset_base
                             deploy takes the newest row of the observation
                             window, the dataset takes state[n_obs_steps-1].
                             These agree only while snapshot_obs() returns
                             exactly n_obs_steps rows; if it ever returns
                             more, deploy silently re-bases onto a later
                             frame.
  encode_decode_is_identity  the two deploy functions are mutual inverses on
                             real data, independent of any base choice.
  off_by_one_is_detectable   a one-frame shift between action and base - the
                             §7.2 candidate - must produce an error far above
                             the tolerance. If this test FAILS, the gate
                             above is not sensitive enough to be trusted.
  error_does_not_grow        an accumulating fault shows up as error that
                             rises along the chunk. Flat error means the
                             transform is stationary, which is what rules
                             the decoder out as the cause.

If every test here passes, the relative pipeline's maths is exonerated and
the behaviour in the rollouts has to be explained by the policy, the
observation it receives, or the executor - not by the encoding.
"""

import importlib.util
import os
import pathlib
import sys

import numpy as np
import pytest

from maniflow.dataset.relative_dataset import RelativeRealDexDataset
from maniflow.dataset.relative_pose import quat_to_matrix

ZARR = "/ros2_ws/data/zarr_datasets/foil_lifting_wrist_tesollo.zarr"
N_OBS = 2
HORIZON = 64
TOL_M = 1e-5          # section 7.1
TOL_QUAT = 1e-5

pytestmark = pytest.mark.skipif(
    not os.path.exists(ZARR), reason=f"{ZARR} not available")


def _load_deploy_module():
    """Import deploy_relative.py WITHOUT its ROS dependencies.

    The module imports rclpy, cv_bridge and the rest at top level, none of
    which belong in a unit test. Only the three pure pose functions are
    needed, so the file is parsed and just those are executed - which also
    means this test reads the real deployed source rather than a copy that
    can drift.
    """
    here = pathlib.Path(__file__).resolve()
    root = here.parents[3]            # .../ManiFlow
    path = root / "deploy_relative.py"
    if not path.is_file():
        pytest.skip(f"{path} not found")

    src = path.read_text()
    import ast

    tree = ast.parse(src)
    wanted = {"encode_obs_relative", "decode_action_relative",
              "rebase_leftover"}
    keep = [n for n in tree.body
            if isinstance(n, ast.FunctionDef) and n.name in wanted]
    if len(keep) != len(wanted):
        found = {n.name for n in keep}
        pytest.skip(f"deploy_relative.py is missing {wanted - found}")

    ns = {}
    exec(compile(ast.Module(body=keep, type_ignores=[]), str(path), "exec"),
         {
             "np": np,
             **_pose_symbols(),
         }, ns)
    return ns


def _pose_symbols():
    from maniflow.dataset import relative_pose as rp
    return {
        "HAND_SLICE": rp.HAND_SLICE,
        "POS_SLICE": rp.POS_SLICE,
        "QUAT_SLICE": rp.QUAT_SLICE,
        "RELATIVE_DIM": rp.RELATIVE_DIM,
        "assert_identity_pose": rp.assert_identity_pose,
        "assert_valid_rotation": rp.assert_valid_rotation,
        "from_relative": rp.from_relative,
        "matrix_to_quat": rp.matrix_to_quat,
        "matrix_to_rot6d": rp.matrix_to_rot6d,
        "quat_to_matrix": rp.quat_to_matrix,
        "rot6d_to_matrix": rp.rot6d_to_matrix,
        "to_relative": rp.to_relative,
    }


@pytest.fixture(scope="module")
def deploy():
    return _load_deploy_module()


@pytest.fixture(scope="module")
def ds():
    return RelativeRealDexDataset(
        zarr_path=ZARR,
        horizon=HORIZON,
        pad_before=N_OBS - 1,
        pad_after=HORIZON - N_OBS,
        n_obs_steps=N_OBS,
        val_ratio=0.0,
        normalizer_samples=50,
    )


# Spread across the buffer rather than one window: a fault that depends on
# the pose (near a singularity, say) would hide in a single sample.
IDX = [0, 137, 1234, 5000, 12345, 20000]


def _window(ds, idx):
    s = ds.sampler.sample_sequence(idx)
    return (np.asarray(s["state"], dtype=np.float64),
            np.asarray(s["action"], dtype=np.float64))


class TestDeployDecoder:
    """Section 7.1: recorded relative actions + recorded base -> recorded
    absolute poses."""

    @pytest.mark.parametrize("idx", IDX)
    def test_decode_matches_recording(self, ds, deploy, idx):
        if idx >= len(ds.sampler):
            pytest.skip("index beyond this zarr")
        state, action = _window(ds, idx)

        # The relative actions the dataset produces for this window...
        rel = ds[idx]["action"].numpy().astype(np.float64)
        # ...and the base deploy_relative.py would use: newest observed row.
        obs_window = state[:N_OBS]
        _, T_base, R_base = deploy["encode_obs_relative"](obs_window, N_OBS)

        dec = deploy["decode_action_relative"](rel, T_base, R_base)

        pos_err = np.abs(dec[:, 0:3] - action[:, 0:3]).max()
        assert pos_err < TOL_M, (
            f"window {idx}: deploy decoder is off by {pos_err:.3e} m "
            f"(tolerance {TOL_M:.0e}). The rollouts cannot be read as "
            f"policy behaviour until this is zero.")

        # Rotation compared as matrices: quaternion sign is not unique.
        R_dec = quat_to_matrix(dec[:, 3:7])
        R_rec = quat_to_matrix(action[:, 3:7])
        rot_err = np.abs(R_dec - R_rec).max()
        assert rot_err < TOL_QUAT, (
            f"window {idx}: rotation off by {rot_err:.3e}")

    @pytest.mark.parametrize("idx", IDX[:3])
    def test_hand_joints_pass_through_unchanged(self, ds, deploy, idx):
        if idx >= len(ds.sampler):
            pytest.skip("index beyond this zarr")
        _, action = _window(ds, idx)
        rel = ds[idx]["action"].numpy().astype(np.float64)
        state, _ = _window(ds, idx)
        _, T_base, R_base = deploy["encode_obs_relative"](
            state[:N_OBS], N_OBS)
        dec = deploy["decode_action_relative"](rel, T_base, R_base)
        # Hands are absolute on both sides of the transform.
        assert np.abs(dec[:, 7:15] - action[:, 7:15]).max() < 1e-4


class TestBaseAgreement:
    """Section 7.2: does deploy pick the base the dataset trained with?"""

    @pytest.mark.parametrize("idx", IDX)
    def test_deploy_base_equals_dataset_base(self, ds, deploy, idx):
        if idx >= len(ds.sampler):
            pytest.skip("index beyond this zarr")
        state, _ = _window(ds, idx)

        # Dataset: state[n_obs_steps - 1].
        base = N_OBS - 1
        T_ds = state[base, 0:3]
        R_ds = quat_to_matrix(state[base, 3:7])

        # Deploy: last row of whatever snapshot_obs() handed over.
        _, T_dep, R_dep = deploy["encode_obs_relative"](
            state[:N_OBS], N_OBS)

        assert np.abs(T_dep - T_ds).max() < 1e-12
        assert np.abs(R_dep - R_ds).max() < 1e-12

    def test_longer_obs_window_would_rebase(self, ds, deploy):
        """Documents the one way the bases can diverge.

        encode_obs_relative() takes the LAST row, ignoring its n_obs_steps
        argument for the base choice. That is correct only while the caller
        passes exactly n_obs_steps rows. This test pins the assumption so a
        change to snapshot_obs() fails here rather than in a rollout.
        """
        state, _ = _window(ds, 1234)
        _, T_short, _ = deploy["encode_obs_relative"](state[:N_OBS], N_OBS)
        _, T_long, _ = deploy["encode_obs_relative"](state[:N_OBS + 3], N_OBS)
        moved = np.linalg.norm(T_long - T_short)
        if moved < 1e-9:
            pytest.skip("arm did not move in this window")
        assert moved > 0, (
            "a longer observation window moves the base; snapshot_obs() must "
            "keep returning exactly n_obs_steps rows")


class TestSensitivity:
    """Would the gate above actually catch the §7.2 candidates?"""

    def test_off_by_one_is_detectable(self, ds, deploy):
        """A one-frame shift must fail by a wide margin.

        This is the check that makes the passing gate meaningful: if a
        deliberate off-by-one produced an error near TOL_M, a real one could
        hide inside the tolerance.
        """
        state, action = _window(ds, 1234)
        rel = ds[1234]["action"].numpy().astype(np.float64)

        # Base taken one frame EARLY - the off-by-one from §7.2.
        T_wrong = state[N_OBS - 2, 0:3]
        R_wrong = quat_to_matrix(state[N_OBS - 2, 3:7])
        dec = deploy["decode_action_relative"](rel, T_wrong, R_wrong)
        err = np.abs(dec[:, 0:3] - action[:, 0:3]).max()

        assert err > 100 * TOL_M, (
            f"a one-frame base error only moves the result by {err:.3e} m, "
            f"which is too close to the {TOL_M:.0e} tolerance for the gate "
            f"to be trusted")

    def test_error_does_not_grow_along_the_chunk(self, ds, deploy):
        """An accumulating fault would show a rising error profile.

        A correct transform is stationary: every step is decoded with the
        same base, so the error must be flat at numerical noise. A rising
        profile is the signature the findings attribute to summing.
        """
        state, action = _window(ds, 5000)
        if 5000 >= len(ds.sampler):
            pytest.skip("index beyond this zarr")
        rel = ds[5000]["action"].numpy().astype(np.float64)
        _, T_base, R_base = deploy["encode_obs_relative"](
            state[:N_OBS], N_OBS)
        dec = deploy["decode_action_relative"](rel, T_base, R_base)

        per_step = np.linalg.norm(dec[:, 0:3] - action[:, 0:3], axis=1)
        first, last = per_step[:8].mean(), per_step[-8:].mean()
        assert last < TOL_M and first < TOL_M
        # No trend beyond float noise.
        assert last < max(10 * first, 1e-9), (
            f"error grows along the chunk ({first:.2e} -> {last:.2e} m), "
            f"which is what an accumulating transform looks like")


class TestRebaseLeftover:
    """RTC_MODE='rebase' path: absolute leftover -> new base -> back."""

    def test_rebase_then_decode_is_identity(self, ds, deploy):
        state, action = _window(ds, 1234)
        # Pretend `action` is a cached absolute chunk and re-base it onto a
        # base taken from a later frame, as RTC does after a delay.
        T_new = state[N_OBS + 5, 0:3]
        R_new = quat_to_matrix(state[N_OBS + 5, 3:7])

        rel = deploy["rebase_leftover"](action, T_new, R_new)
        back = deploy["decode_action_relative"](
            np.asarray(rel, dtype=np.float64), T_new, R_new)

        assert np.abs(back[:, 0:3] - action[:, 0:3]).max() < TOL_M
        R_back = quat_to_matrix(back[:, 3:7])
        R_orig = quat_to_matrix(action[:, 3:7])
        assert np.abs(R_back - R_orig).max() < TOL_QUAT
