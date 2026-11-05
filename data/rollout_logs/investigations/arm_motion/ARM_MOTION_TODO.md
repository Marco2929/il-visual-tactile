# Arm motion: why the arm lags and stutters, and what is left to do

Scope: the UR10e arm's execution of the policy's cartesian stream, as
observed on rollouts 029-050 (`epoch=0040-val_action_mse=9.478029`). The hand
is covered in `data/outputs/ANALYSIS_TODO.md`; nothing here touches it.

All numbers below are measured, and the script that produced each one is
named so it can be re-run. Scripts live in `data/rollout_logs/`, run inside
`ros2_dev` with `/ros2_ws/venv/bin/python`. Every one needing forward
kinematics takes
`--urdf /ros2_ws/install/moveit_ur_hand_config/share/moveit_ur_hand_config/urdf/tesollo_hand_ur10e_flange_workcell.urdf`.

| script | what it answers |
|---|---|
| `damping_check.py` | is slowness damping or a limit? speed vs tracking error |
| `startup_phase.py` | the opening transient, per 100 ms |
| `pump_check.py` | which stage adds unevenness: policy / target_frame / executed |
| `demo_smoothness.py` | are the demos as jerky as the policy? |
| `chunk_density.py` | does chunk merging create the jerk? |
| `seam_vs_speed.py` | chunk-boundary jumps, speed-independent |

## What is already fixed

### Approach ramp (DONE, confirmed good in handling)

`control_ur_tesollo_cartesian.py`: `APPROACH_SPEED_MPS = 0.15`,
`_approach_limit()`.

The first commanded pose sits 523-665 mm from wherever the arm was parked,
almost all of it in y (`startup_phase.py`, r029/033/034/035). The controller
saw half a metre of "tracking error" and leaned on it with full gain:
439 mm/s and 13336 mm/s^2 in r034. `MAX_POSITION_STEP = 0.1` is no defence,
because 0.1 m per tick at 20 Hz permits 2 m/s.

This was never a tuning problem - it is a 0.5 m approach move disguised as a
control error, which is why no `pd_gains`/`iterations` change touched it.
Slew-rate limiting the target fixed it: 58 mm/s and ~1400 mm/s^2 in r035,
and the ramp saturates at its limit (141 mm/s measured in the log), so the
controller is no longer the thing deciding the approach speed.

The ramp defaults to ON and is only disarmed by arriving within 15 mm, so a
missed `/policy_gate` or a node restart mid-run cannot bypass it. Missing TF
now holds instead of publishing an unramped target.

### Controller gain, partially (`p: 1.0 -> 2.0`)

`cartesian_controllers_universal_robots/config/controller_manager.yaml`.
Note this is the file that is actually loaded: `robot.launch.py` hard-codes
it and does **not** declare a `controllers_file` argument, so the
`'controllers_file': cartesian_controller_config` passed by
`ur_tesollo_cartesian.launch.py` is silently ignored. The file it points at,
`il_main/config/cartesian_motion_controller.yaml`, is dead: it sets
`max_trans_vel: 0.05` and `end_effector_link: right_inspire_base_link`, a
link with 0 occurrences in the Tesollo URDF. See TODO 4.

Slowness is velocity damping, not a limit - `ForwardDynamicsSolver` ends each
iteration with `m_current_velocities *= 0.9`, and executed speed is
proportional to tracking error at ~0.8 1/s with no ceiling up to 150 mm
(`damping_check.py`, 325k samples over 8 rollouts). `p: 2.0` brought the
steady-state lag from 18.8 mm to 10-16 mm and the speed from 13.8 to
~28 mm/s.

`iterations` was tried and reverted to 10. It reads like a convergence
setting but the loop closes only ~0.2 % of the error per cycle, so it acts as
a linear speed multiplier whose cost is linear too (500 Hz x N inertia
inversions) and which also weakens the damping: 15 doubled the speed but
pushed executed acceleration p95 from 3416 to 5280 mm/s^2, above the
commanded value. Both the reasoning and the measurements are in the YAML
comments.

### RTC execution horizon (`16 -> 18`)

`third_party/ManiFlow_Policy/ManiFlow/deploy.py`.

Chunks carry 26 actions and overlap by 17, so 18 binds the whole overlap.
Measured with `seam_vs_speed.py`:

| horizon | seam residual | within chunk | ratio |
|---|---|---|---|
| 8 | 22.4 mm | 4.5 mm | 4.3x |
| 16 | 19.0 mm | 7.3 mm | 2.6x |
| **18** | **15.7 mm** | 6.2 mm | **2.5x** |

8 was tried on the theory that the horizon should match the arrival rate
(8 steps). That is backwards: `get_prefix_weights` goes ones -> decay ->
ZERO, so a horizon shorter than the overlap leaves the weight at zero exactly
where the next chunk takes over. The binding has to reach ACROSS the seam.

## TODO

### 1. Chunk merging is the stutter - reduce overlap density

**Cause.** The roughness is created by composing chunks, not by the policy
and not by the controller. Speed CV (std/mean of TCP speed, steady state
t > 3 s):

| | CV | source |
|---|---|---|
| demos, 160 episodes over 4 collections | **0.31** | `demo_smoothness.py` |
| within a single chunk | **0.40** | `chunk_density.py` |
| merged commanded stream | **0.93** | `chunk_density.py` |
| `/target_frame` (after Butterworth + clamp) | 0.70 | `pump_check.py` |
| executed (FK of joint states) | 0.40 | `pump_check.py` |

A single chunk is nearly as smooth as a demo, so the policy learned what it
was shown. The merge doubles the unevenness. The controller only ever
reduces it (0.70 -> 0.40, ratio 0.58), so it is not the source - an earlier
guess that the controller "pumps" the speed was wrong and is disproved by
that ratio.

Mechanism, from `chunk_density.py` on r048/049/050:

* 2 chunks cover a typical step, up to 4
* the winning chunk changes on **8 % of steps** (~1.6 per second at 20 Hz)
* two chunks disagree about the same step by **21.6 mm median**, p95 up to
  124 mm - against a 4.6 mm median local step, i.e. 5x a normal step

So every owner change can jump ~21 mm for no reason in the motion itself.
RTC is supposed to blend this away and partly does (seam residual 15.7 mm at
horizon 18), but 21.6 mm of raw disagreement is too much to hide.

The refill rule is `horizon = delay_steps + min_execution_horizon` with
`min_execution_horizon=20` (deploy.py, call site near the bottom), which with
26-action chunks gives a new chunk every ~8 steps.

**Solution.** Lower `min_execution_horizon` from 20 to ~12. Chunks then
arrive roughly every 14 steps instead of 8, overlap drops from 2-3 to ~2, and
there are fewer owner changes per second.

**Expected effect.** Merged-stream CV should fall from 0.93 toward the 0.40
of a single chunk; owner changes from 8 % toward ~4 %. Commanded acceleration
p95 should drop from ~8600-9000.

**Risk.** The buffer runs closer to empty. Current margin is comfortable -
`starved=0.0%` in steady state in every recent log, with only `starved=3
(9.7%)` in the first second while the buffer fills - but this is the
parameter that spends it. Watch the `[EXEC] ... starved=` line.

**Verify.** `chunk_density.py` and `pump_check.py` on 2-3 new rollouts.
Single-chunk CV is the floor: if merged CV reaches ~0.45 the mechanism is
fully understood and the remainder is the policy's own variation.

### 2. Chunk disagreement at the source (deeper, needs training)

**Cause.** 21.6 mm median disagreement between two inferences 8 steps apart
is large on its own. `n_obs_steps=2` gives the policy very little context, so
near-identical inputs can produce genuinely different solutions - the
multimodality already documented in `ANALYSIS_TODO.md` (`measure_sample_std.py`).
The worst seams cluster at chunk 2-3, t~1.6 s, with `stale=0`, which is where
context is thinnest.

**Solution.** Not a deployment parameter. Candidates: more observation steps,
or a consistency/smoothness term across overlapping predictions.

**Expected effect.** Unknown. This is the ceiling on TODO 1: as long as two
chunks disagree by 21 mm, no scheduling change removes all of the stutter.

**Do not start here** - TODO 1 is a one-line parameter and will show how much
of the gap is pure scheduling.

### 3. Speed is still ~40 % of commanded

**Cause.** Policy commands 120 mm/s median, the arm executes 39 mm/s
(`pump_check.py`). The demos ran at 84 mm/s, so the policy asks for 44 % MORE
than was demonstrated while the arm delivers less than half of it. The
proportional law `v ~ 0.8 * error` means a structural lag is the equilibrium,
not a transient: there is no integrator, so the error never closes.

**Solution, in order of preference.** Raising `p` again is the obvious move
but costs smoothness: `p: 2.0` already took commanded-to-executed
acceleration p95 from 3416 to 8534, i.e. the controller stopped filtering the
seams. Fix TODO 1 first - with a smoother command the same gain buys more
speed at the same jerk. Only then consider `p: 2.5-3.0`, or
`solver.forward_dynamics.link_mass` (default 0.1, never set in our YAML),
which scales acceleration without touching the error term.

**Expected effect.** Model in the deleted `solver_model.py` (conclusions kept
in the YAML comments) predicted gain scaling linearly in both `p` and
1/`link_mass`; the `p: 1.0 -> 2.0` step confirmed it at 1.85x measured.

**Open question.** Whether tracking the demo speed of 84 mm/s is even
desirable, or whether the policy over-commanding by 44 % is itself the thing
to fix. Not investigated.

### 4. Dead config that looks alive (hygiene, do before anyone else tunes)

`il_main/launch/ur_tesollo_cartesian.launch.py` passes `controllers_file`,
but `cartesian_controllers_universal_robots/launch/robot.launch.py` never
declares that argument and hard-codes its own
`config/controller_manager.yaml`. Unknown launch arguments are dropped
silently, so `il_main/config/cartesian_motion_controller.yaml` has no effect.

That file sets `max_trans_vel: 0.05` (5 cm/s) and
`end_effector_link: "right_inspire_base_link"`, which does not exist in the
Tesollo URDF - 0 occurrences, vs 6 for `tool0`. Confirmed inactive by
measurement, not just by reading: executed speed reaches 157 mm/s p95 and
spikes far above, which a 50 mm/s cap would make impossible
(`damping_check.py`).

It cost an hour of this investigation: the file reads as authoritative and I
took it as such until the numbers contradicted it.

**Solution.** Either declare and honour `controllers_file` in
`robot.launch.py`, or delete `il_main/config/cartesian_motion_controller.yaml`
and point the launch file at the real one. Prefer the second - one config,
no ambiguity.

**Expected effect.** No behaviour change. Prevents the next person tuning a
file that does nothing.

### 5. Smoother cutoff - rejected for now, with the reason

`SMOOTHER_CUTOFF_HZ = 8.0` at 20 Hz is 0.8x Nyquist and attenuates a 32 mm
jump to 26.6 mm, i.e. 17 %. 4 Hz would halve the jumps (13.4 mm) while the
median step only falls 4.8 -> 3.6 mm, so it mostly hits the outliers.

Not done, because the commanded stream never actually stops: the longest run
of sub-1 mm commanded steps across 784 steady-state steps is a **single
step** (50 ms), and there are zero dwells of 3+ steps. There is no stop-start
in the command for a low-pass to smooth, so it would buy 75-100 ms of phase
lag against a problem it does not solve. Revisit only if TODO 1 and 2 leave
residual high-frequency jitter.

Also worth knowing: `/target_frame` p95 speed (300 mm/s in r048) exceeds the
policy's own p95 (195 mm/s), so the current filter overshoots at step edges
rather than merely smoothing.

## Instrumentation added

`log_rollout.py` now logs `/target_frame` (`target_frame.jsonl`, plus `tf_t`,
`tf_pose`, `tf_stamp` in the npz; `viz/data.py` exposes `r.tf_t`, `r.tf_pose`,
`r.has_target`, defaulting to empty for older rollouts). Before this the only
logged signals were the policy's raw actions and the measured joints, so
everything the executor does in between had to be re-implemented to be
guessed at. This is what made TODO 1 decidable: without it, "the controller
pumps the speed" and "the command is already uneven" look identical.

Still not logged: the controller's own `publish_state_feedback` output, which
would separate IK solver error from tracking error.