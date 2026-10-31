#!/usr/bin/env python3
"""Shared progress-head handling for the deploy scripts.

The progress head is an OPTIONAL output of the policy: a scalar in [0, 1]
regressed against the PCHIP curve from
dataset_builder/processing/progress.py. When present it can be used purely
for monitoring, or as a termination condition.

Why this lives in its own module
--------------------------------
deploy.py and deploy_relative.py both need it and had drifted into two
copies of the same three methods. More importantly the termination path has
a failure mode that is worth implementing exactly once:

  1. The old gate was `if 'progress_score' in result`. That silently covers
     "checkpoint has no head", "key was renamed" and "head returned
     nothing" - a stop condition that fails open and says nothing. Both
     deploy scripts passed progress_threshold=0.9 to checkpoints whose
     policy class has no progress head at all, and printed no warning.
     So `use_progress` is now declared by the caller and VALIDATED against
     a real policy output; a mismatch in either direction is loud.

  2. A single sample over threshold used to end the episode. The head is
     MSE-trained against a curve clipped to [0, 1], so it is noisy exactly
     where it matters. One spike at 0.91 during the approach would stop a
     run that had not grasped. Hence `patience` consecutive crossings.

  3. /policy_done is what actually stops the robot (the controllers'
     _policy_done_cb). It was a volatile depth-10 publisher written
     immediately before executor.shutdown() + destroy_node(), so the
     message could be dropped and the arm would keep draining the chunk
     that had just been published. Hence TRANSIENT_LOCAL durability plus
     an explicit flush() before teardown.

Relative-pose caveat
--------------------
The progress LABELS are derived from absolute EE height (compute_progress's
hardcoded z_grasp=0.10) plus finger contact. deploy_relative.py zeroes the
newest pose by construction, so absolute z is not recoverable from
agent_pos there. That is fine for a vision-only head - the point cloud is
still in base_link - but a head conditioned on the fused global feature
(which includes agent_pos) cannot see the quantity its labels were defined
from. If the head returns and touches agent_pos, pass T_base in as an extra
input; deploy_relative.py already has it in the loop for decoding.
"""
import time

import numpy as np
import torch
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from std_msgs.msg import Bool, Float64MultiArray
from visualization_msgs.msg import Marker


# The key the policy is expected to put its progress scalar under. Kept as a
# constant so the validation error can name it.
PROGRESS_KEY = "progress_score"


def latched_qos(depth: int = 1) -> QoSProfile:
    """TRANSIENT_LOCAL QoS for one-shot state topics.

    /policy_done and /policy_status are edge-triggered and published once,
    at the moment the run ends. With VOLATILE durability a subscriber that
    is momentarily not matched - or a publisher destroyed right after the
    write - loses the message entirely. TRANSIENT_LOCAL keeps the last
    sample available to late joiners, which also lets log_rollout.py see
    the final status if it attaches after the fact.

    Note this is not a substitute for flush(): durability only serves
    subscribers while the publisher is still alive.
    """
    return QoSProfile(
        depth=depth,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.TRANSIENT_LOCAL,
        history=HistoryPolicy.KEEP_LAST,
    )


def extract_progress(result) -> float:
    """Pull the progress scalar out of a predict_action() result dict.

    Accepts torch tensors, numpy arrays and plain floats, and takes the
    first batch element. Returns None if the key is absent, so the caller
    can distinguish "no head" from "head returned 0.0".
    """
    if not isinstance(result, dict) or PROGRESS_KEY not in result:
        return None

    value = result[PROGRESS_KEY]
    if isinstance(value, torch.Tensor):
        value = value.detach().float().cpu().numpy()
    value = np.asarray(value, dtype=np.float64).reshape(-1)
    if value.size == 0:
        return None
    return float(value[0])


class ProgressTracker:
    """Owns the progress topics, the debounce latch and the done signal.

    Two orthogonal switches, deliberately not collapsed into one:

      use_progress        does the head run / is its output expected at all
      progress_threshold  does that output stop anything (None = monitor only)

    Monitor-only is the right default for a new checkpoint: you get
    /policy_progress and the RViz marker, and nothing can terminate the run
    because of a miscalibrated head.
    """

    def __init__(self, node, use_progress: bool = False,
                 progress_threshold: float = None, patience: int = 3,
                 marker_frame_id: str = "base_link"):
        self.node = node
        self.use_progress = bool(use_progress)
        self.progress_threshold = progress_threshold
        self.patience = max(1, int(patience))
        self.marker_frame_id = marker_frame_id

        self._consecutive = 0
        self._latched = False
        self._last_value = None
        self._validated = False

        # Progress + marker stay VOLATILE: they are continuous telemetry and
        # a late joiner wants the current value, not a stale one. done is
        # latched - it is a one-shot edge that must not be lost.
        self.progress_pub = node.create_publisher(
            Float64MultiArray, '/policy_progress', 10)
        self.marker_pub = node.create_publisher(
            Marker, '/policy_progress_marker', 10)
        self.done_pub = node.create_publisher(
            Bool, '/policy_done', latched_qos())

    # ------------------------------------------------------------------
    # Startup validation
    # ------------------------------------------------------------------

    def validate(self, sample_result):
        """Compare the declared config against a real policy output.

        Called once with the result of a warmup predict_action(). This is
        the same philosophy as _check_wrist_shape_meta: compare intent to
        reality at startup, where it is cheap and loud, instead of letting
        a mismatch degrade into silence.
        """
        self._validated = True
        value = extract_progress(sample_result)
        log = self.node.get_logger()

        if self.use_progress and value is None:
            raise ValueError(
                f"use_progress=True, but the policy produced no "
                f"'{PROGRESS_KEY}' key. This checkpoint has no progress "
                f"head (only the force/pointcloud policy wires up "
                f"ProgressHead). Set use_progress=False, or load a "
                f"checkpoint whose predict_action() returns "
                f"'{PROGRESS_KEY}'.")

        if not self.use_progress:
            if value is not None:
                log.warn(
                    f"The policy returns '{PROGRESS_KEY}' but "
                    f"use_progress=False, so it is ignored and nothing is "
                    f"published. Set use_progress=True to monitor it.")
            else:
                log.info("Progress head: not present, not expected.")
            return

        if not np.isfinite(value):
            raise ValueError(
                f"progress head returned a non-finite value ({value}) "
                f"during warmup; refusing to use it as a stop condition.")

        if not (-0.05 <= value <= 1.05):
            log.warn(
                f"progress head returned {value:.3f} during warmup, which "
                f"is outside the [0, 1] range the labels were clipped to. "
                f"The head may be untrained or mis-normalised - treat the "
                f"threshold with suspicion.")

        if self.progress_threshold is None:
            log.info(
                f"Progress head: ACTIVE, monitor-only (warmup value "
                f"{value:.3f}). No threshold set, the run will not "
                f"self-terminate.")
        else:
            log.info(
                f"Progress head: ACTIVE, will stop at >= "
                f"{self.progress_threshold:.2f} for {self.patience} "
                f"consecutive inferences (warmup value {value:.3f}).")

    # ------------------------------------------------------------------
    # Per-inference update
    # ------------------------------------------------------------------

    def update(self, result) -> bool:
        """Publish progress and report whether the task should stop.

        Returns True only once the threshold has held for `patience`
        consecutive inferences. Call this BEFORE publishing the action
        chunk: a 'done' that arrives after the chunk still lets the
        executor drain a full chunk of motion.

        Note the sampling rate. This runs once per inference, and inference
        is demand-driven (fired when the executor's buffer falls below the
        horizon), so progress is sampled every ~8 control steps and
        irregularly - not at control rate. Fine for termination, but
        /policy_progress should not be read as a 20 Hz signal.
        """
        if not self.use_progress:
            return False

        if not self._validated:
            self.node.get_logger().warn(
                "ProgressTracker.update() called before validate(); "
                "a checkpoint without a progress head would be silently "
                "ignored here.", once=True)

        value = extract_progress(result)
        if value is None:
            self.node.get_logger().error(
                f"'{PROGRESS_KEY}' disappeared from the policy output after "
                f"validation succeeded. Not terminating on a signal that "
                f"comes and goes.", throttle_duration_sec=5.0)
            return False

        self._last_value = value
        self.publish(value)

        if self._latched:
            return True
        if self.progress_threshold is None:
            return False

        if value >= self.progress_threshold:
            self._consecutive += 1
        else:
            if self._consecutive:
                self.node.get_logger().info(
                    f"progress fell back below threshold "
                    f"({value:.3f} < {self.progress_threshold:.2f}) after "
                    f"{self._consecutive}/{self.patience} crossing(s); "
                    f"debounce reset.")
            self._consecutive = 0

        if self._consecutive < self.patience:
            return False

        self._latched = True
        self.node.get_logger().info(
            f"[DONE] progress {value:.3f} >= {self.progress_threshold:.2f} "
            f"for {self.patience} consecutive inferences. Stopping.")
        self.done_pub.publish(Bool(data=True))
        return True

    # ------------------------------------------------------------------
    # Publishing
    # ------------------------------------------------------------------

    def publish(self, progress_score: float):
        msg = Float64MultiArray()
        msg.data = [float(progress_score)]
        self.progress_pub.publish(msg)
        self.marker_pub.publish(self._build_marker(progress_score))

    def _build_marker(self, progress_score: float) -> Marker:
        marker = Marker()
        marker.header.frame_id = self.marker_frame_id
        marker.header.stamp = self.node.get_clock().now().to_msg()
        marker.ns = "policy_progress"
        marker.id = 0
        marker.type = Marker.TEXT_VIEW_FACING
        marker.action = Marker.ADD
        marker.pose.position.x = 0.0
        marker.pose.position.y = 0.0
        marker.pose.position.z = 1.0
        marker.pose.orientation.w = 1.0
        marker.scale.z = 0.08

        clamped = max(0.0, min(1.0, float(progress_score)))
        marker.color.a = 1.0
        marker.color.r = 1.0 - clamped
        marker.color.g = clamped
        marker.color.b = 0.0

        marker.text = f"Progress: {progress_score:.3f}"
        if self.progress_threshold is not None:
            marker.text += f" / {self.progress_threshold:.2f}"
            if self._consecutive:
                marker.text += f" [{self._consecutive}/{self.patience}]"
        marker.lifetime.sec = 1
        return marker

    def render_bar(self, bar_len: int = 20) -> str:
        """Console bar for the inference loop's telemetry."""
        if self._last_value is None:
            return ""
        filled = int(max(0.0, min(1.0, self._last_value)) * bar_len)
        bar = "\u2588" * filled + "\u2591" * (bar_len - filled)
        text = f"[PROGRESS] {bar} {self._last_value:.3f}"
        if self.progress_threshold is not None:
            text += f" / {self.progress_threshold:.2f}"
            if self._consecutive:
                text += f" ({self._consecutive}/{self.patience})"
        return text

    @property
    def last_value(self):
        return self._last_value

    @property
    def done(self) -> bool:
        return self._latched


def flush(seconds: float = 0.25):
    """Give DDS time to deliver before the node is destroyed.

    /policy_done and the final /policy_status are published and then the
    caller tears down the executor and the node. Destroying a publisher
    drops its transient-local history with it, so without a short settle
    the controller can miss the stop and keep executing the last chunk
    (~26 steps at 20 Hz, over a second of motion).
    """
    time.sleep(max(0.0, float(seconds)))