#!/usr/bin/env python3
"""
ManiFlow ROS 2 Action Server for MoveIt Pro Behavior Tree Integration.

Wraps the DRTC policy streaming loop from deploy.py in a standardized
ExecutePolicy.action server.

Features:
  - Supports MoveIt Pro Behavior Tree orchestration (ExecutePolicy Goal, Feedback, Result)
  - Preemption / Cancel handling: Immediately closes policy gate and signals /policy_done,
    triggering TesolloCartesianExecutor schedule flushing.
  - Streaming Feedback: Reports progress_score, latency_ms, buffer_depth, and execution phase.
  - Model caching: Avoids reloading weights and re-warming CUDA if consecutive goals request
    the same checkpoint.
  - Standalone compatibility: Does not break direct usage of deploy.py or /policy_gate topics.
"""
import os
import time
import threading
from typing import Optional, Tuple

import numpy as np
import torch
import rclpy
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from std_msgs.msg import Bool, String
from builtin_interfaces.msg import Duration

from maniflow_msgs.action import ExecutePolicy

from model_serve import init_model, normalize
from cond_preprocess import (PointCloudProcessor, setup_kinematics, compute_fk,
                             preprocess_wrist_rgb, WRIST_RGB_HW)
from rtc_utils import RTCProcessor
from progress_utils import ProgressTracker, flush as flush_dds, latched_qos
from deploy import RealDexEnv, JacobsonKarelsLatencyEstimator, CooldownCounter, sport_mode


class PolicyActionServer(Node):
    def __init__(self):
        super().__init__("policy_action_server")
        self._cbg = ReentrantCallbackGroup()

        # Cached model to avoid repeated cold-starts
        self._cached_model = None
        self._cached_cfg = None
        self._cached_key: Optional[Tuple[str, str, str, int]] = None

        # Gate and Done publishers (mirrors / coordinates with executor)
        self.gate_pub = self.create_publisher(Bool, "/policy_gate", 10)
        self.done_pub = self.create_publisher(Bool, "/policy_done", latched_qos())

        # Action Server
        self._action_server = ActionServer(
            self,
            ExecutePolicy,
            "execute_policy",
            execute_callback=self.execute_callback,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback,
            callback_group=self._cbg,
        )

        self.get_logger().info("ManiFlow Policy Action Server initialized and waiting for goals on /execute_policy")

    def goal_callback(self, goal_request: ExecutePolicy.Goal) -> GoalResponse:
        self.get_logger().info(
            f"Received Goal: task={goal_request.task_name}, ckpt={goal_request.ckpt_file}, "
            f"preload_only={goal_request.preload_only}"
        )
        return GoalResponse.ACCEPT

    def cancel_callback(self, goal_handle) -> CancelResponse:
        self.get_logger().warn("Cancel requested by Behavior Tree client! Preempting execution.")
        # Immediately command safety hold and schedule flush on Cartesian executor
        self.done_pub.publish(Bool(data=True))
        self.gate_pub.publish(Bool(data=False))
        flush_dds()
        return CancelResponse.ACCEPT

    def _get_or_load_model(self, task_name: str, alg_name: str, addition_info: int,
                           run_dir: str, ckpt_file: str, num_inference_steps: int):
        model_key = (task_name, alg_name, run_dir, addition_info, ckpt_file)
        if self._cached_key == model_key and self._cached_model is not None:
            self.get_logger().info("Reusing cached policy in VRAM.")
            policy = self._cached_model
            cfg = self._cached_cfg
        else:
            self.get_logger().info(f"Loading policy into VRAM from {run_dir} (ckpt: {ckpt_file})...")
            cfg, policy = init_model(
                task_name=task_name,
                alg_name=alg_name,
                addition_info=addition_info,
                run_dir=run_dir,
                ckpt_file=ckpt_file,
            )
            self._cached_key = model_key
            self._cached_model = policy
            self._cached_cfg = cfg

        if num_inference_steps > 0:
            policy.num_inference_steps = num_inference_steps

        return cfg, policy

    def execute_callback(self, goal_handle):
        request: ExecutePolicy.Goal = goal_handle.request
        feedback = ExecutePolicy.Feedback()
        result = ExecutePolicy.Result()
        t_start = time.perf_counter()

        task_name = request.task_name or "foil_lifting_wrist"
        alg_name = request.alg_name or "maniflow_multimodal_policy"
        run_dir = request.run_dir or "/ros2_ws/data/outputs/foil_lifting_wrist-maniflow_multimodal_policy-0113_seed0"
        ckpt_file = request.ckpt_file or "epoch=0040-val_action_mse=9.478029"
        addition_info = request.addition_info if request.addition_info != 0 else 1211
        num_inference_steps = request.num_inference_steps if request.num_inference_steps != 0 else 10
        progress_threshold = request.progress_threshold if request.progress_threshold > 0.0 else 0.90
        progress_patience = request.progress_patience if request.progress_patience > 0 else 3
        max_duration_sec = request.max_duration.sec + request.max_duration.nanosec * 1e-9

        feedback.phase = "PRELOADING"
        goal_handle.publish_feedback(feedback)

        try:
            cfg, policy = self._get_or_load_model(
                task_name=task_name,
                alg_name=alg_name,
                addition_info=addition_info,
                run_dir=run_dir,
                ckpt_file=ckpt_file,
                num_inference_steps=num_inference_steps,
            )
        except Exception as e:
            self.get_logger().error(f"Failed to load model: {e}")
            result.success = False
            result.termination_reason = f"LOAD_ERROR: {str(e)}"
            goal_handle.abort()
            return result

        # Preload only requested? (MoveIt Pro Gross motion preparation)
        if request.preload_only:
            self.get_logger().info("Preload-only goal completed successfully.")
            result.success = True
            result.termination_reason = "PRELOADED"
            goal_handle.succeed()
            return result

        # Setup Environment node for observations
        feedback.phase = "WARMING_UP"
        goal_handle.publish_feedback(feedback)

        fps = 20.0
        control_hz = 20.0
        control_period_s = 1.0 / control_hz
        min_execution_horizon = 8
        cooldown_epsilon = 2
        rtc_execution_horizon = 18
        rtc_max_guidance = 5.0
        warmup_steps = 3

        use_progress = progress_threshold is not None and progress_threshold > 0.0
        env = RealDexEnv(cfg=cfg, fps=fps, use_progress=use_progress,
                         progress_threshold=progress_threshold,
                         progress_patience=progress_patience)

        executor = MultiThreadedExecutor()
        executor.add_node(env)
        exec_thread = threading.Thread(target=executor.spin, daemon=True)
        exec_thread.start()

        rtc = RTCProcessor(execution_horizon=rtc_execution_horizon,
                           max_guidance=rtc_max_guidance)

        latency_estimator = JacobsonKarelsLatencyEstimator(alpha=0.1, beta=0.05)
        cooldown = CooldownCounter(epsilon=cooldown_epsilon)
        prev_action_pred = None

        try:
            # 1. Wait for cameras
            self.get_logger().info("Waiting for initial camera observations...")
            while not env.obs_ready():
                if goal_handle.is_cancel_requested:
                    result.success = False
                    result.termination_reason = "CANCELED_WAITING_OBS"
                    goal_handle.canceled()
                    return result
                env.wait_for_obs(timeout=0.2)
                rclpy.spin_once(self, timeout_sec=0.01)

            obs = env.snapshot_obs()
            if obs is None:
                raise RuntimeError("Failed to capture snapshot observation")

            # 2. Warmup CUDA kernels
            for i in range(warmup_steps):
                with torch.no_grad():
                    warmup_result = policy.predict_action(obs)
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()

            if use_progress:
                env.progress.validate(warmup_result)

            env.reset()
            if hasattr(policy, 'reset'):
                policy.reset()

            # 3. Open Policy Gate
            self.gate_pub.publish(Bool(data=True))
            feedback.phase = "STREAMING"
            goal_handle.publish_feedback(feedback)

            # 4. Demand-driven DRTC loop
            last_cooldown_tick_time = time.perf_counter()
            steps_executed = 0
            task_done = False
            term_reason = "SUCCESS"

            while rclpy.ok():
                if goal_handle.is_cancel_requested:
                    self.get_logger().info("BT Goal cancellation acknowledged in streaming loop.")
                    term_reason = "CANCELED"
                    break

                if max_duration_sec > 0.0 and (time.perf_counter() - t_start) > max_duration_sec:
                    self.get_logger().warn(f"Goal exceeded max_duration ({max_duration_sec}s). Timing out.")
                    term_reason = "MAX_DURATION_EXCEEDED"
                    break

                # Tick cooldown
                now = time.perf_counter()
                ticks_due = int((now - last_cooldown_tick_time) / control_period_s)
                if ticks_due > 0:
                    for _ in range(min(ticks_due, 10)):
                        cooldown.tick()
                    last_cooldown_tick_time = now

                buffer_depth = env.get_buffer_depth()
                delay_steps = latency_estimator.estimate_steps(control_period_s)
                horizon = delay_steps + min_execution_horizon

                needs_refill = buffer_depth < horizon
                cd_ready = cooldown.ready()
                obs_fresh = env.obs_ready()

                if not (needs_refill and cd_ready and obs_fresh):
                    time.sleep(0.005)
                    continue

                # Snapshot and RTC
                obs = env.snapshot_obs()
                if obs is None:
                    time.sleep(0.01)
                    continue

                prev_leftover = None
                rtc_delay = 0
                if rtc is not None and prev_action_pred is not None:
                    rtc_delay = latency_estimator.estimate_steps(control_period_s)
                    rtc_delay = max(0, min(rtc_delay, rtc_execution_horizon,
                                           prev_action_pred.shape[1] - 1))
                    prev_leftover = prev_action_pred[:, rtc_delay:, :]

                # GPU Inference
                t_inf_start = time.perf_counter()
                with torch.no_grad():
                    action_queue_raw = policy.predict_action(
                        obs,
                        rtc_processor=rtc,
                        prev_chunk_left_over=prev_leftover,
                        inference_delay=rtc_delay
                    )
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                inf_time = time.perf_counter() - t_inf_start

                latency_estimator.observe(inf_time)
                delay_steps = latency_estimator.estimate_steps(control_period_s)
                cooldown.arm(delay_steps)

                prev_action_pred = action_queue_raw['action_pred'].detach()
                action_queue = normalize(action_queue_raw)

                if request.sport_mode:
                    action_queue = sport_mode(action_queue, speedup=1.3)

                # Progress check
                task_done = env.progress.update(action_queue_raw)
                curr_progress = env.progress.current_score if hasattr(env.progress, 'current_score') else 0.0

                # Feedback publication
                feedback.progress_score = float(curr_progress or 0.0)
                feedback.current_latency_ms = float(inf_time * 1000.0)
                feedback.buffer_depth = int(buffer_depth)
                goal_handle.publish_feedback(feedback)

                if task_done:
                    self.get_logger().info(f"Task completed: progress_score >= {progress_threshold}")
                    term_reason = "PROGRESS_REACHED"
                    env.publish_status("DONE")
                    break

                # Anchor publish
                anchor_step = env.get_buffer_next_step()
                if delay_steps < len(action_queue):
                    future_actions = action_queue[delay_steps:]
                else:
                    future_actions = action_queue[-1:]

                env.publish_action_chunk_with_anchor(future_actions, anchor_step)
                steps_executed += len(future_actions)

        finally:
            # Settle & Cleanup
            self.done_pub.publish(Bool(data=True))
            self.gate_pub.publish(Bool(data=False))
            flush_dds()
            executor.shutdown()
            env.destroy_node()

        total_elapsed = time.perf_counter() - t_start
        dur = Duration()
        dur.sec = int(total_elapsed)
        dur.nanosec = int((total_elapsed - int(total_elapsed)) * 1e9)

        result.elapsed_time = dur
        result.total_steps_executed = steps_executed
        result.termination_reason = term_reason
        result.final_progress_score = float(feedback.progress_score)

        if term_reason in ("PROGRESS_REACHED", "SUCCESS"):
            result.success = True
            goal_handle.succeed()
        elif term_reason == "CANCELED":
            result.success = False
            goal_handle.canceled()
        else:
            result.success = False
            goal_handle.abort()

        return result


def main(args=None):
    rclpy.init(args=args)
    server_node = PolicyActionServer()
    executor = MultiThreadedExecutor()
    executor.add_node(server_node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        server_node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
