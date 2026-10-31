#!/usr/bin/env python3
"""
DG5F temperature watchdog - emergency shutdown on motor overtemperature.

WHY THIS IS DELIBERATELY SIMPLE
-------------------------------
Measurements over ~9 minutes of normal teleoperation (cal_run2.csv) showed
+0.9 C on the worst joint, i.e. about 0.1 C/min. Going from 47 C to the trip
threshold would take hours, so the hand does not get critical during normal
use. A predictive load model therefore solves a problem that rarely occurs,
and its correlation with temperature was NOT supported by the data: the joint
with the highest peak load (rj_dg_2_2, 0.320) rose only +0.5 C, while
rj_dg_5_3 at a quarter of that load rose +0.6 C.

So this script reacts only to the REAL measurement:
/dg5f_<hand>/dynamic_joint_states carries a per-joint "temperature" interface.
It comes from the JointStateBroadcaster, which exports every interface the
hardware component offers, independently of the xacro declarations - that is
why temperature is available even though dg5f_right_ros2_control.xacro only
declares position and velocity.

Thresholds: Tesollo quotes 75 C as the limit. Default trip is 65 C, leaving
10 K of headroom because the case-mounted sensor lags the winding (the winding
heats within seconds, the sensor follows minutes later).

WHAT IT DOES
------------
  >= warn_at   log the affected joint
  >= stop_at   deactivate the controller via switch_controller

Deactivating matters twice over here: according to dg_hardware the firmware
keeps applying the last PWM duty it received, so merely killing the command
stream would leave the hand pressing - and heating - indefinitely.
switch_controller releases the effort cleanly instead.

This script never writes to the command path, so it needs NO topic remap and
no launch changes. It runs as a sidecar alongside normal teleoperation.

USAGE
-----
    bash run_watchdog.sh --table
    bash run_watchdog.sh --stop-at 60 --warn-at 55 --table
    bash run_watchdog.sh --dry-run --table          # warn only, never trip
    HAND=left bash run_watchdog.sh --table

VERIFYING THE SHUTDOWN PATH
---------------------------
To prove the trip actually works without heating the hand, set the threshold
below the current temperature. At ~47 C idle:

    bash run_watchdog.sh --stop-at 40 --warn-at 35 --dry-run   # logs only
    bash run_watchdog.sh --stop-at 40 --warn-at 35              # real trip

Re-arm after a trip:
    ros2 control switch_controllers --activate dg5f_right_controller \\
        -c /dg5f_right/controller_manager
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time

import rclpy
from control_msgs.msg import DynamicJointState
from controller_manager_msgs.srv import SwitchController
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)

SLOPE_WINDOW_S = 60.0   # window used to estimate the rise rate in C/min


class JointTemp:
    """Temperature history of a single joint."""

    def __init__(self) -> None:
        self.temp = float("nan")
        self.peak = 0.0
        self.slope = 0.0
        self._hist: list[tuple[float, float]] = []

    def update(self, temp: float, now: float) -> None:
        self.temp = temp
        self.peak = max(self.peak, temp)
        # Sliding window -> C/min. The sensor quantises to 0.1 C, so a short
        # window would mostly report quantisation noise; hence 60 s.
        self._hist.append((now, temp))
        cutoff = now - SLOPE_WINDOW_S
        while len(self._hist) > 2 and self._hist[0][0] < cutoff:
            self._hist.pop(0)
        if len(self._hist) >= 2:
            t0, v0 = self._hist[0]
            t1, v1 = self._hist[-1]
            if t1 - t0 > 10.0:
                self.slope = (v1 - v0) / (t1 - t0) * 60.0


class TempWatchdog(Node):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__("dg5f_temp_watchdog")
        self.a = args
        ns = f"/dg5f_{args.hand}"
        self.controller = f"dg5f_{args.hand}_controller"
        self.cm = f"{ns}/controller_manager"

        self.joints: dict[str, JointTemp] = {}
        self.tripped = False
        self.t0 = time.monotonic()

        self.csv_file = None
        self.csv_writer = None
        if args.csv:
            self.csv_file = open(args.csv, "w", newline="")
            self.csv_writer = csv.writer(self.csv_file)
            self.csv_writer.writerow(
                ["elapsed_s", "joint", "temp_c", "slope_c_per_min"]
            )

        self.switch_cli = self.create_client(
            SwitchController, f"{self.cm}/switch_controller"
        )

        # Broadcaster publishes at 100 Hz; best-effort depth 1 is plenty.
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.create_subscription(
            DynamicJointState, f"{ns}/dynamic_joint_states", self.on_state, qos
        )
        self.create_timer(args.report_every, self.report)

        self.get_logger().info(
            f"temperature watchdog on {ns}/dynamic_joint_states"
        )
        self.get_logger().info(
            f"warn at {args.warn_at:.0f} C, TRIP at {args.stop_at:.0f} C"
            f"  (vendor limit 75 C)"
            + ("   DRY-RUN: will not deactivate" if args.dry_run else "")
        )

    # ---- measurements ----
    def on_state(self, msg: DynamicJointState) -> None:
        now = time.monotonic()
        for name, iv in zip(msg.joint_names, msg.interface_values):
            if "temperature" not in iv.interface_names:
                continue
            idx = iv.interface_names.index("temperature")
            if idx >= len(iv.values):
                continue
            temp = iv.values[idx]
            # Skip joints that have not reported yet; a 0.0 would otherwise
            # look like a cold motor and mask a real reading.
            if temp <= 0.0 or math.isnan(temp):
                continue
            self.joints.setdefault(name, JointTemp()).update(temp, now)

        self.check()

    def check(self) -> None:
        if self.tripped or not self.joints:
            return
        over = [(j.temp, n) for n, j in self.joints.items()
                if j.temp >= self.a.stop_at]
        if over:
            temp, name = max(over)
            self.trip(name, temp)

    def trip(self, name: str, temp: float) -> None:
        self.tripped = True
        self.get_logger().error(
            f"OVERTEMPERATURE: {name} at {temp:.1f} C "
            f"(threshold {self.a.stop_at:.0f} C) -> deactivating "
            f"{self.controller}, motors will go slack."
        )
        if self.a.dry_run:
            self.get_logger().warning(
                "DRY-RUN: controller left running, would have deactivated here"
            )
            self.tripped = False   # keep observing
            return

        if not self.switch_cli.wait_for_service(timeout_sec=2.0):
            self.get_logger().fatal(
                f"switch_controller unreachable ({self.cm}) - "
                f"DISCONNECT THE HAND MANUALLY NOW!"
            )
            return

        req = SwitchController.Request()
        req.deactivate_controllers = [self.controller]
        req.strictness = SwitchController.Request.BEST_EFFORT
        future = self.switch_cli.call_async(req)
        future.add_done_callback(self._on_switch_done)

    def _on_switch_done(self, future) -> None:
        try:
            ok = future.result().ok
        except Exception as exc:                      # noqa: BLE001
            self.get_logger().fatal(
                f"deactivation failed ({exc}) - DISCONNECT THE HAND MANUALLY!"
            )
            return
        if ok:
            self.get_logger().error(
                "controller deactivated. Let the hand cool, then re-arm:\n"
                f"  ros2 control switch_controllers --activate "
                f"{self.controller} -c {self.cm}"
            )
        else:
            self.get_logger().fatal(
                "switch_controller reported failure - "
                "DISCONNECT THE HAND MANUALLY!"
            )

    # ---- output ----
    def report(self) -> None:
        if not self.joints:
            self.get_logger().warning(
                "no temperature data yet - is the Tesollo stack running?"
            )
            return

        elapsed = time.monotonic() - self.t0
        if self.csv_writer is not None:
            for name, j in sorted(self.joints.items()):
                self.csv_writer.writerow(
                    [f"{elapsed:.1f}", name, f"{j.temp:.1f}", f"{j.slope:.2f}"]
                )
            self.csv_file.flush()

        hot_temp, hot_name = max((j.temp, n) for n, j in self.joints.items())
        rise_val, rise_name = max((j.slope, n) for n, j in self.joints.items())
        head = (
            f"hottest {hot_name} {hot_temp:.1f} C "
            f"(trip {self.a.stop_at:.0f})   "
            f"max rise {rise_val:+.1f} C/min ({rise_name})   "
            f"{len(self.joints)} joints"
        )

        warn = {n for n, j in self.joints.items() if j.temp >= self.a.warn_at}
        if warn:
            detail = ", ".join(
                f"{n} {self.joints[n].temp:.1f}C" for n in sorted(warn)
            )
            self.get_logger().warning(f"{head} | WARNING: {detail}")
        else:
            self.get_logger().info(head)

        if self.a.table:
            self.print_table(elapsed, hot_temp)

    def print_table(self, elapsed: float, hottest: float) -> None:
        """Show every joint, so a single runaway finger cannot hide."""
        rows = sorted(self.joints.items())
        lines = [
            "" if self.a.no_clear else "\033[H\033[2J",
            f"DG5F temperature watchdog   t+{elapsed / 60:5.1f} min   "
            f"warn {self.a.warn_at:.0f} / trip {self.a.stop_at:.0f} C",
            "",
            f"  {'joint':12} {'tempC':>7} {'C/min':>7} {'peak':>7}",
        ]
        for name, j in rows:
            mark = ""
            if j.temp >= self.a.stop_at:
                mark = " <<< TRIP"
            elif j.temp >= self.a.warn_at:
                mark = " <<  warn"
            elif j.temp >= hottest - 0.05:
                mark = " <"
            lines.append(
                f"  {name:12} {j.temp:7.1f} {j.slope:+7.1f} {j.peak:7.1f}{mark}"
            )
        print("\n".join(lines), flush=True)

    def summary(self) -> None:
        if not self.joints:
            return
        print("\npeak temperature per joint:")
        for name, j in sorted(self.joints.items()):
            print(f"  {name:12} {j.peak:5.1f} C")
        peak, name = max((j.peak, n) for n, j in self.joints.items())
        print(f"\n  hottest joint: {name} at {peak:.1f} C")
        if self.a.csv:
            print(f"  CSV: {self.a.csv}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Deactivate the DG5F controller on motor overtemperature."
    )
    p.add_argument("--hand", choices=("right", "left"), default="right")
    p.add_argument("--warn-at", type=float, default=58.0, dest="warn_at",
                   help="temperature in C that triggers warnings (default 58)")
    p.add_argument("--stop-at", type=float, default=65.0, dest="stop_at",
                   help="temperature in C that deactivates the controller "
                        "(default 65; vendor limit is 75)")
    p.add_argument("--report-every", type=float, default=5.0,
                   dest="report_every", help="seconds between reports")
    p.add_argument("--table", action="store_true",
                   help="show all joints as a table")
    p.add_argument("--no-clear", action="store_true", dest="no_clear",
                   help="append tables instead of clearing the screen")
    p.add_argument("--csv", default=None, help="log temperatures to CSV")
    p.add_argument("--dry-run", action="store_true",
                   help="warn only, never deactivate the controller")
    known, _ = p.parse_known_args(argv)
    if known.warn_at > known.stop_at:
        p.error("--warn-at must be lower than --stop-at")
    return known


def main() -> int:
    args = parse_args(sys.argv[1:])
    rclpy.init()
    node = TempWatchdog(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.summary()
        if node.csv_file is not None:
            node.csv_file.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())