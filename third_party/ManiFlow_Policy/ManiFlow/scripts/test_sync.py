#!/usr/bin/env python3
"""
Standalone camera <-> robot synchronization test.

Mirrors the timestamp-matching logic of deploy.py's RobotCameraSync, but
WITHOUT the policy, the point cloud processing, the FK, or anything
hand-related. It only answers: do the ZED colour/depth pair, the wrist
camera and /joint_states line up in time well enough to be treated as one
observation?

What it reports
---------------
Per-topic:   message rate, inter-arrival jitter (from header stamps), and
             delivery lag (now() - stamp) as DIAGNOSTIC ONLY - see below.
Per-triple:  colour<->depth and colour<->wrist stamp spread (how hard the
             ApproximateTimeSynchronizer had to work). This is the figure
             that actually tells you whether the clocks agree.
Per-match:   camera<->joint gap, i.e. the quantity deploy.py warns about at
             35 ms, plus the rate of successfully formed observations.

Why delivery lag is not a clock check
-------------------------------------
An earlier version of this script failed the run when now() - header.stamp
differed across topics. That measurement cannot distinguish a clock offset
from transport and scheduling delay, and in practice it is dominated by the
latter: large depth images arrive later than colour ones even when the ZED
stamps both identically, and a 500 Hz /joint_states gets processed in bursts
by this script's own spin loop, inflating the lag of every message but the
first in each burst.

A clock offset between two sensors shows up as a constant bias in the
difference of their HEADER STAMPS, which survives any amount of transport
jitter. So clock agreement is judged from the matched triples' stamp spread,
and delivery lag is printed but never gated on.

Run
---
    python3 test_sync.py                      # 20 s test, defaults
    python3 test_sync.py --duration 60
    python3 test_sync.py --slop 0.015 --verbose
    python3 test_sync.py --wrist-topic /cam_gemini/color/image_raw

Exit code is 0 if every check passes, 1 otherwise, so it can be used as a
pre-flight gate before launching deploy.py.
"""
import argparse
import collections
import os
import statistics
import sys
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image, JointState
import message_filters


# Same defaults as deploy.py, overridable so this can be pointed at a bag.
COLOR_TOPIC = os.environ.get(
    "ZED_COLOR_TOPIC", "/zed/zed_node/rgb/color/rect/image")
DEPTH_TOPIC = os.environ.get(
    "ZED_DEPTH_TOPIC", "/zed/zed_node/depth/depth_registered")
WRIST_TOPIC = os.environ.get(
    "WRIST_COLOR_TOPIC", "/cam_gemini/color/image_raw")
JOINT_TOPIC = os.environ.get("JOINT_TOPIC", "/joint_states")

# The joint whose presence marks an "arm" message in deploy.py; used the same
# way here so the two scripts agree on which messages count.
ARM_MARKER_JOINT = "shoulder_pan_joint"

# deploy.py's own threshold for "SEVERE SYNC MISMATCH".
JOINT_GAP_WARN_S = 0.035


def stamp_s(stamp) -> float:
    return stamp.sec + stamp.nanosec * 1e-9


class Stat:
    """Running min/max/mean/stdev over a bounded sample window."""

    def __init__(self, maxlen=20000):
        self.samples = collections.deque(maxlen=maxlen)

    def add(self, v):
        self.samples.append(v)

    def __len__(self):
        return len(self.samples)

    def summary(self, scale=1000.0, unit="ms"):
        if not self.samples:
            return "no samples"
        s = list(self.samples)
        mean = statistics.fmean(s) * scale
        sd = statistics.stdev(s) * scale if len(s) > 1 else 0.0
        p95 = sorted(s)[int(0.95 * (len(s) - 1))] * scale
        return (f"n={len(s):5d}  mean={mean:7.2f}{unit}  "
                f"sd={sd:6.2f}{unit}  p95={p95:7.2f}{unit}  "
                f"min={min(s)*scale:7.2f}{unit}  max={max(s)*scale:7.2f}{unit}")


class RateTracker:
    """Message rate and inter-arrival spacing, measured on header stamps."""

    def __init__(self, name):
        self.name = name
        self.count = 0
        self.first_stamp = None
        self.last_stamp = None
        self.gaps = Stat()
        # now() - header.stamp. Transport + scheduling latency, NOT a clock
        # offset: see the module docstring. Printed for diagnosis (a camera
        # driver that stamps on host receipt rather than exposure start will
        # show a suspiciously small value), never used in the verdict.
        self.delivery_lag = Stat()

    def observe(self, t_stamp, t_wall):
        self.count += 1
        if self.first_stamp is None:
            self.first_stamp = t_stamp
        else:
            self.gaps.add(t_stamp - self.last_stamp)
        self.last_stamp = t_stamp
        self.delivery_lag.add(t_wall - t_stamp)

    @property
    def hz(self):
        if self.count < 2 or self.first_stamp is None:
            return 0.0
        span = self.last_stamp - self.first_stamp
        return (self.count - 1) / span if span > 0 else 0.0

    def report(self):
        print(f"  {self.name:<14} {self.count:6d} msgs  {self.hz:6.2f} Hz")
        print(f"    inter-arrival  {self.gaps.summary()}")
        print(f"    delivery lag   {self.delivery_lag.summary()}  "
              f"[diagnostic only]")


class SyncTester(Node):
    def __init__(self, args):
        super().__init__('sync_tester')
        self.args = args

        self.rates = {
            'color': RateTracker('color'),
            'depth': RateTracker('depth'),
            'wrist': RateTracker('wrist'),
            'joints': RateTracker('joints(arm)'),
        }

        # Triple-level spread (what the ApproximateTimeSynchronizer matched).
        self.spread_color_depth = Stat()
        self.spread_color_wrist = Stat()
        self.spread_total = Stat()
        # SIGNED colour-minus-wrist offset. The absolute spread above cannot
        # separate "jitters +-15 ms around zero" from "sits at a constant
        # +15 ms bias"; only the latter is a clock problem, and only the
        # signed mean reveals it.
        self.offset_color_wrist = Stat()
        self.offset_color_depth = Stat()

        # Observation-level: camera stamp vs. nearest joint stamp.
        self.joint_gap = Stat()
        self.joint_gap_warnings = 0
        self.observations = 0
        self.obs_first_t = None
        self.obs_last_t = None

        # Reconstruct deploy.py's buffers so the matching logic is identical.
        self.joint_history = collections.deque(maxlen=500)
        self.camera_buffer = collections.deque(maxlen=15)
        self.triples = 0
        self.dropped_stale_cam = 0

        qos_fast = QoSProfile(depth=500,
                              reliability=ReliabilityPolicy.RELIABLE,
                              history=HistoryPolicy.KEEP_LAST)
        qos_slow = QoSProfile(depth=15,
                              reliability=ReliabilityPolicy.RELIABLE,
                              history=HistoryPolicy.KEEP_LAST)

        self.create_subscription(JointState, JOINT_TOPIC,
                                 self._joint_cb, qos_fast)

        self.sub_color = message_filters.Subscriber(
            self, Image, args.color_topic, qos_profile=qos_slow)
        self.sub_depth = message_filters.Subscriber(
            self, Image, args.depth_topic, qos_profile=qos_slow)
        self.sub_wrist = message_filters.Subscriber(
            self, Image, args.wrist_topic, qos_profile=qos_slow)

        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.sub_color, self.sub_depth, self.sub_wrist],
            queue_size=15, slop=args.slop)
        self.ts.registerCallback(self._camera_sync_cb)

        # Raw per-topic monitors, so a topic that never enters a triple is
        # still visible as "arriving but unmatched" rather than just absent.
        self.create_subscription(
            Image, args.color_topic,
            lambda m: self._raw(m, 'color'), qos_slow)
        self.create_subscription(
            Image, args.depth_topic,
            lambda m: self._raw(m, 'depth'), qos_slow)
        self.create_subscription(
            Image, args.wrist_topic,
            lambda m: self._raw(m, 'wrist'), qos_slow)

        print(f"[sync-test] color : {args.color_topic}")
        print(f"[sync-test] depth : {args.depth_topic}")
        print(f"[sync-test] wrist : {args.wrist_topic}")
        print(f"[sync-test] joints: {JOINT_TOPIC}")
        print(f"[sync-test] slop  : {args.slop*1000:.0f} ms, "
              f"duration {args.duration:.0f} s\n")

    # ---------------- raw monitors ----------------

    def _raw(self, msg, key):
        self.rates[key].observe(stamp_s(msg.header.stamp),
                                self.get_clock().now().nanoseconds / 1e9)

    # ---------------- deploy.py-equivalent pipeline ----------------

    def _joint_cb(self, msg):
        if ARM_MARKER_JOINT not in msg.name:
            return
        self.rates['joints'].observe(
            stamp_s(msg.header.stamp),
            self.get_clock().now().nanoseconds / 1e9)
        self.joint_history.append(msg)
        self._try_process()

    def _camera_sync_cb(self, color_msg, depth_msg, wrist_msg):
        self.triples += 1
        tc = stamp_s(color_msg.header.stamp)
        td = stamp_s(depth_msg.header.stamp)
        tw = stamp_s(wrist_msg.header.stamp)
        self.spread_color_depth.add(abs(tc - td))
        self.spread_color_wrist.add(abs(tc - tw))
        self.spread_total.add(max(tc, td, tw) - min(tc, td, tw))
        self.offset_color_depth.add(tc - td)
        self.offset_color_wrist.add(tc - tw)

        self.camera_buffer.append((color_msg, depth_msg, wrist_msg))
        self._try_process()

    def _try_process(self):
        while self.camera_buffer and self.joint_history:
            color_msg = self.camera_buffer[0][0]
            t_cam = stamp_s(color_msg.header.stamp)
            t_newest_joint = stamp_s(self.joint_history[-1].header.stamp)

            if t_newest_joint < (t_cam - 0.002):
                break  # joints not caught up yet; wait for more

            self.camera_buffer.popleft()

            best = None
            best_diff = float('inf')
            for j in self.joint_history:
                d = abs(t_cam - stamp_s(j.header.stamp))
                if d < best_diff:
                    best_diff, best = d, j

            if best is None:
                continue

            self.joint_gap.add(best_diff)
            if best_diff > JOINT_GAP_WARN_S:
                self.joint_gap_warnings += 1
                if self.args.verbose:
                    print(f"  [warn] camera/joint gap "
                          f"{best_diff*1000:.1f} ms")

            self.observations += 1
            now = time.perf_counter()
            if self.obs_first_t is None:
                self.obs_first_t = now
            self.obs_last_t = now

            if self.args.verbose and self.observations % 20 == 0:
                print(f"  obs #{self.observations}  "
                      f"joint_gap={best_diff*1000:5.1f} ms  "
                      f"triple_spread={self.spread_total.samples[-1]*1000:5.1f} ms")

    # ---------------- reporting ----------------

    def report(self):
        print("\n" + "=" * 68)
        print("PER-TOPIC ARRIVAL")
        print("=" * 68)
        for r in self.rates.values():
            r.report()

        print("\n" + "=" * 68)
        print("CAMERA TRIPLE ALIGNMENT (ApproximateTimeSynchronizer output)")
        print("=" * 68)
        n_color = self.rates['color'].count
        match_pct = (100.0 * self.triples / n_color) if n_color else 0.0
        print(f"  triples formed  {self.triples} "
              f"({match_pct:.1f}% of colour frames)")
        print(f"  color<->depth   {self.spread_color_depth.summary()}")
        print(f"  color<->wrist   {self.spread_color_wrist.summary()}")
        print(f"  total spread    {self.spread_total.summary()}")
        print(f"  signed c-d      {self.offset_color_depth.summary()}")
        print(f"  signed c-w      {self.offset_color_wrist.summary()}")

        # How close the worst matched triple came to being rejected. A p95
        # that is already most of the slop means the match rate above is
        # luck, and will collapse under extra CPU load.
        if len(self.spread_total):
            s = sorted(self.spread_total.samples)
            p95 = s[int(0.95 * (len(s) - 1))]
            worst = s[-1]
            print(f"  slop headroom   p95 uses "
                  f"{100.0*p95/self.args.slop:.0f}% of the "
                  f"{self.args.slop*1000:.0f} ms slop, "
                  f"worst {100.0*worst/self.args.slop:.0f}%")

        print("\n" + "=" * 68)
        print("CAMERA <-> JOINT MATCHING (one observation per triple)")
        print("=" * 68)
        obs_hz = 0.0
        if self.observations > 1 and self.obs_last_t > self.obs_first_t:
            obs_hz = (self.observations - 1) / (self.obs_last_t - self.obs_first_t)
        print(f"  observations    {self.observations}  ({obs_hz:.2f} Hz)")
        print(f"  joint gap       {self.joint_gap.summary()}")
        print(f"  gaps > {JOINT_GAP_WARN_S*1000:.0f} ms     "
              f"{self.joint_gap_warnings} "
              f"({100.0*self.joint_gap_warnings/max(1,self.observations):.1f}%)")

        return self._verdict(match_pct, obs_hz)

    def _verdict(self, match_pct, obs_hz):
        print("\n" + "=" * 68)
        print("VERDICT")
        print("=" * 68)
        ok = True

        def check(label, passed, detail):
            nonlocal ok
            ok = ok and passed
            print(f"  [{'PASS' if passed else 'FAIL'}] {label}: {detail}")

        for key, want in (('color', self.args.min_camera_hz),
                          ('depth', self.args.min_camera_hz),
                          ('wrist', self.args.min_camera_hz),
                          ('joints', self.args.min_joint_hz)):
            r = self.rates[key]
            check(f"{r.name} rate", r.hz >= want,
                  f"{r.hz:.2f} Hz (need >= {want:.0f})")

        check("triple match rate", match_pct >= 70.0,
              f"{match_pct:.1f}% of colour frames matched "
              f"(raise --slop if low)")

        check("observation rate", obs_hz >= self.args.min_camera_hz * 0.7,
              f"{obs_hz:.2f} Hz")

        bad_frac = self.joint_gap_warnings / max(1, self.observations)
        check("camera/joint gap", bad_frac <= 0.02,
              f"{bad_frac*100:.1f}% of observations exceed "
              f"{JOINT_GAP_WARN_S*1000:.0f} ms")

        # Clock agreement, judged on HEADER STAMPS of matched triples.
        #
        # A clock offset is a constant bias, so it appears as a signed mean
        # far from zero while the sd stays small. Symmetric jitter about zero
        # - the normal case for two unsynced cameras free-running at
        # different rates - has a mean near zero and is harmless as long as
        # it fits the slop, which the match-rate check already covers.
        if len(self.offset_color_wrist) > 1:
            bias = statistics.fmean(self.offset_color_wrist.samples)
            jitter = statistics.stdev(self.offset_color_wrist.samples)
            # Threshold relative to jitter: a bias small compared with the
            # frame-to-frame variation is indistinguishable from phase, not
            # evidence of a clock offset.
            biased = abs(bias) > max(0.010, 1.5 * jitter)
            check("clock agreement (color/wrist)", not biased,
                  f"signed stamp offset mean={bias*1000:+.1f} ms, "
                  f"sd={jitter*1000:.1f} ms"
                  + ("" if not biased else
                     " - constant bias dominates jitter, check "
                     "use_sim_time and driver timestamping"))

        # Slop headroom: distinct from match rate, which only reports what
        # already happened. This is whether it will keep happening.
        if len(self.spread_total):
            s = sorted(self.spread_total.samples)
            p95 = s[int(0.95 * (len(s) - 1))]
            frac = p95 / self.args.slop
            check("slop headroom", frac <= 0.8,
                  f"p95 triple spread is {frac*100:.0f}% of slop"
                  + ("" if frac <= 0.8 else
                     f" - raise --slop above {self.args.slop*1000:.0f} ms "
                     f"or match rate will drop under load"))

        print("\n  => " + ("SYNC OK" if ok else "SYNC PROBLEMS, see FAIL lines"))
        return ok


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--duration', type=float, default=20.0,
                   help='test length in seconds')
    p.add_argument('--slop', type=float, default=0.030,
                   help='ApproximateTimeSynchronizer slop in seconds '
                        '(deploy.py uses 0.030)')
    p.add_argument('--color-topic', default=COLOR_TOPIC)
    p.add_argument('--depth-topic', default=DEPTH_TOPIC)
    p.add_argument('--wrist-topic', default=WRIST_TOPIC)
    p.add_argument('--min-camera-hz', type=float, default=10.0)
    p.add_argument('--min-joint-hz', type=float, default=50.0)
    p.add_argument('--verbose', action='store_true')
    args = p.parse_args()

    rclpy.init()
    node = SyncTester(args)
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(node)

    t_end = time.perf_counter() + args.duration
    try:
        while rclpy.ok() and time.perf_counter() < t_end:
            executor.spin_once(timeout_sec=0.1)
    except KeyboardInterrupt:
        print("\n[sync-test] interrupted, reporting what we have")

    ok = node.report()

    executor.shutdown()
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()