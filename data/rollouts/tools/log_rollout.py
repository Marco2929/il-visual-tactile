#!/usr/bin/env python3
"""Record what a policy rollout commanded and what the robot did.

Gated by /policy_gate, the same topic that starts and stops execution, so a
rollout is logged exactly as long as the controller is accepting chunks. ROS 2
topics are one-to-many: this node listens alongside the control node and does
not interfere with it.

WHY THIS EXISTS: between the policy's output and the robot's joints,
deploy.py applies quaternion renormalisation, the 15 -> 27 hand expansion, RTC
prefix guidance and a latency shift. Logging only /joint_states cannot tell a
policy that predicts the wrong height from a chunk pipeline that moves it. So
both sides are recorded:

    /predicted_action_chunk   what was commanded, after all post-processing
    /joint_states             what the robot actually did
    /action_buffer_status     how deep the executor's buffer was, i.e.
                              whether chunks arrived in time at all

The one level NOT captured here is the policy's raw tensor before
normalize()/RTC. That lives only inside the deploy process; logging it would
mean writing from the inference loop. If RTC itself becomes the suspect, add
it there - after the publish, never before.

Runs in its own process on purpose: no GIL contention with inference, no GPU
use, and a crash in the writer cannot take the policy down.

USAGE - start this BEFORE opening the gate, because `ros2 topic pub --once`
is volatile and a subscriber that joins later never sees it:

    python data/rollout_logs/log_rollout.py \
        --ckpt "epoch=0040-val_action_mse=9.478029" \
        --run-dir /ros2_ws/data/outputs/foil_lifting_wrist-maniflow_multimodal_policy-0113_seed0 \
        --inference-steps 10 \
        --notes "pallet at nominal height"

    ros2 topic pub --once /policy_gate std_msgs/msg/Bool "{data: true}"
    ros2 topic pub --once /policy_gate std_msgs/msg/Bool "{data: false}"

Each gate open/close pair writes one rollout directory. Ctrl-C closes a
rollout that is still open.
"""
import argparse
import json
import os
import pathlib
import signal
import subprocess
import time
from datetime import datetime

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState, PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Bool, Float64MultiArray, String

# Mirrors deploy.py: ARM_JOINT_NAMES + FULL_HAND_JOINTS, finger-major. The
# order matters because a snapshot is only written once every one of these is
# present, so the resulting vector is positionally comparable with the
# recorded state in the training data.
ARM_JOINT_NAMES = [
    "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
    "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
]
FULL_HAND_JOINTS = [f"rj_dg_{finger}_{joint}"
                    for finger in range(1, 6)
                    for joint in range(1, 5)]
JOINT_NAMES = ARM_JOINT_NAMES + FULL_HAND_JOINTS

# deploy.py's wire format: 7 pose + 20 hand.
WIRE_ACTION_DIM = 27

DEFAULT_ROOT = os.environ.get(
    "ROLLOUT_LOG_DIR", "/ros2_ws/data/rollouts")

# Same default and env var as deploy.py, so logger and policy cannot end up
# pointed at different cameras. The Orbbec Gemini delivers 640x480; the
# encoder sees 192x256 after preprocess_wrist_rgb()'s INTER_AREA downsample,
# which averages over ~2.5x2.5 blocks. The bag therefore holds what the CAMERA
# saw, not what the MODEL saw - fine for "where was the hand when it closed",
# not sufficient to judge whether a thin foil edge survived the resize.
WRIST_COLOR_TOPIC = os.environ.get(
    "WRIST_COLOR_TOPIC", "/cam_gemini/color/image_raw")

# deploy.py's publish_debug_pointcloud() emits the PROCESSED cloud: 1024
# points, already cropped, FPS-sampled and in base_link - i.e. exactly the
# encoder input. No resolution question here, unlike the wrist image.
DEBUG_PC_TOPIC = "/debug_pc"


def _sanitise(name: str) -> str:
    """Make a checkpoint filename safe as a directory name, reversibly.

    Kept close to the original ('=' and '.' are legal on Linux) so the
    directory can still be matched against the file it came from by eye.
    """
    return name.replace("/", "_").replace(" ", "_").strip()


def _git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(pathlib.Path(__file__).resolve().parent),
            capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


class RolloutLogger(Node):
    def __init__(self, args):
        super().__init__("rollout_logger")
        self.args = args

        # --label exists because --ckpt has to stay the literal .ckpt
        # filename for the lookup below to work, and that filename is often
        # just 'epoch=0040-val_action_mse=...' - identical across runs. The
        # output directory needs to say WHICH run, so it is named separately.
        self.ckpt_label = _sanitise(args.label or args.ckpt)
        target_root = pathlib.Path(args.root)
        if (target_root / "runs").is_dir():
            target_root = target_root / "runs"
        self.root = target_root / self.ckpt_label
        self.root.mkdir(parents=True, exist_ok=True)

        # Checkpoint identity is recorded as size+mtime rather than just the
        # name: two runs can produce files with the same name, and a mislabelled
        # rollout is worse than no rollout. Hashing 680 MB per run is too slow
        # to do by default.
        self.ckpt_stat = None
        if args.run_dir:
            p = pathlib.Path(args.run_dir) / "checkpoints" / (args.ckpt + ".ckpt")
            if p.is_file():
                st = p.stat()
                self.ckpt_stat = {
                    "path": str(p),
                    "size_bytes": st.st_size,
                    "mtime": datetime.fromtimestamp(st.st_mtime).isoformat(),
                }
            else:
                self.get_logger().warn(
                    f"checkpoint not found at {p} - identity will be recorded "
                    f"by name only. --ckpt must be the literal .ckpt filename "
                    f"without the extension; use --label to name the output "
                    f"directory differently. Is --run-dir correct?")
                near = sorted(
                    q.stem for q in (pathlib.Path(args.run_dir)
                                     / "checkpoints").glob("*.ckpt")) \
                    if (pathlib.Path(args.run_dir) / "checkpoints").is_dir() \
                    else []
                if near:
                    self.get_logger().warn(
                        "checkpoints present in that run dir: "
                        + ", ".join(near[:8])
                        + (" ..." if len(near) > 8 else ""))

        # --- rollout state ---
        self.active = False
        self.idx = self._next_index()
        self.dir = None
        self.files = {}
        self.t0_wall = None
        self.counts = {}
        self.bag_process = None

        # Point clouds are accumulated in memory and written once at the end:
        # 1024 x 6 as float16 is 12 KB per frame, so a 25 s rollout at 20 Hz
        # costs ~6 MB. float16 keeps ~3 decimal digits, which on a workspace
        # of +/-1.5 m resolves to well under a millimetre - far finer than the
        # depth sensor itself.
        self.pc_frames = []
        self.pc_times = []
        self.pc_warned = False

        # Latest value per joint, so a snapshot is only taken once all 26 are
        # known. The arm and the hand publish on /joint_states separately
        # (the dg5f driver is remapped onto it), so a single message never
        # carries the full vector - exactly as deploy.py handles it.
        self.latest = {n: None for n in JOINT_NAMES}
        self.warned_missing = False

        qos_fast = QoSProfile(depth=500,
                              reliability=ReliabilityPolicy.RELIABLE,
                              history=HistoryPolicy.KEEP_LAST)
        qos_std = QoSProfile(depth=10,
                             reliability=ReliabilityPolicy.RELIABLE,
                             history=HistoryPolicy.KEEP_LAST)

        self.create_subscription(Bool, "/policy_gate", self._gate_cb, qos_std)
        self.create_subscription(Float64MultiArray, "/predicted_action_chunk",
                                 self._chunk_cb, qos_std)
        # /target_frame is the executor's OUTPUT: chunk actions after the
        # Butterworth, the hand-expansion and the per-tick clamp. Without it
        # the only logged signals are the policy's raw actions and the
        # measured joints, so anything the executor does in between can only
        # be inferred - the smoothing analysis had to re-implement the filter
        # to guess at it. This closes that gap: policy -> target -> executed.
        self.create_subscription(PoseStamped, "/target_frame",
                                 self._target_cb, qos_std)
        self.create_subscription(JointState, "/joint_states",
                                 self._joint_cb, qos_fast)
        self.create_subscription(Float64MultiArray, "/action_buffer_status",
                                 self._buffer_cb, qos_std)
        self.create_subscription(String, "/policy_status",
                                 self._status_cb, qos_std)
        self.create_subscription(Float64MultiArray, "/policy_progress",
                                 self._progress_cb, qos_std)

        if args.log_pc:
            # BEST_EFFORT: deploy.py publishes this on a plain depth-10
            # publisher at observation rate. A RELIABLE subscriber would ask
            # for retransmits of diagnostic frames and could stall the
            # publisher - which sits in the observation callback.
            qos_pc = QoSProfile(depth=10,
                                reliability=ReliabilityPolicy.BEST_EFFORT,
                                history=HistoryPolicy.KEEP_LAST)
            self.create_subscription(PointCloud2, DEBUG_PC_TOPIC,
                                     self._pc_cb, qos_pc)

        self.get_logger().info(f"checkpoint : {args.ckpt}")
        self.get_logger().info(f"output     : {self.root}")
        self.get_logger().info(f"next index : rollout_{self.idx:03d}")
        self.get_logger().info(
            f"wrist bag  : {WRIST_COLOR_TOPIC if args.bag_wrist else 'off'}")
        self.get_logger().info(
            f"debug_pc   : {DEBUG_PC_TOPIC if args.log_pc else 'off'}")
        self.get_logger().info(
            "waiting for /policy_gate=true. Start this node BEFORE publishing "
            "the gate - `pub --once` is volatile and is not replayed.")

    # ------------------------------------------------------------------
    def _next_index(self) -> int:
        existing = [d.name for d in self.root.glob("rollout_*") if d.is_dir()]
        used = []
        for name in existing:
            try:
                used.append(int(name.split("_")[1]))
            except (IndexError, ValueError):
                continue
        return max(used) + 1 if used else 0

    def _stamp_s(self, header) -> float:
        return header.stamp.sec + header.stamp.nanosec * 1e-9

    # ------------------------------------------------------------------
    # Wrist RGB via `ros2 bag record`, as record.py does it
    #
    # A subprocess rather than cv_bridge in this node: image decoding and
    # re-encoding at 20 Hz would compete for the GIL with the callbacks that
    # have to stay responsive, and the bag writer is already optimised for
    # this. zstd message compression keeps 640x480 bgr8 at roughly 2-4 MB/s.
    # ------------------------------------------------------------------
    def _start_bag(self, episode_dir: pathlib.Path):
        bag_path = episode_dir / "wrist_bag"
        qos_path = episode_dir / "qos_overrides.yaml"
        # The camera publishes BEST_EFFORT; without this override the bag
        # subscribes RELIABLE, the QoS contract fails to match and the bag
        # silently records nothing.
        with open(qos_path, "w") as f:
            f.write(f"{WRIST_COLOR_TOPIC}:\n"
                    f"  reliability: best_effort\n"
                    f"  history: keep_last\n"
                    f"  depth: 10\n")
        cmd = [
            "ros2", "bag", "record", WRIST_COLOR_TOPIC,
            "-o", str(bag_path),
            "--qos-profile-overrides-path", str(qos_path),
            "--compression-mode", "message",
            "--compression-format", "zstd",
        ]
        try:
            self.bag_process = subprocess.Popen(
                cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                preexec_fn=os.setsid)
        except Exception as e:
            # Losing the video must not lose the trajectory, which is the
            # part that answers the height question.
            self.bag_process = None
            self.get_logger().error(
                f"could not start `ros2 bag record` ({e}); continuing without "
                f"wrist video.")

    def _stop_bag(self):
        if self.bag_process is None:
            return
        try:
            # SIGINT to the process GROUP: ros2 bag needs the interrupt to
            # close and index the file. SIGKILL leaves an unreadable bag.
            os.killpg(os.getpgid(self.bag_process.pid), signal.SIGINT)
            self.bag_process.wait(timeout=10.0)
        except subprocess.TimeoutExpired:
            self.get_logger().warn("bag did not exit on SIGINT; killing.")
            try:
                os.killpg(os.getpgid(self.bag_process.pid), signal.SIGKILL)
                self.bag_process.wait(timeout=5.0)
            except Exception:
                pass
        except Exception as e:
            self.get_logger().warn(f"error stopping bag: {e}")
        finally:
            self.bag_process = None

    def _rel(self) -> float:
        """Seconds since gate open. Absolute wall time is in meta.json."""
        return time.time() - self.t0_wall if self.t0_wall else 0.0

    def _write(self, key: str, record: dict):
        f = self.files.get(key)
        if f is None:
            return
        f.write(json.dumps(record) + "\n")
        self.counts[key] = self.counts.get(key, 0) + 1

    # ------------------------------------------------------------------
    def _gate_cb(self, msg: Bool):
        if msg.data and not self.active:
            self._start()
        elif not msg.data and self.active:
            self._stop("gate_closed")
        elif msg.data and self.active:
            self.get_logger().warn(
                "gate=true while already recording - ignored. The controller "
                "treats this the same way.")

    def _start(self):
        self.dir = self.root / f"rollout_{self.idx:03d}"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.t0_wall = time.time()
        self.counts = {}

        # Line-buffered jsonl rather than a single array dumped at the end:
        # if the process is killed mid-rollout the data up to that point is
        # still readable. The npz written at _stop() is a convenience, not the
        # source of truth.
        for key in ("chunks", "joint_states", "buffer_status", "events",
                    "target_frame"):
            self.files[key] = open(self.dir / f"{key}.jsonl", "w",
                                   buffering=1)

        meta = {
            "checkpoint": self.args.ckpt,
            "label": self.ckpt_label,
            "checkpoint_file": self.ckpt_stat,
            "run_dir": self.args.run_dir,
            "num_inference_steps": self.args.inference_steps,
            "notes": self.args.notes,
            "rollout_index": self.idx,
            "started_wall": datetime.now().isoformat(),
            "started_unix": self.t0_wall,
            "joint_names": JOINT_NAMES,
            # Logged exactly as received. deploy.py converts the hand block to
            # degrees before handing it to the policy; that conversion is NOT
            # applied here, so the analysis side has to do it. Logging the wire
            # value keeps this node from silently diverging from deploy.
            "joint_units": {"arm": "rad", "hand": "rad"},
            "chunk_format": {
                "data[0]": "anchor_step (-1 = no anchor)",
                "data[1:]": f"flattened (N, {WIRE_ACTION_DIM})",
                "action_layout": "7 pose (xyz + quat wxyz) + 20 hand joints (deg)",
            },
            "git_commit": _git_commit(),
        }
        with open(self.dir / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)

        if self.args.bag_wrist:
            self._start_bag(self.dir)

        self.pc_frames = []
        self.pc_times = []

        self.active = True
        self.get_logger().info(f"[REC] rollout_{self.idx:03d} started")

    def _stop(self, reason: str):
        if not self.active:
            return
        self.active = False
        duration = self._rel()

        # Stopped before the jsonl files are closed so the bag's own flush
        # overlaps with nothing that could still be written.
        self._stop_bag()

        for f in self.files.values():
            try:
                f.close()
            except Exception:
                pass
        self.files = {}

        self._consolidate()

        summary = {
            "rollout_index": self.idx,
            "stop_reason": reason,
            "duration_s": round(duration, 3),
            "counts": self.counts,
            "pc_frames": len(self.pc_times),
            "stopped_wall": datetime.now().isoformat(),
        }
        with open(self.dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        self.get_logger().info(
            f"[REC] rollout_{self.idx:03d} stopped "
            f"{duration:.1f}s chunks={self.counts.get('chunks', 0)} "
            f"joint_samples={self.counts.get('joint_states', 0)}")
        self.idx += 1
        self.dir = None
        self.t0_wall = None

    def _consolidate(self):
        """jsonl -> npz, for analysis without a parser.

        Chunk lengths vary (the latency shift trims the front), so the chunks
        are stored CSR-style - concatenated rows plus an offset array - rather
        than padded into a rectangle that would need a mask to read correctly.
        """
        try:
            rows, anchors, times, offsets = [], [], [], [0]
            path = self.dir / "chunks.jsonl"
            if path.is_file():
                with open(path) as f:
                    for line in f:
                        rec = json.loads(line)
                        a = np.asarray(rec["actions"], dtype=np.float32)
                        rows.append(a)
                        anchors.append(rec["anchor_step"])
                        times.append(rec["t"])
                        offsets.append(offsets[-1] + a.shape[0])

            js_t, js_pos = [], []
            path = self.dir / "joint_states.jsonl"
            if path.is_file():
                with open(path) as f:
                    for line in f:
                        rec = json.loads(line)
                        js_t.append(rec["t"])
                        js_pos.append(rec["positions"])

            tf_t, tf_pose, tf_stamp = [], [], []
            path = self.dir / "target_frame.jsonl"
            if path.is_file():
                with open(path) as f:
                    for line in f:
                        rec = json.loads(line)
                        tf_t.append(rec["t"])
                        tf_pose.append(rec["pose"])
                        tf_stamp.append(rec.get("stamp", 0.0))

            arrays = dict(
                chunk_actions=(np.concatenate(rows, axis=0) if rows
                               else np.zeros((0, WIRE_ACTION_DIM), np.float32)),
                chunk_offsets=np.asarray(offsets, np.int64),
                chunk_anchors=np.asarray(anchors, np.int64),
                chunk_t=np.asarray(times, np.float64),
                js_t=np.asarray(js_t, np.float64),
                js_pos=(np.asarray(js_pos, np.float32) if js_pos
                        else np.zeros((0, len(JOINT_NAMES)), np.float32)),
                joint_names=np.asarray(JOINT_NAMES),
                tf_t=np.asarray(tf_t, np.float64),
                tf_pose=(np.asarray(tf_pose, np.float64) if tf_pose
                         else np.zeros((0, 7), np.float64)),
                tf_stamp=np.asarray(tf_stamp, np.float64),
            )
            if self.pc_frames:
                # Stacked only if every frame has the same point count, which
                # deploy.py's FPS step guarantees. A ragged cloud would mean
                # the preprocessing changed mid-rollout, so it is worth a
                # warning rather than a silent object array.
                shapes = {f.shape for f in self.pc_frames}
                if len(shapes) == 1:
                    arrays["pc"] = np.stack(self.pc_frames, axis=0)
                    arrays["pc_t"] = np.asarray(self.pc_times, np.float64)
                else:
                    self.get_logger().warn(
                        f"point clouds have inconsistent shapes {shapes}; "
                        f"storing the first {len(self.pc_frames)} "
                        f"concatenated instead of stacked.")
                    arrays["pc_flat"] = np.concatenate(self.pc_frames, axis=0)
                    arrays["pc_t"] = np.asarray(self.pc_times, np.float64)

            np.savez_compressed(self.dir / "rollout.npz", **arrays)
        except Exception as e:
            # A failed convenience export must not cost the jsonl, which is
            # already on disk and complete.
            self.get_logger().error(
                f"npz consolidation failed ({e}); jsonl files are intact.")

    # ------------------------------------------------------------------
    def _chunk_cb(self, msg: Float64MultiArray):
        if not self.active:
            return
        data = list(msg.data)
        if not data:
            return
        anchor = data[0]
        flat = data[1:]
        if len(flat) % WIRE_ACTION_DIM != 0:
            self.get_logger().warn(
                f"chunk payload {len(flat)} is not a multiple of "
                f"{WIRE_ACTION_DIM}; logging raw.")
            self._write("chunks", {"t": round(self._rel(), 4),
                                   "anchor_step": int(anchor),
                                   "malformed": True, "raw": flat})
            return
        actions = np.asarray(flat, dtype=np.float32).reshape(
            -1, WIRE_ACTION_DIM)
        self._write("chunks", {
            "t": round(self._rel(), 4),
            "anchor_step": int(anchor),
            "n_steps": int(actions.shape[0]),
            "actions": np.round(actions, 6).tolist(),
        })

    def _target_cb(self, msg: PoseStamped):
        if not self.active:
            return
        p = msg.pose.position
        q = msg.pose.orientation
        self._write("target_frame", {
            "t": round(time.time() - self.t0_wall, 4),
            # Header stamp as well as arrival time: the executor stamps at
            # publish, so the difference is DDS transport, which matters when
            # judging whether a lag is the controller's or the transport's.
            "stamp": round(msg.header.stamp.sec
                           + msg.header.stamp.nanosec * 1e-9, 4),
            "pose": [round(v, 6) for v in
                     (p.x, p.y, p.z, q.x, q.y, q.z, q.w)],
        })

    def _joint_cb(self, msg: JointState):
        for name, pos in zip(msg.name, msg.position):
            if name in self.latest:
                self.latest[name] = pos
        if not self.active:
            return
        # Snapshot on arm messages only, once every joint is known: that is the
        # same trigger deploy.py uses, so the logged vector corresponds to an
        # instant the policy could also have observed.
        if "shoulder_pan_joint" not in msg.name:
            return
        if any(v is None for v in self.latest.values()):
            if not self.warned_missing:
                missing = [n for n, v in self.latest.items() if v is None]
                self.get_logger().warn(
                    f"{len(missing)} joints never seen (e.g. {missing[:4]}); "
                    f"not logging joint states until complete. Is the dg5f "
                    f"driver remapped onto /joint_states?")
                self.warned_missing = True
            return
        self._write("joint_states", {
            "t": round(self._rel(), 4),
            "t_ros": round(self._stamp_s(msg.header), 6),
            "positions": [round(self.latest[n], 6) for n in JOINT_NAMES],
        })

    def _buffer_cb(self, msg: Float64MultiArray):
        if not self.active or len(msg.data) < 2:
            return
        d = list(msg.data)
        self._write("buffer_status", {
            "t": round(self._rel(), 4),
            "depth": int(d[0]),
            "cursor": int(d[1]),
            "next_step": int(d[2]) if len(d) >= 3 else int(d[1]),
        })

    def _pc_cb(self, msg: PointCloud2):
        if not self.active:
            return
        try:
            # read_points gives a structured array; the field names depend on
            # what deploy.py declared (xyz, or xyz+rgb when the cloud carries
            # colour). Read whatever is there instead of assuming six fields.
            names = [f.name for f in msg.fields]
            pts = point_cloud2.read_points(
                msg, field_names=names, skip_nans=False)
            arr = np.stack([np.asarray(pts[n]) for n in names], axis=-1)
            self.pc_frames.append(arr.astype(np.float16))
            self.pc_times.append(round(self._rel(), 4))
        except Exception as e:
            if not self.pc_warned:
                self.pc_warned = True
                self.get_logger().warn(
                    f"could not parse {DEBUG_PC_TOPIC} ({e}); point clouds "
                    f"will not be logged for this rollout.")

    def _status_cb(self, msg: String):
        if self.active:
            self._write("events", {"t": round(self._rel(), 4),
                                   "type": "status", "value": msg.data})

    def _progress_cb(self, msg: Float64MultiArray):
        if self.active and msg.data:
            self._write("events", {"t": round(self._rel(), 4),
                                   "type": "progress",
                                   "value": float(msg.data[0])})


def main():
    ap = argparse.ArgumentParser(
        description="Log a gated policy rollout (commanded vs executed).")
    ap.add_argument("--ckpt", required=True,
                    help="checkpoint filename WITHOUT .ckpt, as passed to "
                         "deploy.py's ckpt_file. Required with no default: a "
                         "silently mislabelled rollout is the problem this "
                         "logger exists to prevent.")
    ap.add_argument("--label", default=None,
                    help="name for the output directory. Defaults to --ckpt. "
                         "Use it when the checkpoint filename alone is "
                         "ambiguous, e.g. --label foil_lifting_relative_e40")
    ap.add_argument("--run-dir", default=None,
                    help="run directory, to record the checkpoint's size and "
                         "mtime as a verifiable identity")
    ap.add_argument("--inference-steps", type=int, default=None,
                    help="num_inference_steps used at deploy time")
    ap.add_argument("--notes", default="",
                    help="free text, e.g. pallet height or lighting")
    ap.add_argument("--root", default=DEFAULT_ROOT,
                    help=f"output root (default {DEFAULT_ROOT})")
    ap.add_argument("--no-bag-wrist", dest="bag_wrist", action="store_false",
                    help=f"do not record {WRIST_COLOR_TOPIC} to a rosbag "
                         f"(~2-4 MB/s with zstd)")
    ap.add_argument("--no-pc", dest="log_pc", action="store_false",
                    help=f"do not log {DEBUG_PC_TOPIC} (~12 KB/frame as "
                         f"float16, ~6 MB per 25 s rollout)")
    ap.set_defaults(bag_wrist=True, log_pc=True)
    args = ap.parse_args()

    rclpy.init()
    node = RolloutLogger(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node._stop("interrupted")
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()