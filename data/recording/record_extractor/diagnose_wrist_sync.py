#!/usr/bin/env python3
"""Diagnose the ZED-to-wrist sync spikes in an extracted episode.

Tests whether the evenly spaced single-frame spikes in the ZED-to-wrist
histogram are a startup ramp (every early ZED frame clamping onto the same
first wrist message) rather than genuine periodic jitter.

Usage:
  python3 diagnose_wrist_sync.py <episode_dir> [--color-topic TOPIC]
"""
import argparse
import json
import os
import sys

import numpy as np
from rosbags.rosbag2 import Reader
from rosbags.typesys import Stores, get_typestore


def header_stamp_ns(msg, fallback):
    try:
        ns = int(msg.header.stamp.sec * 1e9 + msg.header.stamp.nanosec)
        if ns > 0:
            return ns
    except Exception:
        pass
    return fallback


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("episode_dir")
    ap.add_argument("--color-topic", default=None)
    args = ap.parse_args()

    ep = args.episode_dir
    zed_ts = np.load(os.path.join(ep, "frame_timestamps_ns.npy")).astype(np.int64)

    color_topic = args.color_topic
    meta = os.path.join(ep, "wrist_camera_topics.json")
    if color_topic is None and os.path.exists(meta):
        with open(meta) as f:
            color_topic = json.load(f)["color_topic"]
    if color_topic is None:
        sys.exit("No color topic given and no wrist_camera_topics.json found.")

    bag = os.path.join(ep, "realsense_bag")
    typestore = get_typestore(Stores.ROS2_JAZZY)

    wrist_ts = []
    with Reader(bag) as reader:
        conns = [c for c in reader.connections if c.topic == color_topic]
        if not conns:
            sys.exit(f"Topic {color_topic} not in bag. "
                     f"Have: {sorted({c.topic for c in reader.connections})}")
        for conn, ts, raw in reader.messages(connections=conns):
            msg = typestore.deserialize_cdr(raw, conn.msgtype)
            wrist_ts.append(header_stamp_ns(msg, ts))
    wrist_ts = np.sort(np.array(wrist_ts, dtype=np.int64))

    # Reproduce the extractor's nearest-neighbour matching.
    idx = np.array([np.argmin(np.abs(wrist_ts - t)) for t in zed_ts])
    gaps_ms = np.abs(wrist_ts[idx] - zed_ts) / 1e6

    zed_dt = np.diff(zed_ts) / 1e6
    print(f"ZED frames        : {len(zed_ts)}")
    print(f"ZED frame period  : median {np.median(zed_dt):.2f} ms")
    print(f"wrist msgs        : {len(wrist_ts)}")
    print(f"wrist period      : median {np.median(np.diff(wrist_ts))/1e6:.2f} ms")

    lead_ms = (wrist_ts[0] - zed_ts[0]) / 1e6
    print(f"\nfirst wrist msg - first ZED frame = {lead_ms:+.1f} ms")
    print("  (positive => wrist stream started LATE; ZED recorded alone)")

    # How many ZED frames clamp onto the very first wrist message?
    n_clamped = int((idx == 0).sum())
    print(f"ZED frames matched to wrist msg #0: {n_clamped}")
    if n_clamped > 1:
        print("  -> these are duplicates of the same image:")
        for i in range(n_clamped):
            print(f"     frame_{i:04d}  gap {gaps_ms[i]:7.1f} ms")

    print("\nLargest gaps overall:")
    for i in np.argsort(gaps_ms)[-8:][::-1]:
        print(f"  frame_{i:04d}  gap {gaps_ms[i]:7.1f} ms")

    tail = gaps_ms[n_clamped:] if n_clamped > 0 else gaps_ms
    print(f"\nExcluding the {n_clamped} clamped startup frames:")
    print(f"  mean {tail.mean():.2f} ms | max {tail.max():.2f} ms "
          f"| std {tail.std():.2f} ms")

    if n_clamped > 1 and lead_ms > 0:
        print(f"\nCONFIRMED: startup ramp. The bag began ~{lead_ms:.0f} ms after "
              f"the ZED, so the first {n_clamped} wrist frames are stale "
              f"copies of one image.")
    else:
        print("\nNot a startup ramp - the spikes have another cause.")


if __name__ == "__main__":
    main()