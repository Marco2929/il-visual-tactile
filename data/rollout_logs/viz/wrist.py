#!/usr/bin/env python3
"""Read the wrist camera rosbag that log_rollout.py records alongside a run.

This is the stream that answers "what did the camera see when the hand
closed", which no amount of joint data can. It is kept out of data.py because
it is the one part of the loader that can fail for environmental reasons
rather than because the data is bad: the bag is MCAP with per-message zstd
compression holding sensor_msgs/Image, and decoding it needs a reader that is
not guaranteed to be installed.

Three readers are tried, in order of how little they assume:

  1. rosbag2_py   - present inside the ROS container, deserialises properly.
  2. mcap_ros2    - pip-installable, no ROS environment needed. This is the
                    one that makes the bags readable on the host.
  3. mcap + manual CDR - last resort. sensor_msgs/Image is a flat enough
                    layout to unpack by hand, and doing so beats telling the
                    user their 29 GB of video is unreadable.

If none is available the caller gets a clear reason and the rest of the
visualisation carries on without images, which is why every entry point here
returns rather than raises.

Time base: bag timestamps are nanoseconds since the epoch; meta.json's
`started_unix` is the t=0 of js_t and pc_t. Subtracting it puts the frames on
the same axis as every other stream, which is the whole point of loading
them.

One trap, found the hard way: log_rollout.py writes these bags with
`compression_mode: MESSAGE`, and rosbag2_py hands such payloads back STILL
COMPRESSED - they start with the zstd magic 28 b5 2f fd instead of a CDR
encapsulation header. Deserialising that fails with a misleading "invalid
data size". So every payload is checked for the magic and decompressed first;
see _maybe_decompress.
"""
import pathlib
import shutil
import struct
import subprocess

import numpy as np

WRIST_TOPIC = "/cam_gemini/color/image_raw"

INSTALL_HINT = (
    "no MCAP reader available - install one with:\n"
    "    pip install mcap mcap-ros2-support zstandard\n"
    "or run inside the ROS container, where rosbag2_py is present.")

ZSTD_MAGIC = b"\x28\xb5\x2f\xfd"


def _zstd_decompress(buf):
    """Decompress a zstd frame using whatever backend exists.

    Four options rather than one import, because the ROS container ships
    none of the Python zstd bindings but the data is unreadable without
    one, and the `zstd` CLI is often there when the bindings are not.
    """
    try:
        import zstandard
        return zstandard.ZstdDecompressor().decompress(buf)
    except ImportError:
        pass
    except Exception:
        # Frames written without a content-size header need streaming.
        import zstandard
        with zstandard.ZstdDecompressor().stream_reader(buf) as rd:
            return rd.read()
    try:
        import zstd
        return zstd.decompress(buf)
    except ImportError:
        pass
    try:
        import cramjam
        return bytes(cramjam.zstd.decompress(buf))
    except ImportError:
        pass

    exe = shutil.which("zstd") or shutil.which("unzstd")
    if exe:
        r = subprocess.run([exe, "-dc"], input=buf, capture_output=True)
        if r.returncode == 0:
            return r.stdout
        raise RuntimeError(f"zstd CLI failed: {r.stderr[:200]!r}")

    raise ImportError(
        "message-compressed bag needs a zstd backend: "
        "pip install zstandard (or apt install zstd)")


def _maybe_decompress(buf):
    """rosbag2 with compression_mode: MESSAGE returns compressed payloads."""
    if buf[:4] == ZSTD_MAGIC:
        return _zstd_decompress(buf)
    return buf


def bag_path(rollout_dir):
    d = pathlib.Path(rollout_dir) / "wrist_bag"
    return d if (d / "metadata.yaml").exists() else None


def available(rollout_dir):
    return bag_path(rollout_dir) is not None


def _decode_image(width, height, encoding, step, data):
    """sensor_msgs/Image payload -> (H, W, 3) uint8 RGB.

    Only the encodings this camera actually publishes are handled; an
    unknown one returns None rather than a silently wrong picture.
    """
    buf = np.frombuffer(data, dtype=np.uint8)
    enc = encoding.lower()

    if enc in ("rgb8", "bgr8"):
        chan = 3
    elif enc in ("rgba8", "bgra8"):
        chan = 4
    elif enc in ("mono8", "8uc1"):
        chan = 1
    else:
        return None

    # step is the row stride in bytes and may exceed width*chan when rows
    # are padded; indexing by step and slicing is the only correct way.
    if step and step != width * chan:
        if buf.size < step * height:
            return None
        img = buf[:step * height].reshape(height, step)[:, :width * chan]
        img = img.reshape(height, width, chan)
    else:
        if buf.size < width * height * chan:
            return None
        img = buf[:width * height * chan].reshape(height, width, chan)

    if chan == 1:
        img = np.repeat(img, 3, axis=2)
    elif enc.startswith("bgr"):
        img = img[:, :, [2, 1, 0]]
    else:
        img = img[:, :, :3]
    return np.ascontiguousarray(img)


def _unpack_cdr_image(raw):
    """Hand-unpack a CDR-encoded sensor_msgs/Image.

    Layout: 4-byte encapsulation header, then header{stamp{sec:i32,
    nanosec:u32}, frame_id:string}, height:u32, width:u32, encoding:string,
    is_bigendian:u8, step:u32, data:sequence<u8>. Strings and sequences are
    length-prefixed and every field is aligned to its own size, which is the
    part that is easy to get wrong.
    """
    if len(raw) < 4:
        return None
    little = raw[1] & 1
    end = "<" if little else ">"
    off = 4

    def align(n):
        nonlocal off
        pad = (-(off - 4)) % n
        off += pad

    def u32():
        nonlocal off
        align(4)
        v = struct.unpack_from(end + "I", raw, off)[0]
        off += 4
        return v

    def i32():
        nonlocal off
        align(4)
        v = struct.unpack_from(end + "i", raw, off)[0]
        off += 4
        return v

    def string():
        nonlocal off
        n = u32()
        s = raw[off:off + n - 1].decode("utf-8", "replace") if n else ""
        off += n
        return s

    try:
        sec = i32()
        nanosec = u32()
        string()                      # frame_id
        height = u32()
        width = u32()
        encoding = string()
        off += 1                      # is_bigendian
        step = u32()
        n = u32()
        data = raw[off:off + n]
    except (struct.error, IndexError):
        return None

    return dict(stamp=sec + nanosec * 1e-9, height=height, width=width,
                encoding=encoding, step=step, data=data)


# ----------------------------------------------------------- readers -----
def _iter_rosbag2_py(bag, topic):
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from sensor_msgs.msg import Image

    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(bag), storage_id="mcap"),
        rosbag2_py.ConverterOptions("", ""))
    reader.set_filter(rosbag2_py.StorageFilter(topics=[topic]))
    while reader.has_next():
        _, raw, t_ns = reader.read_next()
        raw = _maybe_decompress(raw)
        m = deserialize_message(raw, Image)
        yield t_ns, dict(height=m.height, width=m.width,
                         encoding=m.encoding, step=m.step, data=bytes(m.data))


def _iter_mcap_ros2(bag, topic):
    from mcap_ros2.reader import read_ros2_messages

    for m in read_ros2_messages(str(next(bag.glob("*.mcap"))),
                                topics=[topic]):
        msg = m.ros_msg
        yield m.log_time_ns, dict(height=msg.height, width=msg.width,
                                  encoding=msg.encoding, step=msg.step,
                                  data=bytes(msg.data))


def _iter_mcap_raw(bag, topic):
    from mcap.reader import make_reader

    with open(next(bag.glob("*.mcap")), "rb") as f:
        for schema, channel, message in make_reader(f).iter_messages(
                topics=[topic]):
            img = _unpack_cdr_image(_maybe_decompress(message.data))
            if img is not None:
                yield message.log_time, img


READERS = [("rosbag2_py", _iter_rosbag2_py),
           ("mcap_ros2", _iter_mcap_ros2),
           ("mcap (raw CDR)", _iter_mcap_raw)]


def load_frames(rollout_dir, started_unix, topic=WRIST_TOPIC,
                max_frames=200, size=None):
    """-> (times, frames, note).

    times are seconds relative to started_unix, so they line up with js_t
    and pc_t. frames is a list of (H, W, 3) uint8 RGB arrays.

    Decimation happens on the DECODED count, not by skipping bag entries,
    because the reader has to walk the file sequentially anyway; the cost
    saved is the decode and the memory, which is where it actually is - 2402
    frames at 1280x720 is 6.6 GB in RAM.

    On failure returns ([], [], reason) rather than raising: a missing image
    decoder should not take the rest of the visualisation down with it.
    """
    bag = bag_path(rollout_dir)
    if bag is None:
        return [], [], "no wrist_bag in this rollout"

    errors = []
    for name, it in READERS:
        try:
            entries = list(it(bag, topic))
        except ImportError as e:
            errors.append(f"{name}: not installed ({e.name})")
            continue
        except Exception as e:
            errors.append(f"{name}: {type(e).__name__}: {e}")
            continue

        if not entries:
            errors.append(f"{name}: no messages on {topic}")
            continue

        keep = range(len(entries))
        if max_frames and len(entries) > max_frames:
            keep = np.linspace(0, len(entries) - 1, max_frames).astype(int)

        times, frames = [], []
        for i in keep:
            t_ns, m = entries[i]
            img = _decode_image(m["width"], m["height"], m["encoding"],
                                m["step"], m["data"])
            if img is None:
                continue
            if size:
                import cv2
                img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
            times.append(t_ns / 1e9 - started_unix)
            frames.append(img)

        if frames:
            return (times, frames,
                    f"{len(frames)} of {len(entries)} frames via {name}")
        errors.append(f"{name}: no frame decoded")

    return [], [], "; ".join(errors) or INSTALL_HINT