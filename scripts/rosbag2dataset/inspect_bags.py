#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Inspect one or more ROS1 bag files and report topic inventory.

For each .bag file this prints:
  * overall size, duration, frame count per topic
  * the message type of every topic
  * a tick-mark for every topic we *need* for rosbag2dataset_5hz.py
    (joint_states, end_effector_pose, head_camera rgb+depth, base_scan,
     amcl_icp_pose, tf/tf_static) and for the new particle-cloud feature
     (/particlecloud).

Usage:
    python rosbag2dataset/inspect_bags.py <bag_or_dir> [<bag> ...]
    python rosbag2dataset/inspect_bags.py /Volumes/External/Workspace/datasets

The script first tries ROS1's native ``rosbag`` Python API (which runs
on Linux with a sourced ROS env). If that import fails — e.g. on macOS
where ROS1 is not installed — it falls back to the cross-platform
``rosbags`` library (``pip install rosbags``).
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import Counter, OrderedDict
from typing import Dict, Iterable, List, Tuple


# ---------------------------------------------------------------------------
# Topics we consume / record. Missing topics are flagged so it is immediately
# obvious whether a bag carries what the extractor expects.
# ---------------------------------------------------------------------------

REQUIRED_TOPICS: Dict[str, str] = {
    "/joint_states":                          "sensor_msgs/JointState",
    "/end_effector_pose":                     "geometry_msgs/PoseStamped",
    "/head_camera/rgb/image_raw/compressed":  "sensor_msgs/CompressedImage",
    "/head_camera/depth_registered/image_raw": "sensor_msgs/Image",
    "/base_scan":                             "sensor_msgs/LaserScan",
    "/amcl_icp_pose":                         "geometry_msgs/PoseStamped",
    "/tf":                                    "tf2_msgs/TFMessage",
    "/tf_static":                             "tf2_msgs/TFMessage",
}

# Localization-uncertainty source (new). AMCL historically used
# geometry_msgs/PoseArray; newer distros publish nav_msgs/ParticleCloud
# (PoseWithCovariance[] + weights). Both are accepted downstream.
PARTICLE_TOPICS: Dict[str, Tuple[str, ...]] = {
    "/particlecloud":                         ("geometry_msgs/PoseArray",
                                               "nav_msgs/ParticleCloud"),
    "/amcl/particle_cloud":                   ("geometry_msgs/PoseArray",
                                               "nav_msgs/ParticleCloud"),
}


# ---------------------------------------------------------------------------
# Bag backend
# ---------------------------------------------------------------------------

def _inspect_with_rosbag(path: str):
    """Use the ROS1 native python module (only works on a ROS-enabled host)."""
    import rosbag  # noqa: F401  (Linux-only)

    with rosbag.Bag(path, "r") as bag:
        info = bag.get_type_and_topic_info()
        topics = OrderedDict()
        for tname, tinfo in info.topics.items():
            topics[tname] = {
                "type":  tinfo.msg_type,
                "count": tinfo.message_count,
                "freq":  tinfo.frequency or 0.0,
            }
        return {
            "start":    bag.get_start_time(),
            "end":      bag.get_end_time(),
            "duration": bag.get_end_time() - bag.get_start_time(),
            "size":     bag.size,
            "topics":   topics,
        }


def _inspect_with_rosbags(path: str):
    """Cross-platform fallback via the ``rosbags`` pip package."""
    try:
        from rosbags.rosbag1 import Reader  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "Neither `rosbag` nor `rosbags` is importable. "
            "Install one of:\n"
            "  (Linux/ROS)  source /opt/ros/<distro>/setup.bash\n"
            "  (any OS)     pip install rosbags"
        ) from e

    with Reader(path) as reader:
        topics = OrderedDict()
        counts = Counter()
        for c in reader.connections:
            counts[c.topic] += c.msgcount
            topics[c.topic] = {
                "type":  c.msgtype,
                "count": counts[c.topic],
                "freq":  0.0,  # rosbags doesn't surface per-topic Hz
            }
        dur = (reader.end_time - reader.start_time) / 1e9
        return {
            "start":    reader.start_time / 1e9,
            "end":      reader.end_time   / 1e9,
            "duration": dur,
            "size":     os.path.getsize(path),
            "topics":   topics,
        }


def inspect(path: str) -> dict:
    try:
        return _inspect_with_rosbag(path)
    except ImportError:
        return _inspect_with_rosbags(path)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _fmt_size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:6.1f}{unit}"
        n /= 1024
    return f"{n:.1f}PB"


def _norm_type(t: str) -> str:
    """Normalize ROS type strings across backends.

    ROS1 native prints ``sensor_msgs/JointState``; the ``rosbags`` library
    prints ``sensor_msgs/msg/JointState``. Strip the optional ``msg/`` so
    equality comparisons work across backends.
    """
    return t.replace("/msg/", "/")


def report(bag_path: str, meta: dict) -> None:
    print("=" * 78)
    print(f"BAG  {bag_path}")
    print(f"     size={_fmt_size(meta['size'])}  "
          f"duration={meta['duration']:.1f}s  "
          f"topics={len(meta['topics'])}")
    print("-" * 78)
    print(f"{'topic':<50}{'type':<28}{'count':>8}")
    for t, info in meta["topics"].items():
        print(f"{t:<50}{info['type']:<28}{info['count']:>8}")

    # Coverage of required topics
    missing_required = [t for t in REQUIRED_TOPICS if t not in meta["topics"]]
    part_topics = [t for t in PARTICLE_TOPICS if t in meta["topics"]]

    print("-" * 78)
    print("required topic coverage:")
    for t, expected_type in REQUIRED_TOPICS.items():
        got = meta["topics"].get(t)
        if got is None:
            print(f"  [MISS] {t}  (expected {expected_type})")
        else:
            tag = "OK" if _norm_type(got["type"]) == expected_type \
                else "TYPE?"
            print(f"  [ {tag:<4}] {t}  {got['type']}  ({got['count']} msgs)")

    print("localization-uncertainty (particles):")
    if not part_topics:
        print("  [MISS] none of /particlecloud, /amcl/particle_cloud present")
    else:
        for t in part_topics:
            got = meta["topics"][t]
            ok = _norm_type(got["type"]) in PARTICLE_TOPICS[t]
            tag = "OK" if ok else "TYPE?"
            print(f"  [ {tag:<4}] {t}  {got['type']}  ({got['count']} msgs)")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _iter_bags(paths: Iterable[str]) -> List[str]:
    out: List[str] = []
    for p in paths:
        if os.path.isdir(p):
            for f in sorted(os.listdir(p)):
                if f.endswith(".bag") and not f.startswith("._"):
                    out.append(os.path.join(p, f))
        elif os.path.isfile(p) and p.endswith(".bag"):
            out.append(p)
        else:
            print(f"skip: {p} is not a .bag or directory", file=sys.stderr)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("paths", nargs="+",
                    help="bag files or directories containing bag files")
    args = ap.parse_args()

    bags = _iter_bags(args.paths)
    if not bags:
        print("no .bag files found", file=sys.stderr)
        return 1

    ok = True
    for b in bags:
        try:
            report(b, inspect(b))
        except Exception as e:
            ok = False
            print(f"ERROR reading {b}: {e}", file=sys.stderr)
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
