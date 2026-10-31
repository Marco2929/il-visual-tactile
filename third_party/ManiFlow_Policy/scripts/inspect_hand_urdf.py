"""Dump the actuated-joint structure of a hand URDF.

What an action-space decision actually needs: how many joints are NOT fixed,
how they group per finger, and each one's limits/axis. A 20-DoF claim is only
meaningful once `fixed` and `mimic` joints are excluded -- mimic joints are
coupled to another joint and must NOT get their own action dimension.

    docker exec ros2_dev /ros2_ws/venv/bin/python inspect_hand_urdf.py <urdf>
"""
import re
import sys
import xml.etree.ElementTree as ET
from collections import OrderedDict

path = sys.argv[1]
root = ET.parse(path).getroot()

print(f"=== {path} ===")
print(f"robot name: {root.get('name')}")

links = [l.get("name") for l in root.findall("link")]
print(f"links: {len(links)}")

by_type = OrderedDict()
mimics = []
actuated = []

for j in root.findall("joint"):
    jtype = j.get("type")
    name = j.get("name")
    by_type.setdefault(jtype, []).append(name)

    mimic = j.find("mimic")
    if mimic is not None:
        mimics.append((name, mimic.get("joint"), mimic.get("multiplier"),
                       mimic.get("offset")))
        continue

    if jtype in ("fixed", "floating"):
        continue

    limit = j.find("limit")
    axis = j.find("axis")
    actuated.append({
        "name": name,
        "type": jtype,
        "lower": float(limit.get("lower", "nan")) if limit is not None else None,
        "upper": float(limit.get("upper", "nan")) if limit is not None else None,
        "effort": float(limit.get("effort", "nan")) if limit is not None else None,
        "axis": axis.get("xyz") if axis is not None else None,
        "parent": j.find("parent").get("link") if j.find("parent") is not None else None,
        "child": j.find("child").get("link") if j.find("child") is not None else None,
    })

print("\n--- joints by type ---")
for jtype, names in by_type.items():
    print(f"  {jtype:12s} {len(names)}")

if mimics:
    print("\n--- MIMIC joints (coupled; NOT separate action dims) ---")
    for name, target, mult, off in mimics:
        print(f"  {name:28s} mimics {target}  mult={mult} offset={off}")

print(f"\n--- ACTUATED joints: {len(actuated)} ---")


def finger_of(name):
    """Group by the finger index Tesollo encodes in the joint name
    (e.g. `..._f1_1` -> finger 1). Falls back to 'other'."""
    m = re.search(r"f(\d+)", name)
    return f"finger_{m.group(1)}" if m else "other"


groups = OrderedDict()
for a in actuated:
    groups.setdefault(finger_of(a["name"]), []).append(a)

for g, members in groups.items():
    print(f"\n  [{g}] {len(members)} DoF")
    for a in members:
        lo, up = a["lower"], a["upper"]
        rng = f"[{lo:+.3f}, {up:+.3f}]" if lo is not None else "(no limit)"
        span = f"{up - lo:.3f} rad" if lo is not None else ""
        print(f"    {a['name']:30s} {a['type']:10s} {rng:22s} "
              f"{span:12s} axis={a['axis']}")

print(f"\n{'=' * 60}")
print(f"TOTAL ACTUATED DoF: {len(actuated)}")
print(f"fingers detected:   {len([g for g in groups if g != 'other'])}")