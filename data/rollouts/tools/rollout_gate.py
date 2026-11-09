#!/usr/bin/env python3
"""Rollout Gate CLI: Move robot to designated start position and gate policy execution.

Replaces manual gate toggles:
    ros2 topic pub --once /policy_gate std_msgs/msg/Bool "{data: true}"
    ros2 topic pub --once /policy_gate std_msgs/msg/Bool "{data: false}"

Instead, this tool:
  1. Sends the target start position (P1..P5) and height (mitte, oben, unten)
     to the executor (control_ur_tesollo_cartesian).
  2. The executor smoothly navigates the robot arm directly above the object.
  3. Once the target position is reached (< 15 mm), the policy stream is automatically
     activated (policy_gate = True).
  4. Pressing [ENTER] or Ctrl+C (or running 'rollout_gate.py stop') deactivates
     the policy and holds the arm in place.

USAGE:
    # Interactive menu:
    python3 data/rollouts/tools/rollout_gate.py

    # Direct start with position (default height: mitte):
    python3 data/rollouts/tools/rollout_gate.py start P1
    python3 data/rollouts/tools/rollout_gate.py P1

    # Start with specific height:
    python3 data/rollouts/tools/rollout_gate.py start P2 --height unten
    python3 data/rollouts/tools/rollout_gate.py P3 --height oben

    # Non-blocking / fire-and-forget start:
    python3 data/rollouts/tools/rollout_gate.py start P1 --no-wait

    # Stop policy:
    python3 data/rollouts/tools/rollout_gate.py stop
"""

import argparse
import os
import select
import sys
import threading
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from std_msgs.msg import Bool, String

# Import position definitions
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../src/il_main/il_main")))
try:
    from rollout_positions import POSITIONEN, START_POSITIONS, get_start_pose
except ImportError:
    POSITIONEN = {
        "P1": "Mitte",
        "P2": "vorne links",
        "P3": "vorne rechts",
        "P4": "hinten links",
        "P5": "hinten rechts",
    }


class RolloutGateNode(Node):
    def __init__(self):
        super().__init__("rollout_gate_cli")

        self.cmd_pub = self.create_publisher(String, "/rollout_gate_cmd", 10)
        self.legacy_gate_pub = self.create_publisher(Bool, "/policy_gate", 10)

        self.last_status = "idle"
        self.last_detail = ""
        self.is_active = False
        self.is_moving = False
        self.error_msg = None

        self.status_sub = self.create_subscription(
            String, "/rollout_gate_status", self._status_cb, 10
        )

    def _status_cb(self, msg: String):
        text = msg.data.strip()
        if "|" in text:
            status, detail = text.split("|", 1)
        else:
            status, detail = text, ""

        self.last_status = status
        self.last_detail = detail

        if status == "moving":
            self.is_moving = True
            self.is_active = False
        elif status == "active":
            self.is_active = True
            self.is_moving = False
        elif status == "inactive":
            self.is_active = False
            self.is_moving = False
        elif status == "error":
            self.error_msg = detail

    def send_start(self, position: str, height: str = "mitte"):
        msg = String()
        msg.data = f"start {position} --height {height}"
        # Repeat publish twice to ensure volatile subscribers receive it
        for _ in range(3):
            self.cmd_pub.publish(msg)
            time.sleep(0.05)

    def send_stop(self):
        cmd_msg = String()
        cmd_msg.data = "stop"
        bool_msg = Bool()
        bool_msg.data = False

        for _ in range(3):
            self.cmd_pub.publish(cmd_msg)
            self.legacy_gate_pub.publish(bool_msg)
            time.sleep(0.05)


def run_start_and_monitor(node: RolloutGateNode, position: str, height: str, wait_for_stop: bool):
    pos_code = position.upper() if position.upper() in POSITIONEN else position
    pos_lbl = POSITIONEN.get(pos_code, pos_code)

    print("\n" + "=" * 65)
    print(f" [RolloutGate] START TEST: {pos_code} - {pos_lbl}")
    print(f" [RolloutGate] Ziel-Höhe:  {height.upper()}")
    print("=" * 65)
    print(" --> Sende Fahrbefehl an Roboter-Controller...")

    node.send_start(pos_code, height)

    # Spinner thread for ROS spinning
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    # Wait for moving or active state
    t_start = time.time()
    last_print_dist = None
    timeout_s = 60.0

    print(" --> Fahre Startposition an...")
    while time.time() - t_start < timeout_s:
        if node.error_msg:
            print(f"\n[FEHLER] Controller meldet: {node.error_msg}")
            return False

        if node.is_active:
            break

        if node.is_moving:
            # detail often carries remaining distance
            detail = node.last_detail
            try:
                dist_m = float(detail)
                dist_str = f"{dist_m * 1000:.0f} mm"
                if dist_str != last_print_dist:
                    sys.stdout.write(f"\r     Arm nähert sich Startpose... Rest: {dist_str}   ")
                    sys.stdout.flush()
                    last_print_dist = dist_str
            except ValueError:
                pass

        time.sleep(0.1)

    if not node.is_active:
        print("\n[WARNUNG] Startposition wurde innerhalb des Timeouts nicht bestätigt.")
        print("Überprüfe, ob 'cartesian_inference_control' aktiv ist.")
        return False

    print("\n\n" + "*" * 65)
    print(f" >>> STARTPOSITION {pos_code} ERREICHT! POLICY IST JETZT AKTIV! <<<")
    print("*" * 65)

    if not wait_for_stop:
        print("(--no-wait aktiv: Kehre zur Shell zurück. Stop mit 'rollout_gate.py stop')\n")
        return True

    print("\n>> Drücke [ENTER] oder Strg+C, um die Bewegung zu STOPPEN <<\n")

    try:
        # Wait for keyboard enter
        input()
    except (KeyboardInterrupt, EOFError):
        pass

    print("\n --> Stoppe Policy und halte Arm an...")
    node.send_stop()
    time.sleep(0.3)
    print(" --> [FERTIG] Policy INAKTIV. Roboter hält aktuelle Position.\n")
    return True


def run_interactive(node: RolloutGateNode):
    print("\n" + "=" * 60)
    print("        UR10e Rollout Gate - Startposition wählen")
    print("=" * 60)
    print(" Höhen-Optionen:")
    print("   [m] Mitte (Standard / Anker)")
    print("   [o] Oben")
    print("   [u] Unten")
    print("-" * 60)
    print(" Start-Positionen:")
    for code, lbl in POSITIONEN.items():
        anchor_tag = " (Referenz-Anker)" if code == "P1" else ""
        print(f"   [{code[1]}] {code}: {lbl}{anchor_tag}")
    print("-" * 60)
    print("   [s] STOP (Policy sofort stoppen / Arm anhalten)")
    print("   [q] Abbrechen / Beenden")
    print("=" * 60)

    try:
        choice = input("\nAuswahl eingeben [1-5, s, q] (Standard: 1 = P1 Mitte): ").strip().lower()
    except (KeyboardInterrupt, EOFError):
        return

    if not choice or choice == "1":
        pos = "P1"
    elif choice in ("2", "3", "4", "5"):
        pos = f"P{choice}"
    elif choice in ("p1", "p2", "p3", "p4", "p5"):
        pos = choice.upper()
    elif choice in ("s", "stop"):
        print("\nSende Stopp-Befehl...")
        node.send_stop()
        print("Roboter gestoppt.")
        return
    elif choice in ("q", "quit", "exit"):
        return
    else:
        print(f"Ungültige Eingabe '{choice}'.")
        return

    height = "mitte"
    run_start_and_monitor(node, pos, height, wait_for_stop=True)


def main():
    parser = argparse.ArgumentParser(
        description="Rollout Gate: Automatisches Anfahren der Startposition und Policy-Freigabe.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "action",
        nargs="?",
        default=None,
        help="Aktion: 'start', 'stop', oder Positionskürzel ('P1', 'P2', ...). Wenn leer: interaktives Menü.",
    )
    parser.add_argument(
        "position",
        nargs="?",
        default=None,
        help="Position: P1 (Mitte), P2 (vorne links), P3 (vorne rechts), P4 (hinten links), P5 (hinten rechts).",
    )
    parser.add_argument(
        "--height",
        "-H",
        choices=["mitte", "oben", "unten"],
        default="mitte",
        help="Boxhöhe: 'mitte' (Standard / Anker), 'oben', 'unten'.",
    )
    parser.add_argument(
        "--no-wait",
        action="store_true",
        help="Nicht auf [ENTER] warten, sondern nach Policy-Aktivierung sofort beenden.",
    )

    args = parser.parse_args()

    rclpy.init()
    node = RolloutGateNode()

    try:
        # Determine intent
        action_arg = args.action.lower() if args.action else None

        if action_arg is None:
            # Interactive menu
            run_interactive(node)

        elif action_arg in ("stop", "halt", "pause", "0"):
            print("\n[RolloutGate] Sende STOPP an Controller...")
            node.send_stop()
            print("[RolloutGate] Policy gestoppt. Arm hält Position.\n")

        elif action_arg == "start":
            pos = args.position if args.position else "P1"
            run_start_and_monitor(node, pos, args.height, wait_for_stop=not args.no_wait)

        elif action_arg.upper() in POSITIONEN or action_arg.startswith("p"):
            # Shorthand: rollout_gate.py P1
            pos = action_arg.upper()
            run_start_and_monitor(node, pos, args.height, wait_for_stop=not args.no_wait)

        else:
            print(f"[Fehler] Unbekanntes Argument '{args.action}'. Nutze 'start P1', 'stop' oder keine Argumente für Menü.")

    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
