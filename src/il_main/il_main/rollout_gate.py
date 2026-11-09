#!/usr/bin/env python3
"""ROS 2 wrapper for rollout_gate CLI."""
import os
import sys

# Forward to data/rollouts/tools/rollout_gate.py
tools_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../data/rollouts/tools"))
sys.path.insert(0, tools_dir)

from rollout_gate import main

if __name__ == "__main__":
    main()
