# Data Lifecycle & Architecture Specification (`data/`)

This directory organizes all data, processing pipelines, models, and hardware evaluation logs across the 4 stages of the Imitation Learning lifecycle on the UR10e workcell.

---

## 1. Lifecycle Overview

The directory layout follows the end-to-end data progression from physical teleoperation to policy deployment and validation:

```text
data/
├── recording/                   # Stage 1: Data Acquisition & Teleoperation
│   ├── episodes/                # Raw SVO recordings & ROS 2 bag directories (external mount)
│   ├── record_utils/            # Recording nodes (record.py, labels, camera_info)
│   ├── record_extractor/        # Point cloud, proprioception & RGB extraction
│   └── calibration/             # Hand-eye and extrinsics camera calibration matrices
│
├── datasets/                    # Stage 2: Cleaned Training Datasets
│   ├── configs/                 # YAML dataset recipes (foil_lifting_wrist_*.yaml)
│   ├── dataset_builder/         # Python ETL pipeline for Zarr & LeRobot generation
│   ├── zarr_datasets/           # Cleaned chunked Zarr archives (ManiFlow format)
│   ├── lerobot_datasets/        # Hugging Face LeRobot format
│   └── visualizer/              # Next.js interactive dataset visualizer web UI
│
├── training/                    # Stage 3: Model Weights, Logs & Benchmarks
│   ├── models/                  # Policy checkpoints and configs
│   │   ├── baselines/           # Standard baseline models (foil_lifting_wrist_seed0, etc.)
│   │   ├── ablation/            # Modality ablation experiments (abl_*)
│   │   └── progress_heads/      # Sub-goal progress prediction heads (prg_*)
│   ├── logs/                    # Training execution logs
│   │   ├── queue/               # Multi-GPU dispatcher & worker logs
│   │   └── sync/                # Remote worker rsync logs
│   └── benchmarks/              # Benchmark evaluation metrics and JSON summaries
│
└── rollouts/                    # Stage 4: Hardware Rollouts & Real-World Validation
    ├── runs/                    # Recorded live robot rollouts (MCAP/ROS 2 bags)
    ├── tools/                   # Operational CLI tools (logger, height check, visualizer)
    ├── viz/                     # Python trajectory and metrics plotting library
    └── analysis/                # Markdown protocols (eval.md), reports & diagnostics
```

---

## 2. Detailed Structure by Stage

### Stage 1: Recording (`data/recording/`)
* **`episodes/`**: Mounted externally (e.g. `/mnt/data/data`) to preserve SSD space on the main OS partition. Contains high-throughput raw ZED 2i SVO streams, synchronized force-torque arrays, and robot state.
* **`record_utils/`**: Scripts for teleoperation recording (`record.py`), workspace boundary checks, and camera metadata extraction (`camera_info.py`).
* **`record_extractor/`**: Offline point cloud extraction from neural depth SVO files into compressed numpy arrays.
* **`calibration/`**: Extrinsic calibration matrices mapping the ZED 2i frame into the robot base frame (`base_link`).

### Stage 2: Datasets (`data/datasets/`)
* **`configs/`**: Declarative YAML configurations defining episode filters, sub-sampling rates, action definitions (relative vs. absolute EE, delta wrist vs. delta joint), and state history length.
* **`dataset_builder/`**: The core ETL pipeline parsing recorded episodes, detecting grasp phases, computing relative progress, and writing Zarr arrays.
* **`zarr_datasets/`**: Ready-to-train Zarr files used by ManiFlow (`foil_lifting_box_recorded_action.zarr`, etc.).
* **`visualizer/`**: A modern web viewer for inspecting episodes, 3D point cloud samples, and action distribution plots.

### Stage 3: Training (`data/training/`)
* **`models/`**: Checkpoints grouped by experiment type:
  * `baselines/`: Core policy runs.
  * `ablation/`: Systematic sensor modality ablations (wrist camera vs. front camera vs. point cloud vs. tactile history).
  * `progress_heads/`: Progress head weights trained to estimate the completion curve.
* **`logs/`**: Structured daemon and training stdout/stderr logs.
* **`benchmarks/`**: Machine-readable evaluation results (e.g. validation MSE across horizons and action deltas).

### Stage 4: Rollouts (`data/rollouts/`)
* **`runs/`**: Actual deployment test bags recorded on the physical robot cell.
* **`tools/`**:
  * `compare_foil_height.py`: Real-time ZED 2i plane height verification against dataset ground truth before running an inference test.
  * `log_rollout.py`: CLI logger recording execution lag, commanded vs. executed closure, and lift trajectory.
  * `annotate_rollouts.py`: Success/failure labeling tool.
  * `rollout_viz.py`: Terminal table summarizer and interactive Rerun / HTML dashboard generator.
* **`viz/`**: Python module (`viz/data.py`, `viz/attempts.py`, `viz/pinch.py`, `viz/traj.py`) providing Kinematics forward-solve and pinch-width calculations for the Tesollo DG5F.
* **`analysis/`**:
  * `eval.md`: Hardware evaluation protocol, empirical failure taxonomy, and layer offsets.
  * `investigations/`: In-depth analytical reports (`height_modality`, `arm_motion`).

---

## 3. Workcell Layer Height Reference Table (L3 Bottom Foil Layer)

All heights are measured in the robot base coordinate system (`base_link`), with UR10e mounting flange as origin ($z = 0$). In the dataset, `L3` denotes the lowest foil layer (between cardboard floor and parts):

| Box Height Level | A-Pillar `L3` Mean (Range) | B-Pillar `L3` Mean (Range) | Combined `L3` Mean | Offset above `unten` (`L3`) |
| :--- | :---: | :---: | :---: | :---: |
| **Bottom (`folien_unten`)** | **$-431.3\,\text{mm}$** ($-460.5 \dots -410.9$) | **$-416.7\,\text{mm}$** ($-433.8 \dots -404.2$) | **$-425.8\,\text{mm}$** | **$0.0\,\text{mm}$ (Basis)** |
| **Middle (`folien_mitte`)** | **$-299.0\,\text{mm}$** ($-324.4 \dots -285.3$) | **$-354.9\,\text{mm}$** ($-382.5 \dots -324.5$) | **$-325.3\,\text{mm}$** | **$+62 \dots +132\,\text{mm}$** |
| **Top (`folien_oben`)** | **$-186.5\,\text{mm}$** ($-201.5 \dots -172.9$) | **$-158.0\,\text{mm}$** ($-177.3 \dots -146.6$) | **$-171.2\,\text{mm}$** | **$+245 \dots +259\,\text{mm}$ ($\approx 24\text{--}26\,\text{cm}$)** |

*(All-layer L1–L3 averages for A+B pillar: `folien_unten` = $-398.3\,\text{mm}$, `folien_mitte` = $-302.2\,\text{mm}$, `folien_oben` = $-142.5\,\text{mm}$).*

---

## 4. Key Workflows & Common Commands

### Check live foil height against dataset ground truth:
```bash
# Can be called directly on host (auto-delegates to zed_container) or via bash script:
python3 data/rollouts/tools/compare_foil_height.py            # One-shot 5-frame average
python3 data/rollouts/tools/compare_foil_height.py --watch    # Live 1 Hz monitor

# Alternatively:
./data/rollouts/tools/compare_foil_height.sh
```

### Review hardware rollout performance:
```bash
# Display summary table across all recorded rollouts
docker exec ros2_dev python3 /ros2_ws/data/rollouts/tools/rollout_viz.py table

# Inspect specific rollout with 3D point cloud & pinch width in Rerun
docker exec ros2_dev python3 /ros2_ws/data/rollouts/tools/rollout_viz.py rerun rollout_010 --save /ros2_ws/data/rollouts/analysis/reports/r010.rrd
```
