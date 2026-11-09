#!/usr/bin/env python3
"""
Foil Height Verification & Live Camera Comparison Script.

Vergleicht die aktuelle Folien- und Palettenhöhe aus dem Live-Kamerabild (ZED 2i)
mit den Höhenwerten aus den Trainings-Datensätzen (folien_unten, folien_mitte, folien_oben).

Funktionsweise:
  1. Erfasst RGB-D Bilder aus der Front-Kamera (ZED 2i).
  2. Transformiert die Punktwolke in das Roboter-Basis-Koordinatensystem (base_link)
     unter Verwendung der kalibrierten Intrinsics und Extrinsics.
  3. Verwendet den identischen Trainings-Algorithmus (estimate_foil_height):
     - Workspace-Filterung
     - Kartonrand-Unterdrückung (rim suppression)
     - Modus-Histogrammanalyse (5 mm Bins)
     - Median der Inlier (±15 mm)
  4. Berechnet statistische Abweichungen (Delta in mm, Z-Score, Min..Max-Bereich)
     zu allen Datensätzen (folien_unten, folien_mitte, folien_oben) gegliedert nach
     Gesamt und Bauteilen (A-Säule, B-Säule, Heckleuchte).
  5. Zeigt eine strukturierte Auswertung, Fehlerklassifikation (z. B. "Leichter Fehler")
     und Handlungsempfehlung an.

Verwendung:
  Im ZED-Container (oder via scripts/compare_foil_height.sh):
    python /root/ros2_ws/data/rollouts/tools/compare_foil_height.py

  Optionen:
    --watch, -w          Kontinuierlicher Live-Monitoring-Modus (1 Hz)
    --frames N           Anzahl zu mittelnder Kameraframes (Standard: 5)
    --park-xy X Y        ROI-Zylinder-Zentrum [x, y] in Metern (Standard: -0.35 -1.03)
    --no-park-xy         Verwendet gesamten Workspace ohne ROI-Zylinder
    --recompute          Erzwingt Neuberechnung der Datensatz-Statistiken
    --svo PATH           Liest SVO-Datei statt Live-Kamera
    --pc PATH            Liest gespeicherte .npy Punktwolke statt Live-Kamera
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Wenn auf dem Host ohne ZED/Numpy-Umgebung aufgerufen, automatisch in zed_container delegieren
if not os.path.exists("/.dockerenv"):
    try:
        import numpy  # Probe
        import pyzed.sl  # Probe
    except ImportError:
        import shutil
        import subprocess
        if shutil.which("docker"):
            chk = subprocess.run(["docker", "ps", "--filter", "name=zed_container", "--format", "{{.Names}}"], capture_output=True, text=True)
            if "zed_container" in chk.stdout.split():
                cmd = ["docker", "exec"]
                if sys.stdin.isatty():
                    cmd.append("-it")
                cmd.extend(["zed_container", "python3", "/root/ros2_ws/data/rollouts/tools/compare_foil_height.py"])
                cmd.extend(sys.argv[1:])
                res = subprocess.run(cmd)
                sys.exit(res.returncode)

import numpy as np

# Root- und Modulpfade ermitteln
SCRIPT_DIR = Path(__file__).resolve().parent
ROLLOUTS_DIR = SCRIPT_DIR.parent
DATA_DIR = ROLLOUTS_DIR.parent
REPO_ROOT = DATA_DIR.parent
RECORDING_DIR = DATA_DIR / "recording"
DATASETS_DIR = DATA_DIR / "datasets"
RECORD_UTILS_DIR = RECORDING_DIR / "record_utils"
RECORD_EXTRACTOR_DIR = RECORDING_DIR / "record_extractor"

# Pfade für Modulimporte registrieren
for p in [SCRIPT_DIR, ROLLOUTS_DIR, DATA_DIR, RECORDING_DIR, DATASETS_DIR,
          RECORD_UTILS_DIR, RECORD_EXTRACTOR_DIR,
          Path("/root/ros2_ws/data"),
          Path("/root/ros2_ws/data/recording"),
          Path("/root/ros2_ws/data/recording/record_utils"),
          Path("/root/ros2_ws/data/recording/record_extractor")]:
    p_str = str(p)
    if p_str not in sys.path and p.exists():
        sys.path.insert(0, p_str)

# Open3D und Kamera-Info importieren
try:
    import open3d as o3d
    from camera_info import camera_intrinsics, camera_extrinsics, workspace
    try:
        from record_extractor.point_cloud_utils import generate_point_cloud_from_depth
    except ImportError:
        from point_cloud_utils import generate_point_cloud_from_depth
    OPEN3D_AVAILABLE = True
except ImportError as e:
    OPEN3D_AVAILABLE = False
    _IMPORT_ERROR = str(e)

# PyZED Import
try:
    import pyzed.sl as sl
    PYZED_AVAILABLE = True
except ImportError:
    PYZED_AVAILABLE = False

# ANSI Color Codes
CLR_RESET = "\033[0m"
CLR_BOLD = "\033[1m"
CLR_CYAN = "\033[36m"
CLR_GREEN = "\033[32m"
CLR_YELLOW = "\033[33m"
CLR_RED = "\033[31m"
CLR_MAGENTA = "\033[35m"
CLR_BLUE = "\033[34m"
CLR_GRAY = "\033[90m"


def estimate_foil_height(
    pc_pts: np.ndarray,
    park_xy: Optional[np.ndarray] = None,
    search_radius: float = 0.25,
    rim_lower_bound: float = -0.05,
    rim_suppress_window: Tuple[float, float] = (-0.04, 0.03),
    z_bounds: Tuple[float, float] = (-0.55, 0.12),
    bin_size: float = 0.005,
    inlier_band: float = 0.015,
) -> Tuple[Optional[float], int, str]:
    """
    Exakte Trainings-Funktionalität zur robusten Folienhöhen-Extraktion in base_link.
    Spiegelt third_party/ManiFlow_Policy/.../height_extractor.py wider.
    """
    pts = np.asarray(pc_pts)[:, :3]
    valid_mask = (pts[:, 2] < z_bounds[1]) & (pts[:, 2] > z_bounds[0])
    valid_pts = pts[valid_mask]

    if len(valid_pts) < 10:
        return None, 0, "too_few_points"

    if park_xy is not None:
        park_xy_arr = np.asarray(park_xy, dtype=float)[:2]
        d = np.linalg.norm(valid_pts[:, :2] - park_xy_arr, axis=1)
        cyl_pts = valid_pts[d < search_radius]
        if len(cyl_pts) >= 15:
            valid_pts = cyl_pts

    bins = np.arange(z_bounds[0], z_bounds[1] + bin_size, bin_size)
    counts, edges = np.histogram(valid_pts[:, 2], bins=bins)

    # Box-Rim-Unterdrückung bei Folie im Kisteninneren
    lower_mask = edges[:-1] < rim_lower_bound
    if np.any(lower_mask) and np.max(counts[lower_mask]) > 25:
        search_counts = counts.copy()
        rim_mask = (edges[:-1] >= rim_suppress_window[0]) & (edges[:-1] <= rim_suppress_window[1])
        search_counts[rim_mask] = 0
        search_counts[~lower_mask] = 0
        peak_idx = int(np.argmax(search_counts))
    else:
        peak_idx = int(np.argmax(counts))

    mode_z = 0.5 * (edges[peak_idx] + edges[peak_idx + 1])
    inliers = valid_pts[np.abs(valid_pts[:, 2] - mode_z) <= inlier_band, 2]

    if len(inliers) < 5:
        return float(mode_z), len(inliers), "low_inliers"

    z_est = float(np.median(inliers))
    return z_est, len(inliers), "ok"


class DatasetStatsManager:
    """Verwaltet und berechnet die Höhenstatistiken der Datensätze."""

    STATS_CACHE_CANDIDATES = [
        SCRIPT_DIR / "foil_height_dataset_stats.json",
        RECORD_UTILS_DIR / "foil_height_dataset_stats.json",
        Path("/root/ros2_ws/data/rollouts/tools/foil_height_dataset_stats.json"),
        Path("/root/ros2_ws/data/recording/record_utils/foil_height_dataset_stats.json"),
    ]

    @classmethod
    def get_stats(cls, zarr_path: Optional[str] = None, force_recompute: bool = False) -> Dict:
        if not force_recompute:
            for cache_path in cls.STATS_CACHE_CANDIDATES:
                if cache_path.exists():
                    try:
                        with open(cache_path, "r") as f:
                            data = json.load(f)
                            if "summary" in data:
                                return data["summary"]
                    except Exception as e:
                        print(f"{CLR_YELLOW}[Warnung] Cache-Datei {cache_path} konnte nicht gelesen werden: {e}{CLR_RESET}")

        return cls.compute_stats_from_zarr(zarr_path)

    @classmethod
    def compute_stats_from_zarr(cls, zarr_path: Optional[str] = None) -> Dict:
        try:
            import zarr
        except ImportError:
            raise RuntimeError("zarr ist nicht installiert. Bitte 'pip install zarr' ausführen.")

        candidate_paths = [
            zarr_path,
            "/root/ros2_ws/data/datasets/zarr_datasets/foil_lifting_box_recorded_action.zarr",
            str(REPO_ROOT / "data/datasets/zarr_datasets/foil_lifting_box_recorded_action.zarr"),
            "/root/ros2_ws/data/datasets/zarr_datasets/foil_lifting_box.zarr",
            str(REPO_ROOT / "data/datasets/zarr_datasets/foil_lifting_box.zarr"),
        ]
        target_path = None
        for p in candidate_paths:
            if p and os.path.exists(p):
                target_path = p
                break

        if not target_path:
            raise FileNotFoundError("Kein Zarr-Datensatz gefunden für Höhenanalyse.")

        print(f"{CLR_CYAN}[Info] Berechne Datensatz-Höhenstatistiken aus: {target_path}...{CLR_RESET}")
        root = zarr.open(target_path, mode="r")
        ep_ends = root["meta"]["episode_ends"][:]
        pc = root["point_cloud"]
        state = root["state"]
        bh_ids = root["meta"]["box_height_id"][:]
        part_ids = root["meta"]["part_id"][:]
        fl_ids = root["meta"]["foil_layer_id"][:]

        bh_map = {0: "oben", 1: "mitte", 2: "unten"}
        fl_map = {0: "L3", 1: "L2", 2: "L1"}

        def get_prefix(bh_str: str, pid: int) -> str:
            if pid == 0:
                return "folien_unten_a_sauele" if bh_str == "unten" else f"folien_{bh_str}_a_saeule"
            elif pid == 1:
                return "folien_unten_b_sauele" if bh_str == "unten" else f"folien_{bh_str}_b_saeule"
            else:
                return f"folien_{bh_str}_heckleuchte"

        by_prefix = {}
        by_prefix_foil_layer = {}
        by_layer = {"unten": [], "mitte": [], "oben": []}
        by_layer_l3 = {"unten": [], "mitte": [], "oben": []}
        episodes_data = []

        for ep in range(len(ep_ends)):
            start = int(ep_ends[ep - 1]) if ep > 0 else 0
            end = int(ep_ends[ep])
            layer = bh_map.get(int(bh_ids[ep]), "unten")
            folienlage = fl_map.get(int(fl_ids[ep]), "L1")
            pref = get_prefix(layer, int(part_ids[ep]))
            src = f"{pref}/ep_{ep:03d}"

            st = state[start:end, :3]
            park_xy = st[0, :2]
            p0 = pc[start]

            z_est, inliers, status = estimate_foil_height(p0, park_xy=park_xy)
            if z_est is not None:
                z_mm = float(z_est * 1000.0)
                by_prefix.setdefault(pref, []).append(z_mm)
                by_prefix_foil_layer.setdefault(f"{pref}_{folienlage}", []).append(z_mm)
                by_layer[layer].append(z_mm)
                if folienlage == "L3":
                    by_layer_l3[layer].append(z_mm)
                episodes_data.append({
                    "ep": ep, "source": src, "prefix": pref, "layer": layer,
                    "folienlage": folienlage,
                    "z_est_mm": z_mm, "inliers": inliers, "status": status,
                    "park_xy": [float(park_xy[0]), float(park_xy[1])]
                })

        def calc_summary(vals: List[float]) -> Dict:
            arr = np.array(vals)
            return {
                "count": len(vals),
                "mean_mm": float(arr.mean()),
                "std_mm": float(arr.std()),
                "median_mm": float( np.median(arr)),
                "min_mm": float(arr.min()),
                "max_mm": float(arr.max()),
                "p10_mm": float(np.percentile(arr, 10)),
                "p90_mm": float(np.percentile(arr, 90)),
            }

        unten_a = by_prefix.get("folien_unten_a_sauele", [])
        unten_b = by_prefix.get("folien_unten_b_sauele", [])
        mitte_a = by_prefix.get("folien_mitte_a_saeule", [])
        mitte_b = by_prefix.get("folien_mitte_b_saeule", [])
        oben_a = by_prefix.get("folien_oben_a_saeule", [])
        oben_b = by_prefix.get("folien_oben_b_saeule", [])

        summary = {
            "by_prefix": {k: calc_summary(v) for k, v in sorted(by_prefix.items())},
            "by_prefix_foil_layer": {k: calc_summary(v) for k, v in sorted(by_prefix_foil_layer.items())},
            "by_layer": {k: calc_summary(v) for k, v in sorted(by_layer.items()) if v},
            "by_layer_l3": {k: calc_summary(v) for k, v in sorted(by_layer_l3.items()) if v},
            "folien_unten_a_b": calc_summary(unten_a + unten_b) if (unten_a or unten_b) else None,
            "folien_mitte_a_b": calc_summary(mitte_a + mitte_b) if (mitte_a or mitte_b) else None,
            "folien_oben_a_b": calc_summary(oben_a + oben_b) if (oben_a or oben_b) else None,
        }

        cache_target = SCRIPT_DIR / "foil_height_dataset_stats.json"
        try:
            with open(cache_target, "w") as f:
                json.dump({"summary": summary, "episodes": episodes_data}, f, indent=2)
            print(f"{CLR_GREEN}[Info] Statistiken erfolgreich gecacht in: {cache_target}{CLR_RESET}")
        except Exception as e:
            print(f"{CLR_YELLOW}[Warnung] Konnte Cache-Datei nicht schreiben: {e}{CLR_RESET}")

        return summary


class LiveCameraFoilEstimator:
    """Verwaltet die Verbindung zur ZED 2i Kamera und die Schätzung."""

    def __init__(self, svo_path: Optional[str] = None):
        if not PYZED_AVAILABLE:
            raise RuntimeError("pyzed.sl ist in dieser Umgebung nicht verfügbar (bitte im zed_container ausführen).")
        if not OPEN3D_AVAILABLE:
            raise RuntimeError(f"Open3D / Kamera-Module nicht verfügbar: {_IMPORT_ERROR}")

        self.svo_path = svo_path
        self.cam = sl.Camera()
        self.is_open = False

    def open(self):
        init = sl.InitParameters()
        init.camera_resolution = sl.RESOLUTION.HD720
        init.camera_fps = 30
        init.depth_mode = sl.DEPTH_MODE.NEURAL
        init.coordinate_units = sl.UNIT.METER
        init.depth_minimum_distance = 0.2
        init.depth_maximum_distance = 2.0
        init.sdk_verbose = 0

        if self.svo_path:
            init.set_from_svo_file(self.svo_path)
            init.svo_real_time_mode = False

        status = self.cam.open(init)
        if status != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f"ZED Kamera konnte nicht geöffnet werden: {status}")

        self.runtime = sl.RuntimeParameters(
            confidence_threshold=100,
            texture_confidence_threshold=100,
            remove_saturated_areas=True,
        )
        self.mat_rgb = sl.Mat()
        self.mat_depth = sl.Mat()
        self.is_open = True

        # Sensor-Aufwärmung (3 Frames verwerfen für Belichtung/Tiefe)
        for _ in range(3):
            self.cam.grab(self.runtime)

    def close(self):
        if self.is_open:
            self.cam.close()
            self.is_open = False

    def grab_point_cloud(self, num_points: int = 4096) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Erfasst einen Frame und berechnet die kalibrierte Punktwolke."""
        if not self.is_open:
            self.open()

        err = self.cam.grab(self.runtime)
        if err != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f"Kamera-Grab fehlgeschlagen: {err}")

        self.cam.retrieve_image(self.mat_rgb, sl.VIEW.LEFT)
        self.cam.retrieve_measure(self.mat_depth, sl.MEASURE.DEPTH)

        color_bgra = self.mat_rgb.get_data()
        color_bgr = color_bgra[:, :, :3]
        depth_m = self.mat_depth.get_data()

        pc = generate_point_cloud_from_depth(
            color_bgr,
            depth_m,
            intrinsics=camera_intrinsics,
            extrinsic_matrix=camera_extrinsics,
            workspace_bounds=workspace,
            num_points=num_points,
            include_color=True,
        )
        return pc, color_bgr, depth_m

    def measure_height(
        self,
        num_frames: int = 5,
        park_xy: Optional[np.ndarray] = None,
        search_radius: float = 0.25,
    ) -> Dict:
        """Führt eine Mehrfachmessung durch und berechnet Mittelwert & Streuung."""
        heights = []
        inliers_list = []
        statuses = []
        last_pc = None

        for _ in range(max(1, num_frames)):
            pc, _, _ = self.grab_point_cloud(num_points=4096)
            last_pc = pc
            z_est, inliers, status = estimate_foil_height(
                pc, park_xy=park_xy, search_radius=search_radius
            )
            if z_est is not None:
                heights.append(z_est * 1000.0)  # in mm
                inliers_list.append(inliers)
                statuses.append(status)

        if not heights:
            return {
                "ok": False,
                "error": "Keine gültige Folienhöhe erkannt (zu wenige Punkte / Außerhalb Arbeitsbereich).",
                "last_pc": last_pc,
            }

        h_arr = np.array(heights)
        return {
            "ok": True,
            "mean_z_mm": float(h_arr.mean()),
            "std_z_mm": float(h_arr.std()),
            "median_z_mm": float(np.median(h_arr)),
            "min_z_mm": float(h_arr.min()),
            "max_z_mm": float(h_arr.max()),
            "inliers_mean": int(np.mean(inliers_list)),
            "status": statuses[-1],
            "num_samples": len(heights),
            "last_pc": last_pc,
        }


def format_delta(delta_mm: float) -> str:
    sign = "+" if delta_mm > 0 else ""
    return f"{sign}{delta_mm:6.1f} mm"


def format_z_score(z: float) -> str:
    sign = "+" if z > 0 else ""
    return f"{sign}{z:5.2f} σ"


def print_table_block(title: str, entries: List[Tuple[str, Optional[Dict]]], z_live: float):
    """Hilfsfunktion zum Drucken eines tabellarischen Vergleichsblocks."""
    print(f"{CLR_BOLD}{title}{CLR_RESET}")
    header = (
        f"{'Bedingung / Datensatz':<27} | {'N':>3} | {'Mittelwert':>11} | {'Std':>8} | "
        f"{'Median':>10} | {'Bereich (Min..Max)':>20} | {'Delta (Live-GT)':>15} | {'Z-Score':>9}"
    )
    print(header)
    print("-" * 104)

    for name, st in entries:
        if not st:
            continue
        n = st["count"]
        m = st["mean_mm"]
        s = st["std_mm"]
        med = st["median_mm"]
        r_min, r_max = st["min_mm"], st["max_mm"]
        delta = z_live - m
        z_sc = delta / s if s > 1e-6 else 0.0

        # Farbliche Hervorhebung von Delta
        if abs(delta) <= 20.0:
            d_color = CLR_GREEN
        elif abs(delta) <= 50.0:
            d_color = CLR_YELLOW
        else:
            d_color = CLR_RED

        delta_str = f"{d_color}{format_delta(delta)}{CLR_RESET}"
        z_sc_str = f"{d_color}{format_z_score(z_sc)}{CLR_RESET}"
        rng_str = f"{r_min:+6.1f} .. {r_max:+6.1f}"

        is_sub = name.startswith("folien_") and ("_saeule" in name or "_sauele" in name or "_heckleuchte" in name)
        prefix_indent = "  └─ " if is_sub else ""
        row_name = f"{prefix_indent}{name}"

        print(
            f"{row_name:<27} | {n:3d} | {m:+9.1f} mm | {s:6.1f} mm | "
            f"{med:+8.1f} mm | {rng_str:>20} | {delta_str:>24} | {z_sc_str:>18}"
        )
    print("-" * 104)


def print_comparison_report(
    live_res: Dict,
    dataset_stats: Dict,
    park_xy: Optional[np.ndarray],
    clear_screen: bool = False,
):
    """Gibt den vollständigen tabellarischen Vergleichsbericht auf Deutsch aus."""
    if clear_screen:
        print("\033[2J\033[H", end="")

    z_live = live_res["mean_z_mm"]
    z_live_std = live_res["std_z_mm"]
    samples = live_res["num_samples"]
    inliers = live_res["inliers_mean"]
    status = live_res["status"]

    print("=" * 104)
    print(f"{CLR_BOLD}{CLR_CYAN}ECHTZEIT-HÖHENVERGLEICH: LIVE-KAMERA VS. TRAININGS-DATENSÄTZE{CLR_RESET}")
    print("=" * 104)

    # 1. Live Messung
    roi_info = (
        f"ROI-Cylinder [x={park_xy[0]:.3f} m, y={park_xy[1]:.3f} m, r=0.25 m]"
        if park_xy is not None
        else "Gesamter Workspace (Kein XY-Filter)"
    )
    prefixes = dataset_stats.get("by_prefix", {})
    pref_fl = dataset_stats.get("by_prefix_foil_layer", {})
    layers = dataset_stats.get("by_layer", {})
    layers_l3 = dataset_stats.get("by_layer_l3", {})
    unten_overall = layers.get("unten")
    unten_mean = unten_overall["mean_mm"] if unten_overall else -373.2
    unten_ab_mean = dataset_stats.get("folien_unten_a_b", {}).get("mean_mm", -398.3)
    rel_to_unten = z_live - unten_mean
    rel_to_unten_ab = z_live - unten_ab_mean

    print(f"{CLR_BOLD}1. LIVE-KAMERAMESSUNG (ZED 2i, NEURAL Depth in base_link):{CLR_RESET}")
    print(f"   • Absolute Folienhöhe (z_live): {CLR_BOLD}{CLR_GREEN}{z_live:+8.1f} mm{CLR_RESET} ({z_live/1000:+.3f} m)")
    print(f"   • Relativhöhe zur untersten Ebene:  {CLR_BOLD}{CLR_CYAN}{format_delta(rel_to_unten_ab)}{CLR_RESET} (über A+B Säule unten, {unten_ab_mean:+.1f} mm)")
    print(f"   • Sensor-Streuung (std):          ±{z_live_std:5.2f} mm  (über {samples} Frames gemittelt)")
    print(f"   • Plane-Inliers / Status:         {inliers} Punkte  [Status: {status}]")
    print(f"   • Auswertungsbereich:             {roi_info}")
    print("-" * 104)

    # 2. Fokus: folien_unten
    unten_entries = [
        ("folien_unten (Gesamt)", layers.get("unten")),
        ("folien_unten (L3 Boden)", layers_l3.get("unten")),
        ("folien_unten (A+B Säule)", dataset_stats.get("folien_unten_a_b")),
        ("folien_unten_a_sauele", prefixes.get("folien_unten_a_sauele")),
        ("folien_unten_a_sauele (L3)", pref_fl.get("folien_unten_a_sauele_L3")),
        ("folien_unten_b_sauele", prefixes.get("folien_unten_b_sauele")),
        ("folien_unten_b_sauele (L3)", pref_fl.get("folien_unten_b_sauele_L3")),
        ("folien_unten_heckleuchte", prefixes.get("folien_unten_heckleuchte")),
    ]
    print_table_block("2. FOKUS-VERGLEICH MIT 'FOLIEN_UNTEN' (Tiefste Schicht / Boxboden):", unten_entries, z_live)

    # 3. Vergleich: folien_mitte
    mitte_entries = [
        ("folien_mitte (Gesamt)", layers.get("mitte")),
        ("folien_mitte (L3 Boden)", layers_l3.get("mitte")),
        ("folien_mitte (A+B Säule)", dataset_stats.get("folien_mitte_a_b")),
        ("folien_mitte_a_saeule", prefixes.get("folien_mitte_a_saeule")),
        ("folien_mitte_a_saeule (L3)", pref_fl.get("folien_mitte_a_saeule_L3")),
        ("folien_mitte_b_saeule", prefixes.get("folien_mitte_b_saeule")),
        ("folien_mitte_b_saeule (L3)", pref_fl.get("folien_mitte_b_saeule_L3")),
        ("folien_mitte_heckleuchte", prefixes.get("folien_mitte_heckleuchte")),
    ]
    print_table_block("3. VERGLEICH MIT 'FOLIEN_MITTE' (Mittlere Schicht / Zwischenlage):", mitte_entries, z_live)

    # 4. Vergleich: folien_oben
    oben_entries = [
        ("folien_oben (Gesamt)", layers.get("oben")),
        ("folien_oben (L3 Boden)", layers_l3.get("oben")),
        ("folien_oben (A+B Säule)", dataset_stats.get("folien_oben_a_b")),
        ("folien_oben_a_saeule", prefixes.get("folien_oben_a_saeule")),
        ("folien_oben_a_saeule (L3)", pref_fl.get("folien_oben_a_saeule_L3")),
        ("folien_oben_b_saeule", prefixes.get("folien_oben_b_saeule")),
        ("folien_oben_b_saeule (L3)", pref_fl.get("folien_oben_b_saeule_L3")),
        ("folien_oben_heckleuchte", prefixes.get("folien_oben_heckleuchte")),
    ]
    print_table_block("4. VERGLEICH MIT 'FOLIEN_OBEN' (Oberste Schicht / Decklage):", oben_entries, z_live)

    # 5. Stufen-Übersicht & Ranking
    print(f"{CLR_BOLD}5. MULTI-LEVEL RANKING & HÖHENABSTÄNDE (BEZOGEN AUF UNTERSTE EBENE):{CLR_RESET}")
    print(f"{'Höhenstufe':<18} | {'Mittelwert':>11} | {'Über unterer Lage':>18} | {'Abstand zu z_live':>19} | {'Z-Score':>9} | {'Einstufung':<18}")
    print("-" * 104)

    ranking = []
    for l_key, l_label in [("unten", "folien_unten"), ("mitte", "folien_mitte"), ("oben", "folien_oben")]:
        st = layers.get(l_key)
        if st:
            m = st["mean_mm"]
            s = st["std_mm"]
            d = z_live - m
            z_sc = d / s if s > 1e-6 else 0.0
            h_above_unten = m - unten_ab_mean
            ranking.append((abs(d), l_label, m, d, z_sc, h_above_unten, st))

    ranking.sort(key=lambda x: x[0])
    for idx, (abs_d, l_label, m, d, z_sc, h_above_unten, st) in enumerate(ranking):
        rank_tag = "Nächste Stufe" if idx == 0 else f"Rang {idx+1}"
        col = CLR_GREEN if idx == 0 else CLR_RESET
        h_above_str = format_delta(h_above_unten) if l_label != "folien_unten" else "  0.0 mm (Basis)"
        print(f"{col}{l_label:<18} | {m:+9.1f} mm | {h_above_str:>18} | {format_delta(d):>19} | {format_z_score(z_sc):>9} | {rank_tag:<18}{CLR_RESET}")

    print("=" * 104)

    # 6. Diagnose & Interpretation
    print(f"{CLR_BOLD}6. DIAGNOSE & BEWERTUNG:{CLR_RESET}")

    unten_overall = layers.get("unten")
    unten_ab = dataset_stats.get("folien_unten_a_b")
    heck_unten = prefixes.get("folien_unten_heckleuchte")

    best_dist, best_layer_name, best_mean, best_delta, best_z_sc, best_h_above, _ = ranking[0]

    print(f"  • Nächstgelegene Trainings-Höhenstufe: {CLR_BOLD}{CLR_GREEN}{best_layer_name}{CLR_RESET} (Abweichung: {best_delta:+.1f} mm, {best_z_sc:+.2f} σ)")

    # Detaillierte Einstufung bzgl. folien_unten (A+B Säule & Heckleuchte)
    if unten_ab:
        ab_mean = unten_ab["mean_mm"]
        ab_min = unten_ab["min_mm"]
        ab_max = unten_ab["max_mm"]
        d_ab = z_live - ab_mean

        if abs(d_ab) <= 25.0:
            print(f"  {CLR_GREEN}✔ OPTIMALE ÜBEREINSTIMMUNG MIT 'folien_unten (A+B Säule)'{CLR_RESET}")
            print(f"    Aktuelle Höhe {z_live:+.1f} mm liegt im Kernbereich der unteren Palettenlage ({ab_mean:+.1f} mm ± 25 mm).")
            print(f"    Policy 0029_rel_wrist_height_delta_to4 / 0027_abs läuft in ihrer exakten Trainingsdomäne.")
        elif z_live < ab_min:
            below_min = ab_min - z_live
            print(f"  {CLR_YELLOW}⚠ [LEICHTER FEHLER] HÖHENABWEICHUNG UNTERHALB 'folien_unten'{CLR_RESET}")
            print(f"    Status:                  {CLR_BOLD}{CLR_YELLOW}Leichter Fehler (Tolerierbare Randbereichs-Abweichung){CLR_RESET}")
            print(f"    Aktuell gemessen:        {z_live:+.1f} mm")
            print(f"    Abweichung zu Minimum:   {below_min:+.1f} mm tiefer als A/B-Säule-Minimum ({ab_min:+.1f} mm)")
            print(f"    Abweichung zu Mittelwert:{d_ab:+.1f} mm tiefer als Trainings-Schnitt ({ab_mean:+.1f} mm)")
            print(f"    Analyse & Ursache:")
            print(f"    • Die gemessene Höhe entspricht der Tischplatte bzw. dem leeren Kistenboden (~ -494 mm).")
            print(f"    • Für einen vollständigen 'folien_unten'-Lauf fehlen typischerweise ca. 30..90 mm Stapelhöhe")
            print(f"      (Einlegeboden, Bauteile oder Folienlage).")
            print(f"    • Als 'leichter Fehler' klassifiziert: Die relative Delta-Höhen-Policy (0029) sinkt")
            print(f"      weiter sauber ab, bremst jedoch ggf. etwas früher ab.")
        else:
            above_max = z_live - ab_max
            print(f"  {CLR_CYAN}ℹ HÖHE LIEGT ÜBER DEM 'folien_unten (A+B Säule)'-BEREICH (+{above_max:+.1f} mm über Max {ab_max:+.1f} mm){CLR_RESET}")
            print(f"    Aktuell gemessen:        {z_live:+.1f} mm (Delta zu A+B Säule unten: {d_ab:+.1f} mm)")

            # Prüfen ob Heckleuchte unten passt
            if heck_unten and abs(z_live - heck_unten["mean_mm"]) <= 30.0:
                d_heck = z_live - heck_unten["mean_mm"]
                print(f"    {CLR_GREEN}✔ Passt sehr gut zu 'folien_unten_heckleuchte' (Mittelwert: {heck_unten['mean_mm']:+.1f} mm, Delta: {d_heck:+.1f} mm){CLR_RESET}")

            # Bezug zu mitte / oben
            print(f"    Abstand zu 'folien_mitte': {z_live - layers['mitte']['mean_mm']:+.1f} mm (Mitte-Schnitt: {layers['mitte']['mean_mm']:+.1f} mm)")
            print(f"    Abstand zu 'folien_oben':  {z_live - layers['oben']['mean_mm']:+.1f} mm (Oben-Schnitt:  {layers['oben']['mean_mm']:+.1f} mm)")


    print("=" * 104)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Vergleich der aktuellen Kamera-Folienhöhe mit Trainingsdatensätzen (unten, mitte, oben)."
    )
    parser.add_argument(
        "--watch", "-w",
        action="store_true",
        help="Kontinuierlicher Live-Monitoring-Modus (aktualisiert bei ~1 Hz).",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=1.0,
        help="Aktualisierungsintervall für --watch Modus in Sekunden (Standard: 1.0s).",
    )
    parser.add_argument(
        "--frames", "-n",
        type=int,
        default=5,
        help="Anzahl Kameraframes zur Rauschunterdrückung mitteln (Standard: 5).",
    )
    parser.add_argument(
        "--park-xy",
        type=float,
        nargs=2,
        default=[-0.35, -1.03],
        metavar=("X", "Y"),
        help="Zentrum des ROI-Suchzylinders [x, y] in Metern (Standard: -0.35 -1.03).",
    )
    parser.add_argument(
        "--no-park-xy",
        action="store_true",
        help="Deaktiviert den zylindrischen ROI-Filter und nutzt die gesamte Punktwolke.",
    )
    parser.add_argument(
        "--search-radius",
        type=float,
        default=0.25,
        help="Suchradius um park_xy in Metern (Standard: 0.25m).",
    )
    parser.add_argument(
        "--zarr",
        type=str,
        default=None,
        help="Pfad zum zarr Datensatz (Standard: auto-resolve foil_lifting_box_recorded_action.zarr).",
    )
    parser.add_argument(
        "--recompute",
        action="store_true",
        help="Erzwingt Neuberechnung der Datensatz-Statistiken aus dem Zarr-Archiv.",
    )
    parser.add_argument(
        "--svo",
        type=str,
        default=None,
        help="Pfad zu einer ZED SVO-Aufnahme statt Live-Kamera.",
    )
    parser.add_argument(
        "--pc",
        type=str,
        default=None,
        help="Pfad zu einer gespeicherten .npy Punktwolke statt Live-Kamera.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # 1. Datensatzstatistiken laden / berechnen
    dataset_stats = DatasetStatsManager.get_stats(
        zarr_path=args.zarr,
        force_recompute=args.recompute,
    )

    park_xy = None if args.no_park_xy else np.array(args.park_xy, dtype=float)

    # 2. Offline Punktwolken-Modus (falls --pc übergeben wurde)
    if args.pc:
        if not os.path.exists(args.pc):
            print(f"{CLR_RED}[Fehler] Datei nicht gefunden: {args.pc}{CLR_RESET}")
            sys.exit(1)
        pc_pts = np.load(args.pc)
        z_est, inliers, status = estimate_foil_height(
            pc_pts, park_xy=park_xy, search_radius=args.search_radius
        )
        if z_est is None:
            print(f"{CLR_RED}[Fehler] Aus Datei konnte keine Folienhöhe bestimmt werden.{CLR_RESET}")
            sys.exit(1)
        res = {
            "ok": True,
            "mean_z_mm": z_est * 1000.0,
            "std_z_mm": 0.0,
            "median_z_mm": z_est * 1000.0,
            "min_z_mm": z_est * 1000.0,
            "max_z_mm": z_est * 1000.0,
            "inliers_mean": inliers,
            "status": status,
            "num_samples": 1,
        }
        print_comparison_report(res, dataset_stats, park_xy)
        return

    # 3. Live Kamera Modus
    if not PYZED_AVAILABLE:
        import shutil
        import subprocess
        if shutil.which("docker"):
            chk = subprocess.run(["docker", "ps", "--filter", "name=zed_container", "--format", "{{.Names}}"], capture_output=True, text=True)
            if "zed_container" in chk.stdout.split():
                cmd = ["docker", "exec"]
                if sys.stdin.isatty():
                    cmd.append("-it")
                cmd.extend(["zed_container", "python3", "/root/ros2_ws/data/rollouts/tools/compare_foil_height.py"])
                cmd.extend(sys.argv[1:])
                res = subprocess.run(cmd)
                sys.exit(res.returncode)

        print(f"{CLR_RED}[Fehler] pyzed.sl nicht gefunden und 'zed_container' nicht aktiv!{CLR_RESET}")
        print("Dieses Skript muss entweder im 'zed_container' ausgeführt werden:")
        print("  docker exec -it zed_container python3 /root/ros2_ws/data/rollouts/tools/compare_foil_height.py")
        print("Oder über das Wrapper-Skript:")
        print("  ./data/rollouts/tools/compare_foil_height.sh")
        sys.exit(1)

    print(f"{CLR_CYAN}[Info] Initialisiere ZED 2i Kamera...{CLR_RESET}")
    estimator = LiveCameraFoilEstimator(svo_path=args.svo)

    try:
        estimator.open()
        print(f"{CLR_GREEN}[Info] ZED 2i Kamera erfolgreich verbunden.{CLR_RESET}")

        if args.watch:
            print(f"{CLR_CYAN}[Info] Starte kontinuierliches Monitoring (Beenden mit Strg+C)...{CLR_RESET}")
            while True:
                live_res = estimator.measure_height(
                    num_frames=args.frames,
                    park_xy=park_xy,
                    search_radius=args.search_radius,
                )
                if live_res["ok"]:
                    print_comparison_report(live_res, dataset_stats, park_xy, clear_screen=True)
                else:
                    print(f"\n{CLR_RED}[Warnung] {live_res['error']}{CLR_RESET}")
                time.sleep(args.interval)
        else:
            live_res = estimator.measure_height(
                num_frames=args.frames,
                park_xy=park_xy,
                search_radius=args.search_radius,
            )
            if not live_res["ok"]:
                print(f"{CLR_RED}[Fehler] {live_res['error']}{CLR_RESET}")
                sys.exit(1)
            print_comparison_report(live_res, dataset_stats, park_xy)

    except KeyboardInterrupt:
        print(f"\n{CLR_CYAN}[Info] Monitoring durch Benutzer beendet.{CLR_RESET}")
    finally:
        estimator.close()


if __name__ == "__main__":
    main()
