#!/usr/bin/env python3
"""
Web Dashboard Server (`queue_web.py`) for the Multi-GPU Training Queue & Evaluation Pipeline.

Provides:
- `/`            : Modern dark-mode auto-refreshing HTML/JS dashboard (updates every 3s)
- `/api/status`  : JSON endpoint with live worker GPU telemetry, job progress, ETAs, losses, and post-training evaluations
"""

import argparse
import datetime as dt
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from gpu_queue import (
    DEFAULT_CONFIG_PATH,
    is_in_schedule,
    job_exp_and_rel_dir,
    load_config,
    load_eval_summary,
    now_str,
    parse_log_detailed,
    query_worker_gpu,
)

# Background cache so multiple browser tabs never spam SSH / nvidia-smi
_STATE_LOCK = threading.Lock()
_CACHED_STATE: Dict[str, Any] = {}
_CONFIG_PATH: Path = DEFAULT_CONFIG_PATH


def collect_state(cfg_path: Path) -> Dict[str, Any]:
    cfg = load_config(cfg_path)
    now = dt.datetime.now()
    settings = cfg.get("settings", {})

    daemon_pid_file = Path(settings.get("log_dir", "/tmp")) / "daemon.pid"
    daemon_running = False
    daemon_pid = None
    if daemon_pid_file.exists():
        try:
            pid = int(daemon_pid_file.read_text().strip())
            os.kill(pid, 0)
            daemon_running = True
            daemon_pid = pid
        except Exception:
            pass

    workers_out: List[Dict[str, Any]] = []
    for w_name, w_cfg in cfg.get("workers", {}).items():
        enabled = w_cfg.get("enabled", True)
        sched = w_cfg.get("schedule", "always")
        in_win = is_in_schedule(sched, w_cfg.get("weekend_always", True), now) if enabled else False
        info = query_worker_gpu(w_name, w_cfg) if enabled else {"online": False, "error": "disabled"}

        used_mb = 0
        total_mb = 0
        free_mb = 0
        util_pct = 0
        vram_pct = 0.0
        state_label = "DISABLED"
        gpu_name = "-"

        if enabled:
            if not info.get("online"):
                state_label = "OFFLINE"
            else:
                gpu_name = info.get("gpu_name", "GPU")
                free_mb = info["free_mb"]
                total_mb = info["total_mb"]
                used_mb = total_mb - free_mb
                util_pct = info["util_pct"]
                vram_pct = round((used_mb / max(1, total_mb)) * 100.0, 1)
                if info["is_free"] and in_win:
                    state_label = "READY"
                elif not in_win:
                    state_label = f"WAIT ({sched})"
                else:
                    state_label = "TRAINING / BUSY"

        workers_out.append({
            "name": w_cfg.get("display_name", w_name),
            "type": w_cfg.get("type", "docker"),
            "connection": w_cfg.get("connection", "local" if w_cfg.get("type") == "docker" else "ethernet"),
            "target": w_cfg.get("target", "-"),
            "schedule": sched,
            "window_open": in_win,
            "online": info.get("online", False),
            "gpu_name": gpu_name,
            "used_mb": used_mb,
            "free_mb": free_mb,
            "total_mb": total_mb,
            "vram_pct": vram_pct,
            "util_pct": util_pct,
            "state": state_label,
        })

    jobs_out: List[Dict[str, Any]] = []
    evals_out: List[Dict[str, Any]] = []
    progress_evals_out: List[Dict[str, Any]] = []
    worker_disp_map = {k: v.get("display_name", k) for k, v in cfg.get("workers", {}).items()}

    for idx, job in enumerate(cfg.get("jobs", []), 1):
        st = job.get("status", "pending")
        w = job.get("worker", "-")
        w_disp = worker_disp_map.get(w, w)
        j_type = job.get("type", "policy")
        log_p = Path(job["log_file"]) if job.get("log_file") else None
        det = parse_log_detailed(log_p, job.get("started_at")) if st in ("running", "completed", "failed") else None
        ev = load_eval_summary(job, settings)

        total_ep = job.get("epochs", 80) if j_type == "progress_head" else 80

        if ev is not None:
            if ev.get("job_type") == "progress_head":
                progress_evals_out.append({
                    "job_id": job["id"],
                    "best_epoch": ev.get("best_epoch"),
                    "best_val_loss": ev.get("best_val_loss"),
                    "best_val_mae": ev.get("best_val_mae"),
                    "best_fpr_09": ev.get("best_fpr_09"),
                    "smooth_weight": ev.get("smooth_weight"),
                    "delta_max": ev.get("delta_max"),
                    "checkpoint_file": ev.get("checkpoint_file"),
                })
            else:
                ab = ev.get("modality_ablation_on_best", {})
                evals_out.append({
                    "job_id": job["id"],
                    "best_epoch": ev.get("best_epoch"),
                    "best_ckpt": ev.get("best_ckpt"),
                    "best_val_action_mse": ev.get("best_val_action_mse"),
                    "best_xy_search_epoch": ev.get("best_xy_search_epoch"),
                    "best_xy_search_mse": ev.get("best_xy_search_mse"),
                    "shuffle_wrist_ratio": ab.get("shuffle_wrist", {}).get("ratio_vs_full"),
                    "shuffle_pc_ratio": ab.get("shuffle_pc", {}).get("ratio_vs_full"),
                    "shuffle_state_ratio": ab.get("shuffle_state", {}).get("ratio_vs_full"),
                    "zero_state_ratio": ab.get("zero_state", {}).get("ratio_vs_full"),
                    "checkpoints": ev.get("checkpoints", []),
                })

        overall_pct = 100.0 if st == "completed" else (det["overall_pct"] if det else 0.0)
        jobs_out.append({
            "idx": idx,
            "id": job["id"],
            "type": j_type,
            "task": job.get("task", ""),
            "dataset": job.get("dataset", job.get("split_json", "")),
            "extra_args": job.get("extra_args", ""),
            "status": st,
            "worker": w_disp,
            "started_at": job.get("started_at", "-"),
            "finished_at": job.get("finished_at", "-"),
            "ep": total_ep if st == "completed" else (det["ep"] if det else 0),
            "total_ep": total_ep,
            "ep_pct": 100 if st == "completed" else (det["ep_pct"] if det else 0),
            "overall_pct": round(overall_pct, 1),
            "eta_str": "done" if st == "completed" else (det["eta_str"] if det else "-"),
            "loss": det["loss"] if det else None,
            "it_s": det["it_s"] if det else None,
            "val_loss": det.get("val_loss") if det else None,
            "fpr_09": det.get("fpr_09") if det else None,
            "best_mse": (ev.get("best_val_action_mse") if ev.get("job_type") != "progress_head" else ev.get("best_val_loss")) if ev else (det["best_mse"] if det else None),
            "eval": ev is not None,
        })

    return {
        "timestamp": now.strftime("%Y-%m-%d %H:%M:%S"),
        "daemon": {
            "running": daemon_running,
            "pid": daemon_pid,
        },
        "workers": workers_out,
        "jobs": jobs_out,
        "evaluations": evals_out,
        "progress_evaluations": progress_evals_out,
    }


def _poller_loop(interval: float = 3.0) -> None:
    global _CACHED_STATE
    while True:
        try:
            st = collect_state(_CONFIG_PATH)
            with _STATE_LOCK:
                _CACHED_STATE = st
        except Exception as e:
            with _STATE_LOCK:
                _CACHED_STATE = {"error": str(e), "timestamp": now_str()}
        time.sleep(interval)


INDEX_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Multi-GPU Training & Eval Dashboard</title>
<style>
  :root {
    --bg: #0d1117;
    --panel: #161b22;
    --border: #30363d;
    --text: #e6edf3;
    --muted: #8b949e;
    --accent: #58a6ff;
    --green: #3fb950;
    --yellow: #d29922;
    --red: #f85149;
    --purple: #bc8cff;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0;
    padding: 20px 28px;
    background: var(--bg);
    color: var(--text);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    font-size: 14px;
  }
  header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding-bottom: 16px;
    border-bottom: 1px solid var(--border);
    margin-bottom: 22px;
  }
  h1 {
    font-size: 20px;
    margin: 0;
    display: flex;
    align-items: center;
    gap: 12px;
  }
  .badge {
    display: inline-block;
    padding: 3px 9px;
    border-radius: 999px;
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.4px;
  }
  .badge-green { background: rgba(63,185,80,0.18); color: var(--green); border: 1px solid rgba(63,185,80,0.4); }
  .badge-blue  { background: rgba(88,166,255,0.18); color: var(--accent); border: 1px solid rgba(88,166,255,0.4); }
  .badge-yellow{ background: rgba(210,153,34,0.18); color: var(--yellow); border: 1px solid rgba(210,153,34,0.4); }
  .badge-red   { background: rgba(248,81,73,0.18); color: var(--red); border: 1px solid rgba(248,81,73,0.4); }
  .badge-muted { background: rgba(139,148,158,0.15); color: var(--muted); border: 1px solid rgba(139,148,158,0.3); }

  .grid-workers {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(330px, 1fr));
    gap: 16px;
    margin-bottom: 24px;
  }
  .card {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 16px 18px;
  }
  .card-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 10px;
  }
  .worker-title { font-weight: 600; font-size: 15px; }
  .worker-sub { color: var(--muted); font-size: 12px; margin-bottom: 12px; font-family: monospace; }
  .meter-row { margin-top: 8px; }
  .meter-label {
    display: flex;
    justify-content: space-between;
    font-size: 12px;
    color: var(--muted);
    margin-bottom: 4px;
  }
  .progress-track {
    width: 100%;
    height: 8px;
    background: #21262d;
    border-radius: 4px;
    overflow: hidden;
  }
  .progress-fill {
    height: 100%;
    border-radius: 4px;
    transition: width 0.35s ease;
  }
  h2 {
    font-size: 15px;
    text-transform: uppercase;
    letter-spacing: 0.6px;
    color: var(--muted);
    margin: 22px 0 10px 0;
  }
  table {
    width: 100%;
    border-collapse: collapse;
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 10px;
    overflow: hidden;
  }
  th, td {
    padding: 11px 14px;
    text-align: left;
    border-bottom: 1px solid var(--border);
  }
  th {
    background: #1c2128;
    color: var(--muted);
    font-size: 12px;
    font-weight: 600;
    text-transform: uppercase;
  }
  tr:last-child td { border-bottom: none; }
  .mono { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; font-size: 12.5px; }
  .right { text-align: right; }
</style>
</head>
<body>
  <header>
    <h1>
      <span>Multi-GPU Training & Evaluation Dashboard</span>
      <span id="daemon-badge" class="badge badge-muted">Connecting...</span>
    </h1>
    <div style="color: var(--muted); font-size: 13px;">
      Last updated: <span id="last-updated" class="mono">-</span> (auto-refresh 3s)
    </div>
  </header>

  <h2>GPU Workers</h2>
  <div id="workers-container" class="grid-workers"></div>

  <h2>Training Queue</h2>
  <table>
    <thead>
      <tr>
        <th>#</th>
        <th>Job ID</th>
        <th>Status</th>
        <th>Worker</th>
        <th style="width: 240px;">Progress</th>
        <th>Epoch</th>
        <th>ETA</th>
        <th>Live Telemetry / Best Eval</th>
      </tr>
    </thead>
    <tbody id="jobs-tbody"></tbody>
  </table>

  <h2>Post-Training Policy Evaluation Summary (Validation MSE & Modality Sensitivity)</h2>
  <table>
    <thead>
      <tr>
        <th>Job ID</th>
        <th>Best Checkpoint</th>
        <th class="right">Val Action MSE</th>
        <th class="right">Decision XY Search MSE</th>
        <th class="right">shuffle_wrist</th>
        <th class="right">shuffle_pc</th>
        <th class="right">shuffle_state</th>
        <th class="right">zero_state</th>
      </tr>
    </thead>
    <tbody id="evals-tbody"></tbody>
  </table>

  <div id="progress-evals-section" style="display:none;">
    <h2>Progress Head Evaluation Summary (Validation Loss & Smoothness)</h2>
    <table>
      <thead>
        <tr>
          <th>Job ID</th>
          <th>Best Checkpoint</th>
          <th class="right">Val Loss</th>
          <th class="right">Val MAE</th>
          <th class="right">FPR (&ge;0.9 on GT&lt;0.7)</th>
          <th class="right">Smooth Weight</th>
          <th class="right">Delta Max</th>
        </tr>
      </thead>
      <tbody id="progress-evals-tbody"></tbody>
    </table>
  </div>

<script>
function statusBadge(st) {
  const s = (st || '').toLowerCase();
  if (s === 'completed' || s === 'ready') return `<span class="badge badge-green">${st.toUpperCase()}</span>`;
  if (s === 'running' || s.includes('training')) return `<span class="badge badge-blue">${st.toUpperCase()}</span>`;
  if (s === 'pending' || s.includes('wait')) return `<span class="badge badge-yellow">${st.toUpperCase()}</span>`;
  if (s === 'failed' || s === 'offline') return `<span class="badge badge-red">${st.toUpperCase()}</span>`;
  return `<span class="badge badge-muted">${st}</span>`;
}

async function refresh() {
  try {
    const res = await fetch('/api/status');
    const data = await res.json();
    document.getElementById('last-updated').textContent = data.timestamp || '-';

    const db = document.getElementById('daemon-badge');
    if (data.daemon && data.daemon.running) {
      db.className = 'badge badge-green';
      db.textContent = `Daemon Running (PID ${data.daemon.pid})`;
    } else {
      db.className = 'badge badge-red';
      db.textContent = 'Daemon Stopped';
    }

    // Render workers
    const wc = document.getElementById('workers-container');
    wc.innerHTML = (data.workers || []).map(w => {
      const utilColor = w.util_pct > 60 ? 'var(--accent)' : (w.util_pct > 20 ? 'var(--yellow)' : 'var(--green)');
      const vramColor = w.vram_pct > 80 ? 'var(--red)' : 'var(--purple)';
      return `
        <div class="card">
          <div class="card-header">
            <span class="worker-title">${w.name}</span>
            ${statusBadge(w.state)}
          </div>
          <div class="worker-sub">${w.type}:${w.target} &bull; ${w.gpu_name} &bull; <span style="font-weight:600; color:var(--accent);">${(w.connection || '').toUpperCase()}</span> &bull; sched: ${w.schedule}</div>
          <div class="meter-row">
            <div class="meter-label">
              <span>VRAM Usage</span>
              <span class="mono">${w.used_mb} / ${w.total_mb} MB (${w.vram_pct}%)</span>
            </div>
            <div class="progress-track">
              <div class="progress-fill" style="width:${w.vram_pct}%; background:${vramColor};"></div>
            </div>
          </div>
          <div class="meter-row">
            <div class="meter-label">
              <span>GPU Compute Util</span>
              <span class="mono">${w.util_pct}%</span>
            </div>
            <div class="progress-track">
              <div class="progress-fill" style="width:${w.util_pct}%; background:${utilColor};"></div>
            </div>
          </div>
        </div>
      `;
    }).join('');

    // Render jobs
    const jb = document.getElementById('jobs-tbody');
    jb.innerHTML = (data.jobs || []).map(j => {
      const barColor = j.status === 'completed' ? 'var(--green)' : (j.status === 'running' ? 'var(--accent)' : '#30363d');
      let tele = '-';
      const isPrg = j.type === 'progress_head';
      if (j.status === 'running') {
        const parts = [];
        if (j.loss !== null && j.loss !== undefined) parts.push(`loss=${j.loss.toFixed(4)}`);
        if (j.it_s !== null && j.it_s !== undefined) parts.push(`${j.it_s.toFixed(1)} it/s`);
        if (isPrg) {
          if (j.val_loss !== null && j.val_loss !== undefined) parts.push(`val_loss=${j.val_loss.toFixed(4)}`);
          if (j.fpr_09 !== null && j.fpr_09 !== undefined) parts.push(`fpr=${j.fpr_09.toFixed(1)}%`);
        } else {
          if (j.best_mse !== null && j.best_mse !== undefined) parts.push(`val_mse=${j.best_mse.toFixed(5)}`);
        }
        tele = parts.length ? parts.join(' &bull; ') : 'warming up...';
      } else if (j.status === 'completed') {
        if (isPrg) {
          tele = j.best_mse !== null ? `best val_loss = ${j.best_mse.toFixed(4)}` : 'completed';
        } else {
          tele = j.best_mse !== null ? `best val_action_mse = ${j.best_mse.toFixed(5)}` : 'completed';
        }
      }
      return `
        <tr>
          <td class="mono">${j.idx}</td>
          <td class="mono" style="font-weight:600;">${j.id}</td>
          <td>${statusBadge(j.status)}</td>
          <td class="mono">${j.worker}</td>
          <td>
            <div class="progress-track" style="height:10px;">
              <div class="progress-fill" style="width:${j.overall_pct}%; background:${barColor};"></div>
            </div>
          </td>
          <td class="mono">Ep ${j.ep}/${j.total_ep} (${j.overall_pct}%)</td>
          <td class="mono">${j.eta_str}</td>
          <td class="mono">${tele}</td>
        </tr>
      `;
    }).join('');

    // Render evaluations
    const eb = document.getElementById('evals-tbody');
    const evals = data.evaluations || [];
    if (!evals.length) {
      eb.innerHTML = `<tr><td colspan="8" style="color:var(--muted); text-align:center;">No completed policy evaluations yet.</td></tr>`;
    } else {
      eb.innerHTML = evals.map(ev => {
        const fmtR = v => (v !== null && v !== undefined && !isNaN(v)) ? `${v.toFixed(2)}x` : '-';
        return `
          <tr>
            <td class="mono" style="font-weight:600;">${ev.job_id}</td>
            <td class="mono">Ep ${ev.best_epoch} <span style="color:var(--muted);">(${ev.best_ckpt || ''})</span></td>
            <td class="mono right" style="color:var(--green); font-weight:600;">${ev.best_val_action_mse ? ev.best_val_action_mse.toFixed(5) : '-'}</td>
            <td class="mono right" style="color:var(--accent);">${ev.best_xy_search_mse ? ev.best_xy_search_mse.toFixed(5) : '-'}</td>
            <td class="mono right">${fmtR(ev.shuffle_wrist_ratio)}</td>
            <td class="mono right">${fmtR(ev.shuffle_pc_ratio)}</td>
            <td class="mono right">${fmtR(ev.shuffle_state_ratio)}</td>
            <td class="mono right">${fmtR(ev.zero_state_ratio)}</td>
          </tr>
        `;
      }).join('');
    }

    // Render progress head evaluations
    const pSec = document.getElementById('progress-evals-section');
    const pb = document.getElementById('progress-evals-tbody');
    const pEvals = data.progress_evaluations || [];
    if (pEvals.length > 0) {
      pSec.style.display = 'block';
      pb.innerHTML = pEvals.map(ev => `
        <tr>
          <td class="mono" style="font-weight:600;">${ev.job_id}</td>
          <td class="mono">Ep ${ev.best_epoch}</td>
          <td class="mono right" style="color:var(--green); font-weight:600;">${ev.best_val_loss ? ev.best_val_loss.toFixed(5) : '-'}</td>
          <td class="mono right">${ev.best_val_mae ? ev.best_val_mae.toFixed(5) : '-'}</td>
          <td class="mono right" style="color:var(--accent);">${ev.best_fpr_09 !== null ? (ev.best_fpr_09 * 100).toFixed(2) + '%' : '-'}</td>
          <td class="mono right">${ev.smooth_weight !== null ? ev.smooth_weight : '-'}</td>
          <td class="mono right">${ev.delta_max !== null ? ev.delta_max : '-'}</td>
        </tr>
      `).join('');
    } else {
      pSec.style.display = 'none';
    }
  } catch (e) {
    console.error(e);
  }
}

refresh();
setInterval(refresh, 3000);
</script>
</body>
</html>
"""


class DashboardHandler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args: Any) -> None:
        # Quiet standard request logging
        pass

    def do_GET(self) -> None:
        if self.path.startswith("/api/status"):
            with _STATE_LOCK:
                payload = _CACHED_STATE or collect_state(_CONFIG_PATH)
            body = json.dumps(payload).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            body = INDEX_HTML.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)


def main() -> None:
    global _CONFIG_PATH
    parser = argparse.ArgumentParser(description="Web Dashboard for Multi-GPU Training Queue")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--poll", type=float, default=3.0)
    args = parser.parse_args()

    _CONFIG_PATH = args.config
    # Initial state population
    with _STATE_LOCK:
        global _CACHED_STATE
        _CACHED_STATE = collect_state(_CONFIG_PATH)

    t = threading.Thread(target=_poller_loop, args=(args.poll,), daemon=True)
    t.start()

    server = ThreadingHTTPServer((args.host, args.port), DashboardHandler)
    print(f"[{now_str()}] Multi-GPU Web Dashboard listening on http://{args.host}:{args.port} (http://localhost:{args.port})")
    sys.stdout.flush()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
