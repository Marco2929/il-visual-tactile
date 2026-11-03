#!/usr/bin/env python3
"""One self-contained HTML page for a whole checkpoint's rollouts.

This replaces the earlier pair of pages (compare_*.html and browse_*.html).
They answered two different questions - "do the rollouts agree" and "what
happened in this one" - but each shipped its own copy of plotly.js, so the
pair cost ~11 MB to say less than one page with tabs says now.

Three tabs, each chosen because it answers a question the others cannot:

  overview   scatter across runs. A torque limit predicts a flat ratio line;
             a calibration error predicts a slope. Needs all runs in one
             axis, which no timeline view can do.
  rollout    one run at a time, selected by dropdown or slider, because a
             28-curve overlay hides the shape of any single run. The notes
             field is the label: it records the pallet height and foil
             thickness that were varied deliberately, so it IS the
             independent variable of the experiment.
  table      every number behind the plots, sortable, so a claim made from a
             plot can be checked against the value.

The traces for all rollouts are built once and toggled by visibility rather
than re-rendered, so the page needs no server and works from a file:// URL.
"""
import html

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

from .data import decimate

CMD = "#2a7fff"
EXE = "#e8710a"
Z_C = "#17a673"
MARK = "#d62728"
GREY = "#9aa0a6"

# Commanded closure, executed closure, tracking error, predicted progress,
# gate 0.90, alert 0.80, commanded z, closure marker on z, XY path, XY closure marker.
# Kept as a constant because the visibility bookkeeping depends on it.
TRACES_PER_ROLLOUT = 10


# ------------------------------------------------------------- overview ---
def overview_figure(rollouts, flagged):
    stats = [r.stats() for r in rollouts]
    names = [s["name"] for s in stats]

    def col(k):
        return np.array([s[k] for s in stats], dtype=float)

    cmd, ex = col("peak_cmd"), col("peak_exec")
    ratio, dfc = col("ratio"), col("deficit")
    zc, reopen = col("z_at_close"), col("reopen")

    is_flagged = np.array([n in flagged for n in names])
    good = ~is_flagged

    hover = [
        f"<b>{s['name']}</b><br>{html.escape(s['notes']) or '(no notes)'}<br>"
        f"commanded {s['peak_cmd']:.1f} deg<br>"
        f"executed {s['peak_exec']:.1f} deg<br>"
        f"deficit {s['deficit']:.1f} deg ({s['deficit_pct']:.1f}%)<br>"
        f"ratio {s['ratio']:.4f}<br>"
        f"reopening {s['reopen']:.1f} deg<br>"
        f"z at closure {s['z_at_close']:+.4f} m<br>"
        f"z rise {s['z_rise']:+.3f} m<br>"
        f"duration {s['dur']:.0f} s, {s['steps']} steps<br>"
        f"lag {s['lag']:+.2f} s"
        + (f"<br><i>flagged: {flagged[s['name']]}</i>"
           if s["name"] in flagged else "")
        for s in stats
    ]

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "executed vs commanded closure",
            "ratio vs commanded closure",
            "closure aligned at peak command",
            "commanded z at closure vs deficit",
        ),
        horizontal_spacing=0.09, vertical_spacing=0.13,
    )

    # --- 1: executed vs commanded ---
    for mask, label, colour in ((good, "grasped", CMD),
                                (is_flagged, "flagged", GREY)):
        if not mask.any():
            continue
        fig.add_trace(go.Scatter(
            x=cmd[mask], y=ex[mask], mode="markers", name=label,
            marker=dict(size=11, color=colour,
                        line=dict(width=1, color="white")),
            text=[h for h, m in zip(hover, mask) if m],
            hovertemplate="%{text}<extra></extra>"), row=1, col=1)

    xs = np.array([cmd.min() - 5, cmd.max() + 5])
    fig.add_trace(go.Scatter(x=xs, y=xs, mode="lines", name="perfect",
                             line=dict(dash="dot", color="#444", width=1),
                             hoverinfo="skip"), row=1, col=1)

    # Two competing models of the deficit, drawn with their residuals in the
    # legend. They are nearly indistinguishable (1.95 vs 2.09 deg), so the
    # plot is there to show that honestly rather than to pick a winner.
    if good.sum() > 2:
        s_prop = float(np.median(ratio[good]))
        c_fix = float(np.median(cmd[good] - ex[good]))
        res_prop = float(np.std(ex[good] - s_prop * cmd[good]))
        res_fix = float(np.std(ex[good] - (cmd[good] - c_fix)))
        fig.add_trace(go.Scatter(
            x=xs, y=s_prop * xs, mode="lines",
            name=f"proportional: {s_prop:.4f}x (res {res_prop:.2f})",
            line=dict(color=EXE, width=2), hoverinfo="skip"), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=xs, y=xs - c_fix, mode="lines",
            name=f"fixed offset: x - {c_fix:.2f} (res {res_fix:.2f})",
            line=dict(color="#8e44ad", width=2, dash="dash"),
            hoverinfo="skip"), row=1, col=1)

    # --- 2: ratio vs commanded ---
    for mask, colour in ((good, CMD), (is_flagged, GREY)):
        if not mask.any():
            continue
        fig.add_trace(go.Scatter(
            x=cmd[mask], y=ratio[mask], mode="markers", showlegend=False,
            marker=dict(size=11, color=colour,
                        line=dict(width=1, color="white")),
            text=[h for h, m in zip(hover, mask) if m],
            hovertemplate="%{text}<extra></extra>"), row=1, col=2)
    if good.sum() > 2:
        med = float(np.median(ratio[good]))
        fig.add_hline(y=med, line=dict(color=EXE, width=2),
                      annotation_text=f"median {med:.4f}", row=1, col=2)

    # --- 3: aligned closure overlay ---
    # Aligned at peak COMMANDED closure, not at t=0: the rollouts have
    # different approach durations, so a common start time would smear the
    # one event they all share.
    for r, s in zip(rollouts, stats):
        grey = s["name"] in flagged
        t0 = s["t_cmd_close"]
        t, c = decimate(r.cmd_t - t0, r.cmd_close)
        fig.add_trace(go.Scatter(
            x=t, y=c, mode="lines", name=s["short"], showlegend=False,
            line=dict(width=1.3, color="#cfd8dc" if grey else CMD),
            opacity=0.4 if grey else 0.65,
            hovertemplate=f"<b>{s['name']}</b> commanded<br>"
                          "t%{x:+.2f}s  %{y:.1f} deg<extra></extra>"),
            row=2, col=1)
        te, ce = decimate(r.js_t - t0, r.exec_close)
        fig.add_trace(go.Scatter(
            x=te, y=ce, mode="lines", name=s["short"], showlegend=False,
            line=dict(width=1.0, dash="dot",
                      color="#e0e0e0" if grey else EXE),
            opacity=0.35 if grey else 0.55,
            hovertemplate=f"<b>{s['name']}</b> executed<br>"
                          "t%{x:+.2f}s  %{y:.1f} deg<extra></extra>"),
            row=2, col=1)
    fig.add_vline(x=0, line=dict(color="#444", width=1, dash="dot"),
                  row=2, col=1)

    # --- 4: z at closure vs deficit ---
    # The training set has essentially ONE grasp height (0.3 mm spread over
    # 252 episodes) while the rollouts scatter over 140 mm, so height is a
    # candidate confounder for everything else on this page.
    for mask, colour in ((good, None), (is_flagged, GREY)):
        if not mask.any():
            continue
        fig.add_trace(go.Scatter(
            x=zc[mask], y=dfc[mask], mode="markers", showlegend=False,
            marker=dict(size=11,
                        color=colour if colour else reopen[mask],
                        colorscale=None if colour else "Viridis",
                        showscale=colour is None,
                        colorbar=(None if colour else
                                  dict(title="reopening<br>(deg)", x=1.0,
                                       len=0.42, y=0.21)),
                        line=dict(width=1, color="white")),
            text=[h for h, m in zip(hover, mask) if m],
            hovertemplate="%{text}<extra></extra>"), row=2, col=2)

    fig.update_xaxes(title_text="commanded closure (deg)", row=1, col=1)
    fig.update_yaxes(title_text="executed closure (deg)", row=1, col=1)
    fig.update_xaxes(title_text="commanded closure (deg)", row=1, col=2)
    fig.update_yaxes(title_text="executed / commanded", row=1, col=2)
    fig.update_xaxes(title_text="time relative to peak command (s)",
                     row=2, col=1)
    fig.update_yaxes(title_text="closure, finger 2 sum (deg)", row=2, col=1)
    fig.update_xaxes(title_text="commanded z at closure (m)", row=2, col=2)
    fig.update_yaxes(title_text="deficit (deg)", row=2, col=2)

    fig.update_layout(
        height=860, hovermode="closest", template="plotly_white",
        legend=dict(orientation="h", y=-0.08, x=0),
        margin=dict(t=40, b=80, l=60, r=20),
    )
    return fig


# -------------------------------------------------------------- rollout ---
def _rollout_caption(s, flagged):
    flag = (f" &nbsp;<span style='color:#c0392b'>[FLAGGED: "
            f"{html.escape(flagged[s['name']])}]</span>"
            if s["name"] in flagged else "")
    prog_str = ""
    if s.get("max_prog") is not None:
        alert = " <span style='color:#d95f02;font-weight:bold'>[&ge;80%]</span>" if s.get("reached_80") else ""
        prog_str = f" &nbsp;·&nbsp; progress: max <b>{s['max_prog']:.1%}</b>{alert}, final <b>{s['final_prog']:.1%}</b>"

    return (
        f"<b>{s['name']}</b> &nbsp;|&nbsp; "
        f"<span style='color:#444'>{html.escape(s['notes']) or '(no notes)'}"
        f"</span>{flag}<br>"
        f"<span style='font-size:12px;color:#555'>"
        f"{s['dur']:.0f}s, {s['steps']} steps, stop: {s['stop']} "
        f"&nbsp;·&nbsp; commanded <b>{s['peak_cmd']:.1f}</b> \u2192 "
        f"executed <b>{s['peak_exec']:.1f}</b> deg &nbsp;·&nbsp; "
        f"deficit <b>{s['deficit']:.1f}</b> deg ({s['deficit_pct']:.1f}%), "
        f"ratio {s['ratio']:.4f} &nbsp;·&nbsp; "
        f"reopening {s['reopen']:.1f} deg &nbsp;·&nbsp; "
        f"z at closure <b>{s['z_at_close']:+.4f}</b> m, "
        f"rise after {s['z_rise']:+.3f} m &nbsp;·&nbsp; "
        f"lag {s['lag']:+.2f}s{prog_str}</span>")


def rollout_figure(rollouts, flagged):
    stats = [r.stats() for r in rollouts]

    fig = make_subplots(
        rows=3, cols=2,
        specs=[
            [{"colspan": 2}, None],
            [{"colspan": 2}, None],
            [{}, {}],
        ],
        subplot_titles=(
            "grasp closure: commanded vs executed (deg)",
            "predicted progress (ProgressHeadST, DINOv3)",
            "commanded EE height z (m)",
            "commanded path, top view (m)",
        ),
        vertical_spacing=0.10, horizontal_spacing=0.10,
        row_heights=[0.38, 0.30, 0.32],
    )

    for r, s in zip(rollouts, stats):
        vis = r is rollouts[0]

        t, c = decimate(r.cmd_t, r.cmd_close)
        fig.add_trace(go.Scatter(
            x=t, y=c, mode="lines", name="commanded", visible=vis,
            line=dict(color=CMD, width=2.2), legendgroup="cmd",
            hovertemplate="t=%{x:.2f}s  %{y:.1f} deg"
                          "<extra>commanded</extra>"), row=1, col=1)

        te, ce = decimate(r.js_t, r.exec_close)
        fig.add_trace(go.Scatter(
            x=te, y=ce, mode="lines", name="executed", visible=vis,
            line=dict(color=EXE, width=2.2), legendgroup="exe",
            hovertemplate="t=%{x:.2f}s  %{y:.1f} deg"
                          "<extra>executed</extra>"), row=1, col=1)

        # The tracking error as its own filled trace.
        tt, ee = decimate(r.cmd_t, r.track_err)
        fig.add_trace(go.Scatter(
            x=tt, y=ee, mode="lines", name="executed - commanded",
            visible=vis, line=dict(color="#8e44ad", width=1.2),
            fill="tozeroy", fillcolor="rgba(142,68,173,0.13)",
            legendgroup="err",
            hovertemplate="t=%{x:.2f}s  %{y:+.1f} deg"
                          "<extra>tracking error</extra>"), row=1, col=1)

        # Progress traces (Row 2)
        if r.has_progress and len(r.prog_t) > 0:
            tp, pp = decimate(r.prog_t, r.prog_val)
        else:
            tp, pp = np.zeros(0), np.zeros(0)

        fig.add_trace(go.Scatter(
            x=tp, y=pp, mode="lines", name="progress", visible=vis,
            line=dict(color="#17a673", width=2.2), legendgroup="prog",
            hovertemplate="t=%{x:.2f}s  prog=%{y:.1%}<extra>progress</extra>"),
            row=2, col=1)

        x_span = [float(r.cmd_t[0]), float(r.cmd_t[-1])] if len(r.cmd_t) > 1 else [0.0, 1.0]
        fig.add_trace(go.Scatter(
            x=x_span, y=[0.90, 0.90], mode="lines", name="gate 0.90", visible=vis,
            line=dict(color="#2ca02c", width=1.5, dash="dot"), legendgroup="prog_gate",
            hoverinfo="skip"), row=2, col=1)

        fig.add_trace(go.Scatter(
            x=x_span, y=[0.80, 0.80], mode="lines", name="alert 0.80", visible=vis,
            line=dict(color="#d95f02", width=1.5, dash="dash"), legendgroup="prog_thresh",
            hoverinfo="skip"), row=2, col=1)

        tz, zz = decimate(r.cmd_t, r.cmd_pose[:, 2])
        fig.add_trace(go.Scatter(
            x=tz, y=zz, mode="lines", name="commanded z", visible=vis,
            line=dict(color=Z_C, width=2), legendgroup="z",
            hovertemplate="t=%{x:.2f}s  z=%{y:.4f} m<extra>z</extra>"),
            row=3, col=1)

        # Marks the instant of peak commanded closure, so the grasp height
        # can be read off directly instead of eyeballed.
        fig.add_trace(go.Scatter(
            x=[s["t_cmd_close"]], y=[s["z_at_close"]], mode="markers",
            name="closure", visible=vis, legendgroup="mark",
            marker=dict(size=13, color=MARK, symbol="x-thin",
                        line=dict(width=3, color=MARK)),
            hovertemplate=f"closure at t={s['t_cmd_close']:.2f}s<br>"
                          f"z={s['z_at_close']:+.4f} m<extra></extra>"),
            row=3, col=1)

        # Top view: a grasp aimed next to the foil is visible here rather
        # than inferred from three separate position traces.
        kc = int(np.argmax(r.cmd_close))
        fig.add_trace(go.Scatter(
            x=r.cmd_pose[:, 0], y=r.cmd_pose[:, 1], mode="lines",
            name="xy path", visible=vis, legendgroup="xy",
            line=dict(color=CMD, width=1.6),
            hovertemplate="x=%{x:.4f}  y=%{y:.4f} m<extra>path</extra>"),
            row=3, col=2)
        fig.add_trace(go.Scatter(
            x=[r.cmd_pose[kc, 0]], y=[r.cmd_pose[kc, 1]], mode="markers",
            name="closure xy", visible=vis, showlegend=False,
            legendgroup="mark",
            marker=dict(size=13, color=MARK, symbol="x-thin",
                        line=dict(width=3, color=MARK)),
            hovertemplate="closure at x=%{x:.4f}, y=%{y:.4f} m"
                          "<extra></extra>"), row=3, col=2)

    # Dropdown and slider drive the same update, because picking rollout 23
    # of 28 by dragging is guesswork while scanning neighbours by dragging is
    # exactly what the slider is good at.
    steps, buttons = [], []
    for i, s in enumerate(stats):
        visible = [False] * (len(rollouts) * TRACES_PER_ROLLOUT)
        for k in range(TRACES_PER_ROLLOUT):
            visible[i * TRACES_PER_ROLLOUT + k] = True
        args = [{"visible": visible},
                {"title.text": _rollout_caption(s, flagged)}]
        mark = " !" if s["name"] in flagged else ""
        steps.append(dict(method="update", label=s["short"] + mark,
                          args=args))
        buttons.append(dict(
            method="update", args=args,
            label=f"{s['short']}{mark}  {s['notes'][:30] or '(no notes)'}"))

    fig.update_layout(
        title=dict(text=_rollout_caption(stats[0], flagged),
                   x=0.0, xanchor="left", y=0.985, yanchor="top"),
        updatemenus=[dict(buttons=buttons, direction="down", showactive=True,
                          x=1.0, xanchor="right", y=1.17, yanchor="top",
                          bgcolor="white", bordercolor="#ccc")],
        sliders=[dict(active=0, steps=steps, pad=dict(t=50, b=10),
                      currentvalue=dict(prefix="rollout ",
                                        font=dict(size=14)),
                      x=0.02, len=0.96)],
        height=1020, template="plotly_white", hovermode="x unified",
        margin=dict(t=130, b=130, l=60, r=20),
        legend=dict(orientation="h", y=1.10, x=0),
    )

    fig.update_xaxes(title_text="time (s)", row=1, col=1)
    fig.update_yaxes(title_text="closure (deg)", row=1, col=1)
    fig.update_xaxes(title_text="time (s)", row=2, col=1)
    fig.update_yaxes(title_text="progress [0, 1]", range=[-0.05, 1.05], row=2, col=1)
    fig.update_xaxes(title_text="time (s)", row=3, col=1)
    fig.update_yaxes(title_text="z (m)", row=3, col=1)
    fig.update_xaxes(title_text="x (m)", row=3, col=2)
    fig.update_yaxes(title_text="y (m)", row=3, col=2,
                     scaleanchor="x3", scaleratio=1)

    # Shared axis ranges across rollouts: with autoscale the curves look
    # identical while the scale silently changes, which defeats stepping
    # through them to spot differences.
    def span(arrs, pad_frac=0.05):
        a = np.concatenate(arrs)
        lo, hi = float(a.min()), float(a.max())
        pad = pad_frac * (hi - lo) or 0.01
        return [lo - pad, hi + pad]

    fig.update_yaxes(range=span([r.cmd_close for r in rollouts]
                                + [r.exec_close for r in rollouts]
                                + [r.track_err for r in rollouts]),
                     row=1, col=1)
    fig.update_yaxes(range=span([r.cmd_pose[:, 2] for r in rollouts]),
                     row=3, col=1)
    return fig


# ---------------------------------------------------------------- table ---
COLUMNS = [
    ("short", "rollout", "{}"),
    ("notes", "notes", "{}"),
    ("dur", "dur (s)", "{:.0f}"),
    ("steps", "steps", "{:d}"),
    ("max_prog", "max prog", "{:.1%}"),
    ("final_prog", "final prog", "{:.1%}"),
    ("peak_cmd", "cmd (deg)", "{:.1f}"),
    ("peak_exec", "exec (deg)", "{:.1f}"),
    ("deficit", "deficit (deg)", "{:.1f}"),
    ("deficit_pct", "deficit (%)", "{:.1f}"),
    ("ratio", "ratio", "{:.4f}"),
    ("reopen", "reopen (deg)", "{:.1f}"),
    ("lag", "lag (s)", "{:+.2f}"),
    ("z_at_close", "z close (m)", "{:+.4f}"),
    ("z_rise", "z rise (m)", "{:+.3f}"),
    ("pc_frames", "pc frames", "{:d}"),
]


def table_html(rollouts, flagged):
    head = "".join(
        f'<th data-key="{k}">{html.escape(label)}</th>'
        for k, label, _ in COLUMNS)
    rows = []
    for r in rollouts:
        s = r.stats()
        cls = ' class="flagged"' if s["name"] in flagged else ""
        tds = []
        for k, _, fmt in COLUMNS:
            v = s[k]
            txt = fmt.format(v) if v is not None else ""
            # data-v carries the raw number so the sort is numeric; sorting
            # the formatted string would put 9.5 after 10.0.
            raw = v if isinstance(v, (int, float)) else html.escape(str(v))
            tds.append(f'<td data-v="{raw}">{html.escape(str(txt))}</td>')
        title = (f' title="{html.escape(flagged[s["name"]])}"'
                 if s["name"] in flagged else "")
        rows.append(f"<tr{cls}{title}>{''.join(tds)}</tr>")
    return (f'<table id="stats"><thead><tr>{head}</tr></thead>'
            f"<tbody>{''.join(rows)}</tbody></table>")


# ----------------------------------------------------------------- page ---
PAGE = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>{title}</title>
<style>
  body {{ font-family: -apple-system, "Segoe UI", Roboto, sans-serif;
         margin: 0; padding: 0 24px 40px; color: #202124;
         background: #fff; }}
  header {{ padding: 18px 0 6px; border-bottom: 1px solid #e0e0e0; }}
  h1 {{ font-size: 19px; margin: 0 0 6px; }}
  .sub {{ font-size: 13px; color: #5f6368; line-height: 1.6; }}
  .sub code {{ background: #f1f3f4; padding: 1px 5px; border-radius: 3px; }}
  nav {{ margin: 14px 0 6px; }}
  nav button {{ font: inherit; font-size: 13px; padding: 7px 16px;
               margin-right: 6px; border: 1px solid #dadce0;
               background: #f8f9fa; border-radius: 6px; cursor: pointer; }}
  nav button.active {{ background: #2a7fff; border-color: #2a7fff;
                      color: #fff; font-weight: 600; }}
  section {{ display: none; }}
  section.active {{ display: block; }}
  .note {{ font-size: 12.5px; color: #5f6368; max-width: 980px;
          margin: 10px 0 4px; line-height: 1.6; }}
  table {{ border-collapse: collapse; font-size: 12.5px;
          font-variant-numeric: tabular-nums; margin-top: 12px; }}
  th, td {{ padding: 5px 10px; border-bottom: 1px solid #eceff1;
           text-align: right; white-space: nowrap; }}
  th {{ background: #f8f9fa; cursor: pointer; position: sticky; top: 0;
       text-align: right; border-bottom: 2px solid #dadce0;
       user-select: none; }}
  th:hover {{ background: #eceff1; }}
  td:first-child, th:first-child, td:nth-child(2), th:nth-child(2)
       {{ text-align: left; }}
  tbody tr:hover {{ background: #f1f8ff; }}
  tr.flagged td {{ color: #9aa0a6; font-style: italic; }}
</style></head>
<body>
<header>
  <h1>{title}</h1>
  <div class="sub">{subtitle}</div>
</header>
<nav>
  <button class="active" data-tab="overview">overview</button>
  <button data-tab="rollout">per rollout</button>
  <button data-tab="table">table</button>
</nav>
<section class="active" id="overview">
  <div class="note">{overview_note}</div>
  {overview}
</section>
<section id="rollout">
  <div class="note">{rollout_note}</div>
  {rollout}
</section>
<section id="table">
  <div class="note">{table_note}</div>
  {table}
</section>
<script>
// Plotly sizes a plot to its container, and a hidden container has width 0.
// So a tab that has never been shown needs a resize when it first appears,
// otherwise its plots render 0 px wide.
document.querySelectorAll('nav button').forEach(function (b) {{
  b.addEventListener('click', function () {{
    document.querySelectorAll('nav button').forEach(function (o) {{
      o.classList.remove('active');
    }});
    document.querySelectorAll('section').forEach(function (s) {{
      s.classList.remove('active');
    }});
    b.classList.add('active');
    var sec = document.getElementById(b.dataset.tab);
    sec.classList.add('active');
    sec.querySelectorAll('.js-plotly-plot').forEach(function (p) {{
      Plotly.Plots.resize(p);
    }});
  }});
}});

// Sortable table. Numeric when every cell in the column parses as a number,
// lexicographic otherwise, so the notes column still sorts sensibly.
(function () {{
  var table = document.getElementById('stats');
  if (!table) return;
  var dir = {{}};
  table.querySelectorAll('th').forEach(function (th, i) {{
    th.addEventListener('click', function () {{
      var rows = Array.prototype.slice.call(
        table.tBodies[0].querySelectorAll('tr'));
      var vals = rows.map(function (r) {{
        return r.cells[i].dataset.v;
      }});
      var numeric = vals.every(function (v) {{
        return v !== '' && !isNaN(parseFloat(v));
      }});
      dir[i] = -(dir[i] || 1);
      rows.sort(function (a, b) {{
        var x = a.cells[i].dataset.v, y = b.cells[i].dataset.v;
        var c = numeric ? parseFloat(x) - parseFloat(y)
                        : String(x).localeCompare(String(y));
        return c * dir[i];
      }});
      rows.forEach(function (r) {{ table.tBodies[0].appendChild(r); }});
    }});
  }});
}})();
</script>
</body></html>
"""

OVERVIEW_NOTE = (
    "Each point is one rollout. Top left separates the two models of the "
    "closure deficit: a constant ratio (torque or current limit) predicts "
    "the orange line, a fixed offset (calibration or steady-state error) the "
    "purple one &mdash; their residuals are in the legend, and they are "
    "close enough that this plot is not meant to settle it. Grey points are "
    "runs where the hand never closed on anything; they are drawn but "
    "excluded from every fit and from the headline numbers above.")

ROLLOUT_NOTE = (
    "One rollout at a time &mdash; pick it from the dropdown or scan "
    "neighbours with the slider. The purple band is executed minus "
    "commanded: a band of constant height means a calibration offset, a band "
    "that only opens during fast closing means a rate limit. Axis ranges are "
    "shared across all rollouts, so curves really are comparable as you step "
    "through them. The red cross marks peak commanded closure on both the "
    "height and the top-view plot.")

TABLE_NOTE = (
    "Click a header to sort. Every number the plots are drawn from, so a "
    "claim read off a plot can be checked against the value. Greyed-out, "
    "italic rows are the flagged non-grasps; hover one for the reason.")


def build_page(rollouts, flagged, ckpt):
    """Return the complete HTML for one checkpoint."""
    stats = [r.stats() for r in rollouts]
    good = [s for s in stats if s["name"] not in flagged]

    bits = [f"<b>{len(rollouts)}</b> rollouts"]
    if flagged:
        bits.append(f"{len(flagged)} flagged as non-grasps")
    if len(good) > 2:
        ratio = np.array([s["ratio"] for s in good])
        cmd = np.array([s["peak_cmd"] for s in good])
        dfc = np.array([s["deficit"] for s in good])
        zc = np.array([s["z_at_close"] for s in good])
        bits += [
            f"median ratio <b>{np.median(ratio):.4f}</b> "
            f"(sd {ratio.std():.4f})",
            f"median deficit <b>{np.median(dfc):.1f} deg</b>",
            f"corr(cmd, deficit) {np.corrcoef(cmd, dfc)[0, 1]:+.3f}",
            f"corr(z, deficit) {np.corrcoef(zc, dfc)[0, 1]:+.3f}",
        ]
    infer = {s["infer"] for s in stats if s["infer"] is not None}
    if len(infer) == 1:
        bits.append(f"{infer.pop()} inference steps")

    subtitle = " &nbsp;·&nbsp; ".join(bits)

    # plotly.js is embedded ONCE, in the first figure. Two figures with
    # include_plotlyjs=True each would double a ~3 MB payload for nothing.
    overview = pio.to_html(overview_figure(rollouts, flagged),
                           include_plotlyjs=True, full_html=False,
                           config={"displaylogo": False})
    rollout = pio.to_html(rollout_figure(rollouts, flagged),
                          include_plotlyjs=False, full_html=False,
                          config={"displaylogo": False})

    return PAGE.format(
        title=f"rollouts &middot; {html.escape(ckpt)}",
        subtitle=subtitle,
        overview_note=OVERVIEW_NOTE, overview=overview,
        rollout_note=ROLLOUT_NOTE, rollout=rollout,
        table_note=TABLE_NOTE, table=table_html(rollouts, flagged),
    )