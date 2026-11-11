"""Rebuild all paper figures and numeric tables from the supplied CSV summaries.

Usage:
    python code/draw_figures.py
    python code/draw_figures.py --data-dir report --output-dir report/figures

This script does not fit models or infer unavailable individual predictions.
"""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Patch
from matplotlib.font_manager import FontProperties

BLUE = "#285F8F"
TEAL = "#287D70"
ORANGE = "#BB6B3D"
GRAY = "#8A9299"
SLATE = "#8A9299"
RED = "#B85748"
INK = "#20374C"
plt.rcParams.update({
    "font.family": "cmr10", "font.size": 9, "axes.labelsize": 9,
    "axes.titlesize": 9, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "legend.fontsize": 9, "axes.spines.top": False,
    "axes.spines.right": False, "axes.unicode_minus": False,
    "axes.formatter.use_mathtext": True, "mathtext.fontset": "cm",
    "pdf.fonttype": 42,
})
BOLD = FontProperties(fname=str(Path(matplotlib.get_data_path()) / "fonts/ttf/cmb10.ttf"), size=9)


def save(fig, out, name):
    """Write a high-resolution figure and ensure text stays inside its canvas."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for ax in fig.axes:
        texts = [ax.title, ax.xaxis.label, ax.yaxis.label, *ax.texts,
                 *ax.get_xticklabels(), *ax.get_yticklabels()]
        if not ax.axison:
            texts = [ax.title, *ax.texts]
        for text in texts:
            if not text.get_visible() or not text.get_text():
                continue
            bounds = text.get_window_extent(renderer)
            if bounds.width and bounds.height:
                assert bounds.x0 >= -.5 and bounds.y0 >= -.5, (name, text.get_text())
                assert bounds.x1 <= fig.bbox.x1 + .5 and bounds.y1 <= fig.bbox.y1 + .5, (name, text.get_text())
    fig.savefig(out / f"{name}.pdf")
    plt.close(fig)


def label_number(ax, x, y, text, best=False, **kwargs):
    obj = ax.text(x, y, text, fontsize=9, **kwargs)
    if best:
        obj.set_fontproperties(BOLD)
    return obj


def grid(ax, axis="x"):
    ax.grid(axis=axis, color="#E4E7E9", linewidth=.55)
    ax.set_axisbelow(True)
    ax.tick_params(length=2, pad=3)


def pipeline(out):
    # Sized for Springer LNCS textwidth (12.2 cm = 4.80 in)
    fig, ax = plt.subplots(figsize=(4.85, 3.55))
    fig.subplots_adjust(left=0.005, right=0.995, top=0.995, bottom=0.005)
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")

    INK = "#0F172A"
    MUTED = "#334155"
    LINE_DIV = "#CBD5E1"
    FILLS = {1: "#EEF4FA", 2: "#EEF4FA", 3: "#EAF4EF", 4: "#FDF6EE", 5: "#FDF6EE", 6: "#FDF6EE"}
    BORDERS = {1: "#285F8F", 2: "#285F8F", 3: "#287D70", 4: "#BB6B3D", 5: "#BB6B3D", 6: "#BB6B3D"}
    TITLES = {1: "#1A365D", 2: "#1A365D", 3: "#134E48", 4: "#7C2D12", 5: "#7C2D12", 6: "#7C2D12"}

    w_box = 29.0; gap_x = 5.2; x1 = 1.2
    x2 = x1 + w_box + gap_x  # 35.4
    x3 = x2 + w_box + gap_x  # 69.6
    h_box = 44.5; y_row1 = 53.5; y_row2 = 1.5

    box_coords = {1: (x1, y_row1), 2: (x2, y_row1), 3: (x3, y_row1),
                  4: (x1, y_row2), 5: (x2, y_row2), 6: (x3, y_row2)}
    box_texts = {i: [] for i in range(1, 7)}

    def draw_box_header(x, y, w, h, idx, title, subtitle, t_fs=7.8, s_fs=6.5):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.0,rounding_size=1.2",
                                   facecolor=FILLS[idx], edgecolor=BORDERS[idx], lw=0.95, zorder=2))
        t_obj = ax.text(x + w/2, y + h - 4.2, title, ha="center", va="center",
                        fontsize=t_fs, fontweight="bold", color=TITLES[idx], zorder=4)
        box_texts[idx].append(t_obj)
        s_obj = ax.text(x + w/2, y + h - 8.7, subtitle, ha="center", va="center",
                        fontsize=s_fs, color=MUTED, zorder=4)
        box_texts[idx].append(s_obj)
        ax.plot([x + 1.2, x + w - 1.2], [y + h - 12.2, y + h - 12.2], color=LINE_DIV, lw=0.6, zorder=3)

    def draw_box_items(x, y, w, h, idx, items, item_fs=7.2):
        col = BORDERS[idx]
        x_spine = x + 2.0
        y_positions = [y + 23.5, y + 14.5, y + 5.5]
        ax.plot([x_spine, x_spine], [y_positions[2], y_positions[0]], color=col, lw=0.8, zorder=3)
        for y_pos, item_text in zip(y_positions, items):
            ax.plot([x_spine, x_spine + 1.2], [y_pos, y_pos], color=col, lw=0.8, zorder=3)
            ax.plot(x_spine + 1.2, y_pos, marker="o", markersize=2.4, color=col, zorder=4)
            t = ax.text(x_spine + 2.4, y_pos, item_text, ha="left", va="center",
                        fontsize=item_fs, fontweight="bold", color=INK, zorder=4)
            box_texts[idx].append(t)

    def draw_arrow(x1_a, y1_a, x2_a, y2_a, color="#475569", lw=0.95, mutation_scale=6.5):
        ax.add_patch(FancyArrowPatch((x1_a, y1_a), (x2_a, y2_a), arrowstyle="-|>",
                                     mutation_scale=mutation_scale, color=color, lw=lw,
                                     shrinkA=0.0, shrinkB=0.0, zorder=6))

    # ROW 1: STAGE 1, 2, 3
    draw_box_header(x1, y_row1, w_box, h_box, 1, "PMData Dataset", "16 Participants, 152 Days", t_fs=8.2, s_fs=6.5)
    draw_box_items(x1, y_row1, w_box, h_box, 1, [
        "Fitbit Versa 2",
        "PMSys Daily Surveys",
        "Session-RPE Training"
    ], item_fs=7.2)

    draw_box_header(x2, y_row1, w_box, h_box, 2, "Preprocessing", "Continuous Daily Alignment", t_fs=8.0, s_fs=6.5)
    draw_box_items(x2, y_row1, w_box, h_box, 2, [
        "Daily Calendar Sync",
        "Causal Rolling Windows",
        "Transition Pairing"
    ], item_fs=7.2)

    draw_box_header(x3, y_row1, w_box, h_box, 3, "Feature Engineering", "22 Causal Predictors", t_fs=8.0, s_fs=6.5)
    draw_box_items(x3, y_row1, w_box, h_box, 3, [
        "Subjective Wellness",
        "Wearable Telemetry",
        "Training and Timing"
    ], item_fs=7.2)

    # Horizontal arrows row 1
    draw_arrow(x1 + w_box, y_row1 + h_box/2, x2, y_row1 + h_box/2, color=BORDERS[1], lw=0.95)
    draw_arrow(x2 + w_box, y_row1 + h_box/2, x3, y_row1 + h_box/2, color=BORDERS[3], lw=0.95)

    # CENTRAL DUAL CONNECTOR
    y_split = 49.75
    x_b3_c = x3 + w_box/2; x_b4_c = x1 + w_box/2; x_b6_c = x3 + w_box/2
    bus_col = "#475569"
    ax.plot([x_b3_c, x_b3_c], [y_row1, y_split], color=bus_col, lw=0.9, zorder=5)
    ax.plot([x_b3_c, x_b4_c], [y_split, y_split], color=bus_col, lw=0.9, zorder=5)
    draw_arrow(x_b4_c, y_split, x_b4_c, y_row2 + h_box, color=bus_col, lw=0.95)
    draw_arrow(x_b6_c, y_split, x_b6_c, y_row2 + h_box, color=bus_col, lw=0.95)
    ax.text(50.0, y_split, "1,324 Consecutive Transition Pairs",
            ha="center", va="center", fontsize=7.0, fontweight="bold", color="#1E293B",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="#F1F5F9", edgecolor="#94A3B8", lw=0.6), zorder=6)

    # ROW 2: TRACK 1, 2, 3
    draw_box_header(x1, y_row2, w_box, h_box, 4, "Track 1: Forecasting", "Chronological 70/30 Split", t_fs=7.6, s_fs=6.5)
    draw_box_items(x1, y_row2, w_box, h_box, 4, [
        "Chronological Split",
        "Train-Only Imputation",
        "10 Models and Baseline"
    ], item_fs=7.1)

    y_arrow = y_row2 + h_box/2
    draw_arrow(x1 + w_box, y_arrow, x2, y_arrow, color=BORDERS[4], lw=0.95, mutation_scale=6.5)
    t_explains = ax.text((x1 + w_box + x2)/2, y_arrow + 1.8, "explains",
                         ha="center", va="center", fontsize=4.4, fontweight="bold", color=TITLES[4], zorder=6)

    draw_box_header(x2, y_row2, w_box, h_box, 5, "Track 2: Explainability", "5 Post-Hoc Explainers", t_fs=7.5, s_fs=6.5)
    draw_box_items(x2, y_row2, w_box, h_box, 5, [
        "Top Model Attribution",
        "5 Explainers Tested",
        "Faithfulness Deletion"
    ], item_fs=7.1)

    draw_box_header(x3, y_row2, w_box, h_box, 6, "Track 3: LOPO Transfer", "16 Held-Out Subject Folds", t_fs=7.4, s_fs=6.5)
    draw_box_items(x3, y_row2, w_box, h_box, 6, [
        "16 Held-Out Folds",
        "Generic vs. Personal",
        "Transfer Evaluation"
    ], item_fs=7.1)

    fig.canvas.draw(); renderer = fig.canvas.get_renderer(); inv = ax.transData.inverted()
    for ax_cur in fig.axes:
        texts = [ax_cur.title, *ax_cur.texts]
        for text in texts:
            if not text.get_visible() or not text.get_text(): continue
            bounds = text.get_window_extent(renderer)
            if bounds.width and bounds.height:
                assert bounds.x0 >= -.5 and bounds.y0 >= -.5
                assert bounds.x1 <= fig.bbox.x1 + .5 and bounds.y1 <= fig.bbox.y1 + .5

    for idx, texts in box_texts.items():
        bx, by = box_coords[idx]; bx_right = bx + w_box
        for t in texts:
            bb = t.get_window_extent(renderer).transformed(inv)
            assert bb.x1 <= bx_right + 0.1, f"Box {idx} overflow: {t.get_text()} {bb.x1:.2f} > {bx_right}"
            assert bb.x0 >= bx - 0.1, f"Box {idx} underflow: {t.get_text()} {bb.x0:.2f} < {bx}"

    bb_exp = t_explains.get_window_extent(renderer).transformed(inv)
    assert bb_exp.x0 >= x1 + w_box + 0.1, f"explains left too close: {bb_exp.x0} vs {x1 + w_box}"
    assert bb_exp.y0 >= y_arrow + 0.8, f"explains touches arrow: {bb_exp.y0} vs {y_arrow}"

    save(fig, out, "fig_pipeline")

def people(data, out):
    df = pd.read_csv(data / "table_q1.csv").sort_values("readiness", ascending=False)
    overall = np.average(df.readiness, weights=df.reports)
    y = np.arange(len(df))
    fig, ax = plt.subplots(figsize=(4.8, 3.25))
    ax.barh(y, df.readiness, height=.62,
            color=[TEAL if v == df.readiness.max() else BLUE for v in df.readiness])
    ax.set_yticks(y, [f"{row.pid} (n={row.reports})" for row in df.itertuples()])
    ax.invert_yaxis()
    ax.set_xlim(0, 10); ax.set_xticks([0, 2, 4, 6, 8, 10])
    for i, row in enumerate(df.itertuples()):
        label_number(ax, row.readiness+.12, i, f"{row.readiness:.2f}",
                     best=row.readiness == df.readiness.max(), va="center")
    ax.set_xlabel("Mean reported readiness (0 to 10)")
    fig.text(.60, .965, f"Overall mean = {overall:.2f}", ha="center", va="top", fontsize=9)
    grid(ax)
    fig.subplots_adjust(left=.25, right=.97, bottom=.15, top=.86)
    save(fig, out, "fig_readiness_people")


def availability(data, out):
    df = pd.read_csv(data / "table_cleaning_log.csv").set_index("step")
    total = int(df.loc["person-days", "rows"])
    present = [int(df.loc["days with readiness report", "rows"]), int(df.loc["days with Fitbit steps", "rows"])]
    missing = np.array([total-n for n in present])
    pct = missing/total*100
    fig, ax = plt.subplots(figsize=(4.8, 1.55))
    ax.barh([0, 1], pct, color=[ORANGE, BLUE], height=.45)
    for i, (p, n) in enumerate(zip(pct, missing)):
        ax.text(p+.6, i, f"{p:.1f}% ({n}/{total})", va="center")
    ax.set_yticks([0, 1], ["Readiness report", "Fitbit steps"])
    ax.invert_yaxis(); ax.set_xlim(0, 40); ax.set_xticks([0, 10, 20, 30, 40])
    ax.set_xlabel("Person-days without this measurement (%)")
    grid(ax); fig.subplots_adjust(left=.26, right=.97, bottom=.32, top=.94)
    save(fig, out, "fig_availability")


def hourly(data, out):
    df = pd.read_csv(data / "table_q3.csv")
    fig, ax = plt.subplots(figsize=(4.8, 1.85))
    for part, color, name in [("weekday", BLUE, "Weekday"), ("weekend", ORANGE, "Weekend")]:
        s = df[df.part == part].sort_values("hour")
        assert len(s) == 24
        ax.plot(s.hour, s.steps, marker="o", markersize=2.7, color=color, lw=1.1, label=name)
    ax.set_xticks([0, 6, 12, 18, 23]); ax.set_xlim(0, 23)
    ax.set_xlabel("Hour of day"); ax.set_ylabel("Mean steps")
    grid(ax, "y")
    ax.legend(loc="upper center", bbox_to_anchor=(.5, 1.27), ncol=2, frameon=False)
    fig.subplots_adjust(left=.14, right=.98, top=.76, bottom=.26)
    save(fig, out, "fig_hourly_steps")


def training(data, out):
    df = pd.read_csv(data / "table_q2.csv").set_index("day_type").loc[["rest", "light", "hard"]]
    fig, ax = plt.subplots(figsize=(4.8, 1.6))
    y = np.arange(3)
    ax.barh(y, df.next_readiness, color=BLUE, height=.45)
    for i, row in enumerate(df.itertuples()):
        label_number(ax, row.next_readiness+.12, i, f"{row.next_readiness:.2f}",
                     best=row.next_readiness == df.next_readiness.max(), va="center")
    ax.set_yticks(y, [f"{name.title()} (n={int(n)})" for name, n in zip(df.index, df.n)])
    ax.invert_yaxis(); ax.set_xlim(0, 10); ax.set_xticks([0, 2, 4, 6, 8, 10])
    ax.set_xlabel("Mean next-day readiness (0 to 10)")
    grid(ax); fig.subplots_adjust(left=.24, right=.97, bottom=.31, top=.95)
    save(fig, out, "fig_training_readiness")


def models(data, out):
    df = pd.read_csv(data / "table_rq2.csv").sort_values("MAE")
    fig, ax = plt.subplots(figsize=(4.8, 3.2))
    y = np.arange(len(df))
    ref = float(df.loc[df.model == "Persistence", "MAE"].iloc[0])
    colors = []
    for m, v in zip(df.model, df.MAE):
        if m == "Extra Trees":
            colors.append(TEAL)
        elif m == "Persistence":
            colors.append(ORANGE)
        elif v < ref:
            colors.append(BLUE)
        else:
            colors.append(GRAY)
    ax.barh(y, df.MAE, height=0.62, color=colors)
    ax.axvline(ref, color=ORANGE, ls="--", lw=0.9, zorder=0)
    for i, (m, v) in enumerate(zip(df.model, df.MAE)):
        txt = ax.text(v - 0.025, i, f"{v:.3f}", va="center", ha="right", color="white", fontsize=8.5)
        if m == "Extra Trees":
            txt.set_fontproperties(BOLD)
    ax.set_yticks(y); ax.set_yticklabels(df.model); ax.invert_yaxis()
    ax.set_xlim(0, 1.35); ax.set_xticks([0, 0.3, 0.6, 0.9, 1.2])
    ax.set_xlabel("Test MAE: lower is better")
    legend_elements = [
        Patch(facecolor=TEAL, label="Best (Extra Trees)"),
        Patch(facecolor=BLUE, label="Outperforms baseline"),
        Patch(facecolor=ORANGE, label="Persistence (1.082)"),
        Patch(facecolor=GRAY, label="Underperforms baseline")
    ]
    ax.legend(handles=legend_elements, loc="upper center", bbox_to_anchor=(0.5, 1.25),
              ncol=2, frameon=False, fontsize=8.2, handlelength=1.0, handletextpad=0.4, columnspacing=1.0)
    grid(ax)
    fig.subplots_adjust(left=0.24, right=0.97, top=0.82, bottom=0.14)
    save(fig, out, "fig_model_mae")


def shap(data, out):
    df = pd.read_csv(data / "table_shap_values.csv").head(10)
    domain_map = {
        "ready_base": ("Readiness baseline", BLUE, "///"),
        "readiness": ("Prior-day readiness", BLUE, "///"),
        "rhr": ("Resting heart rate (RHR)", TEAL, ".."),
        "load": ("Daily training load", ORANGE, "xx"),
        "load_28d": ("28-day chronic load", ORANGE, "xx"),
        "sedentary": ("Sedentary duration", TEAL, ".."),
        "steps": ("Daily step count", TEAL, ".."),
        "rhr_dev": ("RHR baseline deviation", TEAL, ".."),
        "ready_dev": ("Readiness deviation", BLUE, "///"),
        "sleep_7d": ("7-day rolling sleep", TEAL, ".."),
    }
    labels = [domain_map[f][0] for f in df.feature]
    colors = [domain_map[f][1] for f in df.feature]
    hatches = [domain_map[f][2] for f in df.feature]

    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    y = np.arange(len(df))
    bars = ax.barh(y, df.mean_abs_shap, height=0.62, color=colors, edgecolor="#1E293B", linewidth=0.6)
    for b, h in zip(bars, hatches):
        b.set_hatch(h)

    for i, v in enumerate(df.mean_abs_shap):
        ax.text(v + 0.012, i, f"{v:.3f}", va="center", ha="left", color=INK, fontsize=9.2, fontweight="medium")

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9.2, color=INK)
    ax.invert_yaxis()
    ax.set_xlim(0, 0.72)
    ax.set_xticks([0, 0.15, 0.30, 0.45, 0.60])
    ax.tick_params(axis="x", labelsize=8.8)
    ax.set_xlabel(r"Mean absolute SHAP value: $\mathrm{mean}(|\mathrm{SHAP\ value}|)$", fontsize=9.0)

    legend_elements = [
        Patch(facecolor=BLUE, hatch="///", edgecolor="#1E293B", linewidth=0.6, label="Subjective wellness"),
        Patch(facecolor=TEAL, hatch="..", edgecolor="#1E293B", linewidth=0.6, label="Wearable telemetry"),
        Patch(facecolor=ORANGE, hatch="xx", edgecolor="#1E293B", linewidth=0.6, label="Training workload")
    ]
    ax.legend(handles=legend_elements, loc="lower center", bbox_to_anchor=(0.42, 1.05),
              ncol=3, frameon=False, fontsize=8.2, handlelength=1.2, handletextpad=0.3, columnspacing=0.8)

    ax.grid(axis="x", color="#E4E7E9", linewidth=0.55, zorder=0)
    fig.subplots_adjust(left=0.36, right=0.96, top=0.88, bottom=0.16)
    save(fig, out, "fig_shap")


def groups(data, out):
    df = pd.read_csv(data / "table_group_share.csv", index_col=0)
    assert np.allclose(df.sum(), 1)
    names = ["TreeSHAP", "KernelSHAP", "LIME", "Permutation", "Gain"]
    fig, ax = plt.subplots(figsize=(4.8, 2.3))
    y = np.arange(5); left = np.zeros(5)
    for group, label, color in [("self-report", "Self-report", BLUE), ("watch", "Wearable", TEAL),
                                ("training log", "Training log", ORANGE), ("calendar", "Calendar", GRAY)]:
        widths = df.loc[group, names].to_numpy()*100
        ax.barh(y, widths, left=left, height=.57, color=color, label=label)
        for i, (w, start) in enumerate(zip(widths, left)):
            if w >= 8:
                ax.text(start+w/2, i, f"{w:.0f}%", ha="center", va="center", color="white", fontsize=9)
        left += widths
    ax.set_yticks(y, names); ax.invert_yaxis(); ax.set_xlim(0, 100)
    ax.set_xticks([0, 25, 50, 75, 100]); ax.set_xlabel("Within-method importance share (%)")
    ax.legend(loc="upper center", bbox_to_anchor=(.41, 1.35), ncol=2, frameon=False, handlelength=1.1)
    fig.subplots_adjust(left=.21, right=.96, top=.72, bottom=.22)
    save(fig, out, "fig_importance_groups")


def deletion(data, out):
    df = pd.read_csv(data / "table_rq3_deletion_curve.csv", index_col=0)
    scores = pd.read_csv(data / "table_rq3_methods.csv", index_col=0)
    methods = scores.drop(index="Random").sort_values("deletion_area", ascending=False).index
    fig, axes = plt.subplots(3, 2, figsize=(4.8, 3.35))
    for ax, name in zip(axes.flat, methods):
        ax.plot(df.index, df[name], marker="o", markersize=2.6, color=BLUE, lw=1)
        ax.plot(df.index, df.Random, color=GRAY, ls="--", lw=.9)
        ax.set_title(name, loc="left", pad=3)
        ax.set_xlim(1, 20); ax.set_ylim(1.08, 1.40)
        ax.set_xticks([1, 5, 12, 20]); ax.set_yticks([1.1, 1.2, 1.3, 1.4])
        grid(ax, "y")
    last = axes.flat[-1]; last.axis("off")
    last.legend([Line2D([], [], color=BLUE, marker="o", markersize=3),
                 Line2D([], [], color=GRAY, ls="--")],
                ["Explainer ranking", "Random ranking"], loc="upper left", frameon=False)
    fig.text(.018, .5, "MAE after deletion", va="center", rotation=90, fontsize=9)
    fig.text(.56, .018, "Number of top-ranked features neutralized", ha="center", fontsize=9)
    fig.subplots_adjust(left=.115, right=.98, top=.91, bottom=.13, hspace=.78, wspace=.3)
    save(fig, out, "fig_deletion_panels")


def lopo(data, out):
    df = pd.read_csv(data / "table_rq3_lopo.csv")
    difference = df.MAE_raw-df.MAE_personal
    fig, ax = plt.subplots(figsize=(4.8, 3.1))
    y = np.arange(len(df))
    ax.barh(y, difference, color=[TEAL if v > 0 else RED for v in difference], height=.66)
    ax.axvline(0, color=INK, lw=.8)
    ax.set_yticks(y, df.pid); ax.invert_yaxis()
    ax.set_xlim(-.27, .74); ax.set_xticks([-.2, 0, .2, .4, .6])
    ax.set_xlabel("MAE decrease: generic minus personalized")
    for i, v in enumerate(difference):
        ax.text(v+(.01 if v>0 else -.01), i, f"{v:+.2f}", va="center", ha="left" if v>0 else "right", fontsize=9)
    fig.legend(handles=[Patch(facecolor=TEAL, label="Improves"), Patch(facecolor=RED, label="Worsens")],
              loc="upper center", bbox_to_anchor=(.55, .99), frameon=False, ncol=2)
    grid(ax); fig.subplots_adjust(left=.12, right=.98, top=.83, bottom=.16)
    save(fig, out, "fig_lopo_changes")


def pred_vs_actual(data, out):
    df = pd.read_csv(data / "table_pred_vs_actual.csv")
    rng = np.random.RandomState(42)
    x_jit = df.y_true + rng.uniform(-0.16, 0.16, size=len(df))
    mae = np.mean(np.abs(df.y_pred - df.y_true))
    r2 = 1 - np.sum((df.y_true - df.y_pred)**2) / np.sum((df.y_true - df.y_true.mean())**2)

    fig, ax = plt.subplots(figsize=(4.8, 3.4))
    ax.set_aspect("equal")

    ax.scatter(x_jit, df.y_pred, s=12, alpha=.38, color=TEAL, edgecolors="none",
               label=f"Extra Trees (MAE = {mae:.3f}, $R^2$ = {r2:.3f})")

    lims = [-.3, 10.3]
    ax.plot(lims, lims, color=ORANGE, ls="--", lw=1.2, label=r"Identity ($y = x$)")

    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xticks([0, 2, 4, 6, 8, 10]); ax.set_yticks([0, 2, 4, 6, 8, 10])
    ax.set_xlabel("Actual readiness score (0 to 10)")
    ax.set_ylabel("Predicted readiness score")

    ax.legend(loc="upper center", bbox_to_anchor=(.5, 1.15), ncol=2, frameon=False, fontsize=8, handlelength=1.1)
    grid(ax, "x"); grid(ax, "y")
    fig.subplots_adjust(left=.18, right=.82, top=.88, bottom=.14)
    save(fig, out, "fig_pred_vs_actual")


def consensus(data, out):
    df = pd.read_csv(data / "table_consensus_features.csv", index_col=0)
    df.columns = ["count"]
    df = df.iloc[::-1]
    name_map = {
        "ready_base": "Readiness baseline",
        "readiness": "Prior-day readiness",
        "load": "Training load",
        "sedentary": "Sedentary minutes",
        "load_28d": "28-day training load",
        "rhr": "Resting heart rate",
        "rhr_dev": "RHR deviation",
        "steps": "Daily steps",
        "ready_dev": "Readiness deviation",
    }
    labels = [name_map.get(n, n) for n in df.index]
    colors = [BLUE if n in ["ready_base", "readiness", "ready_dev"] else ORANGE if n in ["load", "load_28d"] else TEAL for n in df.index]

    fig, ax = plt.subplots(figsize=(4.8, 2.6))
    y = np.arange(len(df))
    ax.barh(y, df["count"], color=colors, height=.60)
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlim(0, 5.8); ax.set_xticks([0, 1, 2, 3, 4, 5])
    ax.set_xlabel("Number of XAI methods ranking feature in top 5")
    grid(ax, "x")
    for i, v in enumerate(df["count"]):
        ax.text(v+.10, i, f"{int(v)}/5", va="center", fontsize=8)
    handles = [Patch(facecolor=BLUE, label="Self-report"), Patch(facecolor=ORANGE, label="Training log"), Patch(facecolor=TEAL, label="Wearable")]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, 1.15), ncol=3, frameon=False, fontsize=8, handlelength=1.0)
    fig.subplots_adjust(left=.27, right=.96, top=.88, bottom=.17)
    save(fig, out, "fig_consensus_features")


def lopo_three_way(data, out):
    df = pd.read_csv(data / "table_rq3_lopo.csv")
    fig, ax = plt.subplots(figsize=(4.8, 3.6))
    y = np.arange(len(df)); h = .25
    ax.barh(y - h, df.MAE_same_as_today, height=h, color=GRAY, alpha=.9, label="Persistence")
    ax.barh(y,     df.MAE_raw,           height=h, color=BLUE, label="Generic features")
    ax.barh(y + h, df.MAE_personal,      height=h, color=TEAL, label="Personalized features")
    ax.set_yticks(y); ax.set_yticklabels(df.pid, fontsize=8)
    ax.invert_yaxis(); ax.set_xlim(0, 2.65); ax.set_xticks([0, .5, 1.0, 1.5, 2.0, 2.5])
    ax.set_xlabel("Leave-One-Person-Out MAE (lower is better)")
    grid(ax, "x")
    ax.legend(loc="upper center", bbox_to_anchor=(.5, 1.12), ncol=3, frameon=False, fontsize=8, handlelength=1.1)
    fig.subplots_adjust(left=.12, right=.96, top=.90, bottom=.12)
    save(fig, out, "fig_lopo_three_way")


def lopo(data, out):
    df = pd.read_csv(data / "table_rq3_lopo.csv")
    difference = df.MAE_raw - df.MAE_personal
    fig, ax = plt.subplots(figsize=(4.8, 3.2))
    y = np.arange(len(df))
    ax.barh(y, difference, color=[TEAL if v > 0 else RED for v in difference], height=.62)
    ax.axvline(0, color=INK, lw=.8)
    ax.set_yticks(y); ax.set_yticklabels(df.pid)
    ax.invert_yaxis()
    ax.set_xlim(-.28, .74); ax.set_xticks([-.2, 0, .2, .4, .6])
    ax.set_xlabel("MAE decrease: generic minus personalized")
    for i, v in enumerate(difference):
        offset = .02 if v > 0 else -.02
        ax.text(v + offset, i, f"{v:+.2f}", va="center", ha="left" if v > 0 else "right", fontsize=8.5)
    ax.legend(handles=[Patch(facecolor=TEAL, label="Improves"), Patch(facecolor=RED, label="Worsens")],
              loc="upper center", bbox_to_anchor=(.5, 1.12), frameon=False, ncol=2, fontsize=8.5)
    grid(ax)
    fig.subplots_adjust(left=.12, right=.97, top=.90, bottom=.14)
    save(fig, out, "fig_lopo_changes")


def athlete_scatter(data, out):
    q1 = pd.read_csv(data / "table_q1.csv")
    lopo = pd.read_csv(data / "table_rq3_lopo.csv")
    df = q1.merge(lopo, on="pid")
    df["adherence"] = df.reports / 152 * 100
    df["cv"] = df.sd / df.readiness
    df["gain"] = df.MAE_raw - df.MAE_personal

    df = df.sort_values("MAE_personal").reset_index(drop=True)
    df["pos"] = np.arange(1, len(df) + 1)

    selected_pids = {"p03", "p16", "p14", "p09", "p04"}

    fig, ax = plt.subplots(figsize=(4.8, 3.5))
    y = np.arange(len(df))

    RED_LINE = "#C0392B"

    for i, row in df.iterrows():
        is_sel = row.pid in selected_pids
        ax.barh(i, row.MAE_personal, height=0.62, color=TEAL if is_sel else GRAY,
                edgecolor=INK, linewidth=0.6, hatch="///" if is_sel else None, zorder=2)
        ax.text(row.MAE_personal + 0.04, i, f"{row.MAE_personal:.2f}", va="center", fontsize=7.5,
                color=INK if is_sel else "#555555",
                fontweight="bold" if is_sel else "normal")

    # Cohort persistence vertical line
    ax.axvline(1.108, color=RED_LINE, ls="--", lw=0.9, zorder=3)

    ylabels = [f"#{int(row.pos):2d}  {row.pid}" for _, row in df.iterrows()]
    ax.set_yticks(y)
    ax.set_yticklabels(ylabels, fontsize=7.8)
    ax.invert_yaxis()

    ax.set_xlim(0, 2.70)
    ax.set_xticks([0, 0.5, 1.0, 1.5, 2.0, 2.5])
    ax.set_xlabel("Personalized LOPO MAE (lower is better)", fontsize=8.5)

    legend_elements = [
        Patch(facecolor=TEAL, hatch="///", edgecolor=INK, label="Selected archetype (5)"),
        Patch(facecolor=GRAY, edgecolor=INK, label="Other participant (11)"),
        Line2D([], [], color=RED_LINE, ls="--", lw=0.9, label="Cohort mean persistence (1.108)"),
    ]
    ax.legend(handles=legend_elements, loc="upper center",
              bbox_to_anchor=(0.5, 1.13), ncol=3, frameon=False, fontsize=7.2,
              handlelength=1.1, handletextpad=0.3, columnspacing=0.8)

    grid(ax, "x")
    fig.subplots_adjust(left=0.18, right=0.97, top=0.88, bottom=0.12)
    save(fig, out, "fig_athlete_ranking")


def athlete_profiles(data, out):
    q1 = pd.read_csv(data / "table_q1.csv")
    lopo = pd.read_csv(data / "table_rq3_lopo.csv")
    df = q1.merge(lopo, on="pid")
    df["adherence"] = (df.reports / 152 * 100).round(0).astype(int)
    df["gain"] = df.MAE_raw - df.MAE_personal
    df = df.sort_values("MAE_personal").reset_index(drop=True)

    top5_pids = {"p03", "p16", "p14", "p09", "p04"}

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5.6, 6.2), gridspec_kw={"height_ratios": [1.30, 1.0]})

    # Panel (a): All 16 participants ranking
    y1 = np.arange(len(df))
    ax1.barh(y1, df["MAE_personal"], height=0.68, color=TEAL, alpha=0.85, zorder=2)

    for i, row in df.iterrows():
        is_top5 = i < 5
        color = INK if is_top5 else "#555555"
        fw = "bold" if is_top5 else "normal"
        ax1.text(row.MAE_personal + 0.035, i, f"{row.MAE_personal:.2f}", va="center", ha="left", fontsize=8.2, color=color, fontweight=fw,
                 bbox=dict(boxstyle="square,pad=0.06", facecolor="white", edgecolor="none", alpha=0.85), zorder=3)

    ax1.axvline(1.108, color=RED, ls="--", lw=1.0, zorder=1)

    ylabels1 = [row.pid for _, row in df.iterrows()]
    ax1.set_yticks(y1)
    ax1.set_yticklabels(ylabels1, fontsize=8.2, color=INK)
    ax1.set_ylim(15.7, -0.85)
    ax1.set_xlim(0, 2.70)
    ax1.set_xticks([0, 0.5, 1.0, 1.5, 2.0, 2.5])
    ax1.tick_params(axis="x", labelsize=8.2)
    ax1.set_xlabel("Personalized LOPO MAE (lower is better)", fontsize=8.8)
    ax1.set_title("(a)", loc="left", fontsize=9.5, fontweight="bold", pad=12)
    ax1.grid(axis="x", color="#E4E7E9", linewidth=0.55, zorder=0)

    # Panel (a) Legend: placed with comfortable gap below title
    handles_a = [
        Patch(facecolor=TEAL, label="Personalized LOPO"),
        Line2D([], [], color=RED, ls="--", lw=1.0, label="Persistence (1.108)")
    ]
    ax1.legend(handles=handles_a, loc="lower center", bbox_to_anchor=(0.55, 1.02),
               ncol=2, frameon=False, fontsize=8.0, handlelength=1.1, handletextpad=0.4, columnspacing=1.2)

    # Panel (b): Paired Top-5
    order = ["p04", "p14", "p09", "p16", "p03"]
    adh_map = {"p04": 91, "p14": 79, "p09": 66, "p16": 54, "p03": 43}
    role_labels = [f"{pid} ({adh_map[pid]}% adherence)" for pid in order]
    sub = df[df.pid.isin(order)].set_index("pid").loc[order]

    y2 = np.arange(len(order))
    h2 = 0.28

    b1 = ax2.barh(y2 - h2/2, sub["MAE_raw"], height=h2*0.88, color=SLATE, alpha=0.85, zorder=2)
    b2 = ax2.barh(y2 + h2/2, sub["MAE_personal"], height=h2*0.88, color=TEAL, zorder=2)

    for i, (g_mae, p_mae) in enumerate(zip(sub["MAE_raw"], sub["MAE_personal"])):
        ax2.text(g_mae + 0.025, i - h2/2, f"{g_mae:.2f}", va="center", ha="left", fontsize=8.2, color="#475569",
                 bbox=dict(boxstyle="square,pad=0.08", facecolor="white", edgecolor="none", alpha=0.85), zorder=3)
        ax2.text(p_mae + 0.025, i + h2/2, f"{p_mae:.2f}", va="center", ha="left", fontsize=8.2, color=INK,
                 bbox=dict(boxstyle="square,pad=0.08", facecolor="white", edgecolor="none", alpha=0.85), zorder=3)

    ax2.axvline(1.108, color=RED, ls="--", lw=1.0, zorder=1)

    for y_sep in (y2[:-1] + y2[1:]) / 2:
        ax2.axhline(y_sep, color="#E5E7EB", ls=":", lw=0.6, zorder=1)

    ax2.set_title("(b)", loc="left", fontsize=9.5, fontweight="bold", pad=12)

    handles_b = [
        Patch(facecolor=SLATE, alpha=0.85, label="Generic LOPO"),
        Patch(facecolor=TEAL, label="Personalized LOPO")
    ]
    ax2.legend(handles=handles_b, loc="lower left", bbox_to_anchor=(0.08, 1.02),
               ncol=2, frameon=False, fontsize=8.0, handlelength=1.1, handletextpad=0.4, columnspacing=1.0)

    x_gain = 1.80
    ax2.text(x_gain, 1.02, "MAE reduction", transform=ax2.get_xaxis_transform(),
             ha="center", va="bottom", fontsize=8.0, color=TEAL, fontweight="bold")
    for i, gain in enumerate(sub["gain"]):
        ax2.text(x_gain, i, f"+{gain:.2f}", ha="center", va="center", fontsize=8.4, color=TEAL, fontweight="bold")

    ax2.set_yticks(y2)
    ax2.set_yticklabels(role_labels, fontsize=8.2, color=INK)
    ax2.set_ylim(4.6, -0.85)
    ax2.set_xlim(0, 2.05)
    ax2.set_xticks([0, 0.3, 0.6, 0.9, 1.2, 1.5])
    ax2.tick_params(axis="x", labelsize=8.2)
    ax2.set_xlabel("Leave-One-Person-Out MAE (lower is better)", fontsize=8.8)
    ax2.grid(axis="x", color="#E4E7E9", linewidth=0.55, zorder=0)

    fig.subplots_adjust(left=0.25, right=0.96, top=0.93, bottom=0.07, hspace=0.45)
    save(fig, out, "fig_athlete_profiles")


def number(value, digits=3, best=False):
    if pd.isna(value):
        return "n/a"
    text = f"{value:.{digits}f}"
    return r"\textbf{"+text+"}" if best else text


def write_tables(data, target):
    target.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(data / "table_rq2.csv")
    assert len(df) == 11 and (df.model == "Persistence").sum() == 1
    rows=[]
    for row in df.itertuples():
        values=[]
        for column, digits, direction in [("MAE",3,"min"),("RMSE",3,"min"),("R2",3,"max"),("sd",5,"min"),("train_s",3,"min")]:
            optimum=getattr(df[column],direction)()
            value=getattr(row,column)
            is_key_opt = (column in ["MAE", "RMSE", "R2"]) and np.isclose(value,optimum,rtol=0,atol=1e-12)
            values.append(number(value,digits,is_key_opt))
        rows.append(row.model+" & "+" & ".join(values)+r"\\")
    table=r'''\begingroup\small\setlength{\tabcolsep}{3pt}
\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}lrrrrr}
\toprule
Model & MAE $\downarrow$ & RMSE $\downarrow$ & $R^2$ $\uparrow$ & \shortstack{MAE seed\\SD $\downarrow$} & \shortstack{Fit (s)\\$\downarrow$}\\
\midrule
'''+"\n".join(rows)+r'''
\bottomrule
\end{tabular*}\endgroup
'''
    (target/"models.tex").write_text(table,encoding="utf-8")
    df=pd.read_csv(data/"table_rq3_methods.csv",index_col=0)
    rows=[]
    method_keys = ["Tree Gain" if "Tree Gain" in df.index else "Gain", "KernelSHAP", "LIME", "Permutation", "TreeSHAP", "Random"]
    for name in method_keys:
        values=[]
        for column,direction in [("MAE_after_top5","max"),("deletion_area","max"),("rank_stability","max"),("seconds","min")]:
            value=df.loc[name,column]
            optimum=getattr(df[column],direction)()
            is_key_opt = (column in ["MAE_after_top5", "deletion_area"]) and not pd.isna(value) and np.isclose(value,optimum,rtol=0,atol=1e-12)
            values.append(number(value,3,is_key_opt))
        display_name = "Gain" if name in ["Gain", "Tree Gain"] else name
        rows.append(display_name+" & "+" & ".join(values)+r"\\")
    table=r'''\begingroup\small\setlength{\tabcolsep}{3pt}
\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}lrrrr}
\toprule
Method & \shortstack{MAE after\\top-5 $\uparrow$} & \shortstack{Deletion\\Area $\uparrow$} & \shortstack{Rank\\stability $\uparrow$} & \shortstack{Time (s)\\$\downarrow$}\\
\midrule
'''+"\n".join(rows)+r'''
\bottomrule
\end{tabular*}\endgroup
'''
    (target/"explainers.tex").write_text(table,encoding="utf-8")
    df=pd.read_csv(data/"table_q2.csv")
    rows=[]
    for row in df.itertuples():
        rows.append(f"{row.day_type.title()} & {row.n} & "+number(row.next_readiness,2,row.next_readiness==df.next_readiness.max())+f" & {row.sleep_min:.1f}"+r"\\")
    table=r'''\begingroup\small\setlength{\tabcolsep}{4pt}
\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}lrrr}
\toprule
Day type & \shortstack{Observed\\pairs $n$} & \shortstack{Readiness score\\$\uparrow$} & \shortstack{Sleep (min)\\descriptive}\\
\midrule
'''+"\n".join(rows)+r'''
\bottomrule
\end{tabular*}\endgroup
'''
    (target/"day_type.tex").write_text(table,encoding="utf-8")
    df=pd.read_csv(data/"table_rq3_lopo.csv")
    summary=pd.DataFrame([
        {"method":name,"macro_MAE":df[col].mean(),"pooled_MAE":np.average(df[col],weights=df.n)}
        for name,col in [("Persistence","MAE_same_as_today"),("Generic features","MAE_raw"),("Personalized features","MAE_personal")]])
    summary["n_participants"]=len(df);summary["n_predictions"]=int(df.n.sum())
    summary.to_csv(target/"lopo_summary.csv",index=False)
    rows=[]
    for row in summary.itertuples():
        rows.append(row.method+" & "+number(row.macro_MAE,3,row.method=="Personalized features")+" & "+number(row.pooled_MAE,3,row.method=="Personalized features")+r"\\")
    table=r'''\begingroup\small\setlength{\tabcolsep}{5pt}
\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}lrr}
\toprule
Variant & \shortstack{Mean participant\\MAE $\downarrow$} & \shortstack{Pooled prediction\\MAE $\downarrow$}\\
\midrule
'''+"\n".join(rows)+r'''
\bottomrule
\end{tabular*}\endgroup
'''
    (target/"lopo.tex").write_text(table,encoding="utf-8")


def main():
    base=Path(__file__).resolve().parent.parent
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir",type=Path,default=base/"report")
    parser.add_argument("--output-dir",type=Path,default=base/"report/figures")
    parser.add_argument("--table-dir",type=Path,default=base/"report/tables")
    args=parser.parse_args();args.output_dir.mkdir(parents=True,exist_ok=True)
    for fn in [shap, athlete_profiles]:
        fn(args.data_dir, args.output_dir)
        print("Built", fn.__name__)
    pipeline(args.output_dir);write_tables(args.data_dir,args.table_dir)
    print("Built pipeline and four data-driven LaTeX tables")
    manifest={"figures":sorted(p.name for p in args.output_dir.glob("*.pdf")),
              "data_directory":str(args.data_dir),"model_training_performed":False,
              "unavailable_raw_predictions_inferred":False}
    (args.output_dir/"figure_manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")


if __name__=="__main__":
    main()
