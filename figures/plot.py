"""WCL 正文两图：只读已冻结的数值。 / Two WCL figures from frozen numerical results."""
import json
import argparse
from math import log
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.path import Path as MarkerPath
from matplotlib.ticker import MaxNLocator
from matplotlib.patches import Ellipse, Patch
import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "outputs"
GREEN, BLUE, CLAY = "#405F53", "#586D84", "#8A6954"
INK, GRID = "#333C39", "#D8DDDA"
BASES = (("joint_lmmse", "Joint LMMSE", GREEN, "o"), ("glsnet", "GLSNet", BLUE, "s"))

plt.rcParams.update({
    "font.family": "Liberation Sans", "mathtext.fontset": "stixsans", "font.size": 8.5,
    "pdf.fonttype": 42, "axes.labelcolor": INK, "text.color": INK,
    "axes.edgecolor": "#89918D", "xtick.color": INK, "ytick.color": INK,
    "axes.linewidth": .55, "axes.spines.top": False, "axes.spines.right": False,
    "axes.axisbelow": True, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "legend.frameon": False, "legend.fontsize": 8,
})


def save(fig, name):
    """保留原生矢量文本。 / Preserve native-size vector text."""
    OUT.mkdir(exist_ok=True, parents=True)
    fig.savefig(OUT / name, metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)


def heading(fig, x, text):
    fig.text(x, .945, text, va="center", size=9, weight="bold")


def theory(data):
    """左图为二维边缘等密度线，右图为直接积分。 / Marginal isodensity contours and direct risk integration."""
    wide, narrow = np.sqrt(data["kappa"])
    fig = plt.figure(figsize=(7.05, 2.28))
    heading(fig, .012, "(a)  Equal absolute errors")
    heading(fig, .502, "(b)  Calibrated bit risk")
    # 左右轴是同一尺度；h 的投影方向不反射。 / Reflect the error only; keep channel and axes fixed.
    for i, (angle, point_x, label) in enumerate(((-45, -.31, "Lower risk"), (45, .31, "Higher risk"))):
        ax = fig.add_axes([.015 + .197 * i, .18, .18, .57])
        ax.set(xlim=(-1.14, 1.14), ylim=(-1.14, 1.14), aspect="equal")
        ax.axis("off")
        ax.plot([-1, 1], [0, 0], lw=.6, color="#9DA5A0")
        ax.plot([0, 0], [-1, 1], lw=.6, color="#9DA5A0")
        ax.add_patch(Ellipse((0, 0), 2 * wide, 2 * narrow, angle=angle,
                             edgecolor=GREEN, facecolor="#E8EEEA", linewidth=1.1, zorder=0))
        ax.annotate("", xy=(.85, .85), xytext=(0, 0),
                    arrowprops={"arrowstyle": "->", "color": INK, "lw": 1, "shrinkA": 0, "shrinkB": 0})
        ax.text(.85, .88, r"$h$", ha="left")
        ax.plot(point_x, .55, "o", ms=4.5, color=CLAY, markeredgecolor="white", markeredgewidth=.45)
        ax.text(.5, -.03, label, transform=ax.transAxes, ha="center", weight="bold")
    fig.text(.205, .82, r"First-branch sign flip:  $(e_1,e_2)\mapsto(-e_1,e_2)$", ha="center")
    fig.text(.205, .11, r"Axes:  $\mathrm{Re}\,e_1$  (horizontal),   $\mathrm{Im}\,e_2$  (vertical)", ha="center")
    schematic_keys = [
        Patch(facecolor="#E8EEEA", edgecolor=GREEN, label="Density contour"),
        Line2D([], [], color=CLAY, marker="o", ms=4, linestyle="none", label="Error draw"),
    ]
    fig.legend(handles=schematic_keys, loc="lower center", bbox_to_anchor=(.205, -.006),
               ncol=2, handlelength=1.35, handletextpad=.45, columnspacing=1.1)
    ax = fig.add_axes([.564, .235, .423, .595])
    ax.grid(axis="y", color=GRID, lw=.55)
    reference, = ax.plot([0, 1], [0, 1], color=INK, ls=(0, (2, 3)), lw=.85, zorder=2.2, label="Quadratic")
    keys = []
    for eps, color, marker, size in ((.003, BLUE, "o", 4.4), (.03, GREEN, "+", 3.5), (.3, CLAY, "s", 3.0)):
        rows = sorted((r for r in data["rows"] if r["epsilon_over_r"] == eps), key=lambda r: r["x"])
        if len(rows) != 9:
            raise ValueError("Expected nine orientations per curve")
        series = ax.errorbar([r["x"] for r in rows], [r["normalized_gap"] for r in rows],
                    yerr=[2 * r["normalized_gap_se"] for r in rows], marker=marker,
                    ms=size, mfc="white", mew=.9, lw=.95, elinewidth=.65, color=color, label=rf"$\epsilon/r={eps:g}$")
        keys.append(series)
    ax.set(xlim=(-.025, 1.025), ylim=(-.035, 1.045), xticks=[0, .25, .5, .75, 1],
           yticks=[0, .5, 1], xlabel=r"Fraction of excess variance along $h$,   $x$", ylabel="Normalized risk gap")
    # 相近曲线不人为偏移；图例覆盖空白处。 / Keep coincident risk curves at their actual coordinates.
    ax.legend(handles=[reference, *keys], loc="upper left", ncol=2, borderaxespad=.05,
              handlelength=1.55, handletextpad=.4, columnspacing=1.0)
    save(fig, "fig1.pdf")


def reception(data):
    """三种估计器共用物理槽；读取既有区间。 / All three estimators share slots; use archived intervals."""
    fig, (error_ax, loss_ax, prediction_ax) = plt.subplots(1, 3, figsize=(7.05, 2.29))
    fig.subplots_adjust(left=.077, right=.987, bottom=.354, top=.855, wspace=.45)
    groups = ("deep", "low", "moderate", "strong")
    profiles = ("adapt_a", "adapt_b", "adapt_c", "adapt_d")
    x = np.arange(len(groups))
    legend_keys = []
    styles = (*BASES, ("dncnn", "DnCNN", CLAY, "^"))
    for i, (base, name, color, marker) in enumerate(styles):
        style = dict(fmt=marker, color=color, ms=4.3, mfc=color if i == 0 else "white",
                     mew=.9, capsize=1.7, elinewidth=.8, zorder=3)
        # 左两幅仅示 A 的增益分组，右幅比较全部剖面。 / Gain bins illustrate A; prediction covers every profile.
        grouped = data[base]["profile_a"]
        loss = [grouped[g]["excess"] for g in groups]
        error = [grouped[g]["energy"] for g in groups]
        prediction_rows = [data[base]["matched"][p] for p in "ABCD"]
        for ax, records, scale in ((error_ax, error, 1000), (loss_ax, loss, 6 / log(2))):
            mean, lo, hi = np.array([[v["mean"], *v["ci95"]] for v in records]).T * scale
            series = ax.errorbar(x + (i - 1) * .075, mean, yerr=[mean - lo, hi - mean],
                                 linestyle=("-", "--", "-.")[i], linewidth=1.1, **style)
            if ax is error_ax:
                series.lines[0].set_label(name)
                legend_keys.append(series.lines[0])
        # 名义剖面不连线；空缺不补零。 / Never connect nominal profiles or fill missing estimates with zero.
        mean, lo, hi = np.array([[v["reduction_percent"], *v["reduction_percent_ci95"]]
                                 for v in prediction_rows]).T
        positions = np.arange(len(profiles)) + (i - 1) * .21
        prediction_ax.errorbar(positions, mean, yerr=[mean - lo, hi - mean],
                               linestyle="none", **style)
    for ax, title in zip((error_ax, loss_ax, prediction_ax),
                         ("(a)  Coefficient error", "(b)  Soft-bit loss", "(c)  Held-out prediction")):
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_color("#7D8983")
        ax.grid(axis="y", color="#DADFDB", lw=.5)
        ax.tick_params(axis="y", length=0, pad=3, labelsize=8)
        ax.tick_params(axis="x", length=0, pad=4, labelsize=8)
        ax.set_xticks(x)
        # 显示全部原始区间和零点，不截断负值。 / Include every interval and zero, including negative effects.
        lower, upper = min(0., ax.dataLim.ymin), max(0., ax.dataLim.ymax)
        pad = .07 * (upper - lower)
        ax.set_ylim(lower - pad if lower < 0 else 0., upper + pad)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5, steps=[1, 2, 5, 10]))
        if lower < 0:
            ax.axhline(0, ls="--", color="#777D79", lw=.7)
        ax.set_title(title, loc="left", size=9, weight="bold", pad=8)
    for ax in (error_ax, loss_ax):
        ax.set(xlim=(-.25, 3.25), xticklabels=["< −10", "−10…−5", "−5…0", "≥ 0"])
    error_ax.set(ylabel=r"Error energy ($\times 10^{-3}$)")
    loss_ax.set(ylabel="Excess loss (bits/data RE)")
    prediction_ax.set(xlim=(-.44, 3.44),
                      xticklabels=list("ABCD"), ylabel="MSPE reduction (%)")
    # 仅前两板共用物理横坐标，第三板为离散剖面。 / Only the first two panels share physical gain bins.
    shared_mid = (error_ax.get_position().x0 + loss_ax.get_position().x1) / 2
    fig.text(shared_mid, .167, r"Profile A: combined gain, $10\log_{10}(P/2)$ (dB)", ha="center")
    fig.text(prediction_ax.get_position().x0 + prediction_ax.get_position().width / 2, .167,
             "NTN-TDL profile", ha="center")
    # 全图共用图例，不把未比较的方法画进来。 / One key across panels; include only measured series.
    bracket = MarkerPath([(-.35, -1), (.35, -1), (0, -1), (0, 1), (-.35, 1), (.35, 1)],
                         [MarkerPath.MOVETO, MarkerPath.LINETO] * 3)
    ci_key = Line2D([], [], marker=bracket, ms=7, mew=.8, color=INK,
                    linestyle="none", label="95% interval")
    fig.legend(handles=[*legend_keys, ci_key], loc="lower center", bbox_to_anchor=(.5, -.004),
               ncol=4, handlelength=2, handletextpad=.5, columnspacing=1.7)
    save(fig, "fig2.pdf")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=HERE / "panels.json")
    parser.add_argument("--theory-result", type=Path, help="Use the reaggregated or reintegrated Fig. 1 coordinates.")
    parser.add_argument("--waveform-result", type=Path, help="Use freshly recomputed wave statistics for Fig. 2.")
    parser.add_argument("--output-dir", type=Path, default=OUT)
    args = parser.parse_args()
    OUT = args.output_dir
    panels = json.loads(args.input.read_text())
    if args.theory_result:
        panels["orientation"] = json.loads(args.theory_result.read_text())["orientation"]
    if args.waveform_result:
        results = json.loads(args.waveform_result.read_text())
        for base, series in panels["waveform"].items():
            for group, values in series["profile_a"].items():
                row = results["A"][base]["groups"][group]
                values["energy"] = row["mean_total_error_energy"]
                values["excess"] = row["excess_over_true_diagonal_reference"]
            series["matched"] = {p: results[p][base]["matched"] for p in "ABCD"}
    theory(panels["orientation"])
    reception(panels["waveform"])
