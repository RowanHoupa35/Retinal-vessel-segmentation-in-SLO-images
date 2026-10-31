"""Figures for the README (English, PNG) and the report (French, PDF).

Reads results/pipeline/per_image.csv and results/analysis/*.csv (run
scripts/ablation.py --test and scripts/analyze_errors.py first); the
cross-validation figures also need results/cross_validation/ (scripts/cv_*.py)
and are skipped without it.

    python scripts/make_figures.py            # results/figures/*.png, with titles
    python scripts/make_figures.py --lang fr  # report/figures/*.pdf, captions in LaTeX
"""

import argparse
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter
from scipy.ndimage import median_filter

from vessels.baselines import frangi_response, threshold_segmentation
from vessels.data import ALL_IDS, TEST_IDS, TUNE_IDS, fov_mask, load_ground_truth, load_image
from vessels.evaluation import evaluate
from vessels.morphology import black_tophat, sup_of_line_openings
from vessels.pipeline import Params, binarize, vessel_response
from vessels.thresholding import rank_threshold
from vessels.visualization import (AXIS, DETECTED, FALSE_POS, GRID, INK, INK_MUTED, INK_SECONDARY,
                                   METHOD_COLORS, MISSED, SURFACE, error_overlay)

ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "results" / "analysis"
PARAMS = Params()
PIPELINE, FRANGI, THRESHOLD = "Morphological pipeline", "Frangi vesselness", "Global threshold"
UNET = "U-Net"
METHODS = (PIPELINE, FRANGI, THRESHOLD)
CV = ROOT / "results" / "cross_validation"
CV_METHODS = {"pipeline": PIPELINE, "unet": UNET, "frangi": FRANGI, "threshold": THRESHOLD}
UNET_THRESHOLD = 0.5  # vessels.unet.THRESHOLD, without importing PyTorch
STEP_NAMES = {"+ area opening": PIPELINE, "Frangi vesselness": FRANGI, "Global threshold": THRESHOLD}
# Operating points chosen on tune (see results/pipeline/ablation.md).
FRANGI_KEPT, THRESHOLD_T = 0.07, 80

# Crops chosen by hand to illustrate each failure type: (image, rows, cols).
STEPS_CROP = ("02", slice(110, 310), slice(200, 400))
FAILURE_CROPS = [("02", slice(190, 318), slice(230, 358)),
                 ("37", slice(150, 278), slice(220, 348)),
                 ("32", slice(40, 168), slice(120, 248))]
COMPARISON_IMAGE = "37"

TEXT = {
    "en": {
        "methods": {PIPELINE: "Morphological pipeline", FRANGI: "Frangi vesselness",
                    THRESHOLD: "Global threshold", UNET: "U-Net"},
        "overlay": ["detected, annotated", "detected, not annotated", "annotated, missed"],
        "cv_columns": ["Image", "Morphological pipeline", "U-Net (seed 0)"],
        "seed_range": "U-Net: range over 3 seeds",
        "centrelines": "Centrelines", "drawn_wide": ", drawn 3 px wide",
        "steps": ["1. Input", "2. Median filter 3 × 3", "3. Black top-hat, r = 4",
                  f"4. Line openings, L = {PARAMS.line_length}", "5. Multi-scale, r = 2 to 5",
                  "6. Hysteresis + area opening", "Ground truth", "Errors on centrelines"],
        "input": "Test image {id}\ninput", "expert": "Ground truth\nexpert annotation",
        "crops": ["Detected but not annotated", "Optic disc", "Missed thin vessels"],
        "image": "image {id}", "recall": "Recall (skeleton)", "precision": "Precision (skeleton)",
        "f1": "F1 (skeleton)", "tune": "Tune images", "test": "Test images",
        "width": "Width of the annotated vessel (share of the annotated length)",
        "width_recall": "Recall (share of centreline found)",
        "titles": {"steps": "Pipeline steps on a crop of test image {id}",
                   "pr": "Precision–recall on the test images",
                   "pr_sub": "Dots: operating points chosen on the tune images",
                   "per_image": "F1 per image", "width": "Recall by vessel width, test images",
                   "cv_per_image": "F1 per image, each image held out (leave-one-out)",
                   "cv_width": "Recall by vessel width, leave-one-out"},
    },
    "fr": {
        "methods": {PIPELINE: "Pipeline morphologique", FRANGI: "Filtre de Frangi",
                    THRESHOLD: "Seuil global", UNET: "U-Net"},
        "overlay": ["détecté, annoté", "détecté, non annoté", "annoté, manqué"],
        "cv_columns": ["Image", "Pipeline morphologique", "U-Net (graine 0)"],
        "seed_range": "U-Net : étendue sur 3 graines",
        "centrelines": "Lignes centrales", "drawn_wide": ", tracées sur 3 px",
        "steps": ["1. Image d'entrée", "2. Filtre médian 3 × 3", "3. Top-hat noir, r = 4",
                  f"4. Ouvertures linéaires, L = {PARAMS.line_length}", "5. Multi-échelle, r = 2 à 5",
                  "6. Hystérésis + ouverture d'aire", "Vérité terrain", "Erreurs (lignes centrales)"],
        "input": "Image de test {id}\nentrée", "expert": "Vérité terrain\nannotation experte",
        "crops": ["Détecté mais non annoté", "Disque optique", "Vaisseaux fins manqués"],
        "image": "image {id}", "recall": "Rappel (squelette)", "precision": "Précision (squelette)",
        "f1": "F1 (squelette)", "tune": "Images de réglage", "test": "Images de test",
        "width": "Largeur du vaisseau annoté (part de la longueur annotée)",
        "width_recall": "Rappel (part de la ligne centrale)",
        "titles": None,
    },
}


class Style:
    def __init__(self, lang):
        self.lang = lang
        self.t = TEXT[lang]
        self.report = lang == "fr"
        self.out = ROOT / ("report/figures" if self.report else "results/figures")
        self.ext = "pdf" if self.report else "png"

    def num(self, value, digits=2):
        text = f"{value:.{digits}f}"
        return text.replace(".", ",") if self.lang == "fr" else text

    def percent(self, value):
        return f"{value * 100:.0f} %" if self.lang == "fr" else f"{value:.0%}"

    def ticks(self, ax, axis="both", digits=1):
        if self.lang != "fr":
            return
        formatter = FuncFormatter(lambda v, _: self.num(v, digits))
        if axis in ("x", "both"):
            ax.xaxis.set_major_formatter(formatter)
        if axis in ("y", "both"):
            ax.yaxis.set_major_formatter(formatter)

    def title(self, target, key, fmt=None, **kwargs):
        if not self.t["titles"]:
            return
        text = self.t["titles"][key].format(**(fmt or {}))
        if isinstance(target, plt.Figure):
            target.suptitle(text, **kwargs)
        else:
            target.set_title(text, **kwargs)

    def save(self, fig, name):
        fig.savefig(self.out / f"{name}.{self.ext}")
        plt.close(fig)


def configure_matplotlib(style):
    family = ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"]
    if style.report:
        try:
            for face in ("Regular", "Bold", "Italic"):
                path = subprocess.run(["kpsewhich", f"LibertinusSans-{face}.otf"],
                                      capture_output=True, text=True, check=True).stdout.strip()
                font_manager.fontManager.addfont(path)
            family = ["Libertinus Sans"] + family
        except (OSError, subprocess.CalledProcessError):
            pass
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.family": "sans-serif", "font.sans-serif": family,
        "font.size": 8.5 if style.report else 9, "text.color": INK,
        "axes.titlesize": 9 if style.report else 10, "axes.titlecolor": INK,
        "axes.titlelocation": "left", "axes.labelcolor": INK_SECONDARY, "axes.edgecolor": AXIS,
        "axes.linewidth": 0.8, "xtick.color": INK_MUTED, "ytick.color": INK_MUTED,
        "xtick.labelcolor": INK_SECONDARY, "ytick.labelcolor": INK_SECONDARY,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
        "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
        "lines.linewidth": 2, "savefig.dpi": 300 if style.report else 200,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.03 if style.report else 0.1,
    })


def overlay_legend(style, fig, y, wide=False, loc="upper center", x=0.5, ncol=3):
    handles = [Patch(facecolor=c, edgecolor=INK_MUTED, linewidth=0.5, label=l)
               for c, l in zip((DETECTED, FALSE_POS, MISSED), style.t["overlay"])]
    title = style.t["centrelines"] + (style.t["drawn_wide"] if wide else "")
    legend = fig.legend(handles=handles, loc=loc, bbox_to_anchor=(x, y), ncol=ncol,
                        handlelength=1.2, columnspacing=1.6, labelcolor=INK_SECONDARY,
                        title=title, title_fontsize=plt.rcParams["font.size"] - 0.5)
    legend.get_title().set_color(INK_SECONDARY)


def show(ax, image, title=None, cmap="gray"):
    ax.imshow(image, cmap=cmap, interpolation="nearest")
    ax.set_xticks([]), ax.set_yticks([]), ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    if title:
        ax.set_title(title, fontsize=plt.rcParams["font.size"])


def stretch(response):
    return np.clip(response / np.percentile(response, 99.5), 0, 1)


def pipeline_steps(style):
    image_id, rows, cols = STEPS_CROP
    image, gt = load_image(image_id), load_ground_truth(image_id)
    mask = fov_mask(image.shape, PARAMS.fov_margin)
    median = median_filter(image, PARAMS.median_size)
    tophat = black_tophat(median, 4)
    lines = sup_of_line_openings(tophat, PARAMS.line_length)
    response = vessel_response(image, mask, PARAMS)
    seg = binarize(response, mask, PARAMS)
    panels = [image, median, stretch(tophat), stretch(lines), stretch(response), seg, gt,
              error_overlay(image, seg, gt)]
    size, top, bottom = ((6.3, 3.75), 0.95, 0.1) if style.report else ((8.4, 4.9), 0.87, 0.08)
    fig, axes = plt.subplots(2, 4, figsize=size)
    fig.subplots_adjust(left=0.01, right=0.99, top=top, bottom=bottom, wspace=0.06, hspace=0.2)
    for ax, title, img in zip(axes.flat, style.t["steps"], panels):
        show(ax, img[rows, cols], title)
    style.title(fig, "steps", fmt={"id": image_id}, x=0.01, y=0.98, ha="left", fontsize=11, color=INK)
    overlay_legend(style, fig, y=0.075 if style.report else 0.07)
    style.save(fig, "pipeline_steps")


def method_comparison(style):
    image_id = COMPARISON_IMAGE
    image, gt = load_image(image_id), load_ground_truth(image_id)
    mask = fov_mask(image.shape, PARAMS.fov_margin)
    segs = {THRESHOLD: threshold_segmentation(image, mask, THRESHOLD_T),
            FRANGI: rank_threshold(frangi_response(image, mask), mask, FRANGI_KEPT),
            PIPELINE: binarize(vessel_response(image, mask, PARAMS), mask, PARAMS)}

    def overlay_title(name, seg):
        return f"{style.t['methods'][name]}\nF1 = {style.num(evaluate(seg, gt)['f1'])}"

    if style.report:
        fig, axes = plt.subplots(2, 3, figsize=(6.3, 4.55))
        fig.subplots_adjust(left=0.005, right=0.995, top=0.94, bottom=0.005, wspace=0.05, hspace=0.16)
        show(axes[0, 0], image, style.t["input"].format(id=image_id))
        show(axes[0, 1], gt, style.t["expert"])
        axes[0, 2].axis("off")
        for ax, (name, seg) in zip(axes[1], segs.items()):
            show(ax, error_overlay(image, seg, gt, thicken=True), overlay_title(name, seg))
        box = axes[0, 2].get_position()
        overlay_legend(style, fig, y=box.y0 + box.height * 0.62, wide=True, loc="center",
                       x=box.x0 + box.width / 2, ncol=1)
    else:
        fig, axes = plt.subplots(1, 5, figsize=(11, 3.1))
        fig.subplots_adjust(left=0.005, right=0.995, top=0.84, bottom=0.2, wspace=0.05)
        show(axes[0], image, style.t["input"].format(id=image_id))
        show(axes[1], gt, style.t["expert"])
        for ax, (name, seg) in zip(axes[2:], segs.items()):
            show(ax, error_overlay(image, seg, gt, thicken=True), overlay_title(name, seg))
        overlay_legend(style, fig, y=0.17, wide=True)
    style.save(fig, "method_comparison")


def failure_crops(style):
    size, top, bottom = ((6.3, 4.75), 0.92, 0.115) if style.report else ((7.6, 5.6), 0.9, 0.12)
    fig, axes = plt.subplots(2, len(FAILURE_CROPS), figsize=size)
    fig.subplots_adjust(left=0.01, right=0.99, top=top, bottom=bottom, wspace=0.05, hspace=0.04)
    for col, (title, (image_id, rows, cols)) in enumerate(zip(style.t["crops"], FAILURE_CROPS)):
        image, gt = load_image(image_id), load_ground_truth(image_id)
        mask = fov_mask(image.shape, PARAMS.fov_margin)
        seg = binarize(vessel_response(image, mask, PARAMS), mask, PARAMS)
        show(axes[0, col], image[rows, cols], f"{title}\n({style.t['image'].format(id=image_id)})")
        show(axes[1, col], error_overlay(image, seg, gt)[rows, cols])
    overlay_legend(style, fig, y=0.095 if style.report else 0.1)
    style.save(fig, "failure_crops")


def precision_recall(style):
    curves = pd.read_csv(ANALYSIS / "pr_curves.csv")
    fig, ax = plt.subplots(figsize=(3.9, 2.85) if style.report else (5.2, 4.4))
    for name in METHODS:
        c = curves[curves.method == name].sort_values("recall")
        color = METHOD_COLORS[name]
        ax.plot(c.recall, c.precision, color=color, label=style.t["methods"][name],
                solid_capstyle="round")
        op = c[c["operating point"]].iloc[0]
        ax.plot(op.recall, op.precision, "o", color=color, markersize=7,
                markeredgecolor=SURFACE, markeredgewidth=2, zorder=3)
        ax.annotate(f"F1 {style.num(op.f1)}", (op.recall, op.precision), xytext=(7, 5),
                    textcoords="offset points", color=INK_SECONDARY, fontsize=8)
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel=style.t["recall"], ylabel=style.t["precision"])
    style.ticks(ax)
    if style.t["titles"]:
        ax.set_title(style.t["titles"]["pr"], pad=20)
        ax.text(0, 1.025, style.t["titles"]["pr_sub"], transform=ax.transAxes,
                color=INK_SECONDARY, fontsize=8, va="bottom")
    ax.legend(loc="center left", bbox_to_anchor=(0, 0.6), labelcolor=INK_SECONDARY)
    style.save(fig, "precision_recall")


def per_image(style):
    d = pd.read_csv(ROOT / "results" / "pipeline" / "per_image.csv", dtype={"image": str})
    d = d[d.step.isin(STEP_NAMES)].assign(method=lambda x: x.step.map(STEP_NAMES))
    order = list(TUNE_IDS) + list(TEST_IDS)
    x = {i: k + (1 if i in TEST_IDS else 0) for k, i in enumerate(order)}
    fig, ax = plt.subplots(figsize=(6.3, 2.9) if style.report else (7.2, 3.6))
    wide = d.pivot_table(index="image", columns="method", values="f1")
    for i in order:
        ax.plot([x[i]] * 2, [wide.loc[i, FRANGI], wide.loc[i, PIPELINE]], color=AXIS,
                linewidth=1, zorder=1)
    for name in METHODS:
        ax.plot([x[i] for i in order], [wide.loc[i, name] for i in order], "o", linestyle="none",
                color=METHOD_COLORS[name], markersize=6.5, markeredgecolor=SURFACE,
                markeredgewidth=1.5, label=style.t["methods"][name], zorder=2)
        last = order[-1]
        ax.annotate(style.t["methods"][name], (x[last], wide.loc[last, name]), xytext=(9, 0),
                    textcoords="offset points", va="center", color=INK_SECONDARY, fontsize=8)
    ax.set_xticks([x[i] for i in order], order)
    ax.set_xlim(-0.6, len(order) + 0.6)
    ax.set_ylim(0.25, 0.9)
    ax.set_ylabel(style.t["f1"])
    style.ticks(ax, "y")
    ax.grid(axis="x", visible=False)
    for ids, key in ((TUNE_IDS, "tune"), (TEST_IDS, "test")):
        ax.text(np.mean([x[i] for i in ids]), 0.9, style.t[key], ha="center", va="bottom",
                color=INK_SECONDARY, fontsize=8)
    ax.axvline(len(TUNE_IDS), color=GRID, linewidth=1)
    style.title(ax, "per_image", pad=16)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.34), ncol=3, labelcolor=INK_SECONDARY)
    style.save(fig, "per_image_f1")


def recall_by_width(style):
    w = pd.read_csv(ANALYSIS / "recall_by_width.csv")
    labels = list(dict.fromkeys(w.width))
    share = w[w.method == PIPELINE].set_index("width")["share of centreline"]
    fig, ax = plt.subplots(figsize=(6.3, 2.9) if style.report else (6.4, 3.6))
    bar = 0.26
    for k, name in enumerate(METHODS):
        r = w[w.method == name].set_index("width").loc[labels, "recall"]
        pos = np.arange(len(labels)) + (k - 1) * bar
        ax.bar(pos, r, width=bar, color=METHOD_COLORS[name], label=style.t["methods"][name],
               edgecolor=SURFACE, linewidth=1.5)
        if name == PIPELINE:
            for p, v in zip(pos, r):
                ax.text(p, v + 0.02, style.num(v), ha="center", color=INK_SECONDARY, fontsize=7.5)
    ax.set_xticks(np.arange(len(labels)),
                  [f"{l}\n({style.percent(share[l])})" for l in labels])
    ax.set_ylabel(style.t["width_recall"])
    ax.set_xlabel(style.t["width"], labelpad=6)
    ax.grid(axis="x", visible=False)
    style.title(ax, "width", pad=10)
    ax.legend(loc="upper left", labelcolor=INK_SECONDARY, ncol=3, bbox_to_anchor=(0, 1.0))
    ax.set_ylim(0, 1.12)
    ax.set_yticks(np.linspace(0, 1, 6))
    style.ticks(ax, "y")
    style.save(fig, "recall_by_width")


def spread_labels(ys, gap):
    """Nudge sorted label positions apart by at least ``gap``; returns them in input order."""
    order = np.argsort(ys)
    placed = np.array(ys, dtype=float)[order]
    for k in range(1, len(placed)):
        placed[k] = max(placed[k], placed[k - 1] + gap)
    out = np.empty_like(placed)
    out[order] = placed
    return out


def cv_per_image(style):
    scores = pd.read_csv(CV / "comparison_per_image.csv", dtype={"image": str})
    seeds = pd.read_csv(CV / "unet_seeds.csv", dtype={"image": str}).set_index("image")
    wide = scores.pivot(index="image", columns="method", values="f1").loc[list(ALL_IDS)]
    x = np.arange(len(wide))
    fig, ax = plt.subplots(figsize=(6.3, 2.9) if style.report else (7.2, 3.6))
    for k, i in enumerate(wide.index):
        ax.plot([k, k], [wide.loc[i, "pipeline"], wide.loc[i, "unet"]], color=AXIS, linewidth=1,
                zorder=1)
        ax.plot([k + 0.18] * 2, [seeds.loc[i, "min"], seeds.loc[i, "max"]],
                color=METHOD_COLORS[UNET], linewidth=1.2, alpha=0.6, zorder=1,
                solid_capstyle="butt")
    for key, name in CV_METHODS.items():
        offset = 0.18 if key == "unet" else 0
        ax.plot(x + offset, wide[key], "o", linestyle="none", color=METHOD_COLORS[name],
                markersize=6.5, markeredgecolor=SURFACE, markeredgewidth=1.5,
                label=style.t["methods"][name], zorder=2)
    last = wide.iloc[-1]
    ys = spread_labels([last[k] for k in CV_METHODS], gap=0.045)
    for (key, name), y in zip(CV_METHODS.items(), ys):
        ax.annotate(style.t["methods"][name], (x[-1] + 0.18, y), xytext=(12, 0),
                    textcoords="offset points", va="center", color=INK_SECONDARY, fontsize=8)
    ax.set_xticks(x, wide.index)
    ax.set_xlim(-0.6, len(x) + 1.6)
    ax.set_ylim(0.25, 0.95)
    ax.set_ylabel(style.t["f1"])
    style.ticks(ax, "y")
    ax.grid(axis="x", visible=False)
    style.title(ax, "cv_per_image", pad=10)
    handles, labels = ax.get_legend_handles_labels()
    handles.append(plt.Line2D([], [], color=METHOD_COLORS[UNET], linewidth=1.2, alpha=0.6))
    labels.append(style.t["seed_range"])
    ax.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.36), ncol=5,
              labelcolor=INK_SECONDARY, columnspacing=1.2, handletextpad=0.4)
    style.save(fig, "cv_per_image")


def cv_recall_by_width(style):
    w = pd.read_csv(CV / "recall_by_width.csv")
    labels = list(dict.fromkeys(w.width))
    fig, ax = plt.subplots(figsize=(6.3, 2.9) if style.report else (6.4, 3.6))
    bar = 0.2
    for k, (key, name) in enumerate(CV_METHODS.items()):
        r = w[w.method == key].set_index("width").loc[labels, "recall"]
        pos = np.arange(len(labels)) + (k - 1.5) * bar
        ax.bar(pos, r, width=bar, color=METHOD_COLORS[name], label=style.t["methods"][name],
               edgecolor=SURFACE, linewidth=1.5)
        if key in ("pipeline", "unet"):
            for p, v in zip(pos, r):
                ax.text(p, v + 0.02, style.num(v), ha="center", color=INK_SECONDARY, fontsize=6.5)
    ax.set_xticks(np.arange(len(labels)), labels)
    ax.set_ylabel(style.t["width_recall"])
    ax.set_xlabel(style.t["width"].split(" (")[0], labelpad=6)
    ax.grid(axis="x", visible=False)
    style.title(ax, "cv_width", pad=10)
    ax.legend(loc="upper left", labelcolor=INK_SECONDARY, ncol=4, bbox_to_anchor=(0, 1.0))
    ax.set_ylim(0, 1.14)
    ax.set_yticks(np.linspace(0, 1, 6))
    style.ticks(ax, "y")
    style.save(fig, "cv_recall_by_width")


def cv_crops(style):
    size = (5.6, 6.0) if style.report else (6.8, 7.2)
    fig, axes = plt.subplots(len(FAILURE_CROPS), 3, figsize=size)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.93, bottom=0.08, wspace=0.05, hspace=0.2)
    for row, (title, (image_id, rows, cols)) in enumerate(zip(style.t["crops"], FAILURE_CROPS)):
        image, gt = load_image(image_id), load_ground_truth(image_id)
        mask = fov_mask(image.shape, PARAMS.fov_margin)
        pipeline = np.load(CV / "segmentations" / f"{image_id}.npz")["pipeline"]
        prob = np.load(CV / "probabilities" / f"{image_id}_seed0.npz")["probability"]
        unet = mask & (prob.astype(np.float32) >= UNET_THRESHOLD)
        panels = [image[rows, cols], error_overlay(image, pipeline, gt)[rows, cols],
                  error_overlay(image, unet, gt)[rows, cols]]
        for col, panel in enumerate(panels):
            label = (f"{title} ({style.t['image'].format(id=image_id)})" if col == 0
                     else (style.t["cv_columns"][col] if row == 0 else None))
            show(axes[row, col], panel, label)
    overlay_legend(style, fig, y=0.065)
    style.save(fig, "cv_crops")


def main():
    global SURFACE
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--lang", choices=sorted(TEXT), default="en")
    style = Style(parser.parse_args().lang)
    if style.report:
        SURFACE = "#ffffff"  # printed page: no tinted chart surface
    configure_matplotlib(style)
    style.out.mkdir(parents=True, exist_ok=True)
    makers = [pipeline_steps, method_comparison, failure_crops, precision_recall, per_image,
              recall_by_width]
    if (CV / "comparison_per_image.csv").exists():
        makers += [cv_per_image, cv_recall_by_width, cv_crops]
    else:
        print("no cross-validation results: run scripts/cv_*.py for the cv_* figures")
    for make in makers:
        make(style)
        print(f"{make.__name__} done")


if __name__ == "__main__":
    main()
