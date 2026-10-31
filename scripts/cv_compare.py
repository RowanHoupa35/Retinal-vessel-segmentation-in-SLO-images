"""Compare the pipeline, the references and the U-Net over the cross-validation.

Reads the outputs of scripts/cv_pipeline.py and scripts/cv_unet.py. Every
image is scored by methods tuned or trained without it. U-Net scores are
averaged over seeds for each image.

- scores of every method, with the skeleton coverage (a recall that does not
  depend on the predicted skeleton length) and its F1;
- paired comparison of the pipeline and the U-Net image by image;
- recall by annotated vessel width;
- annotation style: among the structures the pipeline detects, does the U-Net
  detect the unannotated ones as often as the annotated thin vessels of the
  same local contrast? A network learns what the annotator marks, so it may
  learn to ignore faint vessels the annotator left out.

Writes results/cross_validation/{summary.md, comparison_per_image.csv,
recall_by_width.csv, style_by_contrast.csv, unet_seeds.csv}.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from scipy.stats import binomtest, wilcoxon
from skimage.morphology import skeletonize

from vessels.data import ALL_IDS, fov_mask, load_ground_truth, load_image
from vessels.morphology import black_tophat
from vessels.reporting import mean_std, to_markdown
from vessels.unet import THRESHOLD

OUT = Path(__file__).resolve().parents[1] / "results" / "cross_validation"
METHODS = ["threshold", "frangi", "pipeline", "unet"]
LABELS = {"threshold": "Global threshold", "frangi": "Frangi vesselness",
          "pipeline": "Morphological pipeline", "unet": "U-Net"}
METRICS = ["precision", "recall", "f1", "dice", "covered recall", "covered f1"]
SEEDS = (0, 1, 2)
FOV_MARGIN = 3
NEAR = 2
WIDTH_EDGES = [0, 1.2, 1.7, 2.1, 2.95, np.inf]  # same classes as scripts/analyze_errors.py
WIDTH_LABELS = ["1 px", "2 px", "3 px", "4 px", "≥ 5 px"]
THIN = 1.7
CONTRAST_EDGES = [0, 5, 10, 15, 20, 30, np.inf]


def with_covered_f1(df):
    p, c = df["precision"], df["covered recall"]
    return df.assign(**{"covered f1": np.where(p + c > 0, 2 * p * c / (p + c), 0.0)})


def load_scores():
    pipe = pd.read_csv(OUT / "pipeline_per_image.csv", dtype={"image": str})
    seeds = with_covered_f1(pd.read_csv(OUT / "unet_per_image.csv", dtype={"image": str}))
    unet = seeds.groupby("image", as_index=False)[METRICS].mean().assign(method="unet")
    return pd.concat([with_covered_f1(pipe), unet], ignore_index=True), seeds


def unet_segmentations(image_id, mask):
    return [mask & (np.load(OUT / "probabilities" / f"{image_id}_seed{s}.npz")["probability"]
                    .astype(np.float32) >= THRESHOLD) for s in SEEDS]


def main():
    scores, seeds = load_scores()
    scores.to_csv(OUT / "comparison_per_image.csv", index=False, float_format="%.4f")
    mask = fov_mask((512, 512), FOV_MARGIN)

    # Seed variability of the U-Net.
    spread = seeds.groupby("image").f1.agg(["min", "max", "std"]).reset_index()
    spread.to_csv(OUT / "unet_seeds.csv", index=False, float_format="%.4f")

    # Paired comparison, image by image.
    wide = scores.pivot(index="image", columns="method", values="f1")
    diff = wide["pipeline"] - wide["unet"]
    wide_cov = scores.pivot(index="image", columns="method", values="covered f1")
    diff_cov = wide_cov["pipeline"] - wide_cov["unet"]
    paired = {
        "images": len(diff), "pipeline better": int((diff > 0).sum()),
        "mean difference": diff.mean(), "min difference": diff.min(), "max difference": diff.max(),
        "wilcoxon p": wilcoxon(diff).pvalue,
        "sign test p": binomtest(int((diff > 0).sum()), len(diff)).pvalue,
        "covered: pipeline better": int((diff_cov > 0).sum()),
        "covered: mean difference": diff_cov.mean(), "covered: wilcoxon p": wilcoxon(diff_cov).pvalue,
    }
    pd.DataFrame([paired]).to_csv(OUT / "paired.csv", index=False, float_format="%.4f")

    # Recall by width and annotation style, pooled over the held-out images.
    width_counts = {m: np.zeros((2, len(WIDTH_LABELS))) for m in METHODS}
    style_rows, fp_rows = [], []
    for i in ALL_IDS:
        image, gt = load_image(i), load_ground_truth(i)
        gt_skel = skeletonize(gt)
        half_width = ndi.distance_transform_edt(gt)
        width_class = np.digitize(half_width[gt_skel], WIDTH_EDGES) - 1
        stored = np.load(OUT / "segmentations" / f"{i}.npz")
        segs = {m: [stored[m]] for m in ("threshold", "frangi", "pipeline")}
        segs["unet"] = unet_segmentations(i, mask)
        dist_to_gt = ndi.distance_transform_edt(~gt)
        for m, seg_list in segs.items():
            for seg in seg_list:
                covered = seg[gt_skel]
                for k in range(len(WIDTH_LABELS)):
                    width_counts[m][0, k] += (width_class == k).sum() / len(seg_list)
                    width_counts[m][1, k] += covered[width_class == k].sum() / len(seg_list)
                seg_skel = skeletonize(seg)
                false_pos = seg_skel & ~gt
                fp_rows.append({"image": i, "method": m, "false positives": false_pos.sum(),
                                "far": (false_pos & (dist_to_gt > NEAR)).sum(),
                                "weight": 1 / len(seg_list)})

        # Structures the pipeline detects: annotated thin vessels, and unannotated ones
        # (pipeline centreline more than NEAR px from any annotated vessel).
        pipeline = segs["pipeline"][0]
        contrast = black_tophat(image, 4)
        annotated = gt_skel & (half_width < THIN) & pipeline
        unannotated = skeletonize(pipeline) & (dist_to_gt > NEAR)
        for kind, pixels in (("annotated thin vessel", annotated), ("not annotated", unannotated)):
            bins = np.digitize(contrast[pixels], CONTRAST_EDGES) - 1
            found = np.mean([seg[pixels] for seg in segs["unet"]], axis=0)
            for b in range(len(CONTRAST_EDGES) - 1):
                sel = bins == b
                style_rows.append({"image": i, "kind": kind, "contrast bin": b,
                                   "pixels": sel.sum(), "found by unet": found[sel].sum()})

    widths = pd.DataFrame([{"method": m, "width": label,
                            "centreline": width_counts[m][0, k], "covered": width_counts[m][1, k],
                            "recall": width_counts[m][1, k] / width_counts[m][0, k]}
                           for m in METHODS for k, label in enumerate(WIDTH_LABELS)])
    widths.to_csv(OUT / "recall_by_width.csv", index=False, float_format="%.4f")

    style = pd.DataFrame(style_rows).groupby(["kind", "contrast bin"], as_index=False)[
        ["pixels", "found by unet"]].sum()
    style["rate"] = style["found by unet"] / style["pixels"].where(style["pixels"] > 0)
    style["contrast"] = [f"[{CONTRAST_EDGES[b]:g}, {CONTRAST_EDGES[b + 1]:g})"
                         for b in style["contrast bin"]]
    style.to_csv(OUT / "style_by_contrast.csv", index=False, float_format="%.4f")
    overall = style.groupby("kind")[["pixels", "found by unet"]].sum()
    overall["rate"] = overall["found by unet"] / overall["pixels"]

    fp = pd.DataFrame(fp_rows)
    fp = fp.assign(fp=fp["false positives"] * fp.weight, far_w=fp.far * fp.weight) \
        .groupby("method")[["fp", "far_w"]].sum()
    fp["far share"] = fp.far_w / fp.fp

    # Summary.
    table = scores.groupby("method")[METRICS].agg(list).loc[METHODS]
    rows = [[LABELS[m]] + [mean_std(table.loc[m, c]) for c in METRICS] for m in METHODS]
    summary = pd.DataFrame(rows, columns=["Method", "Precision", "Recall", "F1", "Dice",
                                          "Covered recall", "Covered F1"])
    per_image = wide[METHODS].assign(**{"U-Net seed range": [
        f"{a:.3f}–{b:.3f}" for a, b in spread.set_index("image").loc[wide.index, ["min", "max"]].to_numpy()]})
    per_image = per_image.rename(columns=LABELS).map(lambda v: f"{v:.3f}" if isinstance(v, float) else v)
    width_md = widths.pivot(index="width", columns="method", values="recall").loc[WIDTH_LABELS, METHODS]
    style_md = style.pivot(index="contrast", columns="kind", values="rate")
    style_md = style_md.loc[[f"[{CONTRAST_EDGES[b]:g}, {CONTRAST_EDGES[b + 1]:g})"
                             for b in range(len(CONTRAST_EDGES) - 1)]]
    counts = style.pivot(index="contrast", columns="kind", values="pixels").loc[style_md.index]
    style_md = pd.DataFrame({f"{k} (rate, pixels)": [f"{r:.2f} ({int(n)})" if n else "–"
                                                     for r, n in zip(style_md[k], counts[k])]
                             for k in style_md.columns}, index=style_md.index)
    (OUT / "summary.md").write_text(
        "# Leave-one-out cross-validation\n\n"
        "Each image is scored by methods tuned (pipeline, references) or trained\n"
        "(U-Net, 3 seeds averaged) on the 9 other images. Mean ± standard deviation\n"
        "over the 10 images. Covered recall: share of the annotated skeleton inside\n"
        "the segmentation, which does not depend on the predicted skeleton length.\n\n"
        f"{to_markdown(summary)}\n\n"
        "## Pipeline against U-Net, image by image (F1)\n\n"
        f"{to_markdown(per_image.reset_index())}\n\n"
        f"- Pipeline better on {paired['pipeline better']} of {paired['images']} images; mean "
        f"difference {paired['mean difference']:+.3f} (from {paired['min difference']:+.3f} to "
        f"{paired['max difference']:+.3f}); Wilcoxon signed-rank p = {paired['wilcoxon p']:.3f}, "
        f"sign test p = {paired['sign test p']:.3f}.\n"
        f"- With the covered recall: pipeline better on {paired['covered: pipeline better']} images, "
        f"mean difference {paired['covered: mean difference']:+.3f}, Wilcoxon p = "
        f"{paired['covered: wilcoxon p']:.3f}.\n"
        f"- U-Net seed spread: mean standard deviation of F1 across seeds "
        f"{spread['std'].mean():.3f} per image.\n\n"
        "## Recall by annotated vessel width\n\n"
        f"{to_markdown(width_md.rename(columns=LABELS).map(lambda v: f'{v:.2f}').reset_index())}\n\n"
        "## Annotation style\n\n"
        "Share of the structures detected by the pipeline that the U-Net also segments,\n"
        "by local contrast (black top-hat of the raw image, radius 4, gray levels).\n"
        "Annotated thin vessels are 1 to 2 px wide; \"not annotated\" are pipeline\n"
        f"centreline pixels more than {NEAR} px from any annotated vessel.\n\n"
        f"{to_markdown(style_md.reset_index())}\n\n"
        f"Overall: {overall.loc['annotated thin vessel', 'rate']:.2f} of the annotated thin "
        f"vessels, {overall.loc['not annotated', 'rate']:.2f} of the unannotated structures.\n\n"
        "False positives more than 2 px from any annotated vessel: " +
        ", ".join(f"{LABELS[m]} {fp.loc[m, 'far share']:.0%}" for m in METHODS) + ".\n"
    )
    print((OUT / "summary.md").read_text())


if __name__ == "__main__":
    main()
