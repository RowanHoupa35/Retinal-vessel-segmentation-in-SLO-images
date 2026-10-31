"""Error analysis on the test images. No parameter is chosen here.

- recall by annotated vessel width;
- position of the false positives relative to the annotated vessels;
- local contrast of detected, missed and unannotated structures;
- precision-recall curves through the operating points chosen on tune;
- the course recall against the share of the annotated skeleton covered;
- sensitivity of the metric to deformations of the annotation (all 10
  annotations, no method involved);
- run time of the pipeline.

Also describes the annotations of the 10 images (no method involved).

Writes results/analysis/{recall_by_width.csv, contrast.csv, pr_curves.csv,
false_positives.csv, recall_definitions.csv, metric_sensitivity.csv,
runtime.csv, dataset.csv, frangi_wide.csv, errors.md}.

frangi_wide.csv explains a weakness of the Frangi reference on the tune images:
its coverage of wide vessels with the scales used and with scales up to 6 px.
"""

import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from scipy.ndimage import median_filter
from skimage.morphology import dilation, disk, erosion, reconstruction, skeletonize

from skimage.filters import frangi

from vessels.baselines import FRANGI_SIGMAS, frangi_response, threshold_segmentation
from vessels.data import (ALL_IDS, TEST_IDS, fov_mask, group, load_ground_truth, load_image,
                          TUNE_IDS)
from vessels.evaluation import evaluate
from vessels.morphology import EIGHT_CONNECTED, black_tophat, sup_of_line_openings
from vessels.pipeline import Params, binarize, segment, vessel_response
from vessels.reporting import to_markdown
from vessels.thresholding import hysteresis, rank_threshold

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results" / "analysis"
PARAMS = Params()
# Chosen on tune by scripts/ablation.py (see results/pipeline/ablation.md).
THRESHOLD_T = 80
FRANGI_KEPT = 0.07
PIPELINE, FRANGI, THRESHOLD = "Morphological pipeline", "Frangi vesselness", "Global threshold"

NEAR = 2  # px: a false positive this close to an annotated vessel is an alignment error
# Distance to the vessel border at the centreline takes discrete values on the
# pixel grid (1, 1.41, 2, 2.24, 2.83, 3...); width = 2 * distance - 1.
WIDTH_EDGES = [0, 1.2, 1.7, 2.1, 2.95, np.inf]
WIDTH_LABELS = ["1 px", "2 px", "3 px", "4 px", "≥ 5 px"]
THIN = 1.7  # widths of 1 and 2 px


def sweep(responses, gts, segment, values):
    rows = []
    for v in values:
        scores = [evaluate(segment(responses[i], v), gts[i]) for i in TEST_IDS]
        rows.append({"param": v, **{m: np.mean([s[m] for s in scores])
                                    for m in ("precision", "recall", "f1")}})
    return rows


def recall_definitions(segs, gts):
    """Course recall TP / (TP + FN), with TP counted on the predicted skeleton and
    FN on the annotated one, against the share of the annotated skeleton covered
    (the "topology sensitivity" of clDice, Shit et al. 2021)."""
    rows = []
    for method in (PIPELINE, FRANGI, THRESHOLD):
        for i in TEST_IDS:
            seg, gt = segs[i][method], gts[i]
            seg_skel, gt_skel = skeletonize(seg), skeletonize(gt)
            scores = evaluate(seg, gt, gt_skel)
            covered = (gt_skel & seg).sum() / gt_skel.sum()
            rows.append({"method": method, "image": i, "precision": scores["precision"],
                         "course recall": scores["recall"], "covered recall": covered,
                         "course f1": scores["f1"],
                         "covered f1": 2 * scores["precision"] * covered / (scores["precision"] + covered),
                         "length ratio": (seg_skel & gt).sum() / (gt_skel & seg).sum()})
    return pd.DataFrame(rows)


def metric_sensitivity():
    """Scores of deformed annotations evaluated against the annotation itself."""
    shift = lambda g, k: np.pad(g, ((0, 0), (k, 0)))[:, :g.shape[1]]
    deformations = {
        "annotation itself": lambda g: g,
        "dilated by 1 px": lambda g: dilation(g, disk(1)),
        "dilated by 2 px": lambda g: dilation(g, disk(2)),
        "dilated by 4 px": lambda g: dilation(g, disk(4)),
        "eroded by 1 px": lambda g: erosion(g, disk(1)),
        "shifted by 1 px": lambda g: shift(g, 1),
        "shifted by 2 px": lambda g: shift(g, 2),
    }
    rows = []
    for name, deform in deformations.items():
        for i in ALL_IDS:
            gt = load_ground_truth(i)
            rows.append({"deformation": name, "image": i, **evaluate(deform(gt), gt)})
    return pd.DataFrame(rows).groupby("deformation", sort=False)[
        ["precision", "recall", "f1", "dice"]].mean().reset_index()


def dataset():
    """Share of vessel pixels and centreline widths of each annotation."""
    rows, all_widths, all_fractions = [], [], []
    for i in ALL_IDS:
        gt = load_ground_truth(i)
        widths = 2 * ndi.distance_transform_edt(gt)[skeletonize(gt)] - 1
        all_widths.append(widths)
        all_fractions.append(gt.mean())
        rows.append({"image": i, "group": group(i), "split": "tune" if i in TUNE_IDS else "test",
                     "vessel fraction": gt.mean(), "width p5": np.percentile(widths, 5),
                     "width median": np.median(widths), "width p95": np.percentile(widths, 95),
                     "centreline pixels": len(widths)})
    widths = np.concatenate(all_widths)
    rows.append({"image": "all", "group": "", "split": "", "vessel fraction": np.mean(all_fractions),
                 "width p5": np.percentile(widths, 5), "width median": np.median(widths),
                 "width p95": np.percentile(widths, 95), "centreline pixels": len(widths)})
    return pd.DataFrame(rows)


def frangi_wide_vessels():
    """Share of the centreline of vessels >= 5 px covered by Frangi, tune images."""
    rows = []
    for sigmas in (FRANGI_SIGMAS, (1, 2, 3, 4, 5, 6)):
        covered = total = 0
        for i in TUNE_IDS:
            image, gt = load_image(i), load_ground_truth(i)
            mask = fov_mask(image.shape, PARAMS.fov_margin)
            response = frangi(image.astype(float) / 255, sigmas=sigmas, black_ridges=True)
            response[~mask] = 0
            seg = rank_threshold(response, mask, FRANGI_KEPT)
            wide = skeletonize(gt) & (ndi.distance_transform_edt(gt) >= WIDTH_EDGES[4])
            covered += (wide & seg).sum()
            total += wide.sum()
        rows.append({"sigma max": max(sigmas), "wide vessel coverage": covered / total})
    return pd.DataFrame(rows)


def mean_time(function, repeats):
    function()
    start = time.perf_counter()
    for _ in range(repeats):
        function()
    return (time.perf_counter() - start) / repeats


def runtime():
    """Wall time of segment() on the test images, share of the line openings, and
    hysteresis by labelling against scikit-image's reconstruction."""
    images = [load_image(i) for i in TEST_IDS]
    seconds = np.mean([mean_time(lambda: segment(image), 1) for image in images])
    image = images[0]
    mask = fov_mask(image.shape, PARAMS.fov_margin)
    denoised = median_filter(image, PARAMS.median_size)
    tophats = [black_tophat(denoised, r) for r in PARAMS.tophat_radii]
    lines = sum(mean_time(lambda t=t: sup_of_line_openings(t, PARAMS.line_length), 3)
                for t in tophats)
    response = vessel_response(image, mask, PARAMS)
    candidates = rank_threshold(response, mask, PARAMS.kept_fraction)
    seeds = rank_threshold(response, mask, PARAMS.seed_fraction)
    labelling = mean_time(lambda: hysteresis(response, mask, PARAMS.kept_fraction,
                                             PARAMS.seed_fraction), 20)
    generic = mean_time(lambda: reconstruction(seeds.astype(np.uint8), candidates.astype(np.uint8),
                                               footprint=EIGHT_CONNECTED), 20)
    try:
        cpu = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True,
                             text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        cpu = platform.processor() or platform.machine()
    return pd.DataFrame([{"seconds per image": seconds, "image size": "512 x 512",
                          "line openings share": lines / seconds,
                          "hysteresis labelling ms": labelling * 1000,
                          "hysteresis reconstruction ms": generic * 1000,
                          "hysteresis speed-up": generic / labelling,
                          "cpu": cpu, "python": platform.python_version()}])


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    images = {i: load_image(i) for i in TEST_IDS}
    gts = {i: load_ground_truth(i) for i in TEST_IDS}
    mask = fov_mask(images[TEST_IDS[0]].shape, PARAMS.fov_margin)
    pipe_resp = {i: vessel_response(images[i], mask, PARAMS) for i in TEST_IDS}
    frangi_resp = {i: frangi_response(images[i], mask) for i in TEST_IDS}
    segs = {i: {PIPELINE: binarize(pipe_resp[i], mask, PARAMS),
                FRANGI: rank_threshold(frangi_resp[i], mask, FRANGI_KEPT),
                THRESHOLD: threshold_segmentation(images[i], mask, THRESHOLD_T)}
            for i in TEST_IDS}
    f1 = np.mean([evaluate(segs[i][PIPELINE], gts[i])["f1"] for i in TEST_IDS])
    print(f"Pipeline test F1 = {f1:.3f} (results/pipeline/ablation.md: 0.791)")

    width_rows, contrast = [], {k: [] for k in (
        "annotated thin vessel, detected", "annotated thin vessel, missed",
        "detected, not annotated", "background")}
    near = far = tp = 0
    for i in TEST_IDS:
        gt, gt_skel = gts[i], skeletonize(gts[i])
        half_width = ndi.distance_transform_edt(gt)[gt_skel]
        width_class = np.digitize(half_width, WIDTH_EDGES) - 1
        for method, seg in segs[i].items():
            covered = seg[gt_skel]
            for k, label in enumerate(WIDTH_LABELS):
                in_class = width_class == k
                width_rows.append({"method": method, "width": label, "image": i,
                                   "centreline": in_class.sum(), "covered": covered[in_class].sum()})

        seg = segs[i][PIPELINE]
        seg_skel = skeletonize(seg)
        dist_to_gt = ndi.distance_transform_edt(~gt)
        false_pos = seg_skel & ~gt
        near += (false_pos & (dist_to_gt <= NEAR)).sum()
        far += (false_pos & (dist_to_gt > NEAR)).sum()
        tp += (seg_skel & gt).sum()

        local = black_tophat(images[i], 4)
        thin = gt_skel & (ndi.distance_transform_edt(gt) < THIN)
        contrast["annotated thin vessel, detected"] += list(local[thin & seg])
        contrast["annotated thin vessel, missed"] += list(local[thin & ~seg])
        contrast["detected, not annotated"] += list(local[false_pos & (dist_to_gt > NEAR)])
        contrast["background"] += list(local[mask & ~seg & (dist_to_gt > 5)])

    widths = pd.DataFrame(width_rows).groupby(["method", "width"], sort=False)[["centreline", "covered"]].sum()
    widths["recall"] = widths.covered / widths.centreline
    widths["share of centreline"] = widths.centreline / widths.centreline.groupby("method").transform("sum")
    widths = widths.reset_index()
    widths.to_csv(RESULTS_DIR / "recall_by_width.csv", index=False, float_format="%.4f")

    contrast_table = pd.DataFrame([{"structure": k, "pixels": len(v), "median": np.median(v),
                                    "q1": np.percentile(v, 25), "q3": np.percentile(v, 75)}
                                   for k, v in contrast.items()])
    contrast_table.to_csv(RESULTS_DIR / "contrast.csv", index=False, float_format="%.1f")

    ratio = PARAMS.seed_fraction / PARAMS.kept_fraction
    curves = []
    for method, responses, segment, values, chosen in (
        (PIPELINE, pipe_resp,
         lambda r, k: binarize(r, mask, Params(kept_fraction=k, seed_fraction=ratio * k)),
         [round(k, 2) for k in np.arange(0.03, 0.305, 0.01)], PARAMS.kept_fraction),
        (FRANGI, frangi_resp, lambda r, k: rank_threshold(r, mask, k),
         [round(k, 2) for k in np.arange(0.02, 0.305, 0.01)], FRANGI_KEPT),
        (THRESHOLD, images, lambda img, t: threshold_segmentation(img, mask, t),
         list(range(30, 165, 5)), THRESHOLD_T),
    ):
        for row in sweep(responses, gts, segment, values):
            curves.append({"method": method, **row, "operating point": row["param"] == chosen})
    pd.DataFrame(curves).to_csv(RESULTS_DIR / "pr_curves.csv", index=False, float_format="%.4f")

    width_md = widths.pivot(index="width", columns="method", values="recall").loc[WIDTH_LABELS]
    width_md.insert(0, "share of centreline",
                    widths[widths.method == PIPELINE].set_index("width")["share of centreline"])
    width_md = width_md[["share of centreline", PIPELINE, FRANGI, THRESHOLD]].map(lambda v: f"{v:.2f}")
    contrast_md = contrast_table.assign(
        **{"median [q1, q3]": [f"{m:.0f} [{a:.0f}, {b:.0f}]" for m, a, b in
                               contrast_table[["median", "q1", "q3"]].to_numpy()]})[
        ["structure", "pixels", "median [q1, q3]"]]
    pd.DataFrame([{"outside annotation": near + far, "near": near, "far": far,
                   "near share": near / (near + far), "true positives": tp,
                   "precision": tp / (tp + near + far),
                   "tolerant precision": (tp + near) / (tp + near + far), "near distance": NEAR}]
                 ).to_csv(RESULTS_DIR / "false_positives.csv", index=False, float_format="%.4f")
    data = dataset()
    data.to_csv(RESULTS_DIR / "dataset.csv", index=False, float_format="%.4f")
    frangi_wide_vessels().to_csv(RESULTS_DIR / "frangi_wide.csv", index=False, float_format="%.4f")
    definitions = recall_definitions(segs, gts)
    definitions.to_csv(RESULTS_DIR / "recall_definitions.csv", index=False, float_format="%.4f")
    sensitivity = metric_sensitivity()
    sensitivity.to_csv(RESULTS_DIR / "metric_sensitivity.csv", index=False, float_format="%.4f")
    timing = runtime()
    timing.to_csv(RESULTS_DIR / "runtime.csv", index=False, float_format="%.3f")

    definitions_md = definitions.groupby("method", sort=False)[
        ["precision", "course recall", "covered recall", "course f1", "covered f1"]].mean()
    definitions_md["length ratio"] = definitions.groupby("method", sort=False).apply(
        lambda d: d["length ratio"].mean(), include_groups=False)
    definitions_md = definitions_md.map(lambda v: f"{v:.3f}").reset_index()
    sensitivity_md = sensitivity.assign(**{m: sensitivity[m].map(lambda v: f"{v:.3f}")
                                           for m in ("precision", "recall", "f1", "dice")})
    precision = tp / (tp + near + far)
    (RESULTS_DIR / "errors.md").write_text(
        "# Error analysis (test images)\n\n"
        "## Recall by annotated vessel width\n\n"
        "Share of the annotated centreline covered by each method, pooled over the\n"
        "test images; width is measured on the annotation at the centreline.\n\n"
        f"{to_markdown(width_md.reset_index())}\n\n"
        "## False positives of the pipeline\n\n"
        f"- {near + far} predicted centreline pixels fall outside the annotation;\n"
        f"  {near / (near + far):.0%} lie within {NEAR} px of an annotated vessel (alignment\n"
        f"  or width errors), {far / (near + far):.0%} further away (structures absent from\n"
        "  the annotation).\n"
        f"- Precision is {precision:.3f}; counting the near ones as correct would give\n"
        f"  {(tp + near) / (tp + near + far):.3f}.\n\n"
        "## Local contrast\n\n"
        "Black top-hat of the raw image (disk radius 4), in gray levels, at\n"
        "centreline pixels of each kind of structure. Thin vessels are 1 to 2 px wide.\n\n"
        f"{to_markdown(contrast_md)}\n\n"
        "## Two definitions of recall\n\n"
        "The course recall is TP / (TP + FN) with TP counted on the predicted skeleton\n"
        "and FN on the annotated one. The covered recall is the share of the annotated\n"
        "skeleton inside the prediction. Length ratio: predicted skeleton inside the\n"
        "annotation over annotated skeleton inside the prediction (mean over images).\n\n"
        f"{to_markdown(definitions_md)}\n\n"
        "## Sensitivity of the metric\n\n"
        "Each annotation, deformed, scored against itself; mean over the 10 images.\n\n"
        f"{to_markdown(sensitivity_md)}\n\n"
        "## Run time\n\n"
        f"segment() takes {timing['seconds per image'].item():.2f} s per 512 x 512 image "
        f"({timing['cpu'].item()}, Python {timing['python'].item()}).\n"
    )
    print((RESULTS_DIR / "errors.md").read_text())


if __name__ == "__main__":
    main()
