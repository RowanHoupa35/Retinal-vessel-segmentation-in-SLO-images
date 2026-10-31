"""Ablation study of the morphological pipeline.

Cumulative rows add one step at a time, starting from a bare black top-hat.
Remove-one rows drop one step from the full pipeline. Parameters are chosen
on the tune images (best mean F1 over a grid) with one rule:

- a cumulative row that changes the vessel response re-tunes all response
  parameters introduced so far, since they interact (e.g. the best median
  size changes once line openings are added);
- every row re-tunes the threshold parameters;
- remove-one rows keep the other parameters of the full pipeline (only the
  radius is re-tuned for the single-scale row).

All rows, references included, share the same field of view (course disk
eroded by 3 px). The search itself is in vessels.tuning. Takes a few minutes.

Writes results/pipeline/{ablation.md, ablation.csv, per_image.csv, params.json}.
``--test`` also evaluates every row on the test images with the parameters
chosen on tune; it is meant to be run once, for the final numbers.
"""

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd

from vessels.baselines import frangi_response, threshold_segmentation
from vessels.data import TEST_IDS, TUNE_IDS, group
from vessels.pipeline import binarize, vessel_response
from vessels.reporting import mean_std, to_markdown
from vessels.thresholding import rank_threshold
from vessels.tuning import (BARE, KEPT_FRACTIONS, SINGLE_RADII, THRESHOLDS_T, Data, describe,
                            response_grid, search_pipeline, search_reference)

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results" / "pipeline"
METRICS = ["precision", "recall", "f1", "dice"]
RESPONSE = "all response parameters"

# (label, change applied to the previous parameters, structural grid to search)
CUMULATIVE = [
    ("Black top-hat", {}, RESPONSE),
    ("+ median filter", {"median_size": 3}, RESPONSE),
    ("+ line openings", {"line_length": 15}, RESPONSE),
    ("+ multi-scale", {"tophat_radii": (2, 3, 4)}, RESPONSE),
    ("+ hysteresis", {"seed_fraction": 0.08}, {}),
    ("+ area opening", {"min_area": 50}, {}),
]
REMOVE_ONE = [
    ("without median filter", {"median_size": None}, {}),
    ("without line openings", {"line_length": None}, {}),
    ("single scale", {"tophat_radii": (3,)}, {"tophat_radii": SINGLE_RADII}),
    ("without hysteresis", {"seed_fraction": None}, {}),
    ("without area opening", {"min_area": 0}, {}),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--test", action="store_true", help="also evaluate on the test images")
    args = parser.parse_args()

    tune = Data(TUNE_IDS)
    test = Data(TEST_IDS) if args.test else None
    rows, per_image = [], []

    def add(kind, label, config, segment):
        """segment(data, image_id) -> boolean segmentation."""
        row = {"kind": kind, "step": label, "parameters": config}
        for split, data in (("tune", tune), ("test", test)):
            if data is None:
                continue
            scores = data.scores({i: segment(data, i) for i in data.ids})
            row.update({f"{split} {m}": mean_std(scores[m]) for m in METRICS})
            row[f"{split} f1 mean"] = scores.f1.mean()
            per_image.append(scores.assign(kind=kind, step=label, split=split,
                                           group=[group(i) for i in scores.index])
                             .rename_axis("image").reset_index())
        rows.append(row)
        print(f"{label:24s} tune F1 = {row['tune f1 mean']:.3f}   ({config})", flush=True)

    t = search_reference(tune, lambda image, mask: image, threshold_segmentation, THRESHOLDS_T)
    add("reference", "Global threshold", f"T = {t}",
        lambda d, i: threshold_segmentation(d.images[i], d.mask, t))
    f = search_reference(tune, frangi_response, rank_threshold, KEPT_FRACTIONS)
    add("reference", "Frangi vesselness", f"kept {f:.2f}",
        lambda d, i: rank_threshold(frangi_response(d.images[i], d.mask), d.mask, f))

    def add_pipeline(kind, label, params):
        add(kind, label, describe(params),
            lambda d, i: binarize(vessel_response(d.images[i], d.mask, params), d.mask, params))

    params = BARE
    for label, change, grid in CUMULATIVE:
        params = replace(params, **change)
        params = search_pipeline(tune, params, response_grid(params) if grid == RESPONSE else grid)
        add_pipeline("cumulative", label, params)
    final = params
    for label, change, grid in REMOVE_ONE:
        add_pipeline("remove-one", label, search_pipeline(tune, replace(final, **change), grid))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(rows)
    f1 = table["tune f1 mean"]
    full_f1 = f1[table.step == CUMULATIVE[-1][0]].item()
    table["Δ tune F1"] = np.where(table.kind == "cumulative", f1 - f1.shift(),
                                  np.where(table.kind == "remove-one", f1 - full_f1, np.nan))
    table.loc[table.step == CUMULATIVE[0][0], "Δ tune F1"] = np.nan
    table.to_csv(RESULTS_DIR / "ablation.csv", index=False, float_format="%.4f")
    pd.concat(per_image).to_csv(RESULTS_DIR / "per_image.csv", index=False, float_format="%.4f")
    (RESULTS_DIR / "params.json").write_text(json.dumps(asdict(final), indent=2) + "\n")

    delta = table["Δ tune F1"].map(lambda v: "" if np.isnan(v) else f"{v:+.3f}")
    sections = []
    for split in ("tune", "test") if args.test else ("tune",):
        cols = ["step", "parameters"] + [f"{split} {m}" for m in METRICS]
        for kind, title in (("reference", "Reference methods"),
                            ("cumulative", "Cumulative: one step added per row"),
                            ("remove-one", "Remove-one: full pipeline minus one step")):
            part = table.loc[table.kind == kind, cols].copy()
            part.columns = ["Step", "Parameters (chosen on tune)", "Precision", "Recall", "F1", "Dice"]
            if split == "tune" and kind != "reference":
                part["Δ F1"] = delta[table.kind == kind]
            if split == "test":
                part["F1 on tune"] = table.loc[table.kind == kind, "tune f1"]
            sections.append(f"### {title} ({split} set)\n\n{to_markdown(part)}")
    (RESULTS_DIR / "ablation.md").write_text(
        "# Ablation study\n\n"
        "Mean ± standard deviation over images. Precision and recall are the\n"
        "skeleton-based scores of the course; Dice is pixel-wise. Δ F1 is the\n"
        "change from the previous row (cumulative) or from the full pipeline\n"
        "(remove-one).\n\n" + "\n\n".join(sections) + "\n"
    )
    print(f"\nFinal parameters: {describe(final)}")


if __name__ == "__main__":
    main()
