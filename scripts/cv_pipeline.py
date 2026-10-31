"""Leave-one-out cross-validation of the pipeline and the reference methods.

For each of the 10 images, the pipeline (vessels.tuning.tune_full_pipeline,
the rule of the ablation study), the global threshold and Frangi are tuned on
the 9 other images and evaluated on the held-out one. Folds run in parallel
on the CPU; about ten minutes.

Writes results/cross_validation/{pipeline_folds.csv, pipeline_per_image.csv}
and the held-out segmentations to results/cross_validation/segmentations/.
"""

from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from vessels.baselines import frangi_response, threshold_segmentation
from vessels.data import ALL_IDS
from vessels.evaluation import evaluate, skeleton_coverage
from vessels.pipeline import binarize, vessel_response
from vessels.thresholding import rank_threshold
from vessels.tuning import Data, describe, tune_full_pipeline, tune_references

OUT = Path(__file__).resolve().parents[1] / "results" / "cross_validation"
SEGMENTATIONS = OUT / "segmentations"
WORKERS = 5


def run_fold(held_out):
    train = Data([i for i in ALL_IDS if i != held_out])
    params = tune_full_pipeline(train)
    threshold, kept = tune_references(train)
    test = Data([held_out])
    image, gt, mask = test.images[held_out], test.gts[held_out], test.mask
    segs = {"pipeline": binarize(vessel_response(image, mask, params), mask, params),
            "frangi": rank_threshold(frangi_response(image, mask), mask, kept),
            "threshold": threshold_segmentation(image, mask, threshold)}
    np.savez_compressed(SEGMENTATIONS / f"{held_out}.npz", **segs)
    rows = [{"image": held_out, "method": name, **evaluate(seg, gt),
             "covered recall": skeleton_coverage(seg, gt)} for name, seg in segs.items()]
    fold = {"held_out": held_out, **{k: (list(v) if isinstance(v, tuple) else v)
                                     for k, v in asdict(params).items()},
            "description": describe(params), "threshold T": threshold, "frangi kept": kept}
    return fold, rows


def main():
    SEGMENTATIONS.mkdir(parents=True, exist_ok=True)
    folds, rows = [], []
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        for fold, fold_rows in pool.map(run_fold, ALL_IDS):
            folds.append(fold)
            rows += fold_rows
            print(f"fold {fold['held_out']}: {fold['description']}, T = {fold['threshold T']}, "
                  f"Frangi kept {fold['frangi kept']:.2f}", flush=True)
    pd.DataFrame(folds).to_csv(OUT / "pipeline_folds.csv", index=False)
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "pipeline_per_image.csv", index=False, float_format="%.4f")
    print(table.groupby("method")[["precision", "recall", "f1", "dice"]].agg(["mean", "std"]).round(3))


if __name__ == "__main__":
    main()
