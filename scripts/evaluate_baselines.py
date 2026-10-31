"""Evaluate the reference methods with the tune/test protocol.

Each method has one free parameter. It is swept on the tune images, the value
with the best mean F1 is kept, and only that value is evaluated on the test
images. The course threshold (T = 80) is also reported as is.

Writes results/baselines/{sweep_tune.csv, per_image.csv, summary.md}.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from vessels.baselines import frangi_response, threshold_segmentation
from vessels.data import TEST_IDS, TUNE_IDS, fov_mask, group, load_ground_truth, load_image
from vessels.evaluation import evaluate
from vessels.reporting import to_markdown
from vessels.thresholding import rank_threshold

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results" / "baselines"
METRICS = ["precision", "recall", "f1", "dice"]

METHODS = {
    "threshold": {
        "label": "Global threshold",
        "param": "T",
        "response": lambda image, mask: image,
        "segment": threshold_segmentation,
        "grid": list(range(40, 145, 5)),
    },
    "frangi": {
        "label": "Frangi vesselness",
        "param": "kept fraction",
        "response": frangi_response,
        "segment": rank_threshold,
        "grid": [round(f, 2) for f in np.arange(0.03, 0.205, 0.01)],
    },
}
COURSE_THRESHOLD = 80


def load_samples(ids, method):
    samples = {}
    for image_id in ids:
        image = load_image(image_id)
        mask = fov_mask(image.shape)
        samples[image_id] = {
            "response": method["response"](image, mask),
            "mask": mask,
            "gt": load_ground_truth(image_id),
        }
    return samples


def score(name, method, samples, param, split):
    rows = []
    for image_id, s in samples.items():
        segmentation = method["segment"](s["response"], s["mask"], param)
        rows.append({"method": name, "split": split, "image": image_id,
                     "group": group(image_id), "param": param,
                     **evaluate(segmentation, s["gt"])})
    return rows


def mean_std(df):
    grouped = df.groupby(["method", "split", "param"], sort=False)[METRICS]
    means, stds = grouped.mean(), grouped.std()
    table = pd.DataFrame({m: [f"{a:.3f} ± {b:.3f}" for a, b in zip(means[m], stds[m])]
                          for m in METRICS}, index=means.index)
    return table.reset_index()


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    sweeps, finals = [], []

    for name, method in METHODS.items():
        tune = load_samples(TUNE_IDS, method)
        test = load_samples(TEST_IDS, method)

        sweep = pd.DataFrame([row for p in method["grid"]
                              for row in score(name, method, tune, p, "tune")])
        sweeps.append(sweep)
        best = sweep.groupby("param")["f1"].mean().idxmax()
        print(f"{method['label']}: best {method['param']} on tune = {best}")

        candidates = {f"{name} (tuned)": best}
        if name == "threshold":
            candidates = {"threshold (course T=80)": COURSE_THRESHOLD, **candidates}
        for label, param in candidates.items():
            finals += score(label, method, tune, param, "tune")
            finals += score(label, method, test, param, "test")

    sweep = pd.concat(sweeps)
    sweep.groupby(["method", "param"], sort=False)[METRICS].mean().reset_index() \
        .to_csv(RESULTS_DIR / "sweep_tune.csv", index=False, float_format="%.4f")
    finals = pd.DataFrame(finals)
    finals.to_csv(RESULTS_DIR / "per_image.csv", index=False, float_format="%.4f")

    summary = mean_std(finals)
    test_table = to_markdown(summary[summary.split == "test"].drop(columns="split"))
    tune_table = to_markdown(summary[summary.split == "tune"].drop(columns="split"))
    (RESULTS_DIR / "summary.md").write_text(
        "# Reference methods\n\n"
        "Mean ± standard deviation over images. Precision and recall are the\n"
        "skeleton-based scores of the course; Dice is pixel-wise.\n\n"
        f"## Test set ({', '.join(TEST_IDS)})\n\n{test_table}\n\n"
        f"## Tune set ({', '.join(TUNE_IDS)}), where parameters were chosen\n\n{tune_table}\n"
    )
    print("\nTest set\n" + test_table)


if __name__ == "__main__":
    main()
