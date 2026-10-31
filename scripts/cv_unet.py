"""Leave-one-out cross-validation of the U-Net.

For each of the 10 images, U-Nets are trained from scratch on the 9 other
images, one per seed, with the fixed recipe of vessels.unet, and applied to
the held-out image. Probabilities are binarised at vessels.unet.THRESHOLD
(0.5, fixed a priori).

Writes results/cross_validation/{unet_per_image.csv, unet_training.csv} and
the probability maps to results/cross_validation/probabilities/. Resumable: a
(fold, seed) whose probability map exists is not trained again. About two
hours on an Apple M1 Max.
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from vessels.data import ALL_IDS, fov_mask, load_ground_truth, load_image
from vessels.evaluation import evaluate, skeleton_coverage
from vessels.unet import STEPS, THRESHOLD, default_device, predict, train

OUT = Path(__file__).resolve().parents[1] / "results" / "cross_validation"
PROBABILITIES = OUT / "probabilities"
SEEDS = (0, 1, 2)
FOV_MARGIN = 3


def main():
    PROBABILITIES.mkdir(parents=True, exist_ok=True)
    images = {i: load_image(i) for i in ALL_IDS}
    gts = {i: load_ground_truth(i) for i in ALL_IDS}
    mask = fov_mask(images[ALL_IDS[0]].shape, FOV_MARGIN)
    device = default_device()
    print(f"device {device}, torch {torch.__version__}, {STEPS} steps, seeds {SEEDS}", flush=True)

    for held_out in ALL_IDS:
        train_ids = [i for i in ALL_IDS if i != held_out]
        for seed in SEEDS:
            path = PROBABILITIES / f"{held_out}_seed{seed}.npz"
            if path.exists():
                continue
            start = time.perf_counter()
            model, log = train([images[i] for i in train_ids], [gts[i] for i in train_ids],
                               [mask] * len(train_ids), seed=seed, steps=STEPS, device=device)
            prob = predict(model, images[held_out], mask)
            np.savez_compressed(path, probability=prob.astype(np.float16))
            log = pd.DataFrame(log).assign(held_out=held_out, seed=seed)
            log.to_csv(OUT / "unet_training.csv", mode="a", index=False, float_format="%.5f",
                       header=not (OUT / "unet_training.csv").exists())
            print(f"fold {held_out} seed {seed}: final loss {log.loss.iloc[-1]:.3f}, "
                  f"{time.perf_counter() - start:.0f} s", flush=True)

    rows = []
    for held_out in ALL_IDS:
        for seed in SEEDS:
            prob = np.load(PROBABILITIES / f"{held_out}_seed{seed}.npz")["probability"]
            seg = mask & (prob.astype(np.float32) >= THRESHOLD)
            rows.append({"image": held_out, "seed": seed, **evaluate(seg, gts[held_out]),
                         "covered recall": skeleton_coverage(seg, gts[held_out])})
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "unet_per_image.csv", index=False, float_format="%.4f")
    print(table.groupby("image")[["precision", "recall", "f1", "dice"]].mean().round(3))


if __name__ == "__main__":
    main()
