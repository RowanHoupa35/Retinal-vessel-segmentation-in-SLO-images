"""Calibration of the U-Net training length, on the training loss only.

Trains one U-Net on the 9 images of the first cross-validation fold (image 01
held out) for a long run and logs the training loss. No score is computed on
any image, held out or not: the curve only tells how many steps the loss needs
to settle, which fixes vessels.unet.STEPS before the cross-validation runs.

Writes results/cross_validation/pilot_loss.csv.
"""

from pathlib import Path

import pandas as pd

from vessels.data import ALL_IDS, fov_mask, load_ground_truth, load_image
from vessels.unet import train

OUT = Path(__file__).resolve().parents[1] / "results" / "cross_validation"
PILOT_STEPS = 6000
FOV_MARGIN = 3


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    held_out = ALL_IDS[0]
    ids = [i for i in ALL_IDS if i != held_out]
    images = [load_image(i) for i in ids]
    model, log = train(images, [load_ground_truth(i) for i in ids],
                       [fov_mask(im.shape, FOV_MARGIN) for im in images],
                       seed=0, steps=PILOT_STEPS, log_every=100)
    log = pd.DataFrame(log).assign(held_out=held_out)
    log.to_csv(OUT / "pilot_loss.csv", index=False, float_format="%.5f")
    print(log.iloc[::5].to_string(index=False))


if __name__ == "__main__":
    main()
