"""Loading of the 10-image IOSTAR subset and the tune/test split."""

from pathlib import Path

import numpy as np
from PIL import Image
from skimage.morphology import disk, erosion

DATA_DIR = Path(__file__).resolve().parents[2] / "images_IOSTAR"

# File names are star<id>_<eye><centring>.jpg: eye is OS (left) or OD (right),
# centring is C (macula-centred) or N (optic-disc-centred).
IMAGE_STEMS = {
    "01": "star01_OSC",
    "02": "star02_OSC",
    "03": "star03_OSN",
    "08": "star08_OSN",
    "21": "star21_OSC",
    "26": "star26_ODC",
    "28": "star28_ODN",
    "32": "star32_ODC",
    "37": "star37_ODN",
    "48": "star48_OSN",
}

# Split fixed before developing any method: images sorted by group
# (ODC, ODN, OSC, OSN) then by id, and assigned alternately to tune and test.
# Both halves contain every eye/centring group. Parameters are chosen on TUNE
# only; TEST is used once, for the final numbers.
TUNE_IDS = ("01", "08", "21", "26", "28")
TEST_IDS = ("02", "03", "32", "37", "48")
ALL_IDS = tuple(sorted(IMAGE_STEMS))


def group(image_id):
    """Eye/centring group of an image, e.g. 'OSC'."""
    return IMAGE_STEMS[image_id].split("_")[1]


def load_image(image_id, data_dir=DATA_DIR):
    """Grayscale SLO image as uint8 (vessels are dark)."""
    path = Path(data_dir) / f"{IMAGE_STEMS[image_id]}.jpg"
    return np.asarray(Image.open(path).convert("L"), dtype=np.uint8)


def load_ground_truth(image_id, data_dir=DATA_DIR):
    """Expert vessel annotation as a boolean array.

    The course script used ``np.asarray(Image.open(path)).astype(np.bool_)``.
    For these 1-bit PNGs PIL yields bool arrays whose True bytes are 0xFF
    instead of 0x01; ``astype`` keeps those bytes and scikit-image 0.26's
    ``skeletonize`` then segfaults. Comparing to 0 builds a clean array.
    """
    path = Path(data_dir) / f"GT_{image_id}.png"
    return np.asarray(Image.open(path).convert("L")) > 0


def fov_mask(shape, margin=0):
    """Field of view: the disk inscribed in the image, as in the course script.

    ``margin`` erodes the disk by that many pixels to drop the rim, where the
    black background creates strong artificial contrasts.
    """
    nrows, ncols = shape
    row, col = np.ogrid[:nrows, :ncols]
    mask = (row - nrows / 2) ** 2 + (col - ncols / 2) ** 2 <= (nrows / 2) ** 2
    if margin:
        mask = erosion(mask, disk(margin))
    return mask
