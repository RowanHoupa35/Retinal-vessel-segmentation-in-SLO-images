import numpy as np
from skimage.morphology import dilation, disk

from vessels.evaluation import dice, evaluate


def synthetic_vessels():
    """Two 3-px-wide crossing vessels on a 128 x 128 image."""
    gt = np.zeros((128, 128), dtype=bool)
    gt[20:110, 63:66] = True
    gt[40:43, 10:120] = True
    return gt


def test_perfect_segmentation_scores_one():
    gt = synthetic_vessels()
    assert evaluate(gt, gt) == {"precision": 1.0, "recall": 1.0, "f1": 1.0, "dice": 1.0}


def test_empty_segmentation_scores_zero_without_error():
    gt = synthetic_vessels()
    scores = evaluate(np.zeros_like(gt), gt)
    assert scores["precision"] == scores["recall"] == scores["f1"] == scores["dice"] == 0.0


def test_skeleton_scores_ignore_thickening_but_dice_does_not():
    # Documents a limit of the course metric: a much too thick segmentation
    # keeps a perfect skeleton recall and precision; only Dice drops.
    gt = synthetic_vessels()
    thick = dilation(gt, disk(4))
    scores = evaluate(thick, gt)
    assert scores["recall"] == 1.0
    assert scores["precision"] > 0.9
    assert scores["dice"] < 0.6


def test_dice_of_disjoint_masks_is_zero():
    a = np.zeros((10, 10), dtype=bool)
    b = a.copy()
    a[:5] = True
    b[5:] = True
    assert dice(a, b) == 0.0
