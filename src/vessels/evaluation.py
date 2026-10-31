"""Evaluation metrics.

``skeleton_scores`` reproduces the evaluation function of the course script
(ENSTA MI206, projet 3). It compares centrelines rather than areas:

- precision: share of the predicted skeleton lying inside annotated vessels;
- recall: share of the annotated skeleton covered by the prediction.

Working on skeletons gives a thin capillary the same weight per unit length
as a large vessel, and makes the score insensitive to the imprecise vessel
borders of manual annotations. The flip side is that it barely penalises a
segmentation that is too thick, so the pixel-wise Dice is reported as well.
"""

import numpy as np
from skimage.morphology import skeletonize


def skeleton_scores(segmentation, ground_truth, gt_skeleton=None):
    """Precision and recall of the course evaluation.

    ``gt_skeleton`` can be passed to avoid recomputing it in parameter searches.
    """
    seg_skel = skeletonize(segmentation)
    gt_skel = skeletonize(ground_truth) if gt_skeleton is None else gt_skeleton
    tp = np.sum(seg_skel & ground_truth)
    fp = np.sum(seg_skel & ~ground_truth)
    fn = np.sum(gt_skel & ~segmentation)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return float(precision), float(recall)


def skeleton_coverage(segmentation, ground_truth, gt_skeleton=None):
    """Share of the annotated skeleton inside the segmentation.

    Unlike the course recall, it does not depend on the length of the predicted
    skeleton; it is the "topology sensitivity" of clDice (Shit et al., 2021).
    """
    gt_skel = skeletonize(ground_truth) if gt_skeleton is None else gt_skeleton
    return float((gt_skel & segmentation).sum() / gt_skel.sum()) if gt_skel.any() else 0.0


def f1_score(precision, recall):
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def dice(segmentation, ground_truth):
    """Pixel-wise overlap, sensitive to vessel width unlike the skeleton scores."""
    total = segmentation.sum() + ground_truth.sum()
    return float(2 * np.sum(segmentation & ground_truth) / total) if total else 1.0


def evaluate(segmentation, ground_truth, gt_skeleton=None):
    precision, recall = skeleton_scores(segmentation, ground_truth, gt_skeleton)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1_score(precision, recall),
        "dice": dice(segmentation, ground_truth),
    }
