"""Colours and overlays shared by the figures.

Palette checked with a colour-vision-deficiency validator: the four method
colours are distinguishable pairwise on the light surface, and the two error
colours against the dimmed retina (about #202020). Each method keeps its
colour in every figure.
"""

import numpy as np
from skimage.morphology import dilation, disk, skeletonize

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

METHOD_COLORS = {
    "Morphological pipeline": "#2a78d6",
    "Frangi vesselness": "#eb6834",
    "Global threshold": "#1baf7a",
    "U-Net": "#4a3aa7",
}

# Errors are what the overlays are for: correct centrelines stay recessive.
DETECTED = "#d9d8d2"   # predicted centreline inside an annotated vessel
FALSE_POS = "#d95926"  # predicted centreline outside the annotation
MISSED = "#3987e5"     # annotated centreline not covered by the prediction
OVERLAY_LABELS = {
    DETECTED: "detected, annotated",
    FALSE_POS: "detected, not annotated",
    MISSED: "annotated, missed",
}


def hex_to_rgb(color):
    return tuple(int(color[k:k + 2], 16) for k in (1, 3, 5))


def error_overlay(image, segmentation, ground_truth, dim=0.5, thicken=False):
    """RGB image of the skeleton-level errors counted by the course metric.

    ``thicken`` draws the centrelines 3 px wide, for images shown downscaled.
    """
    out = np.stack([image * dim] * 3, axis=-1).astype(np.uint8)
    seg_skel, gt_skel = skeletonize(segmentation), skeletonize(ground_truth)
    layers = [(seg_skel & ground_truth, DETECTED), (seg_skel & ~ground_truth, FALSE_POS),
              (gt_skel & ~segmentation, MISSED)]
    for pixels, color in layers:
        out[dilation(pixels, disk(1)) if thicken else pixels] = hex_to_rgb(color)
    return out
