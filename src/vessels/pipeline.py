"""Morphological vessel segmentation pipeline.

1. Median filter: removes the fine noise that the top-hat would pick up.
2. Black top-hat: removes the non-uniform background (vignetting).
3. Supremum of line openings: keeps elongated structures only.
4. Multi-scale: steps 2-3 at several disk radii, each response divided by
   its standard deviation in the field of view, then averaged.
5. Hysteresis threshold (geodesic reconstruction from strong seeds).
6. Area opening: removes the remaining small connected components.

Every step can be switched off through ``Params`` for the ablation study.
"""

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import median_filter

from vessels.data import fov_mask
from vessels.morphology import area_opening, black_tophat, sup_of_line_openings
from vessels.thresholding import hysteresis, rank_threshold


@dataclass(frozen=True)
class Params:
    """Defaults are the values chosen on the tune set by scripts/ablation.py."""

    fov_margin: int = 3
    median_size: int | None = 3          # None: no median filter
    tophat_radii: tuple = (2, 3, 4, 5)   # one radius: single scale
    line_length: int | None = 15         # None: no line openings
    kept_fraction: float = 0.12          # share of the field of view kept
    seed_fraction: float | None = 0.09   # None: single threshold, no hysteresis
    min_area: int = 50                   # 0: no area opening


def denoise(image, median_size):
    return median_filter(image, median_size) if median_size else image


def single_scale_response(image, radius, line_length):
    response = black_tophat(image, radius)
    return sup_of_line_openings(response, line_length) if line_length else response


def combine_scales(responses, mask):
    """Average of the responses, each divided by its standard deviation in the mask."""
    return np.mean([r / r[mask].std() for r in responses], axis=0)


def vessel_response(image, mask, params):
    """Grayscale map where vessels are bright."""
    image = denoise(image, params.median_size)
    return combine_scales([single_scale_response(image, r, params.line_length)
                           for r in params.tophat_radii], mask)


def binarize(response, mask, params):
    if params.seed_fraction:
        segmentation = hysteresis(response, mask, params.kept_fraction, params.seed_fraction)
    else:
        segmentation = rank_threshold(response, mask, params.kept_fraction)
    if params.min_area:
        segmentation = area_opening(segmentation, params.min_area)
    return segmentation


def segment(image, params=Params()):
    mask = fov_mask(image.shape, params.fov_margin)
    return binarize(vessel_response(image, mask, params), mask, params)
