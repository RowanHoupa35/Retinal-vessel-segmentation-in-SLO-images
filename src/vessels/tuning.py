"""Grid search of the pipeline and reference parameters on a set of images.

Shared by scripts/ablation.py (tune/test split) and scripts/cv_pipeline.py
(leave-one-out cross-validation), so that both apply exactly the same rule.
"""

import itertools
from dataclasses import replace

import numpy as np
import pandas as pd
from skimage.morphology import skeletonize

from vessels.baselines import frangi_response, threshold_segmentation
from vessels.data import fov_mask, load_ground_truth, load_image
from vessels.evaluation import evaluate
from vessels.morphology import area_opening
from vessels.pipeline import Params, binarize, combine_scales, denoise, single_scale_response
from vessels.thresholding import rank_threshold

FOV_MARGIN = 3

MEDIAN_SIZES = [3, 5]
LINE_LENGTHS = [11, 15, 21]
SINGLE_RADII = [(3,), (4,), (5,), (7,)]
MULTI_RADII = [(2, 4), (2, 3, 4), (2, 4, 6), (2, 3, 4, 5)]
KEPT_FRACTIONS = [round(f, 2) for f in np.arange(0.05, 0.205, 0.01)]
SEED_FRACTIONS = [round(f, 2) for f in np.arange(0.03, 0.105, 0.01)]
MIN_AREAS = [20, 30, 50, 80, 120]
THRESHOLDS_T = list(range(40, 145, 5))

BARE = Params(fov_margin=FOV_MARGIN, median_size=None, tophat_radii=(5,), line_length=None,
              seed_fraction=None, min_area=0)


def describe(p):
    parts = [f"median {p.median_size}"] if p.median_size else []
    parts.append("r = " + ", ".join(map(str, p.tophat_radii)))
    if p.line_length:
        parts.append(f"L = {p.line_length}")
    parts.append(f"kept {p.kept_fraction:.2f}")
    if p.seed_fraction:
        parts.append(f"seeds {p.seed_fraction:.2f}")
    if p.min_area:
        parts.append(f"area ≥ {p.min_area}")
    return ", ".join(parts)


def product(grid):
    return [dict(zip(grid, values)) for values in itertools.product(*grid.values())]


def response_grid(params):
    grid = {"tophat_radii": MULTI_RADII if len(params.tophat_radii) > 1 else SINGLE_RADII}
    if params.median_size:
        grid["median_size"] = MEDIAN_SIZES
    if params.line_length:
        grid["line_length"] = LINE_LENGTHS
    return grid


def threshold_candidates(params):
    """(kept_fraction, seed_fraction) pairs and area values to search for ``params``."""
    seeds = SEED_FRACTIONS if params.seed_fraction else [None]
    pairs = [(k, s) for k in KEPT_FRACTIONS for s in seeds if s is None or s <= k]
    return pairs, MIN_AREAS if params.min_area else [0]


class Data:
    def __init__(self, ids):
        self.ids = tuple(ids)
        self.images = {i: load_image(i) for i in self.ids}
        self.gts = {i: load_ground_truth(i) for i in self.ids}
        self.gt_skeletons = {i: skeletonize(g) for i, g in self.gts.items()}
        self.mask = fov_mask(self.images[self.ids[0]].shape, FOV_MARGIN)
        self._singles = {}

    def response(self, i, params):
        """Same as vessel_response, with the single-scale responses cached."""
        singles = []
        for radius in params.tophat_radii:
            key = (i, params.median_size, radius, params.line_length)
            if key not in self._singles:
                image = denoise(self.images[i], params.median_size)
                self._singles[key] = single_scale_response(image, radius, params.line_length)
            singles.append(self._singles[key])
        return combine_scales(singles, self.mask)

    def clear_cache(self):
        self._singles.clear()

    def scores(self, segmentations):
        return pd.DataFrame([evaluate(segmentations[i], self.gts[i], self.gt_skeletons[i])
                             for i in self.ids], index=self.ids)


def search_pipeline(data, base, grid):
    """Best Params on ``data`` among ``base`` updated with every grid combination."""
    data.clear_cache()
    best, best_f1 = None, -1.0
    for change in product(grid):
        params = replace(base, **change)
        responses = {i: data.response(i, params) for i in data.ids}
        pairs, areas = threshold_candidates(params)
        for kept, seed in pairs:
            p = replace(params, kept_fraction=kept, seed_fraction=seed, min_area=0)
            unfiltered = {i: binarize(responses[i], data.mask, p) for i in data.ids}
            for area in areas:
                segs = {i: area_opening(s, area) if area else s for i, s in unfiltered.items()}
                f1 = data.scores(segs).f1.mean()
                if f1 > best_f1:
                    best, best_f1 = replace(p, min_area=area), f1
    return best


def search_reference(data, response, segment, grid):
    responses = {i: response(data.images[i], data.mask) for i in data.ids}
    return max(grid, key=lambda p: data.scores(
        {i: segment(responses[i], data.mask, p) for i in data.ids}).f1.mean())


def tune_full_pipeline(data):
    """Parameters of the full pipeline chosen on ``data`` by the ablation rule.

    Only the last three cumulative rows of the ablation decide them: the
    multi-scale row re-tunes every response parameter jointly, so the earlier
    rows have no influence, then the hysteresis and area-opening rows re-tune
    the thresholds.
    """
    params = replace(BARE, median_size=3, line_length=15, tophat_radii=(2, 3, 4))
    params = search_pipeline(data, params, response_grid(params))
    params = search_pipeline(data, replace(params, seed_fraction=0.08), {})
    return search_pipeline(data, replace(params, min_area=50), {})


def tune_references(data):
    """Global threshold T and Frangi kept fraction chosen on ``data``."""
    t = search_reference(data, lambda image, mask: image, threshold_segmentation, THRESHOLDS_T)
    kept = search_reference(data, frangi_response, rank_threshold, KEPT_FRACTIONS)
    return t, kept
