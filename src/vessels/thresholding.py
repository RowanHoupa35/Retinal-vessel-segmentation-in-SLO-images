"""Rank-based thresholds of a vessel response.

Thresholds are expressed as the fraction of field-of-view pixels kept,
because the responses have no fixed scale across images.
"""

import numpy as np
from scipy import ndimage as ndi

from vessels.morphology import EIGHT_CONNECTED


def rank_threshold(response, mask, fraction):
    """Keep the ``fraction`` of field-of-view pixels with the highest response."""
    threshold = np.quantile(response[mask], 1 - fraction)
    return mask & (response > threshold)


def hysteresis(response, mask, kept_fraction, seed_fraction):
    """Hysteresis threshold as a geodesic reconstruction.

    Seeds are the top ``seed_fraction`` pixels; the candidates, the top
    ``kept_fraction``, are kept only if they are connected to a seed. Faint
    capillaries attached to the network survive, isolated noise does not.

    On binary images, reconstruction by dilation keeps exactly the connected
    components of the candidates that contain a seed; labelling them is six to
    seven times faster than ``skimage.morphology.reconstruction`` (measured in
    results/analysis/runtime.csv).
    """
    if seed_fraction > kept_fraction:
        raise ValueError("seed_fraction must not exceed kept_fraction")
    candidates = rank_threshold(response, mask, kept_fraction)
    seeds = rank_threshold(response, mask, seed_fraction)
    labels, n = ndi.label(candidates, structure=EIGHT_CONNECTED)
    keep = np.zeros(n + 1, dtype=bool)
    keep[labels[seeds]] = True
    keep[0] = False
    return keep[labels]
