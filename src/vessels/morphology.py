"""Morphological building blocks.

Grayscale operators use ``mode="ignore"``: with scikit-image's default
(``"reflect"``) the padding at the image border makes openings exceed the
image by a few gray levels, which breaks their anti-extensivity and makes a
subsequent reconstruction fail.
"""

import numpy as np
from scipy import ndimage as ndi
from skimage import draw
from skimage.morphology import closing, disk, opening

EIGHT_CONNECTED = np.ones((3, 3), dtype=bool)


def line_footprint(length, angle):
    """Line segment of Euclidean length ``length`` px at ``angle`` degrees.

    One half is drawn from the centre and mirrored, so the segment is always
    centred; rounding both endpoints independently shifts some orientations
    by one pixel.
    """
    half = (length - 1) / 2
    c = int(np.ceil(half))
    theta = np.deg2rad(angle)
    rr, cc = draw.line(c, c, c - int(np.round(np.sin(theta) * half)),
                       c + int(np.round(np.cos(theta) * half)))
    footprint = np.zeros((2 * c + 1, 2 * c + 1), dtype=bool)
    footprint[rr, cc] = True
    footprint[2 * c - rr, 2 * c - cc] = True
    return footprint


def black_tophat(image, radius):
    """Closing minus image: dark structures narrower than the disk, background removed."""
    return closing(image, disk(radius), mode="ignore").astype(float) - image


def sup_of_line_openings(image, length, n_orientations=12):
    """Pointwise maximum of openings by line segments (Zana & Klein, 2001).

    A vessel longer than ``length`` survives the opening whose segment is
    aligned with it; noise and blobs smaller than ``length`` in every
    direction are removed.
    """
    angles = np.arange(n_orientations) * 180 / n_orientations
    return np.max([opening(image, line_footprint(length, a), mode="ignore") for a in angles], axis=0)


def area_opening(binary, min_area):
    """Remove the 8-connected components smaller than ``min_area`` pixels."""
    labels, _ = ndi.label(binary, structure=EIGHT_CONNECTED)
    keep = np.bincount(labels.ravel()) >= min_area
    keep[0] = False
    return keep[labels]
