import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
from skimage.morphology import reconstruction

from vessels.morphology import (EIGHT_CONNECTED, area_opening, black_tophat, line_footprint,
                                sup_of_line_openings)
from vessels.pipeline import Params, segment
from vessels.thresholding import hysteresis, rank_threshold

PARAMS_JSON = Path(__file__).resolve().parents[1] / "results" / "pipeline" / "params.json"


def vessel_image(size=96):
    """Bright background, one dark 3-px vessel, one dark 3 x 3 speck, noise."""
    rng = np.random.default_rng(0)
    image = np.full((size, size), 150.0) + rng.normal(0, 2, (size, size))
    image[:, 40:43] -= 60
    image[20:23, 70:73] -= 60
    return np.clip(image, 0, 255).astype(np.uint8)


@pytest.mark.parametrize("length", [7, 11, 15, 21])
@pytest.mark.parametrize("angle", np.arange(12) * 15)
def test_line_footprint_is_centred_and_symmetric(length, angle):
    fp = line_footprint(length, angle)
    c = fp.shape[0] // 2
    assert fp[c, c]
    assert (fp == fp[::-1, ::-1]).all()


def test_openings_are_anti_extensive_up_to_the_border():
    image = black_tophat(vessel_image(), 4)
    assert (sup_of_line_openings(image, 15) <= image).all()


def test_line_openings_keep_the_vessel_and_remove_the_speck():
    image = black_tophat(vessel_image(), 4)
    opened = sup_of_line_openings(image, 15)
    assert opened[50, 41] > 40       # vessel kept
    assert opened[21, 71] < 10       # speck removed


def test_hysteresis_matches_morphological_reconstruction():
    rng = np.random.default_rng(1)
    response = rng.random((64, 64))
    mask = np.ones_like(response, dtype=bool)
    candidates = rank_threshold(response, mask, 0.4)
    seeds = rank_threshold(response, mask, 0.05)
    expected = reconstruction(seeds.astype(np.uint8), candidates.astype(np.uint8),
                              footprint=EIGHT_CONNECTED).astype(bool)
    assert (hysteresis(response, mask, 0.4, 0.05) == expected).all()


def test_area_opening_removes_small_components_only():
    binary = np.zeros((20, 20), dtype=bool)
    binary[1:3, 1:3] = True           # 4 px
    binary[10:15, 10:15] = True       # 25 px
    kept = area_opening(binary, 10)
    assert not kept[1:3, 1:3].any() and kept[10:15, 10:15].all()


def test_segment_finds_the_synthetic_vessel():
    image = vessel_image(128)
    seg = segment(image, Params(fov_margin=0, kept_fraction=0.05, seed_fraction=0.02, min_area=20))
    assert seg[30:100, 40:43].mean() > 0.9
    assert not seg[20:23, 70:73].any()


@pytest.mark.skipif(not PARAMS_JSON.exists(), reason="run scripts/ablation.py")
def test_default_params_are_the_tuned_ones():
    tuned = json.loads(PARAMS_JSON.read_text())
    assert json.loads(json.dumps(asdict(Params()))) == tuned
