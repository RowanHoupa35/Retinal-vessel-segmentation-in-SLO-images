import numpy as np
import pytest
from skimage.morphology import skeletonize

from vessels.data import (ALL_IDS, DATA_DIR, TEST_IDS, TUNE_IDS, fov_mask, group,
                          load_ground_truth, load_image)

needs_data = pytest.mark.skipif(not DATA_DIR.is_dir(), reason="run scripts/download_data.py")


def test_split_is_a_partition():
    assert set(TUNE_IDS).isdisjoint(TEST_IDS)
    assert sorted(TUNE_IDS + TEST_IDS) == list(ALL_IDS)


def test_split_is_stratified():
    groups = {"ODC", "ODN", "OSC", "OSN"}
    assert {group(i) for i in TUNE_IDS} == groups
    assert {group(i) for i in TEST_IDS} == groups


def test_fov_margin_shrinks_the_disk():
    full, eroded = fov_mask((64, 64)), fov_mask((64, 64), margin=3)
    assert eroded.dtype == bool
    assert eroded.sum() < full.sum()
    assert not (eroded & ~full).any()


@needs_data
@pytest.mark.parametrize("image_id", ALL_IDS)
def test_ground_truth_is_a_clean_boolean_array(image_id):
    gt = load_ground_truth(image_id)
    assert gt.dtype == bool
    assert gt.view(np.uint8).max() == 1  # PIL would give 0xFF bytes
    assert skeletonize(gt).any()  # segfaulted with the course loader


@needs_data
def test_images_match_ground_truth_shape():
    for image_id in ALL_IDS:
        assert load_image(image_id).shape == load_ground_truth(image_id).shape == (512, 512)
