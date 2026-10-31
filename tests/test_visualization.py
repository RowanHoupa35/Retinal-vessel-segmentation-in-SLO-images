import numpy as np

from vessels.visualization import DETECTED, FALSE_POS, MISSED, error_overlay, hex_to_rgb


def test_error_overlay_colours_each_kind_of_centreline():
    gt = np.zeros((40, 40), dtype=bool)
    gt[5:8, 5:35] = True      # annotated and detected
    gt[30:33, 5:35] = True    # annotated, missed
    seg = np.zeros_like(gt)
    seg[5:8, 5:35] = True
    seg[18:21, 5:35] = True   # detected, not annotated
    out = error_overlay(np.zeros(gt.shape, dtype=np.uint8), seg, gt)
    assert tuple(out[6, 20]) == hex_to_rgb(DETECTED)
    assert tuple(out[19, 20]) == hex_to_rgb(FALSE_POS)
    assert tuple(out[31, 20]) == hex_to_rgb(MISSED)
    assert tuple(out[12, 20]) == (0, 0, 0)
