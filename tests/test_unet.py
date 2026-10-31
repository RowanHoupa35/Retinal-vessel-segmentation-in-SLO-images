import numpy as np
import pytest

torch = pytest.importorskip("torch")

from vessels.unet import PatchSampler, UNet, count_parameters, loss_function, normalize, predict


def test_unet_keeps_the_input_size():
    model = UNet()
    assert model(torch.zeros(2, 1, 64, 64)).shape == (2, 1, 64, 64)
    assert 1_000_000 < count_parameters(model) < 3_000_000


def test_normalize_is_centred_in_the_mask_and_zero_outside():
    image = np.arange(100, dtype=np.uint8).reshape(10, 10)
    mask = np.zeros((10, 10), dtype=bool)
    mask[2:8, 2:8] = True
    out = normalize(image, mask)
    assert abs(out[mask].mean()) < 1e-5 and abs(out[mask].std() - 1) < 1e-4
    assert (out[~mask] == 0).all()


def test_patches_keep_image_and_labels_aligned():
    image = np.zeros((64, 64), dtype=np.uint8)
    gt = np.zeros((64, 64), dtype=bool)
    gt[:, 30:33] = True
    image[gt] = 255
    mask = np.ones_like(gt)
    x, y, w = PatchSampler([image], [gt], [mask], seed=0).batch(size=8, patch=32)
    for b in range(8):
        bright = (x[b, 0] > x[b, 0].mean()).numpy()
        assert (bright == (y[b, 0] > 0.5).numpy()).all()


def test_loss_ignores_pixels_outside_the_field_of_view():
    target = torch.zeros(1, 1, 8, 8)
    weight = torch.zeros(1, 1, 8, 8)
    weight[..., :4] = 1
    logits = torch.full((1, 1, 8, 8), -5.0)
    worse = logits.clone()
    worse[..., 4:] = 5.0  # wrong, but outside the field of view
    assert torch.isclose(loss_function(logits, target, weight), loss_function(worse, target, weight))


def test_predict_is_zero_outside_the_mask():
    model = UNet(base=4, depth=2)
    mask = np.zeros((32, 32), dtype=bool)
    mask[4:28, 4:28] = True
    prob = predict(model, np.random.default_rng(0).integers(0, 255, (32, 32)).astype(np.uint8),
                   mask, device=torch.device("cpu"))
    assert prob.shape == (32, 32) and (prob[~mask] == 0).all()
