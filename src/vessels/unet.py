"""A small U-Net (Ronneberger et al., 2015), the learning-based comparison.

The training recipe below was fixed before any cross-validation result was
seen; the only calibration run looked at the training loss alone (see
scripts/cv_unet.py). Nothing here is chosen on held-out images.

Requires PyTorch (``pip install -e ".[learning]"``).
"""

import time

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

PATCH = 128
BATCH = 16
# Set from the training loss of scripts/cv_unet_pilot.py (6000 steps, cosine
# schedule): beyond about 4000 steps each further 1000 steps lowers it by less
# than 5 %, and longer runs mostly fit the 9 training images further, which
# cannot be checked without looking at held-out images.
STEPS = 4000
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
BASE_CHANNELS = 16
DEPTH = 4
THRESHOLD = 0.5  # on the predicted probability, fixed a priori


def default_device():
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


class DoubleConv(nn.Sequential):
    def __init__(self, c_in, c_out):
        super().__init__(
            nn.Conv2d(c_in, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_out, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
        )


class UNet(nn.Module):
    """Encoder-decoder with skip connections; the input side must be a multiple of 2**depth."""

    def __init__(self, base=BASE_CHANNELS, depth=DEPTH):
        super().__init__()
        channels = [base * 2 ** k for k in range(depth + 1)]
        self.down = nn.ModuleList([DoubleConv(1, channels[0])] +
                                  [DoubleConv(channels[k], channels[k + 1]) for k in range(depth)])
        self.up = nn.ModuleList([nn.ConvTranspose2d(channels[k + 1], channels[k], 2, stride=2)
                                 for k in reversed(range(depth))])
        self.merge = nn.ModuleList([DoubleConv(2 * channels[k], channels[k])
                                    for k in reversed(range(depth))])
        self.head = nn.Conv2d(channels[0], 1, 1)

    def forward(self, x):
        skips = []
        for k, block in enumerate(self.down):
            x = block(x if k == 0 else F.max_pool2d(x, 2))
            skips.append(x)
        for up, merge, skip in zip(self.up, self.merge, reversed(skips[:-1])):
            x = merge(torch.cat([up(x), skip], dim=1))
        return self.head(x)


def normalize(image, mask):
    """Zero mean, unit variance inside the field of view; 0 outside."""
    image = image.astype(np.float32)
    values = image[mask]
    return np.where(mask, (image - values.mean()) / values.std(), 0).astype(np.float32)


class PatchSampler:
    """Random training patches with flips, quarter turns and intensity jitter."""

    def __init__(self, images, gts, masks, seed):
        self.x = np.stack([normalize(im, m) for im, m in zip(images, masks)])
        self.y = np.stack(gts).astype(np.float32)
        self.w = np.stack(masks).astype(np.float32)
        self.rng = np.random.default_rng(seed)

    def batch(self, size=BATCH, patch=PATCH):
        n, h, w = self.x.shape
        out = np.empty((3, size, 1, patch, patch), dtype=np.float32)
        for b in range(size):
            i = self.rng.integers(n)
            r, c = self.rng.integers(h - patch + 1), self.rng.integers(w - patch + 1)
            crops = [a[i, r:r + patch, c:c + patch] for a in (self.x, self.y, self.w)]
            k, flip = self.rng.integers(4), self.rng.integers(2)
            crops = [np.rot90(a, k) for a in crops]
            if flip:
                crops = [a[:, ::-1] for a in crops]
            gain, offset = self.rng.uniform(0.9, 1.1), self.rng.uniform(-0.1, 0.1)
            crops[0] = crops[0] * gain + offset * (crops[2] > 0)
            for j in range(3):
                out[j, b, 0] = crops[j]
        return [torch.from_numpy(a) for a in out]


def loss_function(logits, target, weight, eps=1.0):
    """Binary cross-entropy plus soft Dice, both restricted to the field of view."""
    bce = F.binary_cross_entropy_with_logits(logits, target, weight=weight, reduction="sum")
    bce = bce / weight.sum().clamp(min=1)
    prob = torch.sigmoid(logits) * weight
    dice = 1 - (2 * (prob * target).sum() + eps) / (prob.sum() + (target * weight).sum() + eps)
    return bce + dice


def train(images, gts, masks, seed, steps=STEPS, device=None, log_every=50):
    """Train a U-Net from scratch; returns the model and the training-loss log."""
    device = device or default_device()
    torch.manual_seed(seed)
    sampler = PatchSampler(images, gts, masks, seed)
    model = UNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps)
    log, start = [], time.perf_counter()
    running = torch.zeros((), device=device)  # summed on the device: no sync per step
    model.train()
    for step in range(1, steps + 1):
        x, y, w = (t.to(device) for t in sampler.batch())
        loss = loss_function(model(x), y, w)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        schedule.step()
        running += loss.detach()
        if step % log_every == 0:
            log.append({"step": step, "loss": running.item() / log_every,
                        "seconds": time.perf_counter() - start})
            running.zero_()
    return model, log


@torch.no_grad()
def predict(model, image, mask, device=None):
    """Vessel probability on the whole image, 0 outside the field of view."""
    device = device or next(model.parameters()).device
    model.eval()
    x = torch.from_numpy(normalize(image, mask))[None, None].to(device)
    prob = torch.sigmoid(model(x))[0, 0].cpu().numpy()
    return np.where(mask, prob, 0).astype(np.float32)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())
