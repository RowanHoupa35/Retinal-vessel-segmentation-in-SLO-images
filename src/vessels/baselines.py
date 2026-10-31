"""Reference methods the morphological pipeline is compared against."""

from skimage.filters import frangi

# Annotated vessels are 1 to 5 px wide on the tune images (median 3 px).
# On the tune set the F1 changes by less than 0.01 between this range and
# scikit-image's default (1 to 9), so the range matching the widths is kept.
FRANGI_SIGMAS = (1, 1.5, 2, 2.5, 3)


def threshold_segmentation(image, mask, threshold):
    """Course baseline: vessels are the pixels darker than a global threshold."""
    return mask & (image < threshold)


def frangi_response(image, mask, sigmas=FRANGI_SIGMAS):
    """Multi-scale Hessian vesselness (Frangi et al., 1998) of dark vessels."""
    response = frangi(image.astype(float) / 255, sigmas=sigmas, black_ridges=True)
    response[~mask] = 0
    return response
