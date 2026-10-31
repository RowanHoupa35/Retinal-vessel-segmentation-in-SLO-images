# Leave-one-out cross-validation

Each image is scored by methods tuned (pipeline, references) or trained
(U-Net, 3 seeds averaged) on the 9 other images. Mean ± standard deviation
over the 10 images. Covered recall: share of the annotated skeleton inside
the segmentation, which does not depend on the predicted skeleton length.

| Method | Precision | Recall | F1 | Dice | Covered recall | Covered F1 |
|---|---|---|---|---|---|---|
| Global threshold | 0.286 ± 0.048 | 0.616 ± 0.104 | 0.384 ± 0.045 | 0.279 ± 0.040 | 0.676 ± 0.098 | 0.396 ± 0.045 |
| Frangi vesselness | 0.653 ± 0.100 | 0.592 ± 0.064 | 0.615 ± 0.048 | 0.487 ± 0.043 | 0.483 ± 0.091 | 0.546 ± 0.061 |
| Morphological pipeline | 0.845 ± 0.059 | 0.723 ± 0.060 | 0.777 ± 0.029 | 0.731 ± 0.026 | 0.714 ± 0.062 | 0.771 ± 0.031 |
| U-Net | 0.895 ± 0.035 | 0.855 ± 0.053 | 0.873 ± 0.022 | 0.812 ± 0.027 | 0.858 ± 0.051 | 0.874 ± 0.021 |

## Pipeline against U-Net, image by image (F1)

| image | Global threshold | Frangi vesselness | Morphological pipeline | U-Net | U-Net seed range |
|---|---|---|---|---|---|
| 01 | 0.359 | 0.600 | 0.772 | 0.874 | 0.871–0.880 |
| 02 | 0.417 | 0.670 | 0.816 | 0.880 | 0.876–0.881 |
| 03 | 0.350 | 0.643 | 0.791 | 0.873 | 0.871–0.874 |
| 08 | 0.439 | 0.661 | 0.780 | 0.901 | 0.898–0.904 |
| 21 | 0.359 | 0.534 | 0.723 | 0.824 | 0.820–0.830 |
| 26 | 0.400 | 0.562 | 0.747 | 0.856 | 0.847–0.864 |
| 28 | 0.403 | 0.603 | 0.746 | 0.867 | 0.866–0.870 |
| 32 | 0.452 | 0.675 | 0.788 | 0.879 | 0.876–0.885 |
| 37 | 0.350 | 0.579 | 0.797 | 0.901 | 0.898–0.902 |
| 48 | 0.312 | 0.621 | 0.805 | 0.877 | 0.875–0.880 |

- Pipeline better on 0 of 10 images; mean difference -0.097 (from -0.121 to -0.064); Wilcoxon signed-rank p = 0.002, sign test p = 0.002.
- With the covered recall: pipeline better on 0 images, mean difference -0.104, Wilcoxon p = 0.002.
- U-Net seed spread: mean standard deviation of F1 across seeds 0.004 per image.

## Recall by annotated vessel width

| width | Global threshold | Frangi vesselness | Morphological pipeline | U-Net |
|---|---|---|---|---|
| 1 px | 0.50 | 0.33 | 0.37 | 0.60 |
| 2 px | 0.56 | 0.43 | 0.54 | 0.74 |
| 3 px | 0.71 | 0.55 | 0.74 | 0.90 |
| 4 px | 0.75 | 0.55 | 0.86 | 0.95 |
| ≥ 5 px | 0.77 | 0.38 | 0.89 | 0.96 |

## Annotation style

Share of the structures detected by the pipeline that the U-Net also segments,
by local contrast (black top-hat of the raw image, radius 4, gray levels).
Annotated thin vessels are 1 to 2 px wide; "not annotated" are pipeline
centreline pixels more than 2 px from any annotated vessel.

| contrast | annotated thin vessel (rate, pixels) | not annotated (rate, pixels) |
|---|---|---|
| [0, 5) | 0.74 (22) | 0.08 (220) |
| [5, 10) | 0.93 (579) | 0.10 (838) |
| [10, 15) | 0.91 (1715) | 0.18 (1655) |
| [15, 20) | 0.91 (2121) | 0.22 (1566) |
| [20, 30) | 0.94 (2644) | 0.27 (1477) |
| [30, inf) | 0.97 (1451) | 0.34 (453) |

Overall: 0.93 of the annotated thin vessels, 0.21 of the unannotated structures.

False positives more than 2 px from any annotated vessel: Global threshold 88%, Frangi vesselness 90%, Morphological pipeline 74%, U-Net 70%.
