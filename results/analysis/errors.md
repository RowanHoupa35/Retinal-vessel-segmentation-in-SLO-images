# Error analysis (test images)

## Recall by annotated vessel width

Share of the annotated centreline covered by each method, pooled over the
test images; width is measured on the annotation at the centreline.

| width | share of centreline | Morphological pipeline | Frangi vesselness | Global threshold |
|---|---|---|---|---|
| 1 px | 0.15 | 0.52 | 0.45 | 0.37 |
| 2 px | 0.15 | 0.62 | 0.52 | 0.49 |
| 3 px | 0.31 | 0.80 | 0.65 | 0.68 |
| 4 px | 0.24 | 0.89 | 0.64 | 0.70 |
| ≥ 5 px | 0.14 | 0.92 | 0.46 | 0.73 |

## False positives of the pipeline

- 5588 predicted centreline pixels fall outside the annotation;
  23% lie within 2 px of an annotated vessel (alignment
  or width errors), 77% further away (structures absent from
  the annotation).
- Precision is 0.807; counting the near ones as correct would give
  0.851.

## Local contrast

Black top-hat of the raw image (disk radius 4), in gray levels, at
centreline pixels of each kind of structure. Thin vessels are 1 to 2 px wide.

| structure | pixels | median [q1, q3] |
|---|---|---|
| annotated thin vessel, detected | 5085 | 20 [14, 27] |
| annotated thin vessel, missed | 3801 | 8 [5, 13] |
| detected, not annotated | 4329 | 13 [10, 18] |
| background | 590815 | 3 [1, 6] |

## Two definitions of recall

The course recall is TP / (TP + FN) with TP counted on the predicted skeleton
and FN on the annotated one. The covered recall is the share of the annotated
skeleton inside the prediction. Length ratio: predicted skeleton inside the
annotation over annotated skeleton inside the prediction (mean over images).

| method | precision | course recall | covered recall | course f1 | covered f1 | length ratio |
|---|---|---|---|---|---|---|
| Morphological pipeline | 0.807 | 0.782 | 0.776 | 0.791 | 0.788 | 1.039 |
| Frangi vesselness | 0.643 | 0.647 | 0.571 | 0.641 | 0.601 | 1.381 |
| Global threshold | 0.299 | 0.566 | 0.623 | 0.382 | 0.395 | 0.790 |

## Sensitivity of the metric

Each annotation, deformed, scored against itself; mean over the 10 images.

| deformation | precision | recall | f1 | dice |
|---|---|---|---|---|
| annotation itself | 1.000 | 1.000 | 1.000 | 1.000 |
| dilated by 1 px | 0.984 | 1.000 | 0.992 | 0.816 |
| dilated by 2 px | 0.961 | 1.000 | 0.980 | 0.693 |
| dilated by 4 px | 0.889 | 1.000 | 0.941 | 0.529 |
| eroded by 1 px | 1.000 | 0.861 | 0.924 | 0.705 |
| shifted by 1 px | 0.989 | 0.937 | 0.962 | 0.844 |
| shifted by 2 px | 0.787 | 0.752 | 0.769 | 0.695 |

## Run time

segment() takes 0.65 s per 512 x 512 image (Apple M1 Max, Python 3.11.15).
