# Ablation study

Mean ± standard deviation over images. Precision and recall are the
skeleton-based scores of the course; Dice is pixel-wise. Δ F1 is the
change from the previous row (cumulative) or from the full pipeline
(remove-one).

### Reference methods (tune set)

| Step | Parameters (chosen on tune) | Precision | Recall | F1 | Dice |
|---|---|---|---|---|---|
| Global threshold | T = 80 | 0.294 ± 0.024 | 0.617 ± 0.066 | 0.398 ± 0.032 | 0.286 ± 0.026 |
| Frangi vesselness | kept 0.07 | 0.631 ± 0.093 | 0.572 ± 0.041 | 0.598 ± 0.051 | 0.474 ± 0.029 |

### Cumulative: one step added per row (tune set)

| Step | Parameters (chosen on tune) | Precision | Recall | F1 | Dice | Δ F1 |
|---|---|---|---|---|---|---|
| Black top-hat | r = 5, kept 0.09 | 0.651 ± 0.105 | 0.552 ± 0.039 | 0.594 ± 0.051 | 0.555 ± 0.048 |  |
| + median filter | median 5, r = 4, kept 0.14 | 0.727 ± 0.086 | 0.660 ± 0.034 | 0.690 ± 0.037 | 0.676 ± 0.038 | +0.095 |
| + line openings | median 5, r = 3, L = 21, kept 0.11 | 0.857 ± 0.040 | 0.642 ± 0.048 | 0.733 ± 0.040 | 0.695 ± 0.031 | +0.043 |
| + multi-scale | median 3, r = 2, 3, 4, 5, L = 15, kept 0.10 | 0.827 ± 0.072 | 0.685 ± 0.048 | 0.746 ± 0.025 | 0.703 ± 0.019 | +0.013 |
| + hysteresis | median 3, r = 2, 3, 4, 5, L = 15, kept 0.12, seeds 0.08 | 0.799 ± 0.083 | 0.715 ± 0.047 | 0.752 ± 0.032 | 0.719 ± 0.028 | +0.006 |
| + area opening | median 3, r = 2, 3, 4, 5, L = 15, kept 0.12, seeds 0.09, area ≥ 50 | 0.798 ± 0.085 | 0.722 ± 0.050 | 0.754 ± 0.031 | 0.720 ± 0.028 | +0.003 |

### Remove-one: full pipeline minus one step (tune set)

| Step | Parameters (chosen on tune) | Precision | Recall | F1 | Dice | Δ F1 |
|---|---|---|---|---|---|---|
| without median filter | r = 2, 3, 4, 5, L = 15, kept 0.09, seeds 0.09, area ≥ 20 | 0.783 ± 0.101 | 0.641 ± 0.041 | 0.702 ± 0.041 | 0.642 ± 0.029 | -0.053 |
| without line openings | median 3, r = 2, 3, 4, 5, kept 0.15, seeds 0.07, area ≥ 50 | 0.787 ± 0.062 | 0.674 ± 0.045 | 0.724 ± 0.035 | 0.682 ± 0.040 | -0.030 |
| single scale | median 3, r = 3, L = 15, kept 0.12, seeds 0.10, area ≥ 50 | 0.797 ± 0.076 | 0.694 ± 0.040 | 0.739 ± 0.033 | 0.699 ± 0.023 | -0.015 |
| without hysteresis | median 3, r = 2, 3, 4, 5, L = 15, kept 0.12, area ≥ 50 | 0.792 ± 0.081 | 0.724 ± 0.048 | 0.753 ± 0.030 | 0.720 ± 0.028 | -0.001 |
| without area opening | median 3, r = 2, 3, 4, 5, L = 15, kept 0.12, seeds 0.08 | 0.799 ± 0.083 | 0.715 ± 0.047 | 0.752 ± 0.032 | 0.719 ± 0.028 | -0.003 |

### Reference methods (test set)

| Step | Parameters (chosen on tune) | Precision | Recall | F1 | Dice | F1 on tune |
|---|---|---|---|---|---|---|
| Global threshold | T = 80 | 0.299 ± 0.073 | 0.566 ± 0.095 | 0.382 ± 0.056 | 0.278 ± 0.059 | 0.398 ± 0.032 |
| Frangi vesselness | kept 0.07 | 0.643 ± 0.091 | 0.647 ± 0.033 | 0.641 ± 0.044 | 0.528 ± 0.039 | 0.598 ± 0.051 |

### Cumulative: one step added per row (test set)

| Step | Parameters (chosen on tune) | Precision | Recall | F1 | Dice | F1 on tune |
|---|---|---|---|---|---|---|
| Black top-hat | r = 5, kept 0.09 | 0.682 ± 0.090 | 0.630 ± 0.059 | 0.650 ± 0.046 | 0.615 ± 0.036 | 0.594 ± 0.051 |
| + median filter | median 5, r = 4, kept 0.14 | 0.687 ± 0.080 | 0.731 ± 0.064 | 0.703 ± 0.022 | 0.690 ± 0.017 | 0.690 ± 0.037 |
| + line openings | median 5, r = 3, L = 21, kept 0.11 | 0.857 ± 0.034 | 0.694 ± 0.055 | 0.766 ± 0.042 | 0.718 ± 0.037 | 0.733 ± 0.040 |
| + multi-scale | median 3, r = 2, 3, 4, 5, L = 15, kept 0.10 | 0.835 ± 0.060 | 0.747 ± 0.057 | 0.786 ± 0.016 | 0.738 ± 0.020 | 0.746 ± 0.025 |
| + hysteresis | median 3, r = 2, 3, 4, 5, L = 15, kept 0.12, seeds 0.08 | 0.803 ± 0.065 | 0.777 ± 0.050 | 0.786 ± 0.016 | 0.747 ± 0.013 | 0.752 ± 0.032 |
| + area opening | median 3, r = 2, 3, 4, 5, L = 15, kept 0.12, seeds 0.09, area ≥ 50 | 0.807 ± 0.063 | 0.782 ± 0.051 | 0.791 ± 0.016 | 0.749 ± 0.013 | 0.754 ± 0.031 |

### Remove-one: full pipeline minus one step (test set)

| Step | Parameters (chosen on tune) | Precision | Recall | F1 | Dice | F1 on tune |
|---|---|---|---|---|---|---|
| without median filter | r = 2, 3, 4, 5, L = 15, kept 0.09, seeds 0.09, area ≥ 20 | 0.823 ± 0.071 | 0.716 ± 0.053 | 0.763 ± 0.032 | 0.703 ± 0.024 | 0.702 ± 0.041 |
| without line openings | median 3, r = 2, 3, 4, 5, kept 0.15, seeds 0.07, area ≥ 50 | 0.805 ± 0.067 | 0.747 ± 0.040 | 0.773 ± 0.033 | 0.722 ± 0.019 | 0.724 ± 0.035 |
| single scale | median 3, r = 3, L = 15, kept 0.12, seeds 0.10, area ≥ 50 | 0.812 ± 0.074 | 0.752 ± 0.057 | 0.776 ± 0.018 | 0.730 ± 0.019 | 0.739 ± 0.033 |
| without hysteresis | median 3, r = 2, 3, 4, 5, L = 15, kept 0.12, area ≥ 50 | 0.804 ± 0.066 | 0.785 ± 0.052 | 0.792 ± 0.018 | 0.750 ± 0.013 | 0.753 ± 0.030 |
| without area opening | median 3, r = 2, 3, 4, 5, L = 15, kept 0.12, seeds 0.08 | 0.803 ± 0.065 | 0.777 ± 0.050 | 0.786 ± 0.016 | 0.747 ± 0.013 | 0.752 ± 0.032 |
