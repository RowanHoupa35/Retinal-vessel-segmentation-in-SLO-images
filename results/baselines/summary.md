# Reference methods

Mean ± standard deviation over images. Precision and recall are the
skeleton-based scores of the course; Dice is pixel-wise.

## Test set (02, 03, 32, 37, 48)

| method | param | precision | recall | f1 | dice |
|---|---|---|---|---|---|
| threshold (course T=80) | 80 | 0.290 ± 0.063 | 0.570 ± 0.097 | 0.377 ± 0.050 | 0.270 ± 0.057 |
| threshold (tuned) | 80 | 0.290 ± 0.063 | 0.570 ± 0.097 | 0.377 ± 0.050 | 0.270 ± 0.057 |
| frangi (tuned) | 0.07 | 0.613 ± 0.086 | 0.642 ± 0.033 | 0.624 ± 0.045 | 0.518 ± 0.037 |

## Tune set (01, 08, 21, 26, 28), where parameters were chosen

| method | param | precision | recall | f1 | dice |
|---|---|---|---|---|---|
| threshold (course T=80) | 80 | 0.293 ± 0.025 | 0.622 ± 0.067 | 0.398 ± 0.032 | 0.279 ± 0.026 |
| threshold (tuned) | 80 | 0.293 ± 0.025 | 0.622 ± 0.067 | 0.398 ± 0.032 | 0.279 ± 0.026 |
| frangi (tuned) | 0.07 | 0.599 ± 0.086 | 0.567 ± 0.040 | 0.581 ± 0.049 | 0.463 ± 0.026 |
