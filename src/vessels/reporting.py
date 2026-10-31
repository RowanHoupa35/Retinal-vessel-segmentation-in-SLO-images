"""Formatting of result tables."""

import numpy as np


def mean_std(values):
    return f"{np.mean(values):.3f} ± {np.std(values, ddof=1):.3f}"


def to_markdown(df):
    def cell(v):
        return f"{v:g}" if isinstance(v, float) else str(v)

    lines = ["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
    lines += ["| " + " | ".join(cell(v) for v in row) + " |" for row in df.to_numpy()]
    return "\n".join(lines)
