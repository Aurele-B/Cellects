#!/usr/bin/env python3
"""
This script contains functions to compute and display confidence intervals using bootstrap.

Current functions include:
- bootstrap_mean
- plot_confidence_interval
"""

import numpy as np
from numpy.typing import NDArray
from typing import Tuple


def bootstrap_mean(values: NDArray, boot: int=10000) -> Tuple[np.float64, np.float64, np.float64]:
    means = np.zeros(boot)
    for it in np.arange(boot):
        new_values = np.random.choice(values, len(values))
        means[it] = np.mean(new_values)
    average = np.mean(values)
    c025 = np.quantile(means, 0.025)
    c975 = np.quantile(means, 0.975)
    return average, c025, c975

def plot_confidence_interval(x, values: NDArray, ax, color='#2187bb', horizontal_line_width: float=0.25) -> Tuple[np.float64, np.float64, np.float64]:
    m, c025, c975 = bootstrap_mean(values, 10000)
    left = x - horizontal_line_width / 2
    right = x + horizontal_line_width / 2
    top = c975
    bottom = c025
    ax.plot([x, x], [top, bottom], color=color)
    ax.plot([left, right], [top, top], color=color)
    ax.plot([left, right], [bottom, bottom], color=color)
    ax.plot(x, m, 'o', color="black")
    return m, c025, c975
