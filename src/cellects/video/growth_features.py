
"""
"""

from collections import namedtuple
from typing import Tuple
import pandas as pd
import numpy as np
from numpy.typing import NDArray
from scipy import stats
from numba import njit, prange
from numba.typed import List

# Public output container
default_features = {"exp_intercept": pd.NA,               # intercept of best exponential fit (log‑scale)
        "exp_growth_rate_mm2s": pd.NA,        # slope of best exponential fit
        "exp_start": pd.NA,                   # start time (min) of that exponential window
        "exp_end": pd.NA,                     # end   time (min) of that exponential window
        "exp_r_squared": pd.NA,               # Maximum R² among exponential windows
        "lin_intercept": pd.NA,               # intercept of best linear fit
        "lin_growth_rate_mm2s": pd.NA,        # slope of best linear fit
        "lin_start": pd.NA,                   # start time (min) of that linear window
        "lin_end": pd.NA,                     # end   time (min) of that linear window
        "lin_r_squared": pd.NA,               # Maximum R² among linear windows
        "growth_rupture_time_min": pd.NA,
        "growth_rupture_surface_mm2": pd.NA}

# Low‑level helpers (Numba‑compatible)
@njit(inline='always')
def _linregress(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
    """Plain‑Python/Numba implementation of the textbook OLS formulas.
       Returns (slope, intercept, Pearson r)."""
    n = x.shape[0]

    sx = np.sum(x)
    sy = np.sum(y)
    sxy = np.sum(x * y)
    sxx = np.sum(x * x)
    syy = np.sum(y * y)

    denom = n * sxx - sx * sx
    if denom == 0.0:
        return 0.0, sy / n if n > 0 else 0.0, 0.0

    slope = (n * sxy - sx * sy) / denom
    intercept = (sy - slope * sx) / n

    r_num = n * sxy - sx * sy
    r_den = np.sqrt((n * sxx - sx * sx) * (n * syy - sy * sy))
    r = r_num / r_den if r_den != 0.0 else 0.0
    return slope, intercept, r


@njit
def _cluster_means(y: np.ndarray, cluster_len: int) -> np.ndarray:
    """Mean of each non‑overlapping block of length `cluster_len`."""
    n = y.shape[0]
    n_clusters = n // cluster_len
    if n_clusters == 0:
        return np.empty(0, dtype=np.float64)
    out = np.empty(n_clusters, dtype=np.float64)
    for c in range(n_clusters):
        start = c * cluster_len
        stop = start + cluster_len
        out[c] = np.mean(y[start:stop])
    return out


@njit
def _slope_shifts(
    y: np.ndarray,
    cluster_len: int,
    shape: int,
    mean_diff: float,
) -> np.ndarray:
    """
    Detect frames where the sign of the clustered‑mean differences
    changes. Returned indices are **1‑based** frame numbers, multiplied by `cluster_len` as in R.
    """
    n = y.shape[0]
    n_clusters = n // cluster_len
    if n_clusters < 2:                     # need at least 2 clusters
        return np.empty(0, dtype=np.int64)

    clust_means = np.empty(n_clusters, dtype=np.float64)
    for c in range(n_clusters):
        s = c * cluster_len
        clust_means[c] = np.mean(y[s:s + cluster_len])

    diff_len = n_clusters - 1
    diffs = np.empty(diff_len, dtype=np.float64)
    for i in range(diff_len):
        diffs[i] = clust_means[i + 1] - clust_means[i]

    if shape == 1:          # increasing
        cond = diffs < mean_diff
    else:                   # decreasing
        cond = diffs > mean_diff

    idx = np.where(cond)[0]               # 0‑based cluster indices
    if idx.size == 0:
        return np.empty(0, dtype=np.int64)

    out = np.empty(idx.size, dtype=np.int64)
    for k in range(idx.size):
        out[k] = (idx[k] + 1) * cluster_len   # 1‑based × length  ← R behaviour
    return out


@njit(parallel=True)
def _fill_r_squared_matrices(
    y: np.ndarray,
    time_step: float,
    max_start: int,
    max_exp_start: int,
    max_end: int,
    shape_flag: int,
    max_y: float,
    maximal_variation: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build two (n × n) matrices containing the R‑squared of every
    admissible regression window.

    All indices **inside this function are 0‑based**, exactly as NumPy
    uses.  The calling code passes `max_start`, `max_exp_start`,
    `max_end` already converted to 0‑based indices.
    """
    n = y.shape[0]
    exp_r_squared = np.zeros((n, n), dtype=np.float64)
    lin_r_squared = np.zeros((n, n), dtype=np.float64)

    # Window must show “enough” change to be considered (½ * amplitude)
    change_thr = 0.5 * maximal_variation

    for i in prange(max_start + 1):                # i ∈ [0, max_start]
        for j in range(max_start, max_end + 1):    # j ∈ [max_start, max_end]
            if j < i:
                continue

            # 1) enough absolute variation?
            if np.abs(y[j] - y[i]) <= change_thr:
                continue

            # -----------------------------------------------------------------
            # Build the time vector for the (i…j) window
            #   R: seq((frame_i-1)*time_step, (frame_j-1)*time_step, time_step)
            #   Python 0‑based → (i … j) * time_step
            # -----------------------------------------------------------------
            win_len = j - i + 1
            t = np.empty(win_len, dtype=np.float64)
            for k in range(win_len):
                t[k] = (i + k) * time_step

            # -----------------------------------------------------------------
            # Linear regression on raw y (always evaluated)
            # -----------------------------------------------------------------
            y_lin = y[i:j + 1]
            _, _, r_lin = _linregress(t, y_lin)
            lin_r_squared[i, j] = r_lin * r_lin

            # -----------------------------------------------------------------
            # Exponential regression (log‑linear) – only if i < max_exp_start
            # -----------------------------------------------------------------
            if i < max_exp_start:
                if shape_flag == -1:                     # decreasing curve
                    y_exp = -y[i:j + 1] + 2.0 * max_y
                else:                                    # increasing curve
                    y_exp = y[i:j + 1]

                # protect against non‑positive data (should be impossible after
                # preprocessing, but we are defensive)
                eps = 1e-12
                log_y = np.empty(win_len, dtype=np.float64)
                for k in range(win_len):
                    val = y_exp[k]
                    if val <= 0.0:
                        val = eps
                    log_y[k] = np.log(val)

                _, _, r_exp = _linregress(t, log_y)
                exp_r_squared[i, j] = r_exp * r_exp

    return exp_r_squared, lin_r_squared


def find_growth_features(
    y: NDArray,
    time_step: float,
    first_growth: float,
    first_frame: int = 1,
) -> dict:
    """
    Extract some growth descriptors from a time zeries.

    Parameters
    ----------
    y : NDArray
        Raw signal (one measurement per frame). A NumPy 1‑D array.
    time_step : float
        Temporal spacing between successive frames (the R script calls it
        ``time_step`` and later multiplies it by frame indices).
    first_growth : float
        Threshold used to locate the *pseudo‑peak* (the index where the
        curve first rises above ``y[2] + first_growth`` for increasing
        curves, or falls below it for decreasing curves).
        If first_growth is negative the curve is treated as decreasing;
        otherwise it is treated as increasing.
    first_frame : int, default 1
        1‑based index of the frame where analysis should start.
         Frames before this are discarded.

    Returns
    -------
    GrowthFeatures
        A namedtuple with 14 fields; the order matches exactly the column
        order of the data.frame produced by the R implementation.
        If the data are insufficient for regression the numeric fields are
        ``np.nan`` and the two rupture‑related fields contain the string
        ``"censored"`` (again matching the R behaviour).

    Notes
    -----
    * The wrapper calls a handful of internal, JIT‑compiled helpers:
      `_slope_shifts`, `_fill_r_squared_matrices`, and `_linregress`.
      Those helpers must be present in the same module – they are defined
      in the first part of *curve_features.py*.
    * the return type is a lightweight ``namedtuple``.
    """

    if first_frame < 1:
        raise ValueError("first_frame is 1‑based and must be >= 1")
    y = y[first_frame - 1 :]                     # 0‑based slice

    # Decide whether the curve is increasing or decreasing
    if first_growth < 0:
        shape = "decreasing"
        shape_flag = -1
    else:
        shape = "increasing"
        shape_flag = 1

    # Remove a leading zero
    if len(y) > 1 and y[0] == 0.0:
        y = y[1:]

    # Replace interior zeros by the smallest non‑zero value
    zeros = y == 0.0
    if np.any(zeros):
        positive = y[~zeros]
        if positive.size == 0:               # completely zero vector
            raise ValueError("All values are zero; cannot replace zeros")
        smallest = np.min(positive)
        y[zeros] = smallest

    n_frame = len(y)
    if n_frame == 0:
        # Degenerate input – nothing to analyze
        return default_features

    # Basic statistics needed later
    max_y = np.max(y)
    min_y = np.min(y)
    maximal_variation = np.abs(max_y - min_y)          # abs(max-min)
    mean_diff = np.mean(np.diff(y))

    # Detect slope‑shift frames (possible rupture points)
    # _slope_shifts returns **1‑based* frame numbers multiplied by the cluster length.
    cluster_len = 5
    shift_frames_onebased = _slope_shifts(y, cluster_len, shape_flag, mean_diff)
    shift_frames = shift_frames_onebased - 1          # now 0‑based

    # Determine where (if ever) a “rupture” occurs
    # A rupture is the *first* slope‑shift that happens after the signal
    # has moved through 90 % of its total amplitude.
    if shape == "increasing":
        high_thr = y[1] + 0.9 * maximal_variation
        high_idx = np.where(y > high_thr)[0]
    else:
        high_thr = y[1] - 0.9 * maximal_variation
        high_idx = np.where(y < high_thr)[0]

    # Intersection of the two sets gives candidate rupture frames
    cand = np.intersect1d(shift_frames, high_idx, assume_unique=True)
    if cand.size:
        rupture_time_idx = int(cand[0])               # 0‑based
        rupture_time_min = time_step * rupture_time_idx
        rupture_value = float(y[rupture_time_idx])
    else:
        rupture_time_min = "censored"
        rupture_value = "censored"
        rupture_time_idx = n_frame - 1               # we will search up to the end
    max_end = rupture_time_idx                       # upper bound for the regression loops

    # Determine the “growth” windows that bound the exhaustive search
    if shape == "increasing":
        max_start = int(np.where(y > (y[1] + 0.1 * maximal_variation))[0][0])
        max_exp_start = int(np.where(y > (y[1] + 0.01 * maximal_variation))[0][0])
    else:
        max_start = int(np.where(y < (y[1] - 0.1 * maximal_variation))[0][0])
        max_exp_start = int(np.where(y < (y[1] - 0.01 * maximal_variation))[0][0])

    # Guard against pathological series where the thresholds are never crossed
    if max_start < 0 or max_exp_start < 0:
        return default_features

    # Fill the two R‑squared matrices (the O(N²) part)
    exp_r2_mat, lin_r2_mat = _fill_r_squared_matrices(
        y,
        time_step,
        max_start,
        max_exp_start,
        max_end,
        shape_flag,
        max_y,
        maximal_variation,
    )

    # Locate the windows with maximal R² for each model
    exp_flat_idx = int(np.argmax(exp_r2_mat))
    exp_i, exp_j = np.unravel_index(exp_flat_idx, exp_r2_mat.shape)
    best_exp_r2 = float(exp_r2_mat[exp_i, exp_j])

    lin_flat_idx = int(np.argmax(lin_r2_mat))
    lin_i, lin_j = np.unravel_index(lin_flat_idx, lin_r2_mat.shape)
    best_lin_r2 = float(lin_r2_mat[lin_i, lin_j])

    # Perform the *final* regressions on the selected windows
    # Exponential regression (log‑linear)
    if shape == "decreasing":
        # Transform the data exactly as the R code does:
        # y_neg_exp = -y + 2*max_y   → then take log
        y_exp_window = -y[exp_i : exp_j + 1] + 2.0 * max_y
    else:
        y_exp_window = y[exp_i : exp_j + 1]

    # Time vector for the selected exponential window
    t_exp = np.arange(exp_i, exp_j + 1) * time_step

    # Linear regression on log(y) – SciPy gives slope, intercept,
    # rvalue, pvalue, stderr; we need only the first two.
    log_y = np.log(y_exp_window)
    exp_reg = stats.linregress(t_exp, log_y)
    exp_intercept = exp_reg.intercept          # this is the log‑intercept
    exp_slope = exp_reg.slope                  # growth rate (per minute)

    # Linear regression on raw data
    t_lin = np.arange(lin_i, lin_j + 1) * time_step
    lin_reg = stats.linregress(t_lin, y[lin_i : lin_j + 1])
    lin_intercept = lin_reg.intercept
    lin_slope = lin_reg.slope

    # Assemble the final output (exactly the same order as R)
    features = {"exp_intercept": exp_intercept,  # intercept of best exponential fit (log‑scale)
                "exp_growth_rate_mm2s": exp_slope,  # slope of best exponential fit
                "exp_start": time_step * exp_i,  # start time (min) of that exponential window
                "exp_end": time_step * exp_j,  # end   time (min) of that exponential window
                "exp_r_squared": best_exp_r2,  # Maximum R² among exponential windows
                "lin_intercept": lin_intercept,  # intercept of best linear fit
                "lin_growth_rate_mm2s": lin_slope,  # slope of best linear fit
                "lin_start": time_step * lin_i,  # start time (min) of that linear window
                "lin_end": time_step * lin_j,  # end   time (min) of that linear window
                "lin_r_squared": best_lin_r2,  # Maximum R² among linear windows
                "growth_rupture_time_min": rupture_time_min,
                "growth_rupture_surface_mm2": rupture_value}
    return features