#!/usr/bin/env python3
"""2-D Sato and Frangi vesselness filters for grayscale images.

This module provides multiscale tubeness and vesselness filtering for
2-D grayscale images. It computes Gaussian Hessian derivatives, extracts
Hessian eigenvalues, and combines scale-specific responses by taking the
maximum over the requested scales.

The public API includes unmasked filters for full images and a masked
filter for arbitrary binary masks. The masked filter extends the image by
reflection across the mask boundary before computing the Hessian and
zeros the output outside the mask.

Functions
---------
sato_filter : Apply the 2-D Sato tubeness filter.
frangi_filter : Apply the 2-D Frangi vesselness filter.
masked_vessel_filters : Apply a Sato or Frangi filter inside a 2-D mask.

Notes
-----
Input images must be 2-D and all Gaussian scales must be positive. The
unmasked filters use reflective image boundaries. The masked filter uses
a reflective mask boundary and returns zero outside the mask.

The module depends on NumPy, SciPy, and Numba. Performance-critical
per-pixel helper computations are accelerated with Numba.
"""

import math
from numpy.typing import NDArray
import numpy as np
from scipy import ndimage
from numba import prange
from cellects.utils.decorators import njit


def _hessian(image, sigma, mode="reflect"):
    """
    Compute the 2-D Gaussian Hessian derivatives for one scale.

    Parameters
    ----------
    image : ndarray
        2-D grayscale image. Integer and floating-point arrays are converted
        to `float64`.
    sigma : float
        Gaussian scale in pixel units.
    mode : str, optional
        Boundary mode passed to the underlying Gaussian filter. The default
        is `"reflect"`.

    Returns
    -------
    Tuple[ndarray of float64, ndarray of float64, ndarray of float64]
        The Hessian components `hrr`, `hrc`, and `hcc`. Each component has
        the same shape as `image`.

    Notes
    -----
    The Gaussian sigma is divided by `sqrt(2.0)` before filtering.
    The returned `hrc` derivative is symmetric with the corresponding
    column-to-row derivative.

    Examples
    --------
    >>> image = [[0.0, 1.0, 1.0, 0.0]]
    >>> hrr, hrc, hcc = _hessian(image, 1.0)
    >>> print(hrr.shape)
    (1, 4)
    """
    image = np.asarray(image, dtype=np.float64)

    sigma = float(sigma)
    sigma_scaled = sigma / math.sqrt(2.0)

    truncate = 8.0 if sigma > 1.0 else 100.0

    common = dict(sigma=sigma_scaled, mode=mode, cval=0.0, truncate=truncate)

    grad_r = ndimage.gaussian_filter(image, order=(1, 0), **common)
    grad_c = ndimage.gaussian_filter(image, order=(0, 1), **common)
    hrr = ndimage.gaussian_filter(grad_r, order=(1, 0), **common)
    hrc = ndimage.gaussian_filter(grad_r, order=(0, 1), **common)
    hcc = ndimage.gaussian_filter(grad_c, order=(0, 1), **common)
    return hrr, hrc, hcc


@njit(parallel=True, nogil=True, cache=True, fastmath=True)
def _hessian_eigenvalues_2d(hrr, hrc, hcc):
    """Summary
    -------
    Compute the two eigenvalues of a symmetric 2-D Hessian in decreasing order.

    Parameters
    ----------
    hrr : ndarray of float64
        Second Gaussian derivative in the row direction.
    hrc : ndarray of float64
        Mixed row/column Gaussian derivative.
    hcc : ndarray of float64
        Second Gaussian derivative in the column direction.

    Returns
    -------
    Tuple[ndarray of float64, ndarray of float64]
        The eigenvalues `eig0` and `eig1`. `eig0` is the larger algebraic
        eigenvalue and `eig1` is the smaller algebraic eigenvalue.

    Notes
    -----
    This function uses Numba's @njit decorator for performance. The input
    arrays must be 2-D and have identical shapes.

    Examples
    --------
    >>> hrr = np.zeros((1, 2), dtype=np.float64)
    >>> hrc = np.zeros((1, 2), dtype=np.float64)
    >>> hcc = np.zeros((1, 2), dtype=np.float64)
    >>> eig0, eig1 = _hessian_eigenvalues_2d(hrr, hrc, hcc)
    >>> print(eig0.shape, eig1.shape)
    (1, 2) (1, 2)
    """
    h, w = hrr.shape

    eig0 = np.empty((h, w), dtype=np.float64)
    eig1 = np.empty((h, w), dtype=np.float64)

    for r in prange(h):
        for c in range(w):
            a = hrr[r, c]
            b = hrc[r, c]
            d = hcc[r, c]

            m = 0.5 * (a + d)
            s = math.sqrt(b * b + 0.25 * (a - d) * (a - d))

            eig0[r, c] = m + s
            eig1[r, c] = m - s

    return eig0, eig1


@njit(parallel=True, nogil=True, cache=True, fastmath=True)
def _sato_response(eig0, sigma):
    """Summary
    -------
    Compute the 2-D Sato response for one scale.

    Parameters
    ----------
    eig0 : ndarray of float64
        Largest algebraic Hessian eigenvalue.
    sigma : float
        Gaussian scale in pixel units.

    Returns
    -------
    ndarray of float64
        Non-negative Sato response for one scale.

    Notes
    -----
    This function uses Numba's @njit decorator for performance. Positive
    `eig0` values are scaled by `sigma` squared. Non-positive `eig0` values
    produce `0.0`.

    Examples
    --------
    >>> eig0 = np.array([[1.0, -1.0, 0.0, 2.0]], dtype=np.float64)
    >>> result = _sato_response(eig0, 2.0)
    >>> print(result[0, 0], result[0, 3])
    4.0 8.0
    """
    h, w = eig0.shape
    result = np.zeros((h, w), dtype=np.float64)

    s2 = sigma * sigma

    for r in prange(h):
        for c in range(w):
            value = eig0[r, c]

            if value > 0.0:
                result[r, c] = s2 * value

    return result


def sato_filter(greyscale_image, sigmas):
    """Summary
    -------
    Apply the 2-D Sato tubeness filter.

    Parameters
    ----------
    greyscale_image : ndarray
        2-D grayscale image. Integer and floating-point images are accepted.
    sigmas : list[float]
        Positive Gaussian scales in pixel units.

    Returns
    -------
    ndarray of float64
        Floating-point Sato response with the same shape as
        `greyscale_image`.

    Raises
    ------
    ValueError
        If `greyscale_image` is not a 2-D array.
        If `sigmas` is not a non-empty 1-D sequence.
        If any value in `sigmas` is not positive.

    Notes
    -----
    The filter targets dark tubular structures and uses `"reflect"` image
    boundary handling. It matches the scikit-image 0.26.0 2-D formulation
    with black ridges. The per-pixel helper computations use Numba-compiled
    functions for performance.

    Examples
    --------
    >>> image = [[0.0, 1.0, 1.0, 0.0]]
    >>> sigmas = [1.0, 2.0]
    >>> result = sato_filter(image, sigmas)
    >>> print(result.shape)
    (1, 4)
    """
    image = np.asarray(greyscale_image)

    if image.ndim != 2:
        raise ValueError("greyscale_image must be a 2-D array.")

    sigmas = np.asarray(sigmas, dtype=np.float64)

    if sigmas.ndim != 1 or sigmas.size == 0:
        raise ValueError("sigmas must be a non-empty 1-D sequence.")

    if np.any(sigmas <= 0):
        raise ValueError("all sigmas must be positive.")

    image = image.astype(np.float64, copy=False)

    result = np.zeros_like(image)

    for sigma in sigmas:
        hrr, hrc, hcc = _hessian(image, sigma, mode="reflect")
        eig0, _ = _hessian_eigenvalues_2d(hrr, hrc, hcc)
        response = _sato_response(eig0, sigma)
        np.maximum(result, response, out=result)

    return result


@njit(parallel=True, nogil=True, cache=True, fastmath=True)
def _frangi_response(eig0, eig1, alpha, beta, gamma):
    """Summary
    -------
    Compute the 2-D Frangi response for one scale.

    Parameters
    ----------
    eig0 : ndarray of float64
        Larger algebraic Hessian eigenvalue.
    eig1 : ndarray of float64
        Smaller algebraic Hessian eigenvalue.
    alpha : float
        Reserved for compatibility with the 3-D Frangi formulation. In the
        2-D response the plate term is `1.0`, so this parameter does not
        affect the result.
    beta : float
        Smoothing parameter for the ridge/blob ratio term.
    gamma : float
        Structuredness threshold. Must be positive.

    Returns
    -------
    ndarray of float64
        Frangi response in `[0.0, 1.0]` for one scale.

    Notes
    -----
    This function uses Numba's @njit decorator for performance. The ridge
    ratio uses the eigenvalue with the smaller absolute magnitude in the
    numerator.

    Examples
    --------
    >>> eig0 = np.zeros((1, 2), dtype=np.float64)
    >>> eig1 = np.zeros((1, 2), dtype=np.float64)
    >>> result = _frangi_response(eig0, eig1, 0.5, 0.5, 1.0)
    >>> print(result[0, 0])
    0.0
    """
    h, w = eig0.shape
    result = np.zeros((h, w), dtype=np.float64)

    beta2 = 2.0 * beta * beta
    gamma2 = 2.0 * gamma * gamma

    for r in prange(h):
        for c in range(w):
            e0 = eig0[r, c]
            e1 = eig1[r, c]

            if abs(e0) <= abs(e1):
                lambda1 = e0
                lambda2 = e1
            else:
                lambda1 = e1
                lambda2 = e0

            if lambda2 < 1e-10:
                lambda2 = 1e-10

            rb = abs(lambda1) / lambda2

            s2 = e0 * e0 + e1 * e1

            # In 2-D, r_a = infinity, so the plate term is exactly 1.
            blobness = math.exp(-(rb * rb) / beta2)

            structuredness = 1.0 - math.exp(-s2 / gamma2)
            result[r, c] = blobness * structuredness

    return result


def frangi_filter(greyscale_image, sigmas):
    """Summary
    -------
    Apply the 2-D Frangi vesselness filter.

    Parameters
    ----------
    greyscale_image : ndarray
        2-D grayscale image. Integer and floating-point images are accepted.
    sigmas : list[float]
        Positive Gaussian scales in pixel units.

    Returns
    -------
    ndarray of float64
        Floating-point Frangi response in `[0.0, 1.0]` with the same
        shape as `greyscale_image`.

    Raises
    ------
    ValueError
        If `greyscale_image` is not a 2-D array.
        If `sigmas` is not a non-empty 1-D sequence.
        If any value in `sigmas` is not positive.

    Notes
    -----
    The filter targets dark tubular structures and uses `"reflect"` image
    boundary handling. It matches the scikit-image 0.26.0 2-D formulation
    with black ridges. The structuredness parameter is chosen automatically
    from the largest Hessian norm of the first scale.

    Examples
    --------
    >>> image = [[0.0, 1.0, 1.0, 0.0]]
    >>> sigmas = [1.0, 2.0]
    >>> result = frangi_filter(image, sigmas)
    >>> print(result.shape)
    (1, 4)
    """
    image = np.asarray(greyscale_image)

    if image.ndim != 2:
        raise ValueError("greyscale_image must be a 2-D array.")

    sigmas = np.asarray(sigmas, dtype=np.float64)

    if sigmas.ndim != 1 or sigmas.size == 0:
        raise ValueError("sigmas must be a non-empty 1-D sequence.")

    if np.any(sigmas <= 0):
        raise ValueError("all sigmas must be positive.")

    image = image.astype(np.float64, copy=False)

    alpha = 0.5
    beta = 0.5

    result = np.zeros_like(image, dtype=np.float64)

    gamma = None

    for sigma in sigmas:
        hrr, hrc, hcc = _hessian(image, sigma, mode="reflect")

        eig0, eig1 = _hessian_eigenvalues_2d(hrr, hrc, hcc)

        if gamma is None:
            s = np.sqrt(eig0 * eig0 + eig1 * eig1)

            gamma = 0.5 * np.max(s)

            if gamma == 0.0:
                gamma = 1.0

        response = _frangi_response(eig0, eig1, alpha, beta, gamma)

        np.maximum(result, response, out=result)

    return result

@njit(parallel=True, nogil=True, cache=True, fastmath=True)
def _reflect_points(mask: NDArray[bool], boundary_y: NDArray[np.int32], boundary_x: NDArray[np.int32], distance, radius):
    """Summary
    -------
    Reflect outside-mask pixels across their nearest mask boundary.

    Parameters
    ----------
    mask : ndarray of bool
        2-D binary mask. `True` pixels are inside the mask.
    boundary_y : ndarray of int32
        Row index of the nearest mask boundary for each pixel.
    boundary_x : ndarray of int32
        Column index of the nearest mask boundary for each pixel.
    distance : ndarray of float
        Distance of each pixel from the nearest mask boundary.
    radius : int or float
        Maximum reflection distance in pixels. Pixels farther than this
        distance map to their nearest boundary.

    Returns
    -------
    Tuple[ndarray of int32, ndarray of int32]
        The row and column index arrays `map_y` and `map_x` used to sample
        the reflected image.

    Notes
    -----
    This function uses Numba's @njit decorator for performance. Pixels
    inside `mask` map to themselves. Repeated reflection is limited to 64
    iterations.

    Examples
    --------
    >>> mask = np.array([[False, True, True, False]], dtype=bool)
    >>> boundary_y = np.zeros((1, 4), dtype=np.int32)
    >>> boundary_x = np.arange(4, dtype=np.int32).reshape(1, 4)
    >>> distance = np.zeros((1, 4), dtype=np.float64)
    >>> map_y, map_x = _reflect_points(
    ...     mask, boundary_y, boundary_x, distance, 4.0)
    >>> print(map_y.shape, map_x.shape)
    (1, 4) (1, 4)
    """
    h, w = mask.shape

    map_y = np.empty((h, w), dtype=np.int32)
    map_x = np.empty((h, w), dtype=np.int32)

    for r in prange(h):
        for c in range(w):
            if mask[r, c]:
                map_y[r, c] = r
                map_x[r, c] = c
                continue

            if distance[r, c] > radius:
                map_y[r, c] = boundary_y[r, c]
                map_x[r, c] = boundary_x[r, c]
                continue

            y = np.int64(r)
            x = np.int64(c)

            for _ in range(64):
                by = boundary_y[y, x]
                bx = boundary_x[y, x]

                y = 2 * by - y
                x = 2 * bx - x

                if y < 0:
                    y = 0
                elif y >= h:
                    y = h - 1

                if x < 0:
                    x = 0
                elif x >= w:
                    x = w - 1

                if mask[y, x]:
                    break

            map_y[r, c] = y
            map_x[r, c] = x

    return map_y, map_x


def _reflect_mask_image(image, mask_bool, radius):
    """Summary
    -------
    Extend an image by reflection across an arbitrary binary mask.

    Parameters
    ----------
    image : ndarray
        2-D image to extend. Values inside `mask_bool` are preserved.
    mask_bool : ndarray of bool
        2-D binary mask. `True` pixels define the preserved region.
    radius : int or float
        Maximum reflection distance in pixels. Pixels farther than this
        distance from the mask boundary map to their nearest boundary.

    Returns
    -------
    ndarray of float64
        A copy of `image` converted to `float64`. Pixels outside
        `mask_bool` are replaced by reflected samples.

    Raises
    ------
    ValueError
        If `mask_bool` contains no `True` pixels.

    Notes
    -----
    The resulting image can be used to impose a reflective boundary
    condition for masked filtering.

    Examples
    --------
    >>> image = np.array([[0.0, 1.0, 2.0, 0.0]])
    >>> mask = np.array([[False, True, True, False]])
    >>> extended = _reflect_mask_image(image, mask, 8.0)
    >>> print(extended.shape)
    (1, 4)
    """

    if not np.any(mask_bool):
        raise ValueError("mask contains no non-zero pixels.")

    eroded = ndimage.binary_erosion(mask_bool, structure=np.ones((3, 3), dtype=bool), border_value=0)

    boundary = mask_bool & ~eroded

    distance, indices = ndimage.distance_transform_edt(~boundary, return_indices=True)

    boundary_y = np.ascontiguousarray(indices[0].astype(np.int32))
    boundary_x = np.ascontiguousarray(indices[1].astype(np.int32))

    map_y, map_x = _reflect_points(np.ascontiguousarray(mask_bool), boundary_y, boundary_x, distance, radius)

    extended = np.asarray(image, dtype=np.float64).copy()

    outside = ~mask_bool

    extended[outside] = image[map_y[outside], map_x[outside]]

    return extended


def masked_vessel_filters(greyscale_image, filter_name, sigmas, mask_bool):
    """Summary
    -------
    Apply a Sato or Frangi vesselness filter inside a 2-D mask.

    Parameters
    ----------
    greyscale_image : ndarray
        2-D grayscale image. Integer and floating-point images are accepted.
    filter_name : str
        Filter to apply. Must be `"Frangi"` or `"Sato"`.
    sigmas : list[float]
        Positive Gaussian scales in pixel units.
    mask_bool : ndarray of bool
        2-D binary mask. `True` pixels define the filtering domain.

    Returns
    -------
    ndarray of float64
        Floating-point vesselness response with the same shape as
        `greyscale_image`. Values outside `mask_bool` are `0.0`.

    Raises
    ------
    ValueError
        If `greyscale_image` is not a 2-D array.
        If `mask_bool` is not a 2-D array.
        If `greyscale_image` and `mask_bool` do not have identical shapes.
        If `filter_name` is not `"Frangi"` or `"Sato"`.
        If `sigmas` is not a non-empty 1-D sequence.
        If any value in `sigmas` is not positive.
        If `mask_bool` contains no `True` pixels.

    Notes
    -----
    Image borders use reflective boundary handling. The mask boundary is
    extended by reflection before computing the Gaussian Hessian. Repeated
    reflections are used for concave or thin mask regions. The reflection
    mapping uses Numba-compiled helpers for performance.

    Examples
    --------
    >>> image = [[0.0, 1.0, 1.0, 0.0]]
    >>> mask = [[False, True, True, False]]
    >>> result = masked_vessel_filters(image, "Frangi", [1.0, 2.0], mask)
    >>> print(result.shape)
    (1, 4)
    >>> print(result[0, 0])
    0.0
    """
    image = np.asarray(greyscale_image)
    mask_bool = np.asarray(mask_bool)

    if image.ndim != 2:
        raise ValueError("greyscale_image must be a 2-D array.")

    if mask_bool.ndim != 2:
        raise ValueError("mask must be a 2-D array.")

    if image.shape != mask_bool.shape:
        raise ValueError("greyscale_image and mask must have identical shapes.")

    if filter_name not in ("Frangi", "Sato"):
        raise ValueError('filter_name must be either "Frangi" or "Sato".')

    sigmas = np.asarray(sigmas, dtype=np.float64)

    if sigmas.ndim != 1 or sigmas.size == 0:
        raise ValueError("sigmas must be a non-empty 1-D sequence.")

    if np.any(sigmas <= 0):
        raise ValueError("all sigmas must be positive.")

    if not np.any(mask_bool):
        raise ValueError("mask contains no non-zero pixels.")

    image_float = image.astype(np.float64, copy=False)

    if np.any(sigmas <= 1.0):
        truncate = 100.0
    else:
        truncate = 8.0

    radius = np.ceil(truncate * float(np.max(sigmas)))

    extended = _reflect_mask_image(image_float, mask_bool, radius)

    if filter_name == "Frangi":
        result = frangi_filter(extended, sigmas)
    else:
        result = sato_filter(extended, sigmas)

    result[~mask_bool] = 0.0

    return result
