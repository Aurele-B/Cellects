#!/usr/bin/env python3
"""
2-D image filters with Numba-accelerated helper kernels.

This module provides 2-D grayscale image filtering functions for
sharpening, smoothing, edge detection, median filtering, Hessian-based
vesselness, and masked filtering. Public functions accept integer or
floating-point images and return NumPy arrays.

The filters include classical linear filters, scale-space tubeness and
vesselness filters, a frequency-domain Butterworth filter, and helper
functions for masked Sato or Frangi filtering.

Functions
---------
sharpen_filter : Apply a 3x3 sharpen kernel.
mexican_hat_filter : Apply a 5x5 Mexican-hat kernel.
gaussian_filter : Apply Gaussian smoothing.
butterworth_filter : Apply a Butterworth filter.
farid_filter : Compute Farid edge magnitude.
hessian_filter : Compute a Hessian vesselness response.
laplace_filter : Apply Laplacian convolution.
median_filter : Apply median filtering.
meijering_filter : Compute a Meijering vesselness response.
prewitt_filter : Compute Prewitt edge magnitude.
roberts_filter : Compute Roberts edge magnitude.
scharr_filter : Compute Scharr edge magnitude.
sobel_filter : Compute Sobel edge magnitude.
sato_filter : Apply the Sato tubeness filter.
frangi_filter : Apply the Frangi vesselness filter.
masked_vessel_filters : Apply Sato or Frangi inside a mask.

Notes
-----
Boundary handling is specific to each filter. Some functions use
Numba-compiled loops for per-pixel Hessian and mask reflection work.
"""
from numpy.typing import NDArray
import math
import cv2
import numpy as np
from cellects.utils.decorators import njit
from numba import prange
from scipy import ndimage


__all__ = [
    "sharpen_filter",
    "mexican_hat_filter",
    "gaussian_filter",
    "butterworth_filter",
    "farid_filter",
    "hessian_filter",
    "laplace_filter",
    "median_filter",
    "meijering_filter",
    "prewitt_filter",
    "roberts_filter",
    "scharr_filter",
    "sobel_filter",
    "sato_filter",
    "frangi_filter",
]

def sharpen_filter(image: np.ndarray) -> np.ndarray:
    """
    Summary
    -------
    Apply a 3x3 sharpen filter to a 2-D image.

    Parameters
    ----------
    image : ndarray
        2-D grayscale image.

    Returns
    -------
    ndarray
        Sharpened image with the same shape as `image`.

    Examples
    --------
    >>> result = sharpen_filter(np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.uint8))
    >>> print(result.shape)
    # (2, 2)
    """
    return cv2.filter2D(image, -1, np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]]))

def mexican_hat_filter(image: np.ndarray) -> np.ndarray:
    """
    Summary
    -------
    Apply a 5x5 Mexican-hat filter to a 2-D image.

    Parameters
    ----------
    image : ndarray
        2-D grayscale image.

    Returns
    -------
    ndarray
        Filtered image with the same shape as `image`.

    Examples
    --------
    >>> result = mexican_hat_filter(np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.uint8))
    >>> print(result.shape)
    # (2, 2)
    """
    return cv2.filter2D(image, -1, np.array(
            [[0, 0, -1, 0, 0], [0, -1, -2, -1, 0], [-1, -2, 16, -2, -1], [0, -1, -2, -1, 0], [0, 0, -1, 0, 0]]))

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

def _float_image(image):
    image = np.asarray(image)
    if image.ndim != 2:
        raise ValueError("image must be a 2-D array.")
    return np.ascontiguousarray(image, dtype=np.float64)


def _image01(image):
    image = np.asarray(image)
    if image.ndim != 2:
        raise ValueError("image must be a 2-D array.")

    if image.dtype.kind == "u":
        return np.ascontiguousarray(
            image.astype(np.float64) / np.iinfo(image.dtype).max
        )
    if image.dtype.kind == "i":
        info = np.iinfo(image.dtype)
        scale = max(abs(info.min), abs(info.max))
        return np.ascontiguousarray(image.astype(np.float64) / scale)

    return np.ascontiguousarray(image, dtype=np.float64)


def _sigmas(sigmas):
    sigmas = np.asarray(sigmas, dtype=np.float64)
    if sigmas.ndim == 0:
        sigmas = sigmas.reshape(1)
    if sigmas.ndim != 1 or sigmas.size == 0:
        raise ValueError("sigmas must be a non-empty 1-D sequence.")
    if np.any(sigmas <= 0):
        raise ValueError("all sigmas must be positive.")
    return sigmas


def gaussian_filter(image, sigma=1.0, preserve_range=False, mode="nearest", cval=0.0, truncate=4.0):
    """
    Summary
    -------
    Apply Gaussian smoothing to a 2-D image.

    Extended Description
    --------------------
    When `preserve_range` is ``False``, unsigned integer input is scaled to
    [0.0, 1.0], signed integer input is scaled by the largest absolute
    endpoint, and floating-point input is unchanged. When `preserve_range`
    is ``True``, integer input is converted to `float64` without range
    scaling.

    Parameters
    ----------
    image : array_like
        2-D grayscale image.
    sigma : float
        Gaussian standard deviation.
    preserve_range : bool
        If ``True``, integer input is not rescaled; if ``False``, integer
        input is rescaled before filtering.
    mode : str
        Boundary mode passed to the Gaussian filter.
    cval : float
        Constant value used for boundary extension.
    truncate : float
        Gaussian kernel truncation parameter.

    Returns
    -------
    ndarray of float64
        Smoothed image with the same shape as `image`.

    Raises
    ------
    ValueError
        If `image` is not a 2-D array.

    Examples
    --------
    >>> result = gaussian_filter(np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.uint8), 1.0)
    >>> print(result.shape)
    # (2, 2)
    """
    image = _float_image(image) if preserve_range else _image01(image)
    return np.ascontiguousarray(
        ndimage.gaussian_filter(image, sigma=sigma, mode=mode, cval=cval, truncate=truncate),
        dtype=np.float64,
    )


def butterworth_filter(image, cutoff_frequency_ratio=0.005, high_pass=False, order=2.0, squared_butterworth=True, npad=0, preserve_range=False):
    """
    Summary
    -------
    Apply a Butterworth low-pass or high-pass filter to a 2-D image.

    Extended Description
    --------------------
    The filter is applied in the frequency domain. Integer input is
    converted to `float64` without range scaling. `preserve_range` is
    accepted for interface compatibility but does not affect this function.

    Parameters
    ----------
    image : array_like
        2-D grayscale image.
    cutoff_frequency_ratio : float
        Cutoff ratio relative to each image dimension. Must be greater than
        0.0 and less than or equal to 0.5.
    high_pass : bool
        If ``True``, apply a high-pass response; if ``False``, apply a
        low-pass response.
    order : float
        Butterworth filter order.
    squared_butterworth : bool
        If ``True``, use the squared Butterworth response; if ``False``, use
        the square-root response.
    npad : int
        Number of edge pixels to pad before filtering and crop after
        filtering.
    preserve_range : bool
        Accepted for compatibility; this function does not change behavior.

    Returns
    -------
    ndarray of float64
        Filtered image with the same shape as `image`.

    Raises
    ------
    ValueError
        If `cutoff_frequency_ratio` is not in (0.0, 0.5], if `order` is
        not positive, if `npad` is negative, or if `image` is not a 2-D
        array.

    Examples
    --------
    >>> result = butterworth_filter(np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.uint8))
    >>> print(result.shape)
    # (2, 2)
    """

    image = _float_image(image)
    cutoff_frequency_ratio = float(cutoff_frequency_ratio)
    order = float(order)
    npad = int(npad)

    if cutoff_frequency_ratio <= 0 or cutoff_frequency_ratio > 0.5:
        raise ValueError("cutoff_frequency_ratio must be in (0, 0.5].")
    if order <= 0:
        raise ValueError("order must be positive.")
    if npad < 0:
        raise ValueError("npad must be >= 0.")

    if npad:
        crop = (slice(npad, -npad), slice(npad, -npad))
        image = np.pad(image, npad, mode="edge")
    else:
        crop = None

    h, w = image.shape

    fy = np.arange(-(h - 1) // 2, (h - 1) // 2 + 1, dtype=np.float64) / (h * cutoff_frequency_ratio)
    fx = np.arange(-(w - 1) // 2, (w - 1) // 2 + 1, dtype=np.float64) / (w * cutoff_frequency_ratio)

    fy = np.fft.ifftshift(fy * fy)
    fx = np.fft.ifftshift(fx * fx)

    q2 = (fy[:, None] + fx[None, :]) ** order

    if high_pass:
        response = q2 / (1.0 + q2)
    else:
        response = 1.0 / (1.0 + q2)

    if not squared_butterworth:
        response = np.sqrt(response)

    result = np.fft.irfftn(
        np.fft.rfftn(image) * response[:, :w // 2 + 1],
        s=image.shape,
        axes=(0, 1),
    )

    if crop is not None:
        result = result[crop]

    return np.ascontiguousarray(result, dtype=np.float64)


_SOBEL_E = np.array([-1.0, 0.0, 1.0])
_SOBEL_S = np.array([1.0, 2.0, 1.0]) / 4.0

_PREWITT_E = np.array([-1.0, 0.0, 1.0])
_PREWITT_S = np.ones(3) / 3.0

_SCHARR_E = np.array([-1.0, 0.0, 1.0])
_SCHARR_S = np.array([3.0, 10.0, 3.0]) / 16.0

_FARID_E = np.array([
    0.109603762960254,
    0.276690988455557,
    0.0,
    -0.276690988455557,
    -0.109603762960254,
])

_FARID_S = np.array([
    0.0376593171958126,
    0.249153396177344,
    0.426374573253687,
    0.249153396177344,
    0.0376593171958126,
])


def _edge_filter(image, edge, smooth, mode, cval):
    image = _image01(image)

    k0 = np.outer(edge, smooth)
    k1 = k0.T

    a = ndimage.convolve(image, k0, mode=mode, cval=float(cval))
    b = ndimage.convolve(image, k1, mode=mode, cval=float(cval))

    return np.ascontiguousarray(
        np.sqrt(a * a + b * b) / math.sqrt(2.0),
        dtype=np.float64,
    )


def sobel_filter(image, mode="reflect", cval=0.0):
    """
    Summary
    -------
    Compute Sobel edge magnitude for a 2-D image.

    Parameters
    ----------
    image : array_like
        2-D grayscale image.
    mode : str
        Boundary mode used for convolution.
    cval : float
        Constant value used for boundary extension.

    Returns
    -------
    ndarray of float64
        Edge magnitude with the same shape as `image`.

    Raises
    ------
    ValueError
        If `image` is not a 2-D array.

    Notes
    -----
    Unsigned integer input is scaled to [0.0, 1.0]; signed integer input is
    scaled by the largest absolute endpoint. The two directional responses
    are combined and divided by sqrt(2).

    Examples
    --------
    >>> result = sobel_filter(np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.uint8)
    >>> print(result.shape)
    # (2, 2)
    """
    return _edge_filter(image, _SOBEL_E, _SOBEL_S, mode, cval)


def prewitt_filter(image, mode="reflect", cval=0.0):
    """
    Summary
    -------
    Compute Prewitt edge magnitude for a 2-D image.

    Parameters
    ----------
    image : array_like
        2-D grayscale image.
    mode : str
        Boundary mode used for convolution.
    cval : float
        Constant value used for boundary extension.

    Returns
    -------
    ndarray of float64
        Edge magnitude with the same shape as `image`.

    Raises
    ------
    ValueError
        If `image` is not a 2-D array.

    Notes
    -----
    Unsigned integer input is scaled to [0.0, 1.0]; signed integer input is
    scaled by the largest absolute endpoint. The two directional responses
    are combined and divided by sqrt(2).

    Examples
    --------
    >>> result = prewitt_filter(np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.uint8)
    >>> print(result.shape)
    # (2, 2)
    """
    return _edge_filter(image, _PREWITT_E, _PREWITT_S, mode, cval)


def scharr_filter(image, mode="reflect", cval=0.0):
    """
    Summary
    -------
    Compute Scharr edge magnitude for a 2-D image.

    Parameters
    ----------
    image : array_like
        2-D grayscale image.
    mode : str
        Boundary mode used for convolution.
    cval : float
        Constant value used for boundary extension.

    Returns
    -------
    ndarray of float64
        Edge magnitude with the same shape as `image`.

    Raises
    ------
    ValueError
        If `image` is not a 2-D array.

    Notes
    -----
    Unsigned integer input is scaled to [0.0, 1.0]; signed integer input is
    scaled by the largest absolute endpoint. The two directional responses
    are combined and divided by sqrt(2).

    Examples
    --------
    >>> result = scharr_filter(np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.uint8)
    >>> print(result.shape)
    # (2, 2)
    """
    return _edge_filter(image, _SCHARR_E, _SCHARR_S, mode, cval)


def farid_filter(image, mode="reflect", cval=0.0):
    """
    Summary
    -------
    Compute Farid edge magnitude for a 2-D image.

    Parameters
    ----------
    image : array_like
        2-D grayscale image.
    mode : str
        Boundary mode used for convolution.
    cval : float
        Constant value used for boundary extension.

    Returns
    -------
    ndarray of float64
        Edge magnitude with the same shape as `image`.

    Raises
    ------
    ValueError
        If `image` is not a 2-D array.

    Notes
    -----
    Unsigned integer input is scaled to [0.0, 1.0]; signed integer input is
    scaled by the largest absolute endpoint. The two directional responses
    are combined and divided by sqrt(2).

    Examples
    --------
    >>> result = farid_filter(np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.uint8))
    >>> print(result.shape)
    # (2, 2)
    """
    return _edge_filter(image, _FARID_E, _FARID_S, mode, cval)


_ROBERTS_A = np.array([[1.0, 0.0], [0.0, -1.0]])
_ROBERTS_B = np.array([[0.0, 1.0], [-1.0, 0.0]])


def roberts_filter(image, mode="reflect", cval=0.0):
    """
    Summary
    -------
    Compute Roberts edge magnitude for a 2-D image.

    Parameters
    ----------
    image : array_like
        2-D grayscale image.
    mode : str
        Boundary mode used for convolution.
    cval : float
        Constant value used for boundary extension.

    Returns
    -------
    ndarray of float64
        Edge magnitude with the same shape as `image`.

    Raises
    ------
    ValueError
        If `image` is not a 2-D array.

    Notes
    -----
    Uses two 2x2 diagonal difference kernels. Unsigned integer input is
    scaled to [0.0, 1.0]; signed integer input is scaled by the largest
    absolute endpoint. The two directional responses are combined and divided
    by sqrt(2).

    Examples
    --------
    >>> result = roberts_filter(np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.uint8)
    >>> print(result.shape)
    # (2, 2)
    """
    image = _image01(image)

    a = ndimage.convolve(image, _ROBERTS_A, mode=mode, cval=float(cval))
    b = ndimage.convolve(image, _ROBERTS_B, mode=mode, cval=float(cval))

    return np.ascontiguousarray(
        np.sqrt(a * a + b * b) / math.sqrt(2.0),
        dtype=np.float64,
    )


def laplace_filter(image, ksize=3):
    """
    Summary
    -------
    Apply Laplacian convolution to a 2-D image.

    Parameters
    ----------
    image : array_like
        2-D grayscale image.
    ksize : int
        Odd convolution kernel size.

    Returns
    -------
    ndarray of float64
        Laplacian response with the same shape as `image`.

    Raises
    ------
    ValueError
        If `image` is not a 2-D array or if `ksize` is not a positive odd
        integer.

    Notes
    -----
    `ksize=3` uses the standard 3x3 Laplacian. Larger odd values use a
    cross-shaped kernel. Unsigned integer input is scaled to [0.0, 1.0];
    signed integer input is scaled by the largest absolute endpoint.

    Examples
    --------
    >>> result = laplace_filter(np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.uint8))
    >>> print(result.shape)
    # (2, 2)
    """

    image = _image01(image)
    ksize = int(ksize)

    if ksize <= 0 or ksize % 2 == 0:
        raise ValueError("ksize must be a positive odd integer.")

    if ksize == 3:
        kernel = np.array([
            [0.0, 1.0, 0.0],
            [1.0, -4.0, 1.0],
            [0.0, 1.0, 0.0],
        ])
    else:
        kernel = np.zeros((ksize, ksize), dtype=np.float64)
        c = ksize // 2
        kernel[c, :] = 1.0
        kernel[:, c] = 1.0
        kernel[c, c] = -2.0 * (ksize - 1)

    return np.ascontiguousarray(
        ndimage.convolve(image, kernel, mode="reflect"),
        dtype=np.float64,
    )


def median_filter(image, footprint=None, mode="nearest", cval=0.0):
    """
    Summary
    -------
    Apply median filtering to a 2-D image.

    Parameters
    ----------
    image : array_like
        2-D grayscale image.
    footprint : ndarray of bool or None
        Median filter footprint. If ``None``, a 3x3 all-`True` footprint is
        used.
    mode : str
        Boundary mode passed to the median filter.
    cval : float
        Constant value used for boundary extension.

    Returns
    -------
    ndarray
        Median-filtered image with the same shape as `image`. The returned
        dtype follows the input dtype and the underlying median filter.

    Raises
    ------
    ValueError
        If `image` is not a 2-D array.

    Examples
    --------
    >>> result = median_filter(np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.uint8))
    >>> print(result.shape)
    # (2, 2)
    """
    image = np.asarray(image)

    if image.ndim != 2:
        raise ValueError("image must be a 2-D array.")

    if footprint is None:
        footprint = np.ones((3, 3), dtype=bool)

    return ndimage.median_filter(
        image,
        footprint=np.asarray(footprint, dtype=bool),
        mode=mode,
        cval=float(cval),
    )


def _hessian(image, sigma, mode="reflect"):
    image = _float_image(image)

    sigma = float(sigma)
    if sigma <= 0:
        raise ValueError("sigma must be positive.")

    s = sigma / math.sqrt(2.0)
    truncate = 8.0 if sigma > 1.0 else 100.0

    common = dict(sigma=s, mode=mode, cval=0.0, truncate=truncate)

    gr = ndimage.gaussian_filter(image, order=(1, 0), **common)
    gc = ndimage.gaussian_filter(image, order=(0, 1), **common)

    hrr = ndimage.gaussian_filter(gr, order=(1, 0), **common)
    hrc = ndimage.gaussian_filter(gr, order=(0, 1), **common)
    hcc = ndimage.gaussian_filter(gc, order=(0, 1), **common)

    return hrr, hrc, hcc


@njit(parallel=True, nogil=True, cache=True, fastmath=True)
def _hessian_eigenvalues_2d(hrr, hrc, hcc):
    h, w = hrr.shape
    eig0 = np.empty((h, w), dtype=np.float64)
    eig1 = np.empty((h, w), dtype=np.float64)

    for r in prange(h):
        for c in range(w):
            a = hrr[r, c]
            b = hrc[r, c]
            d = hcc[r, c]
            m = 0.5 * (a + d)
            s = math.sqrt(b * b + 0.25 * (a - d) ** 2)
            eig0[r, c] = m + s
            eig1[r, c] = m - s

    return eig0, eig1


def hessian_filter(image, sigmas=(1, 3, 5), alpha=0.5, beta=0.5, gamma=15.0, black_ridges=True):
    """
    Summary
    -------
    Compute a Hessian-based vesselness response over multiple scales.

    Extended Description
    --------------------
    For `black_ridges=``True``, dark linear structures produce positive
    responses. For `black_ridges=``False``, the input is negated before
    filtering so that bright structures are detected. Responses are combined
    by taking the maximum over `sigmas`. Locations with no positive response
    are set to 1.0.

    Parameters
    ----------
    image : array_like
        2-D grayscale image.
    sigmas : float or sequence[float]
        Positive Gaussian scales.
    alpha : float
        Accepted for compatibility; this parameter has no effect in this
        2-D implementation.
    beta : float
        Smoothing parameter for the ridge/blob ratio term.
    gamma : float
        Structuredness threshold.
    black_ridges : bool
        If ``True``, filter dark ridges; if ``False``, filter bright ridges.

    Returns
    -------
    ndarray of float64
        Vesselness-like response with the same shape as `image`. Vessel
        locations contain values in [0.0, 1.0], and background locations are
        1.0.

    Raises
    ------
    ValueError
        If `image` is not a 2-D array or if `sigmas` is invalid.

    Notes
    -----
    Uses a Numba-compiled helper for per-pixel eigenvalue computation.

    Examples
    --------
    >>> result = hessian_filter(np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.uint8), [1.0])
    >>> print(result.shape)
    # (2, 2)
    """

    image = _float_image(image)

    if not black_ridges:
        image = -image

    sigmas = _sigmas(sigmas)
    result = np.zeros_like(image)

    for sigma in sigmas:
        hrr, hrc, hcc = _hessian(image, sigma)
        e0, e1 = _hessian_eigenvalues_2d(hrr, hrc, hcc)

        if np.all(np.abs(e0) < np.abs(e1)):
            e0, e1 = e1, e0

        # Sort by absolute eigenvalue per pixel.
        a = np.abs(e0) <= np.abs(e1)
        lambda1 = np.where(a, e0, e1)
        lambda2 = np.where(a, e1, e0)

        lambda2 = np.maximum(lambda2, 1e-10)
        rb = np.abs(lambda1) / lambda2
        s2 = e0 * e0 + e1 * e1

        response = np.exp(-(rb * rb) / (2.0 * beta * beta))
        response *= 1.0 - np.exp(-s2 / (2.0 * gamma * gamma))

        if black_ridges:
            response[lambda2 <= 0] = 0.0
        else:
            response[lambda2 >= 0] = 0.0

        np.maximum(result, response, out=result)

    result[result <= 0] = 1.0
    return result


def meijering_filter(image, sigmas=(1, 3, 5), alpha=None, black_ridges=True):
    """
    Summary
    -------
    Compute a Meijering vesselness response over multiple scales.

    Extended Description
    --------------------
    For `black_ridges=``True``, dark linear structures produce positive
    responses. For `black_ridges=``False``, the input is negated before
    filtering so that bright structures are detected. Each scale response is
    normalized by its maximum before combining scales by maximum.

    Parameters
    ----------
    image : array_like
        2-D grayscale image.
    sigmas : float or sequence[float]
        Positive Gaussian scales.
    alpha : float or None
        Meijering weighting parameter. If ``None``, `1.0/3.0` is used.
    black_ridges : bool
        If ``True``, filter dark ridges; if ``False``, filter bright ridges.

    Returns
    -------
    ndarray of float64
        Vesselness response in [0.0, 1.0] with the same shape as `image`.

    Raises
    ------
    ValueError
        If `image` is not a 2-D array or if `sigmas` is invalid.

    Examples
    --------
    >>> result = meijering_filter([[0.0, 1.0], [1.0, 0.0]], [1.0])
    >>> print(result.shape)
    # (2, 2)
    """
    image = _float_image(image)

    if not black_ridges:
        image = -image

    sigmas = _sigmas(sigmas)

    if alpha is None:
        alpha = 1.0 / 3.0

    result = np.zeros_like(image)

    for sigma in sigmas:
        hrr, hrc, hcc = _hessian(image, sigma)
        e0, e1 = _hessian_eigenvalues_2d(hrr, hrc, hcc)

        a = e0 + alpha * e1
        b = alpha * e0 + e1

        response = np.where(
            np.abs(a) >= np.abs(b),
            a,
            b,
        )

        response = np.maximum(response, 0.0)

        maximum = response.max()
        if maximum > 0:
            response /= maximum

        np.maximum(result, response, out=result)

    return result

def sharpen_filter(image):
    kernel = np.array([
        [-1, -1, -1],
        [-1,  9, -1],
        [-1, -1, -1],
    ], dtype=np.float64)

    return cv2.filter2D(image, -1, kernel)


def mexican_hat_filter(image):
    kernel = np.array([
        [0, 0, -1, 0, 0],
        [0, -1, -2, -1, 0],
        [-1, -2, 16, -2, -1],
        [0, -1, -2, -1, 0],
        [0, 0, -1, 0, 0],
    ], dtype=np.float64)

    return cv2.filter2D(image, -1, kernel)
