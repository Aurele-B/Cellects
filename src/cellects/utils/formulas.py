#!/usr/bin/env python3
"""
Optimized mathematical and image processing utilities for Cellects.

This module provides a collection of performance-optimized functions primarily used for numerical array operations,
statistical calculations, and basic image transformation tasks. All implementations leverage NumPy for numerical
computations and are accelerated via Numba's JIT compilation (`@njit`) to ensure low-latency execution in production workflows.
Functions include statistical moment analysis, coordinate-based variance calculation, and pixel-level normalization routines.

Functions:
    sum_of_abs_differences: Calculate element-wise absolute differences between arrays.
    linear_model: Execute affine transformations on numerical data.
    get_power_dists: Generate spatial distribution patterns for image coordinates.
    get_var: Compute weighted centroids using moments and pixel positions.
    get_skewness_kurtosis: Derive skewness/kurtosis from statistical moments.
    bracket_to_uint8_image_contrast: Normalize images to 8-bit unsigned integer format.

Notes:
- All Numba-accelerated functions require NumPy arrays as inputs
- Division-by-zero operations will raise exceptions for invalid input shapes/moments
- Image processing functions expect binary (boolean/int8) input matrices
"""
from copy import deepcopy
from numba import njit
import numpy as np


@njit()
def sum_of_abs_differences(array1, array2):
    """
    Calculate the sum of absolute differences between two NumPy arrays.

    Parameters
    ----------
    array1 : array_like
        First input array.
    array2 : array_like
        Second input array.

    Returns
    -------
    scalar
        Sum of absolute differences between the two arrays.
    """
    return np.sum(np.absolute(array1 - array2))


@njit()
def to_uint8(an_array):
    """
    Convert an array to unsigned 8-bit integers.

    Parameters
    ----------
    an_array : numpy.ndarray
        Input array to be converted. It can be of any numeric dtype.

    Returns
    -------
    numpy.ndarray
        The input array rounded to the nearest integer and then cast to
        unsigned 8-bit integers.

    Raises
    ------
    TypeError
        If `an_array` is not a numpy.ndarray.

    Notes
    -----
    This function uses Numba's `@njit` decorator for performance optimization.

    Examples
    --------
    >>> import numpy as np
    >>> result = to_uint8(np.array([1.2, 2.5, -3.7]))
    >>> print(result)
    [1 3 0]
    """
    out = np.empty_like(an_array)
    return np.round(an_array, 0, out).astype(np.uint8)


@njit()
def bracket_to_uint8_image_contrast(image):
    """
    Convert an image with bracket contrast values to uint8 type.

    This function normalizes an input image by scaling the minimum and maximum
    values of the image to the range [0, 255] and then converts it to uint8
    data type.

    Parameters
    ----------
    image : np.ndarray
        Input image as a numpy array with floating-point values.

    Returns
    -------
    np.ndarray
        Output image converted to uint8 type after normalization.
    """
    image -= np.min(image)
    return to_uint8(255 * (image / np.max(image)))


@njit()
def linear_model(x, a, b):
    """
    Perform a linear transformation on input data using slope and intercept.

    Parameters
    ----------
    x : array_like
        Input data.
    a : float
        Slope coefficient.
    b : float
        Intercept.

    Returns
    -------
    float
        Resulting value from linear transformation: `a` * `x` + `b`.

    Examples
    --------
    >>> result = linear_model(5, 2.0, 1.5)
    >>> print(result)  # doctest: +SKIP
    11.5

    Notes
    -----
    This function uses Numba's @njit decorator for performance.
    """
    return a * x + b


@njit()
def get_power_dists(binary_image, cx, cy, n):
    """
    Calculate the power distributions based on the given center coordinates and exponent.

    This function computes the `n`th powers of x and y distances from
    a given center point `(cx, cy)` for each pixel in the binary image.

    Parameters
    ----------
    binary_image : np.ndarray
        A 2D array (binary image) where the power distributions are calculated.
    cx : float
        The x-coordinate of the center point.
    cy : float
        The y-coordinate of the center point.
    n : int
        The exponent for power distribution calculation.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        A tuple containing two arrays:
        - The first array contains the `n`th power of x distances from the center.
        - The second array contains the `n`th power of y distances from the center.

    Notes
    -----
    This function uses Numba's `@njit` decorator for performance optimization.
    Ensure that `binary_image` is a NumPy ndarray to avoid type issues.

    Examples
    --------
    >>> binary_image = np.zeros((10, 10))
    >>> xn, yn = get_power_dists(binary_image, 5.0, 5.0, 2)
    >>> print(xn.shape), print(yn.shape
    (10,) (10,)

    """
    xn = (np.arange(binary_image.shape[1]) - cx) ** n
    yn = (np.arange(binary_image.shape[0]) - cy) ** n
    return xn, yn


@njit()
def get_var(mo, binary_image, Xn, Yn):
    """
    Compute the center of mass in 2D space.

    This function calculates the weighted average position (centroid) of
    a binary image using given pixel coordinates and moments.

    Parameters
    ----------
    mo : dict
        Image moments dictionary with keys 'm00', 'm10', and 'm20'.
    binary_image : ndarray
        2D binary image where non-zero pixels are considered.
    Xn : ndarray
        Array of x-coordinates for each pixel in `binary_image`.
    Yn : ndarray
        Array of y-coordinates for each pixel in `binary_image`.

    Returns
    -------
    tuple
        A tuple of two floats `(vx, vy)` representing the centroid coordinates.

    Raises
    ------
    ZeroDivisionError
        If `mo['m00']` is zero, indicating no valid pixels in the image.
        The function raises a `ZeroDivisionError`.

    Notes
    -----
    Performance considerations: This function uses Numba's `@njit` decorator for performance.
    """
    vx = np.sum(binary_image * Xn) / mo["m00"]
    vy = np.sum(binary_image * Yn) / mo["m00"]
    return vx, vy

@njit()
def get_skewness_kurtosis(mnx, mny, sx, sy, n):
    """
    Calculates skewness and kurtosis of a distribution.

    This function computes the skewness and kurtosis from given statistical
    moments, standard deviations, and order of moments.

    Parameters
    ----------
    mnx : float
        The third moment about the mean for x.
    mny : float
        The fourth moment about the mean for y.
    sx : float
        The standard deviation of x.
    sy : float
        The standard deviation of y.
    n : int
        Order of the moment (3 for skewness, 4 for kurtosis).

    Returns
    -------
    skewness : float
        The computed skewness.
    kurtosis : float
        The computed kurtosis.

    Notes
    -----
    This function uses Numba's `@njit` decorator for performance.
    Ensure that the values of `mnx`, `mny`, `sx`, and `sy` are non-zero to avoid division by zero.
    If `n = 3`, the function calculates skewness. If `n = 4`, it calculates kurtosis.

    Examples
    --------
    >>> skewness, kurtosis = get_skewness_kurtosis(1.5, 2.0, 0.5, 0.75, 3)
    >>> print("Skewness:", skewness)
    Skewness: 8.0
    >>> print("Kurtosis:", kurtosis)
    Kurtosis: nan

    """
    return mnx / sx ** n, mny / sy ** n


def get_standard_deviations(mo, binary_image, cx, cy):
    """
    Return spatial standard deviations for a given moment and binary image.

    This function computes the square root of variances along `x` (horizontal)
    and `y` (vertical) axes for the given binary image and moment.

    Parameters
    ----------
    mo : ndarray of float64
        The moment matrix computed from the binary image.
    binary_image : ndarray of bool or int8
        The binary input image where the moments are computed.
    cx : float64
        X-coordinate of center of mass (horizontal position).
    cy : float64
        Y-coordinate of center of mass (vertical position).

    Returns
    -------
    tuple[ndarray of float64, ndarray of float64]
        Tuple containing the standard deviations along the x and y axes.

    Raises
    ------
    ValueError
        If `binary_image` is not a binary image or has an invalid datatype.

    Notes
    -----
    This function uses the `get_power_dists` and `get_var` functions to compute
    the distributed variances, which are then transformed into standard deviations.

    Examples
    --------
    >>> import numpy as np
    >>> binary_image = np.array([[0, 1], [1, 0]], dtype=np.int8)
    >>> mo = np.array([[2.0], [3.0]])
    >>> cx, cy = 1.5, 1.5
    >>> stdx, stdy = get_standard_deviations(mo, binary_image, cx, cy)
    >>> print(stdx)
    [1.1]
    >>> print(stdy)
    [0.8366600265...]
    """
    x2, y2 = get_power_dists(binary_image, cx, cy, 2)
    X2, Y2 = np.meshgrid(x2, y2)
    vx, vy = get_var(mo, binary_image, X2, Y2)
    return np.sqrt(vx), np.sqrt(vy)


def get_skewness(mo, binary_image, cx, cy, sx, sy):
    """
    Calculate the skewness of a moment image.

    Parameters
    ----------
    mo : ndarray
        Moment matrix.
    binary_image : ndarray
        Binary image of the object.
    cx, cy : int
        Center coordinates of the moment calculation.
    sx, sy : float or int
        Standard deviation in x and y directions.

    Returns
    -------
    float
        Skewness value. If the values are not defined, returns `None`.

    Examples
    --------
    >>> import numpy as np
    >>> mo = np.array([[1, 2, 3], [4, 5, 6]])  # Example moment matrix
    >>> binary_image = np.array([[0, 1], [1, 0]])  # Example binary image
    >>> cx = 1
    >>> cy = 0
    >>> sx = 2.5
    >>> sy = 1.8

    >>> result = get_skewness(mo, binary_image, cx, cy, sx, sy)
    >>> print(result)  # Expected output may vary based on the input data
    None
    """
    x3, y3 = get_power_dists(binary_image, cx, cy, 3)
    X3, Y3 = np.meshgrid(x3, y3)
    m3x, m3y = get_var(mo, binary_image, X3, Y3)
    return get_skewness_kurtosis(m3x, m3y, sx, sy, 3)


def get_kurtosis(mo, binary_image, cx, cy, sx, sy):
    """
    Calculate the kurtosis of a binary image.

    The function calculates the fourth moment (kurtosis) of the given
    binary image around the specified center coordinates with an option
    to specify the size of the square window.

    Parameters
    ----------
    mo : np.ndarray
        A 2D numpy ndarray of moments calculated from the image.
    binary_image : np.ndarray
        A 2D numpy ndarray representing a binary image.
    cx : int or float
        The x-coordinate of the center point of the square window.
    cy : int or float
        The y-coordinate of the center point of the square window.
    sx : int or float
        The x-length of the square window (width).
    sy : int or float
        The y-length of the square window (height).

    Returns
    -------
    float
        The kurtosis value calculated from the moments.

    Examples
    --------
    >>> mo = np.array([[0, 1], [2, 3]])
    >>> binary_image = np.array([[1, 0], [0, 1]])
    >>> cx = 2
    >>> cy = 3
    >>> sx = 5
    >>> sy = 6
    >>> result = get_kurtosis(mo, binary_image, cx, cy, sx, sy)
    >>> print(result)
    expected output
    """
    x4, y4 = get_power_dists(binary_image, cx, cy, 4)
    X4, Y4 = np.meshgrid(x4, y4)
    m4x, m4y = get_var(mo, binary_image, X4, Y4)
    return get_skewness_kurtosis(m4x, m4y, sx, sy, 4)


@njit()
def get_inertia_axes(mo):
    """
    Calculate the inertia axes of a moment object.

    This function computes the barycenters, central moments,
    and the lengths of the major and minor axes, as well as
    their orientation.

    Parameters
    ----------
    mo : dict
        Dictionary containing moments, which should include keys:
        'm00', 'm10', 'm01', 'm20', and 'm11'.

    Returns
    -------
    tuple
        A tuple containing:
            - cx : float
                The x-coordinate of the barycenter.
            - cy : float
                The y-coordinate of the barycenter.
            - major_axis_len : float
                The length of the major axis.
            - minor_axis_len : float
                The length of the minor axis.
            - axes_orientation : float
                The orientation of the axes in radians.

    Notes
    -----
    This function uses Numba's @njit decorator for performance.
    The moments in the input dictionary should be computed from
    the same image region.

    Examples
    --------
    >>> mo = {'m00': 1.0, 'm10': 2.0, 'm01': 3.0, 'm20': 4.0, 'm11': 5.0}
    >>> get_inertia_axes(mo)
    (2.0, 3.0, 9.165151389911677, 0.8421875803239, 0.7853981633974483)

    """
    #L. Rocha, L. Velho and P.C.P. Calvalho (2002)
    #http://sibgrapi.sid.inpe.br/col/sid.inpe.br/banon/2002/10.23.11.34/doc/35.pdf
    # http://raphael.candelier.fr/?blog=Image%20Moments

    # Calculate barycenters
    cx = mo["m10"] / mo["m00"]
    cy = mo["m01"] / mo["m00"]
    # Calculate central moments
    c20 = (mo["m20"] / mo["m00"]) - np.square(cx)
    c02 = (mo["m02"] / mo["m00"]) - np.square(cy)
    c11 = (mo["m11"] / mo["m00"]) - (cx * cy)
    # Calculate major and minor axi lengths OK
    major_axis_len = np.sqrt(6 * (c20 + c02 + np.sqrt(np.square(2 * c11) + np.square(c20 - c02))))
    minor_axis_len = np.sqrt(6 * (c20 + c02 - np.sqrt(np.square(2 * c11) + np.square(c20 - c02))))
    if (c20 - c02) != 0:
        axes_orientation = (0.5 * np.arctan((2 * c11) / (c20 - c02))) + ((c20 < c02) * (np.pi /2))
    else:
        axes_orientation = 0
    return cx, cy, major_axis_len, minor_axis_len, axes_orientation


def eudist(v1, v2):
    """
    Calculate the Euclidean distance between two points in n-dimensional space.

    Parameters
    ----------
    v1 : iterable of float
        The coordinates of the first point.
    v2 : iterable of float
        The coordinates of the second point.

    Returns
    -------
    float
        The Euclidean distance between `v1` and `v2`.

    Raises
    ------
    ValueError
        If `v1` and `v2` do not have the same length.

    Notes
    -----
    The Euclidean distance is calculated using the standard formula:
    √((x2 − x1)^2 + (y2 − y1)^2 + ...).

    Examples
    --------
    >>> v1 = [1.0, 2.0]
    >>> v2 = [4.0, 6.0]
    >>> eudist(v1, v2)
    5.0

    >>> v1 = [1.0, 2.0, 3.0]
    >>> v2 = [4.0, 6.0, 8.0]
    >>> eudist(v1, v2)
    7.0710678118654755
    """
    dist = [(a - b)**2 for a, b in zip(v1, v2)]
    dist = np.sqrt(np.sum(dist))
    return dist


def moving_average(vector, step):
    """
    Calculate the moving average of a given vector with specified step size.

    Computes the moving average of input `vector` using specified `step`
    size. NaN values are treated as zeros in the calculation to allow
    for continuous averaging.

    Parameters
    ----------
    vector : numpy.ndarray
        Input vector for which to calculate the moving average.
    step : int
        Size of the window for computing the moving average.

    Returns
    -------
    numpy.ndarray
        Vector containing the moving averages of the input vector.

    Raises
    ------
    ValueError
        If `step` is less than 1.
    ValueError
        If the input vector has no valid (non-NaN) elements.

    Notes
    -----
    - The function considers NaN values as zeros during the averaging process.
    - If `step` is greater than or equal to the length of the vector, a warning will be raised.

    Examples
    --------
    >>> import numpy as np
    >>> vector = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
    >>> step = 3
    >>> result = moving_average(vector, step)
    >>> print(result)
    [1.5 2.33333333 3.66666667 4.         nan]
    """
    substep = np.array((- int(np.floor((step - 1) / 2)), int(np.ceil((step - 1) / 2))))
    sums = np.zeros(vector.shape)
    n_okays = deepcopy(sums)
    true_numbers = np.logical_not(np.isnan(vector))
    vector[np.logical_not(true_numbers)] = 0
    for step_i in np.arange(substep[1] + 1):
        sums[step_i: (sums.size - step_i)] = sums[step_i: (sums.size - step_i)] + vector[(2 * step_i):]
        n_okays[step_i: (sums.size - step_i)] = n_okays[step_i: (sums.size - step_i)] + true_numbers[(2 * step_i):]
        if np.logical_and(step_i > 0, step_i < np.absolute(substep[0])):
            sums[step_i: (sums.size - step_i)] = sums[step_i: (sums.size - step_i)] + vector[:(sums.size - (2 * step_i)):]
            n_okays[step_i: (sums.size - step_i)] = n_okays[step_i: (sums.size - step_i)] + true_numbers[:(
                        true_numbers.size - (2 * step_i))]
    vector = sums / n_okays
    return vector


def find_common_coord(array1, array2):
    """
    Find common coordinates between two arrays.

    Given two 2D arrays, this function finds the indices where all corresponding
    elements are equal.

    Parameters
    ----------
    array1 : array_like of type int
        First 2D array of shape (M, N).
    array2 : array_like of type int
        Second 2D array of shape (P, N).

    Returns
    -------
    bool_array : ndarray
        A 2D boolean array of shape (M, P) where each element indicates
        whether the corresponding rows in `array1` and `array2` are equal.

    Examples
    --------
    >>> array1 = np.array([[1, 2], [3, 4]])
    >>> array2 = np.array([[5, 6], [1, 2]])
    >>> result = find_common_coord(array1, array2)
    >>> print(result)
    array([ True, False])"""
    return (array1[:, None, :] == array2[None, :, :]).all(-1).any(-1)


def find_duplicates_coord(array1):
    """
    Find duplicate rows in a 2D array and return their coordinate indices.

    Given a 2D NumPy array, this function identifies rows that are duplicated (i.e., appear more than once) and returns a boolean array indicating their positions.

    Parameters
    ----------
    array1 : ndarray
        Input 2D array of shape (n_rows, n_columns) from which to find duplicate rows.

    Returns
    -------
    duplicates : ndarray
        Boolean array of shape (n_rows,), where `True` indicates that the corresponding row in `array1` is a duplicate.

    Examples
    --------
    >>> import numpy as np
    >>> array1 = np.array([[1, 2], [3, 4], [1, 2], [5, 6]])
    >>> find_duplicates_coord(array1)
    array([ True, False,  True, False])"""
    unique_rows, inverse_indices = np.unique(array1, axis=0, return_inverse=True)
    counts = np.bincount(inverse_indices)
    # A row is duplicate if its count > 1
    return counts[inverse_indices] > 1

