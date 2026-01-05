#!/usr/bin/env python3
"""
Statistical and geometric analysis tools for numerical arrays.

This module provides a collection of functions and unit tests for calculating distances,
statistical properties (skewness, kurtosis), array transformations, and image moment-based
analysis. The tools are optimized for applications involving binary images, coordinate data,
and mathematical modeling operations where performance-critical calculations benefit from
vectorized or JIT-compiled implementations.

Functions:
eudist : Calculate Euclidean distance between two vectors
to_uint8 : Convert array to 8-bit unsigned integers using NumBA acceleration
translate_dict : Transform dictionary structures into alternative formats
linear_model : Compute y = a*x + b regression model values (JIT-compiled)
moving_average : Calculate sliding window averages with specified step size
get_var : Derive variance from image moments and spatial coordinates
find_common_coord : Identify shared coordinate pairs between two arrays
get_skewness/get_kurtosis : Calculate third/fourth standardized moment statistics
sum_of_abs_differences : Compute total absolute differences between arrays (JIT)
bracket_to_uint8_image_contrast : Convert images to 8-bit with contrast normalization
find_duplicates_coord : Locate rows with duplicate coordinate values
get_power_dists : Generate radial distance measures from image centers
get_inertia_axes : Calculate principal axes of inertia for binary shapes

Notes:
- All Numba-accelerated functions require congruent NumPy arrays as inputs
- Image processing functions expect binary (boolean/int8) input matrices
"""
from copy import deepcopy
import pandas as pd
from cellects.utils.decorators import njit
import numpy as np
from numpy.typing import NDArray
from typing import Tuple


@njit()
def sum_of_abs_differences(array1: NDArray, array2: NDArray):
    """
    Compute the sum of absolute differences between two arrays.

    Parameters
    ----------
    array1 : NDArray
        The first input array.
    array2 : NDArray
        The second input array.

    Returns
    -------
    int
        Sum of absolute differences between elements of `array1` and `array2`.

    Examples
    --------
    >>> arr1 = np.array([1.2, 2.5, -3.7])
    >>> arr2 = np.array([12, 25, -37])
    >>> result = sum_of_abs_differences(arr1, arr2)
    >>> print(result)
    66.6
    """
    return np.sum(np.absolute(array1 - array2))


@njit()
def to_uint8(an_array: NDArray):
    """
    Convert an array to unsigned 8-bit integers.

    Parameters
    ----------
    an_array : ndarray
        Input array to be converted. It can be of any numeric dtype.

    Returns
    -------
    ndarray
        The input array rounded to the nearest integer and then cast to
        unsigned 8-bit integers.

    Raises
    ------
    TypeError
        If `an_array` is not a ndarray.

    Notes
    -----
    This function uses Numba's `@njit` decorator for performance optimization.

    Examples
    --------
    >>> result = to_uint8(np.array([1.2, 2.5, -3.7]))
    >>> print(result)
    [1 3 0]
    """
    out = np.empty_like(an_array)
    return np.round(an_array, 0, out).astype(np.uint8)


@njit()
def bracket_to_uint8_image_contrast(image: NDArray):
    """
    Convert an image with bracket contrast values to uint8 type.

    This function normalizes an input image by scaling the minimum and maximum
    values of the image to the range [0, 255] and then converts it to uint8
    data type.

    Parameters
    ----------
    image : ndarray
        Input image as a numpy array with floating-point values.

    Returns
    -------
    ndarray of uint8
        Output image converted to uint8 type after normalization.

    Examples
    --------
    >>> image = np.random.randint(0, 255, (10, 10), dtype=np.uint8)
    >>> res = bracket_to_uint8_image_contrast(image)
    >>> print(res)

    >>> image = np.zeros((10, 10), dtype=np.uint8)
    >>> res = bracket_to_uint8_image_contrast(image)
    >>> print(res)
    """
    image -= image.min()
    if image.max() == 0:
        return np.zeros_like(image, dtype=np.uint8)
    else:
        return to_uint8(255 * (image / np.max(image)))

@njit()
def linear_model(x: NDArray, a: float, b: float) -> float:
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
def get_power_dists(binary_image: np.ndarray, cx: float, cy: float, n: int):
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
    >>> print(xn.shape), print(yn.shape)
    (10,) (10,)
    """
    xn = (np.arange(binary_image.shape[1]) - cx) ** n
    yn = (np.arange(binary_image.shape[0]) - cy) ** n
    return xn, yn


@njit()
def get_var(mo: dict, binary_image: NDArray, Xn: NDArray, Yn: NDArray) -> Tuple[float, float]:
    """
    Compute the center of mass in 2D space.

    This function calculates the weighted average position (centroid) of
    a binary image using given pixel coordinates and moments.

    Parameters
    ----------
    mo : dict
        Dictionary containing moments of binary image.
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
    if mo['m00'] == 0:
        vx, vy = 0., 0.
    else:
        vx = np.sum(binary_image * Xn) / mo["m00"]
        vy = np.sum(binary_image * Yn) / mo["m00"]
    return vx, vy


@njit()
def get_skewness_kurtosis(mnx: float, mny: float, sx: float, sy: float, n: int) -> Tuple[float, float]:
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
    if sx == 0:
        fx = 0
    else:
        fx = mnx / sx ** n

    if sy == 0:
        fy = 0
    else:
        fy = mny / sy ** n

    return fx, fy


def get_standard_deviations(mo: dict, binary_image: NDArray, cx: float, cy: float) -> Tuple[float, float]:
    """
    Return spatial standard deviations for a given moment and binary image.

    This function computes the square root of variances along `x` (horizontal)
    and `y` (vertical) axes for the given binary image and moment.

    Parameters
    ----------
    mo : dict
        Dictionary containing moments of binary image.
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


def get_skewness(mo: dict, binary_image: NDArray, cx: float, cy: float, sx: float, sy: float) -> Tuple[float, float]:
    """Calculate skewness of the given moment.

    This function computes the skewness based on the third moments
    and the central moments of a binary image.

    Parameters
    ----------
    mo : dict
        Dictionary containing moments of binary image.
    binary_image : ndarray
        Binary image as a 2D numpy array.
    cx : float
        Description of parameter `cx`.
    cy : float
        Description of parameter `cy`.
    sx : float
        Description of parameter `sx`.
    sy : float
        Description of parameter `sy`.

    Returns
    -------
    Tuple[float, float]
        Tuple containing skewness values.

    Examples
    --------
    >>> result = get_skewness(mo=example_mo, binary_image=binary_img,
    ... cx=0.5, cy=0.5, sx=1.0, sy=1.0)
    >>> print(result)
    (skewness_x, skewness_y)  # Example output
    """
    x3, y3 = get_power_dists(binary_image, cx, cy, 3)
    X3, Y3 = np.meshgrid(x3, y3)
    m3x, m3y = get_var(mo, binary_image, X3, Y3)
    return get_skewness_kurtosis(m3x, m3y, sx, sy, 3)


def get_kurtosis(mo: dict, binary_image: NDArray, cx: float, cy: float, sx: float, sy: float) -> Tuple[float, float]:
    """
    Calculate the kurtosis of a binary image.

    The function calculates the fourth moment (kurtosis) of the given
    binary image around the specified center coordinates with an option
    to specify the size of the square window.

    Parameters
    ----------
    mo : dict
        Dictionary containing moments of binary image.
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
def get_inertia_axes(mo: dict) -> Tuple[float, float, float, float, float]:
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
        axes_orientation = 0.
    return cx, cy, major_axis_len, minor_axis_len, axes_orientation


def eudist(v1, v2) -> float:
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


def moving_average(vector: NDArray, step: int) -> NDArray[float]:
    """
    Calculate the moving average of a given vector with specified step size.

    Computes the moving average of input `vector` using specified `step`
    size. NaN values are treated as zeros in the calculation to allow
    for continuous averaging.

    Parameters
    ----------
    vector : ndarray
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


def find_common_coord(array1: NDArray[int], array2: NDArray[int]) -> NDArray[bool]:
    """Find common coordinates between two arrays.

    This function compares the given 2D `array1` and `array2`
    to determine if there are any common coordinates.

    Parameters
    ----------
    array1 : ndarray of int
        A 2D numpy ndarray.
    array2 : ndarray of int
        Another 2D numpy ndarray.

    Returns
    -------
    out : ndarray of bool
        A boolean numpy ndarray where True indicates common
        coordinates.

    Examples
    --------
    >>> array1 = np.array([[1, 2], [3, 4]])
    >>> array2 = np.array([[5, 6], [1, 2]])
    >>> result = find_common_coord(array1, array2)
    >>> print(result)
    array([ True, False])"""
    return (array1[:, None, :] == array2[None, :, :]).all(-1).any(-1)


def find_duplicates_coord(array1: NDArray[int]) -> NDArray[bool]:
    """
    Find duplicate rows in a 2D array and return their coordinate indices.

    Given a 2D NumPy array, this function identifies rows that are duplicated (i.e., appear more than once) and returns a boolean array indicating their positions.

    Parameters
    ----------
    array1 : ndarray of int
        Input 2D array of shape (n_rows, n_columns) from which to find duplicate rows.

    Returns
    -------
    duplicates : ndarray of bool
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

def detect_first_move(size_dynamics: NDArray, growth_threshold)-> int:
    """
    Detects the first move in a time series where the value exceeds the initial value by a given threshold.

    Parameters
    ----------
    size_dynamics : numpy.ndarray
        The time series data of dynamics.
    growth_threshold: int or float
        The threshold value for detecting the first move.

    Returns
    -------
    int or pandas.NA
        The index of the first move where the condition is met.
        Returns `pandas.NA` if no such index exists.

    Examples
    --------
    >>> size_dynamics = np.array([10, 12, 15, 18])
    >>> growth_threshold = 5
    >>> detect_first_move(size_dynamics, growth_threshold)
    2
    """
    first_move = pd.NA
    thresh_reached = np.nonzero(size_dynamics >= (size_dynamics[0] + growth_threshold))[0]
    if len(thresh_reached) > 0:
        first_move = thresh_reached[0]
    return first_move

@njit()
def get_newly_explored_area(binary_vid: NDArray[np.uint8]) -> NDArray:
    """
    Get newly explored area in a binary video.

    Calculate the number of new pixels that have become active (==1) from
    the previous frame in a binary video representation.

    Parameters
    ----------
    binary_vid : np.ndarray
        The current frame of the binary video.

    Returns
    -------
    np.ndarray
        An array containing the number of new active pixels for each row.

    Notes
    -----
    This function uses Numba's @njit decorator for performance.

    Examples
    --------
    >>> binary_vid=np.zeros((4, 5, 5), dtype=np.uint8)
    >>> binary_vid[:2, 3, 3] = 1
    >>> binary_vid[1, 4, 3] = 1
    >>> binary_vid[2, 3, 4] = 1
    >>> binary_vid[3, 2, 3] = 1
    >>> get_newly_explored_area(binary_vid)
    array([0, 1, 1, 1])

    >>> binary_vid=np.zeros((5, 5), dtype=np.uint8)[None, :, :]
    >>> get_newly_explored_area(binary_vid)
    array([0])
    """
    return ((binary_vid - binary_vid[0, ...]) == 1).reshape(binary_vid.shape[0], - 1).sum(1)

def get_contour_width_from_im_shape(im_shape: Tuple) -> int:
    """
    Calculate the contour width based on image shape.

    Parameters
    ----------
    im_shape : tuple of int, two items
        The dimensions of the image.

    Returns
    -------
    int
        The calculated contour width.
    """
    return np.max((np.round(np.log10(np.max(im_shape)) - 2).astype(int), 2))

def scale_coordinates(coord: NDArray, scale: Tuple, dims: Tuple) -> Tuple[NDArray[np.int64], np.int64, np.int64, np.int64, np.int64]:
    """
    Scale coordinates based on given scale factors and dimensions.

    Parameters
    ----------
    coord : numpy.ndarray
        A 2x2 array of coordinates to be scaled.
    scale : tuple of float
        Scaling factors for the x and y coordinates, respectively.
    dims : tuple of int
        Maximum dimensions (height, width) for the scaled coordinates.

    Returns
    -------
    numpy.ndarray
        Scaled and rounded coordinates.
    int
        Minimum y-coordinate.
    int
        Maximum y-coordinate.
    int
        Minimum x-coordinate.
    int
        Maximum x-coordinate.

    Examples
    --------
    >>> coord = np.array(((47, 38), (59, 37)))
    >>> scale = (0.92, 0.87)
    >>> dims = (245, 300, 3)
    >>> scaled_coord, min_y, max_y, min_x, max_x = scale_coordinates(coord, scale, dims)
    >>> scaled_coord
    array([[43, 33],
           [54, 32]])
    >>> min_y, max_y
    (np.int64(43), np.int64(54))
    >>> min_x, max_x
    (np.int64(32), np.int64(33))

    Notes
    -----
    This function assumes that the input coordinates are in a specific format
    and will fail if not. The scaling factors should be positive.
    """
    coord = np.array(((np.round(coord[0][0] * scale[0]), np.round(coord[0][1] * scale[1])),
                    (np.round(coord[1][0] * scale[0]), np.round(coord[1][1] * scale[1]))), dtype=np.int64)
    min_y = np.max((0, np.min(coord[:, 0])))
    max_y = np.min((dims[0], np.max(coord[:, 0])))
    min_x = np.max((0, np.min(coord[:, 1])))
    max_x = np.min((dims[1], np.max(coord[:, 1])))
    return coord, min_y, max_y, min_x, max_x


