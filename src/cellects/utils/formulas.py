#!/usr/bin/env python3
""" This script contains formula functions,
they mainly are simple mathematical expressions, used many times by Cellects.
This is part of the lower level of the software. When possible, these functions are optimized using Numba's decorator
@njit.
    - linear_model
    - get_divisor
    - eudist
    - cart2pol
    - pol2cart
    - get_peak_number
    - cohen_d_95
    - cohen_d
    - SlopeDeviation
    - acf_fft
    - moving_average
    - max_cum_sum_from_rolling_window
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
    Convert array values to uint8.

    Round the input array values and convert them to unsigned 8-bit integers.

    Parameters
    ----------
    an_array : numpy.ndarray
        Input array whose values are to be converted.

    Returns
    -------
    numpy.ndarray
        The numpy.ndarray of uint8 type containing rounded values of the input
        array.

    Raises
    ------
    TypeError
        If the input is not a numpy.ndarray.
    ValueError
        If the array contains values that cannot be rounded.

    Notes
    -----
    This function uses Numba's @njit decorator for performance.

    Examples
    --------
    >>> import numpy as np
    >>> an_array = np.array([1.2, 2.7, 3.4])
    >>> to_uint8(an_array)
    array([1, 3, 3], dtype=uint8)

    >>> an_array = np.array([-1.2, -2.7, 3.4])
    >>> to_uint8(an_array)
    array([0, 0, 3], dtype=uint8)
    """
    out = np.empty_like(an_array)
    return np.round(an_array, 0, out).astype(np.uint8)


@njit()
def bracket_to_uint8_image_contrast(image):
    """
    Convert a float image with dynamic range from [min, max] to an 8-bit integer
    image with a specified contrast.

    Parameters
    ----------
    image : numpy.ndarray
        Input image as a 2D NumPy array of floats.

    Returns
    -------
    numpy.ndarray
        Output image as a 2D NumPy array of uint8.

    Raises
    ------
    TypeError
        If `image` is not a NumPy array.
    ValueError
        If `image` contains non-finite values.

    Notes
    -----
    This function uses Numba’s `@njit` decorator for performance.

    Examples
    --------

    >>> image = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float64)
    >>> result = bracket_to_uint8_image_contrast(image)
    >>> print(result)
    [[ 64 128]
     [192 255]]
    """
    image -= np.min(image)
    return to_uint8(255 * (image / np.max(image)))


@njit()
def linear_model(x, a, b):
    """
    Compute the y values of a linear model
    :param x: vector of x values
    :param a: slope
    :param b: intercep
    :return: y values
    """
    return a * x + b


@njit()
def get_power_dists(binary_image, cx, cy, n):
    """
    Compute the power n of the distance of each row/column with the barycenter of the shape
    :param binary_image: a binary image
    :type binary_image: uint8
    :param cx: x coordinate of the barycenter
    :param cy: y coordinate of the barycenter
    :param n: power
    :return: a vector of these power distances
    """
    xn = (np.arange(binary_image.shape[1]) - cx) ** n
    yn = (np.arange(binary_image.shape[0]) - cy) ** n
    return xn, yn


@njit()
def get_var(mo, binary_image, Xn, Yn):
    """
    Compute the variance of the shape in a binary image, over the y and x axes
    :param mo: the moments of the shape
    :param binary_image: a binary image
    :param Xn:
    :param Yn:
    :return: variance of the shape over the y and x axes
    """
    vx = np.sum(binary_image * Xn) / mo["m00"]
    vy = np.sum(binary_image * Yn) / mo["m00"]
    return vx, vy

@njit()
def get_skewness_kurtosis(mnx, mny, sx, sy, n):
    """
    General formula to compute both the skewness and kurtosis over the y and x axes
    :param mnx:
    :param mny:
    :param sx:
    :param sy:
    :param n:
    :return: x_skewness, y_skewness or x_kurtosis, y_kurtosis
    """
    return mnx / sx ** n, mny / sy ** n


def get_standard_deviations(mo, binary_image, cx, cy):
    """
    Compute the standard deviation of the shape in a binary image, over the y and x axes
    :param mo: the moments of the shape
    :param binary_image: a binary image
    :param cx: x coordinate of the barycenter
    :param cx: y coordinate of the barycenter
    :return: standard deviation of the shape over the y and x axes

    """
    x2, y2 = get_power_dists(binary_image, cx, cy, 2)
    X2, Y2 = np.meshgrid(x2, y2)
    vx, vy = get_var(mo, binary_image, X2, Y2)
    return np.sqrt(vx), np.sqrt(vy)


def get_skewness(mo, binary_image, cx, cy, sx, sy):
    """
    Compute the skewness of the shape in a binary image, over the y and x axes
    :param mo: the moments of the shape
    :param binary_image: a binary image
    :param cx: x coordinate of the barycenter
    :param cy: y coordinate of the barycenter
    :param sx: standard deviation of the shape over the x axis
    :param sy: standard deviation of the shape over the y axis
    :return: x_skewness, y_skewness
    """
    x3, y3 = get_power_dists(binary_image, cx, cy, 3)
    X3, Y3 = np.meshgrid(x3, y3)
    m3x, m3y = get_var(mo, binary_image, X3, Y3)
    return get_skewness_kurtosis(m3x, m3y, sx, sy, 3)


def get_kurtosis(mo, binary_image, cx, cy, sx, sy):
    """
     Compute the kurtosis of the shape in a binary image, over the y and x axes
    :param mo: the moments of the shape
    :param binary_image: a binary image
    :param cx: x coordinate of the barycenter
    :param cy: y coordinate of the barycenter
    :param sx: standard deviation of the shape over the x axis
    :param sy: standard deviation of the shape over the y axis
    :return: x_kurtosis, y_kurtosis
    """
    x4, y4 = get_power_dists(binary_image, cx, cy, 4)
    X4, Y4 = np.meshgrid(x4, y4)
    m4x, m4y = get_var(mo, binary_image, X4, Y4)
    return get_skewness_kurtosis(m4x, m4y, sx, sy, 4)


@njit()
def get_inertia_axes(mo):
    """
    Compute the inertia axes (major and minor axes) and orientation of the shape in a binary image
    :param mo: the moments of the shape
    :return: x coordinate of the barycenter, y coordinate of the barycenter, major axis length, minor axis length,
    axes orientation
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
    Compute the euclidian distance between two points
    :param v1: coordinates of the point 1
    :param v2: coordinates of the point 2
    :return: euclidian distance
    """
    dist = [(a - b)**2 for a, b in zip(v1, v2)]
    dist = np.sqrt(np.sum(dist))
    return dist


def cart2pol(x, y):
    """
    Convert a point's coordinates from cartesian to polar
    :param x: coordinate over the x axis
    :param y: coordinate over the y axis
    :return: distance, angle
    """
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)


def pol2cart(rho, phi):
    """
    Convert a point's coordinates from polar to cartesian
    :param rho: distance
    :param phi: angle
    :return: x coordinate, y coordinate
    """
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return x, y


def cohen_d_95(vector_1, vector_2, nbboot=100000):
    """
    Compute the 95% confidence interval around a cohen d using bootstrap
    :param vector_1:
    :param vector_2:
    :param nbboot:
    :return:
    """
    boot = np.zeros(nbboot, dtype=int)
    n1 = len(vector_1)
    n2 = len(vector_2)
    for i in np.arange(nbboot):
        v1bis = np.random.choice(vector_1, size=n1, replace=True)
        v2bis = np.random.choice(vector_2, size=n2, replace=True)
        boot[i] = cohen_d(v1bis,v2bis)
    effect_low_top = np.append(cohen_d(vector_1, vector_2), np.quantile(boot, (0.025, 0.975)))
    return effect_low_top


def cohen_d(vector_1, vector_2):
    """
    Compute the Cohen d between two vectors
    :param vector_1:
    :param vector_2:
    :return: Cohen d
    """
    m1 = np.mean(vector_1)
    m2 = np.mean(vector_2)
    s1 = np.std(vector_1)
    s2 = np.std(vector_2)
    n1 = len(vector_1)
    n2 = len(vector_2)
    spooled = np.sqrt(((n2 - 1) * s2 ** 2 + (n1 - 1) * s1 ** 2) / (n1 + n2 - 2))
    return (m2 - m1) / spooled


def moving_average(vector, step):
    """
    Compute the moving averate on a vector, given a step
    :param vector: the vector to average/smooth
    :param step: the window size to compute averages
    :return: the averaged/smoothed vector
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


def max_cum_sum_from_rolling_window(side_length, window_step):
    """
    Calculates the maximum cumulative sum from a rolling window across a square grid.

    This function computes the squared result of dividing `side_length` by `window_step`,
    rounded up to the nearest integer. It represents the theoretical upper limit of
    cumulative values achievable when applying a rolling window mechanism over a square
    grid with uniform spacing.

    Parameters
    ----------
    side_length : int or float
        Total length of one side of the square grid.
    window_step : int or float
        Spacing between consecutive windows along the grid axis. Must be positive and
        smaller than `side_length`.

    Returns
    -------
    int or float
        Squared value representing maximum cumulative sum based on window distribution.

    Notes
    -----
    The ceiling operation ensures full coverage of the grid when dividing into discrete
    windows, preventing underestimation due to partial remainder windows.
    """
    return np.square(np.ceil(side_length / window_step))


def find_common_coord(array1, array2):
    """
    Compares coordinates between two arrays to find matching rows from array1 in array2.

    Parameters
    ----------
    array1 : numpy.ndarray
        First 2D coordinate array (shape `(n_coords, n_dims)`)
    array2 : numpy.ndarray
        Second 2D coordinate array (shape `(m_coords, n_dims)`)

    Returns
    -------
    numpy.ndarray
        Boolean array with shape `(n_coords,)` where True indicates that corresponding row in `array1`
        exists as a matching row in `array2`. Comparison is done element-wise across all dimensions.
    """
    return (array1[:, None, :] == array2[None, :, :]).all(-1).any(-1)


def find_duplicates_coord(array1):
    """
    Detect duplicate rows in a 2D array by comparing row occurrences.

    Returns boolean mask indicating which rows are duplicated (appear more than once) along the first axis of input array. Uses inverse indices mapping to track original positions during deduplication process.

    Parameters
    ----------
    array1 : numpy.ndarray
        Input array with shape (N, M) containing coordinates or values where N is number of rows and M is row dimension

    Returns
    -------
    duplicates_mask : numpy.ndarray
        Boolean array with same first dimension as input. True at index i indicates that the corresponding row in array1 occurs more than once.

    See Also
    --------
    numpy.unique : Used for finding unique rows and generating inverse indices mapping.
    numpy.bincount : Counts occurrences of each unique row based on inverse indices.
    """
    unique_rows, inverse_indices = np.unique(array1, axis=0, return_inverse=True)
    counts = np.bincount(inverse_indices)
    # A row is duplicate if its count > 1
    return counts[inverse_indices] > 1

def remove_excedent_duplicates_coord(array1):
    # np.unique(array1, axis=0)
    unique_rows, inverse_indices = np.unique(array1, axis=0, return_inverse=True)
    to_remove = []
    seen_indices = []
    for i in inverse_indices:
        if i in seen_indices:
            to_remove.append(True)
        else:
            to_remove.append(False)
            seen_indices.append(i)
    return np.delete(array1, to_remove, 0)
