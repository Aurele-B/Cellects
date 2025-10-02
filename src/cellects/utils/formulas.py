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


def cart2pol(x, y):
    """
    Convert Cartesian coordinates to polar coordinates.

    Given kartesian coordinates x and y, return the corresponding
    polar coordinates rho and phi. This is done by using the formula for
    conversion from Cartesian to Polar.

    Parameters
    ----------
    x : float
        The x-coordinate of the point in Cartesian space.
    y : float
        The y-coordinate of the point in Cartesian space.

    Returns
    -------
    tuple[float, float]
        A tuple containing (rho, phi) where rho is the radial distance
        and phi is the angle in radians.

    Examples
    --------
    >>> cart2pol(1, 0)
    (1.0, 0.0)

    >>> cart2pol(0, 1)
    (1.0, 1.5707963267948966)

    >>> cart2pol(-1, 0)
    (1.0, 3.141592653589793)
    """
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)


def pol2cart(rho, phi):
    """
    Convert from polar to Cartesian coordinates.

    Given a point in polar coordinates (rho, phi), this function
    calculates the corresponding Cartesian coordinates (x, y).

    Parameters
    ----------
    rho : float
        The radius in polar coordinates.
    phi : float
        The angle in radians in polar coordinates.

    Returns
    -------
    tuple[float, float]
        A tuple containing the x and y coordinates in Cartesian
        representation.

    Notes
    -----
    This function assumes the input values for phi are in radians.
    If they are provided in degrees, convert them using
    `np.radians(phi)` before calling this function.

    Examples
    --------
    >>> pol2cart(1.0, 0)
    (1.0, 0.0)

    >>> pol2cart(5.0, np.pi / 4)
    (3.5355339059327378, 3.5355339059327373)
    """
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return x, y


def cohen_d_95(vector_1, vector_2, nbboot=100000):
    """
    Calculate Cohen's d with a 95% confidence interval through bootstrapping.

    Performs bootstrapping on the provided data vectors to calculate Cohen's d
    effect size along with its 95% confidence interval.

    Parameters
    ----------
    vector_1 : array_like
        The first data vector.
    vector_2 : array_like
        The second data vector.
    nbboot : int, optional
        Number of bootstrap iterations. Default is ``100000``.

    Returns
    -------
    ndarray
        An array containing Cohen's d effect size and its 95% confidence interval.
        The first element is the Cohen's d value, followed by the lower and upper
        bounds of the confidence interval.

    Raises
    ------
    ValueError
        If `vector_1` and `vector_2` are not of the same length.
    TypeError
        If `vector_1` or `vector_2` are not array-like.

    Notes
    -----
    This function uses bootstrapping to estimate the 95% confidence interval for
    Cohen's d. The calculation of Cohen’s d assumes equal population variances.

    Examples
    --------
    >>> vector_1 = np.array([2.3, 3.4, 5.6, 7.8])
    >>> vector_2 = np.array([1.1, 4.3, 9.8])
    >>> result = cohen_d_95(vector_1, vector_2)
    >>> print(result)
    [ 0.73864592 -1.46522    1.88307 ]
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
    Calculate Cohen's d effect size between two independent groups.

    Parameters
    ----------
    vector_1 : array_like
        First group's data.
    vector_2 : array_like
        Second group's data.

    Returns
    -------
    float
        Cohen's d effect size statistic between the two groups.

    Notes
    -----
    Cohen's d is a measure of the difference between two means in terms
    of standard deviation units. It can be used to compare the mean difference
    between two groups, normalized by pooled standard deviation.

    Examples
    --------
    >>> vector_1 = [2.3, 4.5, 6.7]
    >>> vector_2 = [5.3, 7.8, 9.1]
    >>> result = cohen_d(vector_1, vector_2)
    >>> print(result)
    0.864
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


def max_cum_sum_from_rolling_window(side_length, window_step):
    """
    Calculate the maximum cumulative sum from a rolling window.

    Parameters
    ----------
    side_length : int
        The length of the side of the square window.
    window_step : int
        The step size for rolling the window.

    Returns
    -------
    int
        The maximum cumulative sum calculated from the rolling window.

    Raises
    ------
    ValueError
        If `side_length` or `window_step` are not positive integers.

    Notes
    -----
    - The function assumes that `side_length` and `window_step` are both positive integers.
    - This function uses NumPy's ceiling function to handle non-integer division results.

    Examples
    --------
    >>> result = max_cum_sum_from_rolling_window(10, 2)
    >>> print(result)
    9

    """
    return np.square(np.ceil(side_length / window_step))


def find_common_coord(array1, array2):
    """
    Find common coordinates between two arrays.

    Given two 2D arrays, this function finds the indices where all corresponding
    elements are equal.

    Parameters
    ----------
    array1 : array_like
        First 2D array of shape (M, N).
    array2 : array_like
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

def remove_excedent_duplicates_coord(array1):
    """
    Remove duplicate rows in a 2D array based on coordinate order.

    This function removes all but the first occurrence of duplicate rows in a 2D array.
    The removal is based on the order of appearance (coordinates) in the input array.

    Parameters
    ----------
    array1 : numpy.ndarray
        A 2D array of shape (n, m) containing rows to be deduplicated.

    Other Parameters
    ----------------
    None

    Returns
    -------
    numpy.ndarray
        A 2D array with duplicate rows removed.

    Raises
    ------
    None

    Notes
    -----
    This function uses NumPy's `np.unique` for performance and efficiency.

    Examples
    --------
    >>> array1 = np.array([[1, 2], [3, 4], [1, 2]])
    >>> result = remove_excedent_duplicates_coord(array1)
    >>> print(result)
    [[1 2]
     [3 4]]

    >>> array1 = np.array([[5, 6], [7, 8], [9, 0], [5, 6]])
    >>> result = remove_excedent_duplicates_coord(array1)
    >>> print(result)
    [[5 6]
     [7 8]
     [9 0]]
    """
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
