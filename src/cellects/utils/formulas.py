#!/usr/bin/env python3
""" This script contains formula functions
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
"""
from numpy import sum, absolute, max, min, empty_like, round_, uint8, meshgrid, sqrt, arange,square, arctan, pi, isnan, arctan2, cos, sin, mean, random, append, quantile, std, array, floor, ceil, zeros, logical_not, logical_and
from numba import njit


@njit()
def sum_of_abs_differences(array1, array2):
    """
    Compute the sum of absolute differences between two arrays of the same size and type
    :param array1:
    :param array2:
    :return: sum of absolute differences
    """
    return sum(absolute(array1 - array2))


@njit()
def to_uint8(an_array):
    """
    Round and convert an array into uint8
    :param an_array:
    :return:
    :rtype: uint8
    """
    out = empty_like(an_array)
    return round_(an_array, 0, out).astype(uint8)


@njit()
def bracket_to_uint8_image_contrast(image):
    """
    Increased an image contrast by:
    Standardizing values so that all values become distributed between 0 and 255
    :param image: any array
    :return: an image with increased contrast
    :rtype: uint8
    """
    image = image - min(image)
    return to_uint8(255 * (image / max(image)))


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
    xn = (arange(binary_image.shape[1]) - cx) ** n
    yn = (arange(binary_image.shape[0]) - cy) ** n
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
    vx = sum(binary_image * Xn) / mo["m00"]
    vy = sum(binary_image * Yn) / mo["m00"]
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
    X2, Y2 = meshgrid(x2, y2)
    vx, vy = get_var(mo, binary_image, X2, Y2)
    return sqrt(vx), sqrt(vy)


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
    X3, Y3 = meshgrid(x3, y3)
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
    X4, Y4 = meshgrid(x4, y4)
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
    c20 = (mo["m20"] / mo["m00"]) - square(cx)
    c02 = (mo["m02"] / mo["m00"]) - square(cy)
    c11 = (mo["m11"] / mo["m00"]) - (cx * cy)
    # Calculate major and minor axi lengths OK
    # major_axis_len = sqrt(8 * (c20 + c02 + sqrt(4 * square(c11) + square(c20 - c02))))
    # minor_axis_len = sqrt(8 * (c20 + c02 - sqrt(4 * square(c11) + square(c20 - c02))))
    major_axis_len = sqrt(6 * (c20 + c02 + sqrt(square(2 * c11) + square(c20 - c02))))
    minor_axis_len = sqrt(6 * (c20 + c02 - sqrt(square(2 * c11) + square(c20 - c02))))
    if (c20 - c02) != 0:
        axes_orientation = (0.5 * arctan((2 * c11) / (c20 - c02))) + ((c20 < c02) * (pi /2))
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
    dist = sqrt(sum(dist))
    return dist


def cart2pol(x, y):
    """
    Convert a point's coordinates from cartesian to polar
    :param x: coordinate over the x axis
    :param y: coordinate over the y axis
    :return: distance, angle
    """
    rho = sqrt(x**2 + y**2)
    phi = arctan2(y, x)
    return(rho, phi)


def pol2cart(rho, phi):
    """
    Convert a point's coordinates from polar to cartesian
    :param rho: distance
    :param phi: angle
    :return: x coordinate, y coordinate
    """
    x = rho * cos(phi)
    y = rho * sin(phi)
    return x, y


def cohen_d_95(vector_1, vector_2, nbboot=100000):
    """
    Compute the 95% confidence interval around a cohen d using bootstrap
    :param vector_1:
    :param vector_2:
    :param nbboot:
    :return:
    """
    boot = zeros(nbboot, dtype=int)
    n1 = len(vector_1)
    n2 = len(vector_2)
    for i in arange(nbboot):
        v1bis = random.choice(vector_1, size=n1, replace=True)
        v2bis = random.choice(vector_2, size=n2, replace=True)
        boot[i] = cohen_d(v1bis,v2bis)
    effect_low_top = append(cohen_d(vector_1, vector_2), quantile(boot, (0.025, 0.975)))
    return effect_low_top


def cohen_d(vector_1, vector_2):
    """
    Compute the Cohen d between two vectors
    :param vector_1:
    :param vector_2:
    :return: Cohen d
    """
    m1 = mean(vector_1)
    m2 = mean(vector_2)
    s1 = std(vector_1)
    s2 = std(vector_2)
    n1 = len(vector_1)
    n2 = len(vector_2)
    spooled = sqrt(((n2 - 1) * s2 ** 2 + (n1 - 1) * s1 ** 2) / (n1 + n2 - 2))
    return (m2 - m1) / spooled


def moving_average(vector, step):
    """
    Compute the moving averate on a vector, given a step
    :param vector: the vector to average/smooth
    :param step: the window size to compute averages
    :return: the averaged/smoothed vector
    """
    substep = array((- int(floor((step - 1) / 2)), int(ceil((step - 1) / 2))))
    sums = zeros(vector.shape)
    n_okays = sums.copy()
    true_numbers = logical_not(isnan(vector))
    vector[logical_not(true_numbers)] = 0
    for step_i in arange(substep[1] + 1):
        sums[step_i: (sums.size - step_i)] = sums[step_i: (sums.size - step_i)] + vector[(2 * step_i):]
        n_okays[step_i: (sums.size - step_i)] = n_okays[step_i: (sums.size - step_i)] + true_numbers[(2 * step_i):]
        if logical_and(step_i > 0, step_i < absolute(substep[0])):
            sums[step_i: (sums.size - step_i)] = sums[step_i: (sums.size - step_i)] + vector[:(sums.size - (2 * step_i)):]
            n_okays[step_i: (sums.size - step_i)] = n_okays[step_i: (sums.size - step_i)] + true_numbers[:(
                        true_numbers.size - (2 * step_i))]
    vector = sums / n_okays
    return vector
