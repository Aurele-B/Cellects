"""Module for computing shape descriptors from binary images.

This module provides a framework for calculating various geometric and statistical
descriptors of shapes in binary images through configurable dictionaries and a core class.
Supported metrics include area, perimeter, axis lengths, orientation, and more.
Descriptor computation is controlled via category dictionaries (e.g., `descriptors_categories`)
and implemented as methods in the ShapeDescriptors class.

Classes
-------
ShapeDescriptors : Class to compute various descriptors for a binary image

Notes
-----
Relies on OpenCV and NumPy for image processing operations.
Shape descriptors: The following names, lists and computes all the variables describing a shape in a binary image.
If you want to allow the software to compute another variable:
1) In the following dicts and list, you need to:
        add the variable name and whether to compute it (True/False) by default
2) In the ShapeDescriptors class:
        add a method to compute that variable
3) In the init method of the ShapeDescriptors class
    attribute a None value to the variable that store it
    add a if condition in the for loop to compute that variable when its name appear in the wanted_descriptors_list
"""
import cv2
import numpy as np
from copy import deepcopy
from cellects.utils.utilitarian import translate_dict
from cellects.utils.formulas import get_inertia_axes, get_standard_deviations, get_skewness, get_kurtosis

descriptors_categories = {'area': True, 'perimeter': False, 'circularity': False, 'rectangularity': False,
                          'total_hole_area': False, 'solidity': False, 'convexity': False, 'eccentricity': False,
                          'euler_number': False, 'standard_deviation_xy': False, 'skewness_xy': False,
                          'kurtosis_xy': False, 'major_axes_len_and_angle': True, 'iso_digi_analysis': False,
                          'oscilacyto_analysis': False, 'network_analysis': False, 'graph_extraction': False,
                          'fractal_analysis': False
                          }

descriptors_names_to_display = ['Area', 'Perimeter', 'Circularity', 'Rectangularity', 'Total hole area',
                                'Solidity', 'Convexity', 'Eccentricity', 'Euler number', 'Standard deviation xy',
                                'Skewness xy', 'Kurtosis xy', 'Major axes lengths and angle',
                                'Growth transitions', 'Oscillations', 'Network', 'Graph',
                                'Fractals'
                                ]#, 'Oscillating cluster nb and size'

from_shape_descriptors_class = {'area': True, 'perimeter': False, 'circularity': False, 'rectangularity': False,
               'total_hole_area': False, 'solidity': False, 'convexity': False, 'eccentricity': False,
               'euler_number': False, 'standard_deviation_y': False, 'standard_deviation_x': False,
               'skewness_y': False, 'skewness_x': False, 'kurtosis_y': False, 'kurtosis_x': False,
               'major_axis_len': True, 'minor_axis_len': True, 'axes_orientation': True
                               }

descriptors = deepcopy(from_shape_descriptors_class)
descriptors.update({'cluster_number': False, 'mean_cluster_area': False, 'minkowski_dimension': False,
                    'vertices_number': False, 'edges_number': False})



class ShapeDescriptors:
    """
        This class takes :
        - a binary image of 0 and 1 drawing one shape
        - a list of descriptors to calculate from that image
        ["area", "perimeter", "circularity", "rectangularity", "total_hole_area", "solidity", "convexity",
         "eccentricity", "euler_number",

        "standard_deviation_y", "standard_deviation_x", "skewness_y", "skewness_x", "kurtosis_y", "kurtosis_x",
        "major_axis_len", "minor_axis_len", "axes_orientation",

        "mo", "contours", "min_bounding_rectangle", "convex_hull"]

        Be careful! mo, contours, min_bounding_rectangle, convex_hull,
        standard_deviations, skewness and kurtosis are not atomics
    https://www.researchgate.net/publication/27343879_Estimators_for_Orientation_and_Anisotropy_in_Digitized_Images
    """

    def __init__(self, binary_image, wanted_descriptors_list):
        """
        Class to compute various descriptors for a binary image.

        Parameters
        ----------
        binary_image : ndarray
            Binary image used to compute the descriptors.
        wanted_descriptors_list : list
            List of strings with the names of the wanted descriptors.

        Attributes
        ----------
        binary_image : ndarray
            The binary image.
        descriptors : dict
            Dictionary containing the computed descriptors.
        mo : float or None, optional
            Moment of inertia (default is `None`).
        area : int or None, optional
            Area of the object (default is `None`).
        contours : ndarray or None, optional
            Contours of the object (default is `None`).
        min_bounding_rectangle : tuple or None, optional
            Minimum bounding rectangle of the object (default is `None`).
        convex_hull : ndarray or None, optional
            Convex hull of the object (default is `None`).
        major_axis_len : float or None, optional
            Major axis length of the object (default is `None`).
        minor_axis_len : float or None, optional
            Minor axis length of the object (default is `None`).
        axes_orientation : float or None, optional
            Orientation of the axes (default is `None`).
        sx : float or None, optional
            Standard deviation in x-axis (default is `None`).
        kx : float or None, optional
            Kurtosis in x-axis (default is `None`).
        skx : float or None, optional
            Skewness in x-axis (default is `None`).
        perimeter : float or None, optional
            Perimeter of the object (default is `None`).
        convexity : float or None, optional
            Convexity of the object (default is `None`).

        Examples
        --------
        >>> binary_image = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=np.uint8)
        >>> wanted_descriptors_list = ["area", "perimeter"]
        >>> SD = ShapeDescriptors(binary_image, wanted_descriptors_list)
        >>> SD.descriptors
        {'area': np.uint64(9), 'perimeter': 8.0}
        """
        # Give a None value to each parameters whose presence is assessed before calculation (less calculus for speed)
        self.mo = None
        self.area = None
        self.contours = None
        self.min_bounding_rectangle = None
        self.convex_hull = None
        self.major_axis_len = None
        self.minor_axis_len = None
        self.axes_orientation = None
        self.sx = None
        self.kx = None
        self.skx = None
        self.perimeter = None
        self.convexity = None

        self.binary_image = binary_image
        if self.binary_image.dtype == 'bool':
            self.binary_image = self.binary_image.astype(np.uint8)

        self.descriptors = {i: np.empty(0, dtype=np.float64) for i in wanted_descriptors_list}
        self.get_area()

        for name in self.descriptors.keys():
            if name == "mo":
                self.get_mo()
                self.descriptors[name] = self.mo
            elif name == "area":
                self.descriptors[name] = self.area
            elif name == "contours":
                self.get_contours()
                self.descriptors[name] = self.contours
            elif name == "min_bounding_rectangle":
                self.get_min_bounding_rectangle()
                self.descriptors[name] = self.min_bounding_rectangle
            elif name == "major_axis_len":
                self.get_major_axis_len()
                self.descriptors[name] = self.major_axis_len
            elif name == "minor_axis_len":
                self.get_minor_axis_len()
                self.descriptors[name] = self.minor_axis_len
            elif name == "axes_orientation":
                self.get_inertia_axes()
                self.descriptors[name] = self.axes_orientation
            elif name == "standard_deviation_y":
                self.get_standard_deviations()
                self.descriptors[name] = self.sy
            elif name == "standard_deviation_x":
                self.get_standard_deviations()
                self.descriptors[name] = self.sx
            elif name == "skewness_y":
                self.get_skewness()
                self.descriptors[name] = self.sky
            elif name == "skewness_x":
                self.get_skewness()
                self.descriptors[name] = self.skx
            elif name == "kurtosis_y":
                self.get_kurtosis()
                self.descriptors[name] = self.ky
            elif name == "kurtosis_x":
                self.get_kurtosis()
                self.descriptors[name] = self.kx
            elif name == "convex_hull":
                self.get_convex_hull()
                self.descriptors[name] = self.convex_hull
            elif name == "perimeter":
                self.get_perimeter()
                self.descriptors[name] = self.perimeter
            elif name == "circularity":
                self.get_circularity()
                self.descriptors[name] = self.circularity
            elif name == "rectangularity":
                self.get_rectangularity()
                self.descriptors[name] = self.rectangularity
            elif name == "total_hole_area":
                self.get_total_hole_area()
                self.descriptors[name] = self.total_hole_area
            elif name == "solidity":
                self.get_solidity()
                self.descriptors[name] = self.solidity
            elif name == "convexity":
                self.get_convexity()
                self.descriptors[name] = self.convexity
            elif name == "eccentricity":
                self.get_eccentricity()
                self.descriptors[name] = self.eccentricity
            elif name == "euler_number":
                self.get_euler_number()
                self.descriptors[name] = self.euler_number

    """
        The following methods can be called to compute parameters for descriptors requiring it
    """

    def get_mo(self):
        """
        Get moments of a binary image.

        Calculate the image moments for a given binary image using OpenCV's
        `cv2.moments` function and then translate these moments into a formatted
        dictionary.

        Notes
        -----
        This function assumes the binary image has already been processed and is in a
        suitable format for moment calculation.

        Returns
        -------
        None

        Examples
        --------
       >>> SD = ShapeDescriptors(np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=np.uint8), ["mo"])
        >>> print(SD.mo["m00"])
        9.0
        """
        self.mo = translate_dict(cv2.moments(self.binary_image))

    def get_area(self):
        """
        Calculate the area of a binary image by summing its pixel values.

        This function computes the area covered by white pixels (value 1) in a binary image,
        which is equivalent to counting the number of 'on' pixels.

        Notes
        -----
        Sums values in `self.binary_image` and stores the result in `self.area`.

        Returns
        -------
        None

        Examples
        --------
        >>> SD = ShapeDescriptors(np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=np.uint8), ["area"])
        >>> print(SD.area)
        9.0
        """
        self.area = self.binary_image.sum()

    def get_contours(self):
        """
        Find and process the largest contour in a binary image.

        Retrieves contours from a binary image, calculates the Euler number,
        and identifies the largest contour based on its length.

        Notes
        -----
        This function modifies the internal state of the `self` object to store
        the largest contour and Euler number.

        Returns
        -------
        None

        Examples
        --------
        >>> SD = ShapeDescriptors(np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=np.uint8), ["euler_number"])
        >>> print(len(SD.contours))
        8
        """
        if self.area == 0:
            self.euler_number = 0.
            self.contours = np.array([], np.uint8)
        else:
            contours, hierarchy = cv2.findContours(self.binary_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            nb, shapes = cv2.connectedComponents(self.binary_image, ltype=cv2.CV_16U)
            self.euler_number = (nb - 1) - len(contours)
            self.contours = contours[0]
            if len(contours) > 1:
                all_lengths = np.zeros(len(contours))
                for i, contour in enumerate(contours):
                    all_lengths[i] = len(contour)
                self.contours = contours[np.argmax(all_lengths)]

    def get_min_bounding_rectangle(self):
        """
        Retrieve the minimum bounding rectangle from the contours of an image.

        This method calculates the smallest area rectangle that can enclose
        the object outlines present in the image, which is useful for
        object detection and analysis tasks.

        Notes
        -----
        - The bounding rectangle is calculated only if contours are available.
          If not, they will be retrieved first before calculating the rectangle.

        Raises
        ------
        RuntimeError
            If the contours are not available and cannot be retrieved,
            indicating a problem with the image or preprocessing steps.

        Returns
        -------
        None

        Examples
        --------
        >>> SD = ShapeDescriptors(np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=np.uint8), ["rectangularity"])
        >>> print(len(SD.min_bounding_rectangle))
        3
        """
        if self.area == 0:
            self.min_bounding_rectangle = np.array([], np.uint8)
        else:
            if self.contours is None:
                self.get_contours()
            if len(self.contours) == 0:
                self.min_bounding_rectangle = np.array([], np.uint8)
            else:
                self.min_bounding_rectangle = cv2.minAreaRect(self.contours)  # ((cx, cy), (width, height), angle)

    def get_inertia_axes(self):
        """
        Calculate and set the moments of inertia properties of an object.

        This function computes the centroid, major axis length,
        minor axis length, and axes orientation for an object. It
        first ensures that the moments of inertia (`mo`) attribute is available,
        computing them if necessary, before using the `get_inertia_axes` function.

        Returns
        -------
        None

            This method sets the following attributes:
            - `cx` : float
                The x-coordinate of the centroid.
            - `cy` : float
                The y-coordinate of the centroid.
            - `major_axis_len` : float
                The length of the major axis.
            - `minor_axis_len` : float
                The length of the minor axis.
            - `axes_orientation` : float
                The orientation angle of the axes.

        Raises
        ------
        ValueError
            If there is an issue with the moments of inertia computation.

        Notes
        -----
        This function modifies in-place the object's attributes related to its geometry.

        Examples
        --------
        >>> SD = ShapeDescriptors(np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=np.uint8), ["major_axis_len"])
        >>> print(SD.axes_orientation)
        0.0
        """
        if self.mo is None:
            self.get_mo()
        if self.area == 0:
            self.cx, self.cy, self.major_axis_len, self.minor_axis_len, self.axes_orientation = 0, 0, 0, 0, 0
        else:
            self.cx, self.cy, self.major_axis_len, self.minor_axis_len, self.axes_orientation = get_inertia_axes(self.mo)

    def get_standard_deviations(self):
        """
        Calculate and store standard deviations along x and y (sx, sy).

        Notes
        -----
        Requires centroid and moments; values are stored in `self.sx` and `self.sy`.

        Returns
        -------
        None

        Examples
        --------
       >>> SD = ShapeDescriptors(np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=np.uint8), ["standard_deviation_x", "standard_deviation_y"])
        >>> print(SD.sx, SD.sy)
        0.816496580927726 0.816496580927726
        """
        if self.sx is None:
            if self.axes_orientation is None:
                self.get_inertia_axes()
            self.sx, self.sy = get_standard_deviations(self.mo, self.binary_image, self.cx, self.cy)

    def get_skewness(self):
        """
        Calculate and store skewness along x and y (skx, sky).

        This function computes the skewness about the x-axis and y-axis of
        an image. Skewness is a measure of the asymmetry of the probability
        distribution of values in an image.

        Notes
        -----
        Requires standard deviations; values are stored in `self.skx` and `self.sky`.

        Returns
        -------
        None

        Examples
        --------
        >>> SD = ShapeDescriptors(np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=np.uint8), ["skewness_x", "skewness_y"])
        >>> print(SD.skx, SD.sky)
        0.0 0.0
        """
        if self.skx is None:
            if self.sx is None:
                self.get_standard_deviations()

            self.skx, self.sky = get_skewness(self.mo, self.binary_image, self.cx, self.cy, self.sx, self.sy)

    def get_kurtosis(self):
        """
        Calculates the kurtosis of the image moments.

        Kurtosis is a statistical measure that describes the shape of
        a distribution's tails in relation to its overall shape. It is
        used here in the context of image moments analysis.

        Notes
        -----
        This function first checks if the kurtosis values have already been calculated.
        If not, it calculates them using the `get_kurtosis` function.

        Examples
        --------
        >>> SD = ShapeDescriptors(np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=np.uint8), ["kurtosis_x", "kurtosis_y"])
        >>> print(SD.kx, SD.ky)
        1.5 1.5
        """
        if self.kx is None:
            if self.sx is None:
                self.get_standard_deviations()

            self.kx, self.ky = get_kurtosis(self.mo, self.binary_image, self.cx, self.cy, self.sx, self.sy)

    def get_convex_hull(self):
        """
        Compute and store the convex hull of the object's contour.

        Notes
        -----
        Stores the result in `self.convex_hull`. Computes contours if needed.

        Returns
        -------
        None

        Examples
        --------
        >>> SD = ShapeDescriptors(np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=np.uint8), ["solidity"])
        >>> print(len(SD.convex_hull))
        4
        """
        if self.area == 0:
            self.convex_hull = np.array([], np.uint8)
        else:
            if self.contours is None:
                self.get_contours()
            self.convex_hull = cv2.convexHull(self.contours)

    def get_perimeter(self):
        """
        Compute and store the contour perimeter length.

        Notes
        -----
        Computes contours if needed and stores the length in `self.perimeter`.

        Returns
        -------
        None

        Examples
        --------
        >>> SD = ShapeDescriptors(np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=np.uint8), ["perimeter"])
        >>> print(SD.perimeter)
        8.0
        """
        if self.area == 0:
            self.perimeter = 0.
        else:
            if self.contours is None:
                self.get_contours()
            if len(self.contours) == 0:
                self.perimeter = 0.
            else:
                self.perimeter = cv2.arcLength(self.contours, True)

    def get_circularity(self):
        """
        Compute and store circularity: 4πA / P².

        Notes
        -----
        Uses `self.area` and `self.perimeter`; stores result in `self.circularity`.

        Returns
        -------
        None

        Examples
        --------
         >>> SD = ShapeDescriptors(np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=np.uint8), ["circularity"])
        >>> print(SD.circularity)
        1.7671458676442586
        """
        if self.area == 0:
            self.circularity = 0.
        else:
            if self.perimeter is None:
                self.get_perimeter()
            self.circularity = (4 * np.pi * self.binary_image.sum()) / np.square(self.perimeter)

    def get_rectangularity(self):
        """
        Compute and store rectangularity: area / bounding-rectangle-area.

        Notes
        -----
        Uses `self.binary_image` and `self.min_bounding_rectangle`. Computes the MBR if needed.

        Returns
        -------
        None

        Examples
        --------
        >>> SD = ShapeDescriptors(np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=np.uint8), ["rectangularity"])
        >>> print(SD.rectangularity)
        2.25
        """
        if self.area == 0:
            self.rectangularity = 0.
        else:
            if self.min_bounding_rectangle is None:
                self.get_min_bounding_rectangle()
            bounding_rectangle_area = self.min_bounding_rectangle[1][0] * self.min_bounding_rectangle[1][1]
            if bounding_rectangle_area == 0:
                self.rectangularity = 0.
            else:
                self.rectangularity = self.binary_image.sum() / bounding_rectangle_area

    def get_total_hole_area(self):
        """
        Calculate the total area of holes in a binary image.

        This function uses connected component labeling to detect and
        measure the area of holes in a binary image.

        Returns
        -------
        float
            The total area of all detected holes in the binary image.

        Notes
        -----
        This function assumes that the binary image has been pre-processed
        and that holes are represented as connected components of zero
        pixels within the foreground

        Examples
        --------
        >>> SD = ShapeDescriptors(np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=np.uint8), ["total_hole_area"])
        >>> print(SD.total_hole_area)
        0
        """
        nb, new_order = cv2.connectedComponents(1 - self.binary_image)
        if nb > 2:
            self.total_hole_area = (new_order > 1).sum()
        else:
            self.total_hole_area = 0.

    def get_solidity(self):
        """
        Compute and store solidity: contour area / convex hull area.

        Extended Summary
        ----------------
        The solidity is a dimensionless measure that compares the area of a shape to
        its convex hull. A solidity of 1 means the contour is fully convex, while a
        value less than 1 indicates concavities.

        Notes
        -----
        If the convex hull area is 0 or absent, solidity is set to 0.

        Returns
        -------
        None

        Examples
        --------
        >>> SD = ShapeDescriptors(np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=np.uint8), ["solidity"])
        >>> print(SD.solidity)
        1.0
        """
        if self.area == 0:
            self.solidity = 0.
        else:
            if self.convex_hull is None:
                self.get_convex_hull()
            if len(self.convex_hull) == 0:
                self.solidity = 0.
            else:
                hull_area = cv2.contourArea(self.convex_hull)
                if hull_area == 0:
                    self.solidity = 0.
                else:
                    self.solidity = cv2.contourArea(self.contours) / hull_area

    def get_convexity(self):
        """
        Compute and store convexity: convex hull perimeter / contour perimeter.

        Notes
        -----
        Requires `self.perimeter` and `self.convex_hull`.

        Returns
        -------
        None

        Examples
        --------
        >>> SD = ShapeDescriptors(np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=np.uint8), ["convexity"])
        >>> print(SD.convexity)
        1.0
        """
        if self.perimeter is None:
            self.get_perimeter()
        if self.convex_hull is None:
            self.get_convex_hull()
        if self.perimeter == 0 or len(self.convex_hull) == 0:
            self.convexity = 0.
        else:
            self.convexity = cv2.arcLength(self.convex_hull, True) / self.perimeter

    def get_eccentricity(self):
        """
        Compute and store eccentricity from major and minor axis lengths.

        Notes
        -----
        Calls `get_inertia_axes()` if needed and stores result in `self.eccentricity`.

        Returns
        -------
        None

        Examples
        --------
        >>> SD = ShapeDescriptors(np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=np.uint8), ["eccentricity"])
        >>> print(SD.eccentricity)
        0.0
        """
        self.get_inertia_axes()
        if self.major_axis_len == 0:
            self.eccentricity = 0.
        else:
            self.eccentricity = np.sqrt(1 - np.square(self.minor_axis_len / self.major_axis_len))

    def get_euler_number(self):
        """
        Ensure contours are computed; stores Euler number in `self.euler_number` via `get_contours()`.

        Returns
        -------
        None

        Notes
        -----
        Euler number is computed in `get_contours()` as `(components - 1) - len(contours)`.
        """
        if self.contours is None:
            self.get_contours()

    def get_major_axis_len(self):
        """
        Ensure the major axis length is computed and stored in `self.major_axis_len`.

        Returns
        -------
        None

        Notes
        -----
        Triggers `get_inertia_axes()` if needed.

        Examples
        --------
        >>> SD = ShapeDescriptors(np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]], dtype=np.uint8), ["major_axis_len"])
        >>> print(SD.major_axis_len)
        2.8284271247461907
        """
        if self.major_axis_len is None:
            self.get_inertia_axes()

    def get_minor_axis_len(self):
        """
        Ensure the minor axis length is computed and stored in `self.minor_axis_len`.

        Returns
        -------
        None

        Notes
        -----
        Triggers `get_inertia_axes()` if needed.

        Examples
        --------
        >>> SD = ShapeDescriptors(np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]], dtype=np.uint8), ["minor_axis_len"])
        >>> print(SD.minor_axis_len)
        0.0
        """
        if self.minor_axis_len is None:
            self.get_inertia_axes()

    def get_axes_orientation(self):
        """
        Ensure the axes orientation angle is computed and stored in `self.axes_orientation`.

        Returns
        -------
        None

        Notes
        -----
        Calls `get_inertia_axes()` if orientation is not yet computed.

        Examples
        --------
        >>> SD = ShapeDescriptors(np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]], dtype=np.uint8), ["axes_orientation"])
        >>> print(SD.axes_orientation)
        1.5707963267948966
        """
        if self.axes_orientation is None:
            self.get_inertia_axes()

