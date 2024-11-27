"""
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
from copy import deepcopy
from numpy import square, sqrt, empty, zeros, float64, uint8, argmax, sum, pi
from cv2 import convexHull, arcLength, contourArea, minAreaRect, moments, findContours, RETR_TREE, CHAIN_APPROX_NONE, connectedComponents, CV_16U
from cellects.utils.utilitarian import translate_dict
from cellects.utils.formulas import get_inertia_axes, get_standard_deviations, get_skewness, get_kurtosis
from cellects.image_analysis.morphological_operations import cc

descriptors_categories = {'area': True, 'perimeter': False, 'circularity': False, 'rectangularity': False,
                          'total_hole_area': False, 'solidity': False, 'convexity': False, 'eccentricity': False,
                          'euler_number': False, 'standard_deviation_xy': False, 'skewness_xy': False,
                          'kurtosis_xy': False, 'major_axes_len_and_angle': True, 'iso_digi_analysis': False,
                          'oscilacyto_analysis': False, 'network_detection': False, 'fractal_analysis': False,
                          }

descriptors_names_to_display = ['Area', 'Perimeter', 'Circularity', 'Rectangularity', 'Total hole area',
                                'Solidity', 'Convexity', 'Eccentricity', 'Euler number', 'Standard deviation xy',
                                'Skewness xy', 'Kurtosis xy', 'Major axes lengths and angle',
                                'Related to growth transitions', 'Related to oscillations', 'Related to network',
                                'Related to fractals'
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
            self.binary_image = self.binary_image.astype(uint8)

        self.descriptors = {i: empty(0, dtype=float64) for i in wanted_descriptors_list}

        self.get_area()

        for name in self.descriptors.keys():
            if self.area == 0:
                self.descriptors[name] = 0
            else:
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
        self.mo = translate_dict(moments(self.binary_image))

    def get_area(self):
        self.area = self.binary_image.sum()

    def get_contours(self):
        contours, hierarchy = findContours(self.binary_image, RETR_TREE, CHAIN_APPROX_NONE)
        nb, shapes = connectedComponents(self.binary_image, ltype=CV_16U)
        self.euler_number = (nb - 1) - len(contours)
        self.contours = contours[0]
        if len(contours) > 1:
            all_lengths = zeros(len(contours))
            for i, contour in enumerate(contours):
                all_lengths[i] = len(contour)
            self.contours = contours[argmax(all_lengths)]

        # self.contours = contours[0]
        # if len(contours) > 1:
        #     for i in arange(1, len(contours)):
        #         self.contours = concatenate((self.contours, contours[i]))
            # self.contours = max(self.contours, key=contourArea)

    def get_min_bounding_rectangle(self):
        if self.contours is None:
            self.get_contours()
        self.min_bounding_rectangle = minAreaRect(self.contours)  # ((cx, cy), (width, height), angle)

    def get_inertia_axes(self):
        if self.mo is None:
            self.get_mo()

        self.cx, self.cy, self.major_axis_len, self.minor_axis_len, self.axes_orientation = get_inertia_axes(self.mo)

    def get_standard_deviations(self):
        if self.sx is None:
            if self.axes_orientation is None:
                self.get_inertia_axes()
            self.sx, self.sy = get_standard_deviations(self.mo, self.binary_image, self.cx, self.cy)

    def get_skewness(self):
        if self.skx is None:
            if self.sx is None:
                self.get_standard_deviations()

            self.skx, self.sky = get_skewness(self.mo, self.binary_image, self.cx, self.cy, self.sx, self.sy)

    def get_kurtosis(self):
        if self.kx is None:
            if self.sx is None:
                self.get_standard_deviations()

            self.kx, self.ky = get_kurtosis(self.mo, self.binary_image, self.cx, self.cy, self.sx, self.sy)

    def get_convex_hull(self):
        if self.contours is None:
            self.get_contours()
        self.convex_hull = convexHull(self.contours)

    """
       The following methods are shape descriptors calculus
    """

    def get_perimeter(self):
        if self.contours is None:
            self.get_contours()
        self.perimeter = arcLength(self.contours, True)

    def get_circularity(self):
        if self.perimeter is None:
            self.get_perimeter()
        self.circularity = (4 * pi * self.binary_image.sum()) / square(self.perimeter)

    def get_rectangularity(self):
        if self.min_bounding_rectangle is None:
            self.get_min_bounding_rectangle()
        bounding_rectangle_area = self.min_bounding_rectangle[1][0] * self.min_bounding_rectangle[1][1]
        if bounding_rectangle_area != 0:
            self.rectangularity = self.binary_image.sum() / bounding_rectangle_area
        else:
            self.rectangularity = 1
        # with self.min_bounding_rectangle[0] the x and y coordinates of the center point of the box
        # and self.min_bounding_rectangle[1] the widht and height of the box
        # box = boxPoints(self.min_bounding_rectangle)
        # box = int0(box)
        # img_to_display = drawContours(self.binary_image, [box], 0, (0, 0, 255), 2)
        # imtoshow = resize(img_to_display.astype(uint8)*120, (1000, 1000))
        # imshow('Rough detection', imtoshow)
        # waitKey(1)

    def get_total_hole_area(self):
        new_order, stats, centers = cc(1 - self.binary_image)
        self.total_hole_area = sum(stats[2:, 4])#  / self.binary_image.sum()

    def get_solidity(self):
        """
            :return: The ratio between the convex hull area and the shape area
        """
        if self.convex_hull is None:
            self.get_convex_hull()
        conv_h_cont_area = contourArea(self.convex_hull)
        if conv_h_cont_area != 0:
            self.solidity = contourArea(self.contours) / contourArea(self.convex_hull)
        else:
            self.solidity = 1

    def get_convexity(self):
        """
        :return: The ratio between the convex hull perimeter and the shape perimeter
        https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_contours/py_contour_features/py_contour_features.html
        """
        if self.perimeter is None:
            self.get_perimeter()
        if self.convex_hull is None:
            self.get_convex_hull()
        self.convexity = arcLength(self.convex_hull, True) / self.perimeter

    def get_eccentricity(self):
        self.get_inertia_axes()
        self.eccentricity = sqrt(1 - square(self.minor_axis_len / self.major_axis_len))

    def get_euler_number(self):
        if self.contours is None:
            self.get_contours()

    def get_major_axis_len(self):
        if self.major_axis_len is None:
            self.get_inertia_axes()

    def get_minor_axis_len(self):
        if self.minor_axis_len is None:
            self.get_inertia_axes()

    def get_axes_orientation(self):
        if self.axes_orientation is None:
            self.get_inertia_axes()

