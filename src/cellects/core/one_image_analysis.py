#!/usr/bin/env python3
"""
This script contains the OneImageAnalysis class
OneImageAnalysis is a class containing many tools to analyze one image

An image can be coded in different color spaces, such as RGB, HSV, etc. These color spaces code the color of each pixel as three numbers, ranging from 0 to 255. Our aim is to find a combination of these three numbers that provides a single intensity value for each pixel, and which maximizes the contrast between the organism and the background. To increase the flexibility of our algorithm, we use more than one color space to look for these combinations. In particular, we use the RGB, LAB, HSV, LUV, HLS and YUV color spaces. What we call a color space combination is a transformation combining several channels of one or more color spaces.
To find the optimal color space combination, Cellects uses one image (which we will call “seed image”). The software selects by default the first image of the sequence as seed image, but the user can select a different image where the cells are more visible.
Cellects has a fully automatic algorithm to select a good color space combination, which proceeds in four steps:

First, it screens every channel of every color space. For instance, it converts the image into grayscale using the second channel of the color space HSV, and segments that grayscale image using Otsu thresholding. Once a binary image is computed from every channel, Cellects only keep the channels for which the number of connected components is lower than 10000, and the total area detected is higher than 100 pixels but lower than 0.75 times the total size of the image. By doing so, we eliminate the channels that produce the most noise.

In the second step, Cellects uses all the channels that pass the first filter and tests all possible pairwise combinations. Cellects combines channels by summing their intensities and re-scaling the result between 0 and 255. It then performs the segmentation on these combinations, and filters them with the same criteria as in the first step.

The third step uses the previously selected channels and combinations that produce the highest and lowest detected surface to make logical operations between them. It applies the AND operator between the two results having the highest surface, and the OR operator between the two results having the lowest surface. It thus generates another two candidate segmentations, which are added to the ones obtained in the previous steps.

In the fourth step, Cellects works under the assumption that the image contains multiple similar arenas containing a collection of objects with similar size and shape, and keeps the segmentations whose standard error of the area is smaller than ten times the smallest area standard error across all segmentations. To account for cases in which the experimental setup induces segmentation errors in one particular direction, Cellects also keeps the segmentation with minimal width standard error across all segmentations, and the one with minimal height standard error across all segmentations. All retained segmentations are shown to the user, who can then select the best one.

As an optional step, Cellects can refine the choice of color space combination, using the last image of the sequence instead of the seed image. In order to increase the diversity of combinations explored, this optional analysis is performed in a different way than for the seed image. Also, this refining can use information from the segmentation of the seed frame and from the geometry of the arenas to rank the quality of the segmentation emerging from each color space combination. To generate these combinations, Cellects follows four steps.
The first step is identical to the first step of the previously described automatic algorithm (in section 1) and starts by screening every possible channel and color space.

The second step aims to find combinations that consider many channels, rather than those with only one or two. To do that, it creates combinations that consist of the sum of all channels except one. It then filters these combinations in the same way as for the previous step. Then, all surviving combinations are retained, and also undergo the same process in which one more channel is excluded, and the process continues until reaching single-channel combinations. This process thus creates new combinations that include any number of channels.

The third step filters these segmentations, keeping those that fulfill the following criteria: (1) The number of connected components is higher than the number of arenas and lower than 10000. (2) The detected area covers less than 99% of the image. (2) Less than 1% of the detected area falls outside the arenas. (4) Each connected component of the detected area covers less than 75% of the image.

Finally, the fourth step ranks the remaining segmentations using the following criteria: If the user labeled any areas as “cell”, the ranking will reflect the amount of cell pixels in common between the segmentation and the user labels. If the user did not label any areas as cells but labeled areas as background, the ranking will reflect the number of background pixels in common. Otherwise, the ranking will reflect the number of pixels in common with the segmentation of the first image.


"""

import logging
import os
from copy import deepcopy
import numpy as np
import cv2  # named opencv-python
import multiprocessing.pool as mp
from numba.typed import List as TList
from numba.typed import Dict as TDict
from cellects.image_analysis.morphological_operations import cross_33, Ellipse
from cellects.image_analysis.image_segmentation import get_color_spaces, combine_color_spaces, apply_filter, otsu_thresholding, get_otsu_threshold
from cellects.image_analysis.one_image_analysis_threads import SaveCombinationThread, ProcessFirstImage
from cellects.utils.formulas import bracket_to_uint8_image_contrast


class OneImageAnalysis:
    """
        This class takes a 3D matrix (2 space and 1 color [BGR] dimensions),
        Its methods allow image
        - conversion to any bgr/hsv/lab channels
        - croping
        - rotating
        - filtering using some of the mainly used techniques:
            - Gaussian, Median, Bilateral, Laplacian, Mexican hat
        - segmenting using thresholds or kmeans
        - shape selection according to horizontal size or shape ('circle' vs 'quadrilateral')

        ps: A viewing method displays the image before and after the most advanced modification made in instance
    """
    def __init__(self, image):
        self.image = image
        if len(self.image.shape) == 2:
            self.already_greyscale = True
        else:
            self.already_greyscale = False
        self.image2 = None
        self.binary_image2 = None
        self.drift_correction_already_adjusted: bool = False
        # Create empty variables to fill in the following functions
        self.binary_image = np.zeros(self.image.shape[:2], dtype=np.uint8)
        self.previous_binary_image = None
        self.validated_shapes = np.zeros(self.image.shape[:2], dtype=np.uint8)
        self.centroids = 0
        self.shape_number = 0
        self.concomp_stats = 0
        self.y_boundaries = None
        self.x_boundaries = None
        self.crop_coord = None
        self.cropped: bool = False
        self.subtract_background = None
        self.subtract_background2 = None
        self.im_combinations = None
        self.bgr = image
        self.colorspace_list = TList(("bgr", "lab", "hsv", "luv", "hls", "yuv"))
        self.spot_shapes = None
        self.all_c_spaces = TDict()
        self.hsv = None
        self.hls = None
        self.lab = None
        self.luv = None
        self.yuv = None
    """
        I/ Image modification for segmentation through thresholding
        This part contain methods to convert, visualize, filter and threshold one image.
    """
    def convert_and_segment(self, c_space_dict, color_number=2, biomask=None,
                            backmask=None, subtract_background=None, subtract_background2=None, grid_segmentation=False,
                            lighter_background=None, side_length=20, step=5, int_variation_thresh=None, mask=None,
                            filter_spec=None):

        if self.already_greyscale:
            self.segmentation(logical='None', color_number=2, biomask=biomask, backmask=backmask,
                              grid_segmentation=grid_segmentation, lighter_background=lighter_background,
                              side_length=side_length, step=step, int_variation_thresh=int_variation_thresh, mask=mask,
                              filter_spec=filter_spec)
        else:
            if len(self.all_c_spaces) == 0:
                self.all_c_spaces = get_color_spaces(self.bgr)
            # if c_space_dict['logical'] != 'None':
            first_dict = TDict()
            second_dict = TDict()
            for k, v in c_space_dict.items():
                 if k != 'logical' and v.sum() > 0:
                    if k[-1] != '2':
                        first_dict[k] = v
                    else:
                        second_dict[k[:-1]] = v
            logging.info(first_dict)
            self.image = combine_color_spaces(first_dict, self.all_c_spaces, subtract_background)
            if len(second_dict) > 0:
                self.image2 = combine_color_spaces(second_dict, self.all_c_spaces, subtract_background2)
                self.segmentation(logical=c_space_dict['logical'], color_number=color_number, biomask=biomask,
                                  backmask=backmask, grid_segmentation=grid_segmentation,
                                  lighter_background=lighter_background, side_length=side_length, step=step,
                                  int_variation_thresh=int_variation_thresh, mask=mask, filter_spec=filter_spec)

            else:

                self.segmentation(logical='None', color_number=color_number, biomask=biomask,
                                  backmask=backmask, grid_segmentation=grid_segmentation,
                                  lighter_background=lighter_background, side_length=side_length, step=step,
                                  int_variation_thresh=int_variation_thresh, mask=mask, filter_spec=filter_spec)


    def segmentation(self, logical='None', color_number=2, biomask=None, backmask=None, bio_label=None, bio_label2=None, grid_segmentation=False, lighter_background=None, side_length=20, step=5, int_variation_thresh=None, mask=None, filter_spec=None):
        if filter_spec is not None and filter_spec["filter1_type"] != "":
            self.image = apply_filter(self.image, filter_spec["filter1_type"], filter_spec["filter1_param"])
        if (color_number > 2):
            self.kmeans(color_number, biomask, backmask, logical, bio_label, bio_label2)
        elif grid_segmentation:
            if lighter_background is None:
                self.binary_image = otsu_thresholding(self.image)
                lighter_background = self.binary_image.sum() > (self.binary_image.size / 2)
            if int_variation_thresh is None:
                int_variation_thresh =100 - (np.ptp(self.image) * 90 / 255)
            self.grid_segmentation(lighter_background, side_length, step, int_variation_thresh, mask)
        else:
            # logging.info("Segment the image using Otsu thresholding")
            self.binary_image = otsu_thresholding(self.image)
            if self.previous_binary_image is not None:
                if (self.binary_image * (1 - self.previous_binary_image)).sum() > (self.binary_image * self.previous_binary_image).sum():
                    # Ones of the binary image have more in common with the background than with the specimen
                    self.binary_image = 1 - self.binary_image
                # self.binary_image = self.correct_with_previous_binary_image(self.binary_image.copy())

            if logical != 'None':
                # logging.info("Segment the image using Otsu thresholding")
                if filter_spec is not None and filter_spec["filter2_type"] != "":
                    self.image2 = apply_filter(self.image2, filter_spec["filter2_type"], filter_spec["filter2_param"])
                self.binary_image2 = otsu_thresholding(self.image2)
                if self.previous_binary_image is not None:
                    if (self.binary_image2 * (1 - self.previous_binary_image)).sum() > (
                            self.binary_image2 * self.previous_binary_image).sum():
                        self.binary_image2 = 1 - self.binary_image2
                    # self.binary_image2 = self.correct_with_previous_binary_image(self.binary_image2.copy())

        if logical != 'None':
            if logical == 'Or':
                self.binary_image = np.logical_or(self.binary_image, self.binary_image2)
            elif logical == 'And':
                self.binary_image = np.logical_and(self.binary_image, self.binary_image2)
            elif logical == 'Xor':
                self.binary_image = np.logical_xor(self.binary_image, self.binary_image2)
            self.binary_image = self.binary_image.astype(np.uint8)


    def correct_with_previous_binary_image(self, binary_image):
        # If binary image is more than twenty times bigger or smaller than the previous binary image:
        # otsu thresholding failed, we use a threshold of 127 instead
        if binary_image.sum() > self.previous_binary_image.sum() * 20 or binary_image.sum() < self.previous_binary_image.sum() * 0.05:
            binary_adaptive = cv2.adaptiveThreshold(bracket_to_uint8_image_contrast(self.image), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
            # from skimage import filters
            # threshold_value = filters.threshold_li(self.image)
            # binary_image = self.image >= threshold_value
            binary_image = self.image >= 127
            # And again, make sure than these pixels are shared with the previous binary image
            if (binary_image * (1 - self.previous_binary_image)).sum() > (binary_image * self.previous_binary_image).sum():
                binary_image = 1 - binary_image
        return binary_image.astype(np.uint8)


    def get_largest_shape(self):
        shape_number, shapes, stats, centroids = cv2.connectedComponentsWithStats(self.binary_image)
        sorted_area = np.sort(stats[1:, 4])
        self.validated_shapes = np.zeros(self.binary_image.shape, dtype=np.uint8)
        self.validated_shapes[np.nonzero(shapes == np.nonzero(stats[:, 4] == sorted_area[-1])[0])] = 1

    def generate_subtract_background(self, c_space_dict):
        logging.info("Generate background using the generate_subtract_background method of OneImageAnalysis class")
        if len(self.all_c_spaces) == 0 and not self.already_greyscale:
            self.all_c_spaces = get_color_spaces(self.bgr)
        self.convert_and_segment(c_space_dict, grid_segmentation=False)
        # self.image = generate_color_space_combination(c_space_dict, self.all_c_spaces)
        disk_size = int(np.floor(np.sqrt(np.min(self.bgr.shape[:2])) / 2))
        disk = np.uint8(Ellipse((disk_size, disk_size)).create())
        self.subtract_background = cv2.morphologyEx(self.image, cv2.MORPH_OPEN, disk)
        if self.image2 is not None:
            self.subtract_background2 = cv2.morphologyEx(self.image2, cv2.MORPH_OPEN, disk)

    def check_if_image_border_attest_drift_correction(self):
        t = np.all(self.binary_image[0, :])
        b = np.all(self.binary_image[-1, :])
        l = np.all(self.binary_image[:, 0])
        r = np.all(self.binary_image[:, -1])
        if (t and b) or (t and r) or (t and l) or (t and r) or (b and l) or (b and r) or (l and r):
            cc_nb, shapes = cv2.connectedComponents(self.binary_image)
            if cc_nb == 2:
                return True
            else:
                return False
        else:
            return False

    def adjust_to_drift_correction(self, logical):
        if not self.drift_correction_already_adjusted:
            self.drift_correction_already_adjusted = True

            mask = cv2.dilate(self.binary_image, kernel=cross_33)
            mask -= self.binary_image
            mask = np.nonzero(mask)

            drift_correction = np.mean(self.image[mask[0], mask[1]])
            self.image[np.nonzero(self.binary_image)] = drift_correction
            threshold = get_otsu_threshold(self.image)
            binary = (self.image > threshold)
            # while np.any(binary * self.binary_image) and threshold > 1: #binary.sum() > self.binary_image.sum()
            #     threshold -= 1
            #     binary1 = (self.image > threshold)
            #     binary2 = np.logical_not(binary1)
            #     if binary1.sum() < binary2.sum():
            #         binary = binary1
            #     else:
            #         binary = binary2
            self.binary_image = binary.astype(np.uint8)

            if self.image2 is not None:
                drift_correction2 = np.mean(self.image2[mask[0], mask[1]])
                self.image2[np.nonzero(self.binary_image)] = drift_correction2
                threshold = get_otsu_threshold(self.image2)
                binary1 = (self.image2 > threshold)
                binary2 = np.logical_not(binary1)
                if binary1.sum() < binary2.sum():
                    binary = binary1
                else:
                    binary = binary2
                while np.any(binary * self.binary_image2) and threshold > 1:  # binary.sum() > self.binary_image.sum()
                    threshold -= 1
                    binary1 = (self.image2 > threshold)
                    binary2 = np.logical_not(binary1)
                    if binary1.sum() < binary2.sum():
                        binary = binary1
                    else:
                        binary = binary2
                self.binary_image2 = binary.astype(np.uint8)
                if logical == 'Or':
                    self.binary_image = np.logical_or(self.binary_image, self.binary_image2)
                elif logical == 'And':
                    self.binary_image = np.logical_and(self.binary_image, self.binary_image2)
                elif logical == 'Xor':
                    self.binary_image = np.logical_xor(self.binary_image, self.binary_image2)
                self.binary_image = self.binary_image.astype(np.uint8)

    def set_spot_shapes_and_size_confint(self, spot_shape):
        self.spot_size_confints = np.arange(0.75, 0.00, - 0.05)# np.concatenate((np.arange(0.75, 0.00, - 0.05), np.arange(0.05, 0.00, -0.005)))#
        if spot_shape is None:
            self.spot_shapes = np.tile(["circle", "rectangle"], len(self.spot_size_confints))
            self.spot_size_confints = np.repeat(self.spot_size_confints, 2)
        else:
            self.spot_shapes = np.repeat(spot_shape, len(self.spot_size_confints))

    def find_first_im_csc(self, sample_number=None, several_blob_per_arena=True,  spot_shape=None, spot_size=None, kmeans_clust_nb=None, biomask=None, backmask=None, color_space_dictionaries=None, carefully=False):
        logging.info(f"Prepare color space lists, dictionaries and matrices")
        if len(self.all_c_spaces) == 0:
            self.all_c_spaces = get_color_spaces(self.bgr)
        if color_space_dictionaries is None:
            if carefully:
                colorspace_list = ["bgr", "lab", "hsv", "luv", "hls", "yuv"]
            else:
                colorspace_list = ["lab", "hsv"]
            color_space_dictionaries = TList()
            for i, c_space in enumerate(colorspace_list):
                for i in np.arange(3):
                    channels = np.array((0, 0, 0), dtype=np.int8)
                    channels[i] = 1
                    csc_dict = TDict()
                    csc_dict[c_space] = channels
                    color_space_dictionaries.append(csc_dict)

        # if not several_blob_per_arena:
        self.set_spot_shapes_and_size_confint(spot_shape)

        self.combination_features = np.zeros((len(color_space_dictionaries) + 50, 11), dtype=np.uint32)
        # ["c1", "c2", "c3", "unaltered_cc_nb", "concomp_nb", "total_area", "width_std", "height_std", "centrodist_std", "biosum", "backsum"]
        unaltered_cc_nb, cc_nb, area, width_std, height_std, area_std, biosum, backsum = 3, 4, 5, 6, 7, 8, 9, 10
        self.saved_images_list = TList()
        self.converted_images_list = TList()
        self.saved_color_space_list = list()
        self.saved_csc_nb = 0
        self.save_combination_thread = SaveCombinationThread(self)
        get_one_channel_result = True
        combine_channels = False

        for csc_dict in color_space_dictionaries:
            logging.info(f"Try detection with each color space channel, one by one. Currently analyzing {csc_dict}")
            list_args = [self, get_one_channel_result, combine_channels, csc_dict, several_blob_per_arena,
                         sample_number, spot_size, kmeans_clust_nb, biomask, backmask, None]
            ProcessFirstImage(list_args)
            # logging.info(csc_dict)

        if sample_number is not None and carefully:
            # tic = default_timer()
            # Try to add csc together
            # possibilities = np.arange(len(self.saved_color_space_list))
            possibilities = []
            if self.saved_csc_nb > 6:
                different_color_spaces = np.unique(self.saved_color_space_list)
                for color_space in different_color_spaces:
                    csc_idx = np.nonzero(np.isin(self.saved_color_space_list, color_space))[0]
                    possibilities.append(csc_idx[0] + np.argmin(self.combination_features[csc_idx, area_std]))
                if len(possibilities) < 6:
                    remaining_possibilities = np.arange(len(self.saved_color_space_list))
                    remaining_possibilities = remaining_possibilities[np.logical_not(np.isin(remaining_possibilities, possibilities))]
                    while len(possibilities) < 6:
                        new_possibility = np.argmin(self.combination_features[remaining_possibilities, area_std])
                        possibilities.append(new_possibility)
                        remaining_possibilities = remaining_possibilities[remaining_possibilities != new_possibility]


            pool = mp.ThreadPool(processes=os.cpu_count() - 1)
            get_one_channel_result = False
            combine_channels = True
            list_args = [[self, get_one_channel_result, combine_channels, i, several_blob_per_arena, sample_number,
                          spot_size, kmeans_clust_nb, biomask, backmask, possibilities] for i in possibilities]
            for process_i in pool.imap_unordered(ProcessFirstImage, list_args):
                pass

        # Get the most and the least covered images and the 2 best biomask and backmask scores
        # To try combinations of those
        if self.saved_csc_nb > 1:
            coverage = np.argsort(self.combination_features[:self.saved_csc_nb, area])
            most1 = coverage[-1]; most2 = coverage[-2]
            least1 = coverage[0]; least2 = coverage[1]
            if biomask is not None:
                bio_sort = np.argsort(self.combination_features[:self.saved_csc_nb, biosum])
                bio1 = bio_sort[-1]; bio2 = bio_sort[-2]
            if backmask is not None:
                back_sort = np.argsort(self.combination_features[:self.saved_csc_nb, backsum])
                back1 = back_sort[-1]; back2 = back_sort[-2]

            # Try a logical And between the most covered images
            # Should only need one instanciation
            process_i = ProcessFirstImage(
                [self, False, False, None, several_blob_per_arena, sample_number, spot_size, kmeans_clust_nb, biomask, backmask, None])
            process_i.binary_image = np.logical_and(self.saved_images_list[most1], self.saved_images_list[most2]).astype(np.uint8)
            process_i.image = self.converted_images_list[most1]
            process_i.process_binary_image()
            process_i.csc_dict = {list(self.saved_color_space_list[most1].keys())[0]: self.combination_features[most1, :3],
                        "logical": "And",
                        list(self.saved_color_space_list[most2].keys())[0] + "2": self.combination_features[most2, :3]}
            process_i.unaltered_concomp_nb = np.min(self.combination_features[(most1, most2), unaltered_cc_nb])
            process_i.total_area = process_i.binary_image.sum()
            self.save_combination_features(process_i)
            process_i.image = self.converted_images_list[least1]
            process_i.binary_image = np.logical_or(self.saved_images_list[least1], self.saved_images_list[least2]).astype(np.uint8)
            process_i.process_binary_image()
            process_i.csc_dict = {list(self.saved_color_space_list[least1].keys())[0]: self.combination_features[least1, :3],
                        "logical": "Or",
                        list(self.saved_color_space_list[least2].keys())[0] + "2": self.combination_features[least2, :3]}
            process_i.unaltered_concomp_nb = np.max(self.combination_features[(least1, least2), unaltered_cc_nb])
            process_i.total_area = process_i.binary_image.sum()
            self.save_combination_features(process_i)

            # self.save_combination_features(csc_dict, unaltered_concomp_nb, self.binary_image.sum(), biomask, backmask)

            # If most images are very low in biosum or backsum, try to mix them together to improve that score
            # Do a logical And between the two best biomasks
            if biomask is not None:
                if not np.all(np.isin((bio1, bio2), (most1, most2))):
                    process_i.image = self.converted_images_list[bio1]
                    process_i.binary_image = np.logical_and(self.saved_images_list[bio1], self.saved_images_list[bio2]).astype(
                    np.uint8)
                    process_i.process_binary_image()
                    process_i.csc_dict = {list(self.saved_color_space_list[bio1].keys())[0]: self.combination_features[bio1, :3],
                                "logical": "And",
                                list(self.saved_color_space_list[bio2].keys())[0] + "2": self.combination_features[bio2,:3]}
                    process_i.unaltered_concomp_nb = np.min(self.combination_features[(bio1, bio2), unaltered_cc_nb])
                    process_i.total_area = process_i.binary_image.sum()

                    self.save_combination_features(process_i)

            # Do a logical And between the two best backmask
            if backmask is not None:

                if not np.all(np.isin((back1, back2), (most1, most2))):
                    process_i.image = self.converted_images_list[back1]
                    process_i.binary_image = np.logical_and(self.saved_images_list[back1], self.saved_images_list[back2]).astype(
                    np.uint8)
                    process_i.process_binary_image()
                    process_i.csc_dict = {list(self.saved_color_space_list[back1].keys())[0]: self.combination_features[back1, :3],
                                "logical": "And",
                                list(self.saved_color_space_list[back2].keys())[0] + "2": self.combination_features[back2,:3]}
                    process_i.unaltered_concomp_nb = np.min(self.combination_features[(back1, back2), unaltered_cc_nb])
                    process_i.total_area = process_i.binary_image.sum()
                    self.save_combination_features(process_i)
            # Do a logical Or between the best biomask and the best backmask
            if biomask is not None and backmask is not None:
                if not np.all(np.isin((bio1, back1), (least1, least2))):
                    process_i.image = self.converted_images_list[bio1]
                    process_i.binary_image = np.logical_and(self.saved_images_list[bio1], self.saved_images_list[back1]).astype(
                        np.uint8)
                    process_i.process_binary_image()
                    process_i.csc_dict = {list(self.saved_color_space_list[bio1].keys())[0]: self.combination_features[bio1, :3],
                                "logical": "Or",
                                list(self.saved_color_space_list[back1].keys())[0] + "2": self.combination_features[back1, :3]}
                    process_i.unaltered_concomp_nb = np.max(self.combination_features[(bio1, back1), unaltered_cc_nb])
                    # self.save_combination_features(csc_dict, unaltered_concomp_nb, self.binary_image.sum(), biomask,
                    #                                backmask)
                    process_i.total_area = self.binary_image.sum()
                    self.save_combination_features(process_i)

            if self.save_combination_thread.is_alive():
                self.save_combination_thread.join()
            self.combination_features = self.combination_features[:self.saved_csc_nb, :]
            # Only keep the row that filled conditions
            # Save all combinations if they fulfill the following conditions:
            #   - Their conncomp number is lower than 3 times the smaller conncomp number.
            #   - OR The minimal area variations
            #   - OR The minimal width variations
            #   - OR The minimal height variations
            #   - AND/OR their segmentation fits with biomask and backmask
            width_std_fit = self.combination_features[:, width_std] == np.min(self.combination_features[:, width_std])
            height_std_fit = self.combination_features[:, height_std] == np.min(self.combination_features[:, height_std])
            area_std_fit = self.combination_features[:, area_std] < np.min(self.combination_features[:, area_std]) * 10
            fit = np.logical_or(np.logical_or(width_std_fit, height_std_fit), area_std_fit)
            biomask_fit = np.ones(self.saved_csc_nb, dtype=bool)
            backmask_fit = np.ones(self.saved_csc_nb, dtype=bool)
            if biomask is not None or backmask is not None:
                if biomask is not None:
                    biomask_fit = self.combination_features[:, biosum] > 0.9 * len(biomask[0])
                if backmask is not None:
                    backmask_fit = self.combination_features[:, backsum] > 0.9 * len(backmask[0])
                # First test a logical OR between the precedent options and the mask fits.
                fit = np.logical_or(fit, np.logical_and(biomask_fit, backmask_fit))
                # If this is not stringent enough, use a logical AND and increase progressively the proportion of pixels that
                # must match the biomask and the backmask
                if np.sum(fit) > 5:
                    to_add = 0
                    while np.sum(fit) > 5 and to_add <= 0.25:
                        if biomask is not None:
                            biomask_fit = self.combination_features[:, biosum] > (0.75 + to_add) * len(biomask[0])
                        if backmask is not None:
                            backmask_fit = self.combination_features[:, backsum] > (0.75 + to_add) * len(backmask[0])
                        test_fit = np.logical_and(fit, np.logical_and(biomask_fit, backmask_fit))
                        if np.sum(test_fit) != 0:
                            fit = test_fit
                        to_add += 0.05
        else:
            self.combination_features = self.combination_features[:self.saved_csc_nb, :]
            fit = np.array([True])
        # If saved_csc_nb is too low, try bool operators to mix them together to fill holes for instance
        # Order the table according to the number of shapes that have been removed by filters
        # cc_efficiency_order = np.argsort(self.combination_features[:, unaltered_cc_nb] - self.combination_features[:, cc_nb])
        cc_efficiency_order = np.argsort(self.combination_features[:, area_std])
        # Save and return a dictionnary containing the selected color space combinations
        # and their corresponding binary images

        # first_im_combinations = [i for i in np.arange(fit.sum())]
        self.im_combinations = []
        for saved_csc in cc_efficiency_order:
            if fit[saved_csc]:
                self.im_combinations.append({})
                # self.im_combinations.append({})
                # self.im_combinations[len(self.im_combinations) - 1]["csc"] = self.saved_color_space_list[saved_csc]
                self.im_combinations[len(self.im_combinations) - 1]["csc"] = {}
                self.im_combinations[len(self.im_combinations) - 1]["csc"]['logical'] = 'None'
                for k, v in self.saved_color_space_list[saved_csc].items():
                    self.im_combinations[len(self.im_combinations) - 1]["csc"][k] = v
                # self.im_combinations[len(self.im_combinations) - 1]["csc"] = {list(self.saved_color_space_list[saved_csc])[0]: self.combination_features[saved_csc, :3]}

                if backmask is not None:
                    shape_number, shapes = cv2.connectedComponents(self.saved_images_list[saved_csc], connectivity=8)
                    if np.any(shapes[backmask]):
                        shapes[np.isin(shapes, np.unique(shapes[backmask]))] = 0
                        self.saved_images_list[saved_csc] = (shapes > 0).astype(np.uint8)
                if biomask is not None:
                    self.saved_images_list[saved_csc][biomask] = 1
                if backmask is not None or biomask is not None:
                    self.combination_features[saved_csc, cc_nb], shapes = cv2.connectedComponents(self.saved_images_list[saved_csc], connectivity=8)
                    self.combination_features[saved_csc, cc_nb] -= 1
                self.im_combinations[len(self.im_combinations) - 1]["binary_image"] = self.saved_images_list[saved_csc]
                self.im_combinations[len(self.im_combinations) - 1]["shape_number"] = self.combination_features[saved_csc, cc_nb]
                self.im_combinations[len(self.im_combinations) - 1]["converted_image"] = self.converted_images_list[saved_csc]

        # logging.info(default_timer()-tic)
        self.saved_color_space_list = []
        self.saved_images_list = None
        self.converted_images_list = None
        self.combination_features = None

    # def save_combination_features(self, process_i):
    #     if self.save_combination_thread.is_alive():
    #         self.save_combination_thread.join()
    #     self.save_combination_thread = SaveCombinationThread(self)
    #     self.save_combination_thread.process_i = process_i
    #     self.save_combination_thread.start()

    def save_combination_features(self, process_i):
        self.saved_images_list.append(process_i.validated_shapes)
        self.converted_images_list.append(np.round(process_i.image).astype(np.uint8))
        self.saved_color_space_list.append(process_i.csc_dict)
        self.combination_features[self.saved_csc_nb, :3] = list(process_i.csc_dict.values())[0]
        self.combination_features[
            self.saved_csc_nb, 3] = process_i.unaltered_concomp_nb - 1  # unaltered_cc_nb
        self.combination_features[self.saved_csc_nb, 4] = process_i.shape_number  # cc_nb
        self.combination_features[self.saved_csc_nb, 5] = process_i.total_area  # area
        self.combination_features[self.saved_csc_nb, 6] = np.std(process_i.stats[1:, 2])  # width_std
        self.combination_features[self.saved_csc_nb, 7] = np.std(process_i.stats[1:, 3])  # height_std
        self.combination_features[self.saved_csc_nb, 8] = np.std(process_i.stats[1:, 4])  # area_std
        if process_i.biomask is not None:
            self.combination_features[self.saved_csc_nb, 9] = np.sum(
                process_i.validated_shapes[process_i.biomask[0], process_i.biomask[1]])
        if process_i.backmask is not None:
            self.combination_features[self.saved_csc_nb, 10] = np.sum(
                (1 - process_i.validated_shapes)[process_i.backmask[0], process_i.backmask[1]])
        self.saved_csc_nb += 1

    def update_current_images(self, current_combination_id):
        self.image = self.im_combinations[current_combination_id]["converted_image"]
        self.validated_shapes = self.im_combinations[current_combination_id]["binary_image"]

    def find_last_im_csc(self, concomp_nb, total_surfarea, max_shape_size, out_of_arenas=None, ref_image=None,
                                subtract_background=None, kmeans_clust_nb=None, biomask=None, backmask=None,
                                color_space_dictionaries=None, carefully=False):
        if len(self.all_c_spaces) == 0:
            self.all_c_spaces = get_color_spaces(self.bgr)
        if color_space_dictionaries is None:
            if carefully:
                colorspace_list = TList(("bgr", "lab", "hsv", "luv", "hls", "yuv"))
            else:
                colorspace_list = TList(("lab", "hsv"))
            color_space_dictionaries = TList()
            for i, c_space in enumerate(colorspace_list):
                for i in np.arange(3):
                    channels = np.array((0, 0, 0), dtype=np.int8)
                    channels[i] = 1
                    csc_dict = TDict()
                    csc_dict[c_space] = channels
                    color_space_dictionaries.append(csc_dict)
        if ref_image is not None:
            ref_image = cv2.dilate(ref_image, cross_33)
        else:
            ref_image = np.ones(self.bgr.shape[:2], dtype=np.uint8)
        if out_of_arenas is not None:
            out_of_arenas_threshold = 0.01 * out_of_arenas.sum()
        else:
            out_of_arenas = np.zeros(self.bgr.shape[:2], dtype=np.uint8)
            out_of_arenas_threshold = 1
        self.combination_features = np.zeros((len(color_space_dictionaries) + 50, 9), dtype=np.uint32)
        cc_nb_idx, area_idx, out_of_arenas_idx, surf_in_common_idx, biosum_idx, backsum_idx = 3, 4, 5, 6, 7, 8
        self.saved_images_list = TList()
        self.converted_images_list = TList()
        self.saved_color_space_list = list()
        self.saved_csc_nb = 0
        self.save_combination_thread = SaveCombinationThread(self)

        # One channel processing
        potentials = TDict()
        for csc_dict in color_space_dictionaries:
            self.image = combine_color_spaces(csc_dict, self.all_c_spaces, subtract_background)
            # self.generate_color_space_combination(c_space_dict, subtract_background)
            if kmeans_clust_nb is not None and (biomask is not None or backmask is not None):
                self.kmeans(kmeans_clust_nb, biomask, backmask)
            else:
                self.binary_image = otsu_thresholding(self.image)
            surf = np.sum(self.binary_image)
            if surf < total_surfarea:
                # nb, shapes = cv2.connectedComponents(oia.binary_image)
                nb, shapes = cv2.connectedComponents(self.binary_image)
                # outside_pixels = np.sum(oia.binary_image * out_of_arenas)
                outside_pixels = np.sum(self.binary_image * out_of_arenas)
                if outside_pixels < out_of_arenas_threshold:
                    if (nb > concomp_nb[0]) and (nb < concomp_nb[1]):
                        # in_common = np.sum(ref_image * oia.binary_image)
                        in_common = np.sum(ref_image * self.binary_image)
                        if in_common > 0:
                            nb, shapes, stats, centroids = cv2.connectedComponentsWithStats(self.binary_image)
                            nb -= 1
                            if np.all(np.sort(stats[:, 4])[:-1] < max_shape_size):
                                # oia.viewing()
                                c_space = list(csc_dict.keys())[0]
                                self.converted_images_list.append(self.image)
                                self.saved_images_list.append(self.binary_image)
                                self.saved_color_space_list.append(csc_dict)
                                self.combination_features[self.saved_csc_nb, :3] = csc_dict[c_space]
                                self.combination_features[self.saved_csc_nb, cc_nb_idx] = nb
                                self.combination_features[self.saved_csc_nb, area_idx] = surf
                                self.combination_features[self.saved_csc_nb, out_of_arenas_idx] = outside_pixels
                                self.combination_features[self.saved_csc_nb, surf_in_common_idx] = in_common
                                if biomask is not None:
                                    self.combination_features[self.saved_csc_nb, biosum_idx] = np.sum(
                                        self.binary_image[biomask[0], biomask[1]])
                                if backmask is not None:
                                    self.combination_features[self.saved_csc_nb, backsum_idx] = np.sum(
                                        (1 - self.binary_image)[backmask[0], backmask[1]])
                                if np.isin(c_space, list(potentials.keys())):
                                    potentials[c_space] += csc_dict[c_space]
                                else:
                                    potentials[c_space] = csc_dict[c_space]
                                self.saved_csc_nb += 1
        if len(potentials) > 0:
            # All combination processing

            # Add a combination of all selected channels :
            self.saved_color_space_list.append(potentials)
            # all_potential_combinations.append(potentials)
            self.image = combine_color_spaces(potentials, self.all_c_spaces, subtract_background)
            # self.generate_color_space_combination(potentials, subtract_background)
            if kmeans_clust_nb is not None and (biomask is not None or backmask is not None):
                self.kmeans(kmeans_clust_nb, biomask, backmask)
            else:
                self.binary_image = otsu_thresholding(self.image)
            # self.thresholding()
            surf = self.binary_image.sum()
            nb, shapes = cv2.connectedComponents(self.binary_image)
            nb -= 1
            outside_pixels = np.sum(self.binary_image * out_of_arenas)
            in_common = np.sum(ref_image * self.binary_image)
            self.converted_images_list.append(self.image)
            self.saved_images_list.append(self.binary_image)
            self.saved_color_space_list.append(potentials)
            self.combination_features[self.saved_csc_nb, :3] = list(potentials.values())[0]
            self.combination_features[self.saved_csc_nb, cc_nb_idx] = nb
            self.combination_features[self.saved_csc_nb, area_idx] = surf
            self.combination_features[self.saved_csc_nb, out_of_arenas_idx] = outside_pixels
            self.combination_features[self.saved_csc_nb, surf_in_common_idx] = in_common
            if biomask is not None:
                self.combination_features[self.saved_csc_nb, biosum_idx] = np.sum(
                    self.binary_image[biomask[0], biomask[1]])
            if backmask is not None:
                self.combination_features[self.saved_csc_nb, backsum_idx] = np.sum(
                    (1 - self.binary_image)[backmask[0], backmask[1]])
            self.saved_csc_nb += 1
        # current = {"total_area": surf, "concomp_nb": nb, "out_of_arenas": outside_pixels,
        #            "surf_in_common": in_common}
        # combination_features = combination_features.append(current, ignore_index=True)

        # All combination processing
        # Try to remove color space one by one
        i = 0
        original_length = len(potentials)
        while np.logical_and(len(potentials) > 1, i < original_length // 2):
            color_space_to_remove = TList()
            # The while loop until one col space remains or the removal of one implies a strong enough area change
            previous_c_space = list(potentials.keys())[-1]
            for c_space in potentials.keys():
                try_potentials = potentials.copy()
                try_potentials.pop(c_space)
                if i > 0:
                    try_potentials.pop(previous_c_space)
                self.image = combine_color_spaces(try_potentials, self.all_c_spaces, subtract_background)
                # self.generate_color_space_combination(try_potentials, subtract_background)
                if kmeans_clust_nb is not None and (biomask is not None or backmask is not None):
                    self.kmeans(kmeans_clust_nb, biomask, backmask)
                else:
                    self.binary_image = otsu_thresholding(self.image)
                # self.thresholding()
                surf = np.sum(self.binary_image)
                if surf < total_surfarea:
                    nb, shapes = cv2.connectedComponents(self.binary_image)
                    outside_pixels = np.sum(self.binary_image * out_of_arenas)
                    if outside_pixels < out_of_arenas_threshold:
                        if (nb > concomp_nb[0]) and (nb < concomp_nb[1]):
                            in_common = np.sum(ref_image * self.binary_image)
                            if in_common > 0:
                                nb, shapes, stats, centroids = cv2.connectedComponentsWithStats(self.binary_image)
                                nb -= 1
                                if np.all(np.sort(stats[:, 4])[:-1] < max_shape_size):
                                    # If a color space remove fits in the requirements, we store its values
                                    self.converted_images_list.append(self.image)
                                    self.saved_images_list.append(self.binary_image)
                                    self.saved_color_space_list.append(try_potentials)
                                    self.combination_features[self.saved_csc_nb, cc_nb_idx] = nb
                                    self.combination_features[self.saved_csc_nb, area_idx] = surf
                                    self.combination_features[self.saved_csc_nb, out_of_arenas_idx] = outside_pixels
                                    self.combination_features[self.saved_csc_nb, surf_in_common_idx] = in_common
                                    if biomask is not None:
                                        self.combination_features[self.saved_csc_nb, biosum_idx] = np.sum(
                                            self.binary_image[biomask[0], biomask[1]])
                                    if backmask is not None:
                                        self.combination_features[self.saved_csc_nb, backsum_idx] = np.sum(
                                            (1 - self.binary_image)[backmask[0], backmask[1]])
                                    self.saved_csc_nb += 1
                                    # all_potential_combinations.append(try_potentials)
                                    # current = {"total_area": surf, "concomp_nb": nb, "out_of_arenas": outside_pixels,
                                    #            "surf_in_common": in_common}
                                    # combination_features = combination_features.append(current, ignore_index=True)
                                    color_space_to_remove.append(c_space)
                                    if i > 0:
                                        color_space_to_remove.append(previous_c_space)
                # If it does not (if it did not pass every "if" layers), we definitely remove that color space
                previous_c_space = c_space
            color_space_to_remove = np.unique(color_space_to_remove)
            for remove_col_space in color_space_to_remove:
                potentials.pop(remove_col_space)
            i += 1
        if np.logical_and(len(potentials) > 0, i > 1):
            self.converted_images_list.append(self.image)
            self.saved_images_list.append(self.binary_image)
            self.saved_color_space_list.append(potentials)
            self.combination_features[self.saved_csc_nb, :3] = list(potentials.values())[0]
            self.combination_features[self.saved_csc_nb, cc_nb_idx] = nb
            self.combination_features[self.saved_csc_nb, area_idx] = surf
            self.combination_features[self.saved_csc_nb, out_of_arenas_idx] = outside_pixels
            self.combination_features[self.saved_csc_nb, surf_in_common_idx] = in_common
            if biomask is not None:
                self.combination_features[self.saved_csc_nb, biosum_idx] = np.sum(
                    self.binary_image[biomask[0], biomask[1]])
            if backmask is not None:
                self.combination_features[self.saved_csc_nb, backsum_idx] = np.sum(
                    (1 - self.binary_image)[backmask[0], backmask[1]])
            self.saved_csc_nb += 1
            # all_potential_combinations.append(potentials)
            # current = {"total_area": surf, "concomp_nb": nb, "out_of_arenas": outside_pixels,
            #            "surf_in_common": in_common}
            # combination_features = combination_features.append(current, ignore_index=True)

        self.combination_features = self.combination_features[:self.saved_csc_nb, :]
        # Among all potentials, select the best one, according to criterion decreasing in importance
        # a = combination_features.sort_values(by=["surf_in_common"], ascending=False)
        # self.channel_combination = all_potential_combinations[a[:1].index[0]]
        cc_efficiency_order = np.argsort(self.combination_features[:, surf_in_common_idx])

        # Save and return a dictionnary containing the selected color space combinations
        # and their corresponding binary images

        self.im_combinations = []
        for saved_csc in cc_efficiency_order:
            if len(self.saved_color_space_list[saved_csc]) > 0:
                self.im_combinations.append({})
                self.im_combinations[len(self.im_combinations) - 1]["csc"] = {}
                self.im_combinations[len(self.im_combinations) - 1]["csc"]['logical'] = 'None'
                for k, v in self.saved_color_space_list[saved_csc].items():
                    self.im_combinations[len(self.im_combinations) - 1]["csc"][k] = v
                self.im_combinations[len(self.im_combinations) - 1]["binary_image"] = self.saved_images_list[saved_csc]
                self.im_combinations[len(self.im_combinations) - 1]["converted_image"] = np.round(self.converted_images_list[
                    saved_csc]).astype(np.uint8)
        self.saved_color_space_list = []
        self.saved_images_list = None
        self.converted_images_list = None
        self.combination_features = None

    """
        Thresholding is a very simple and fast segmentation method. Kmeans can be implemented in a function bellow
    """
    def thresholding(self, luminosity_threshold=None, lighter_background=None):
        if luminosity_threshold is not None:
            binarymg = np.zeros(self.image.shape, dtype=np.uint8)
            if lighter_background:
                binarymg[self.image < luminosity_threshold] = 1
            else:
                binarymg[self.image > luminosity_threshold] = 1
        else:
            ret, binarymg = cv2.threshold(self.image, 0, 1, cv2.THRESH_OTSU)
        #binarymg = binarymg - 1
        # Make sure that blobs are 1 and background is 0
        if np.sum(binarymg) > np.sum(1 - binarymg):
            binarymg = 1 - binarymg
        self.binary_image = binarymg

    def kmeans(self, cluster_number, biomask=None, backmask=None, logical='None', bio_label=None, bio_label2=None):
        image = self.image.reshape((-1, 1))
        image = np.float32(image)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        compactness, label, center = cv2.kmeans(image, cluster_number, None, criteria, attempts=10, flags=cv2.KMEANS_RANDOM_CENTERS)
        kmeans_image = np.uint8(label.flatten().reshape(self.image.shape[:2]))
        sum_per_label = np.zeros(cluster_number)
        self.binary_image = np.zeros(self.image.shape[:2], np.uint8)
        if self.previous_binary_image is not None:
            binary_images = []
            image_scores = np.zeros(cluster_number, np.uint64)
            for i in range(cluster_number):
                binary_image_i = np.zeros(self.image.shape[:2], np.uint8)
                binary_image_i[np.nonzero(kmeans_image == i)] = 1
                image_scores[i] = (binary_image_i * self.previous_binary_image).sum()
                binary_images.append(binary_image_i)
            self.binary_image[np.nonzero(kmeans_image == np.argmax(image_scores))] = 1
        elif bio_label is not None:
            self.binary_image[np.nonzero(kmeans_image == bio_label)] = 1
            self.bio_label = bio_label
        else:
            if biomask is not None:
                all_labels = kmeans_image[biomask[0], biomask[1]]
                for i in range(cluster_number):
                    sum_per_label[i] = (all_labels == i).sum()
                self.bio_label = np.nonzero(sum_per_label == np.max(sum_per_label))
            elif backmask is not None:
                all_labels = kmeans_image[backmask[0], backmask[1]]
                for i in range(cluster_number):
                    sum_per_label[i] = (all_labels == i).sum()
                self.bio_label = np.nonzero(sum_per_label == np.min(sum_per_label))
            else:
                for i in range(cluster_number):
                    sum_per_label[i] = (kmeans_image == i).sum()
                self.bio_label = np.nonzero(sum_per_label == np.min(sum_per_label))
            self.binary_image[np.nonzero(kmeans_image == self.bio_label)] = 1

        if logical != 'None':
            image = self.image2.reshape((-1, 1))
            image = np.float32(image)
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
            compactness, label, center = cv2.kmeans(image, cluster_number, None, criteria, attempts=10,
                                                    flags=cv2.KMEANS_RANDOM_CENTERS)
            kmeans_image = np.uint8(label.flatten().reshape(self.image.shape[:2]))
            sum_per_label = np.zeros(cluster_number)
            self.binary_image2 = np.zeros(self.image.shape[:2], np.uint8)
            if self.previous_binary_image is not None:
                binary_images = []
                image_scores = np.zeros(cluster_number, np.uint64)
                for i in range(cluster_number):
                    binary_image_i = np.zeros(self.image.shape[:2], np.uint8)
                    binary_image_i[np.nonzero(kmeans_image == i)] = 1
                    image_scores[i] = (binary_image_i * self.previous_binary_image).sum()
                    binary_images.append(binary_image_i)
                self.binary_image2[np.nonzero(kmeans_image == np.argmax(image_scores))] = 1
            elif bio_label2 is not None:
                self.binary_image2[np.nonzero(kmeans_image == bio_label2)] = 1
                self.bio_label2 = bio_label2
            else:
                if biomask is not None:
                    all_labels = kmeans_image[biomask[0], biomask[1]]
                    for i in range(cluster_number):
                        sum_per_label[i] = (all_labels == i).sum()
                    self.bio_label2 = np.nonzero(sum_per_label == np.max(sum_per_label))
                elif backmask is not None:
                    all_labels = kmeans_image[backmask[0], backmask[1]]
                    for i in range(cluster_number):
                        sum_per_label[i] = (all_labels == i).sum()
                    self.bio_label2 = np.nonzero(sum_per_label == np.min(sum_per_label))
                else:
                    for i in range(cluster_number):
                        sum_per_label[i] = (kmeans_image == i).sum()
                    self.bio_label2 = np.nonzero(sum_per_label == np.min(sum_per_label))
                self.binary_image2[np.nonzero(kmeans_image == self.bio_label2)] = 1

    def binarize_k_means_product(self, grey_idx):
        binarization = np.zeros_like(self.binary_image)
        binarization[np.nonzero(self.binary_image == grey_idx)] = 1
        self.binary_image = binarization

    def grid_segmentation(self, lighter_background, side_length=8, step=2, int_variation_thresh=20, mask=None):
        """
        Segment small squares of the images to detect local intensity valleys
        This method segment the image locally using otsu thresholding on a rolling window
        :param side_length: The size of the window to detect the blobs
        :type side_length: uint8
        :param step:
        :type step: uint8
        :return:
        """
        if len(self.image.shape) == 3:
            print("Image is not Grayscale")
        if mask is None:
            min_y = 0
            min_x = 0
            y_size = self.image.shape[0]
            x_size = self.image.shape[1]
            max_y = y_size + 1
            max_x = x_size + 1
            mask = np.ones_like(self.image)
        else:
            y, x = np.nonzero(mask)
            min_y = np.min(y)
            if (min_y - 20) >= 0:
                min_y -= 20
            else:
                min_y = 0
            max_y = np.max(y) + 1
            if (max_y + 20) < mask.shape[0]:
                max_y += 20
            else:
                max_y = mask.shape[0] - 1
            min_x = np.min(x)
            if (min_x - 20) >= 0:
                min_x -= 20
            else:
                min_x = 0
            max_x = np.max(x) + 1
            if (max_x + 20) < mask.shape[1]:
                max_x += 20
            else:
                max_x = mask.shape[1] - 1
            y_size = max_y - min_y
            x_size = max_x - min_x
        grid_image = np.zeros((y_size, x_size), np.uint64)
        homogeneities = np.zeros((y_size, x_size), np.uint64)
        cropped_mask = mask[min_y:max_y, min_x:max_x]
        cropped_image = self.image[min_y:max_y, min_x:max_x]
        # will be more efficient if it only loops over a zoom on self.mask == 1
        for to_add in np.arange(0, side_length, step):
            y_windows = np.arange(0, y_size, side_length)
            x_windows = np.arange(0, x_size, side_length)
            y_windows += to_add
            x_windows += to_add
            for y_start in y_windows:
                # y_start = 4
                if y_start < self.image.shape[0]:
                    y_end = y_start + side_length
                    if y_end < self.image.shape[0]:
                        for x_start in x_windows:
                            if x_start < self.image.shape[1]:
                                x_end = x_start + side_length
                                if x_end < self.image.shape[1]:
                                    if np.any(cropped_mask[y_start:y_end, x_start:x_end]):
                                        potential_detection = cropped_image[y_start:y_end, x_start:x_end]
                                        if np.any(potential_detection):
                                            if np.ptp(potential_detection[np.nonzero(potential_detection)]) < int_variation_thresh:
                                                homogeneities[y_start:y_end, x_start:x_end] += 1

                                            threshold = get_otsu_threshold(potential_detection)
                                            if lighter_background:
                                                net_coord = np.nonzero(potential_detection < threshold)
                                            else:
                                                net_coord = np.nonzero(potential_detection > threshold)
                                            grid_image[y_start + net_coord[0], x_start + net_coord[1]] += 1

        self.binary_image = np.zeros(self.image.shape, np.uint8)
        self.binary_image[min_y:max_y, min_x:max_x] = (grid_image >= (side_length // step)).astype(np.uint8)
        self.binary_image[min_y:max_y, min_x:max_x][homogeneities >= (((side_length // step) // 2) + 1)] = 0


    """
        III/ Use validated shapes to exclude from analysis the image parts that are far from them
        i.e. detect projected shape boundaries over both axis and determine crop coordinates
    """
    def get_crop_coordinates(self, are_zigzag=None):
        logging.info("Project the image on the y axis to detect rows of arenas")
        self.y_boundaries, y_max_sum = self.projection_to_get_peaks_boundaries(axis=1)
        logging.info("Project the image on the x axis to detect columns of arenas")
        self.x_boundaries, x_max_sum = self.projection_to_get_peaks_boundaries(axis=0)
        logging.info("Get crop coordinates using the get_crop_coordinates method of OneImageAnalysis class")
        row_number = len(np.nonzero(self.y_boundaries)[0]) // 2
        col_number = len(np.nonzero(self.x_boundaries)[0]) // 2
        if (x_max_sum / col_number) * 2 < (y_max_sum / row_number):
            are_zigzag = "columns"
        elif (x_max_sum / col_number) > (y_max_sum / row_number) * 2:
            are_zigzag = "rows"
        else:
            are_zigzag = None
        # here automatically determine if are zigzag
        x_boundary_number = (self.x_boundaries == 1).sum()
        if x_boundary_number > 1:
            if x_boundary_number < 4:
                x_interval = np.absolute(np.max(np.diff(np.where(self.x_boundaries == 1)[0]))) // 2
            else:
                if are_zigzag == "columns":
                    x_interval = np.absolute(np.max(np.diff(np.where(self.x_boundaries == 1)[0][::2]))) // 2
                else:
                    x_interval = np.absolute(np.max(np.diff(np.where(self.x_boundaries == 1)[0]))) // 2
            cx_min = np.where(self.x_boundaries == - 1)[0][0] - x_interval.astype(int)
            cx_max = np.where(self.x_boundaries == 1)[0][col_number - 1] + x_interval.astype(int)
            if cx_min < 0: cx_min = 0
            if cx_max > len(self.x_boundaries): cx_max = len(self.x_boundaries) - 1
        else:
            cx_min = 0
            cx_max = len(self.x_boundaries) - 1

        y_boundary_number = (self.y_boundaries == 1).sum()
        if y_boundary_number > 1:
            if y_boundary_number < 4:
                y_interval = np.absolute(np.max(np.diff(np.where(self.y_boundaries == 1)[0]))) // 2
            else:
                if are_zigzag == "rows":
                    y_interval = np.absolute(np.max(np.diff(np.where(self.y_boundaries == 1)[0][::2]))) // 2
                else:
                    y_interval = np.absolute(np.max(np.diff(np.where(self.y_boundaries == 1)[0]))) // 2
            cy_min = np.where(self.y_boundaries == - 1)[0][0] - y_interval.astype(int)
            cy_max = np.where(self.y_boundaries == 1)[0][row_number - 1] + y_interval.astype(int)
            if cy_min < 0: cy_min = 0
            if cy_max > len(self.y_boundaries): cy_max = len(self.y_boundaries) - 1
        else:
            cy_min = 0
            cy_max = len(self.y_boundaries) - 1

        self.crop_coord = [cy_min, cy_max, cx_min, cx_max]
        return are_zigzag
        # plt.imshow(self.image)
        #plt.scatter(cx_min,cy_min)
        #plt.scatter(cx_max, cy_max)

    def projection_to_get_peaks_boundaries(self, axis):
        sums = np.sum(self.validated_shapes, axis)
        slopes = np.greater(sums, 0)
        slopes = np.append(0, np.diff(slopes))
        coord = np.nonzero(slopes)[0]
        for ci in np.arange(len(coord)):
            if ci % 2 == 0:
                slopes[coord[ci]] = - 1
        return slopes, sums.max()

    def jackknife_cutting(self, changes):
        """
        This function compare the mean distance between each 1 in a vector of 0.
        Since a few irregular intervals affect less the median that the mean,
        It try to remove each 1, one by one to see if it reduce enough the difference between mean and median.
        If the standard error of that difference is higher than 2,
        we remove each point whose removal decrease that difference by half of the median of these differences.
        i.e. differences between jackkniffed means and original median of the distance between each 1.
        """
        indices = np.nonzero(changes)[0]
        indices_to_remove = np.zeros(len(indices), dtype=bool)
        # To test the impact of a removal, changes must contain at least four 1.
        if len(indices) > 3:
            jackknifed_mean = np.zeros(np.sum(changes == 1))
            for dot_i in np.arange(len(indices)):
                steep = changes == 1
                steep[indices[dot_i]] = False
                new_indices = np.where(steep == 1)[0]
                if dot_i != 0:
                    new_indices[dot_i:] = indices[(dot_i + 1):] - (indices[dot_i] - indices[dot_i - 1])
                jackknifed_mean[dot_i] = np.mean(np.diff(new_indices))
            improving_cuts = np.absolute(jackknifed_mean - np.median(np.diff(indices)))
            if np.std(improving_cuts) > 2:
                improving_cuts = np.argwhere(improving_cuts < 0.5 * np.median(improving_cuts))
                indices_to_remove[improving_cuts] = 1
        return indices_to_remove

    def automatically_crop(self, crop_coord):
        if not self.cropped:
            logging.info("Crop using the automatically_crop method of OneImageAnalysis class")
            self.cropped = True
            self.image = self.image[crop_coord[0]:crop_coord[1], crop_coord[2]:crop_coord[3], ...]
            self.bgr = deepcopy(self.bgr[crop_coord[0]:crop_coord[1], crop_coord[2]:crop_coord[3], ...])
            if len(self.all_c_spaces) > 0:
                self.all_c_spaces = get_color_spaces(self.bgr)
            if self.im_combinations is not None:
                for i in np.arange(len(self.im_combinations)):
                    self.im_combinations[i]["binary_image"] = self.im_combinations[i]["binary_image"][crop_coord[0]:crop_coord[1], crop_coord[2]:crop_coord[3]]
                    self.im_combinations[i]["converted_image"] = self.im_combinations[i]["converted_image"][crop_coord[0]:crop_coord[1], crop_coord[2]:crop_coord[3]]
            self.binary_image = self.binary_image[crop_coord[0]:crop_coord[1], crop_coord[2]:crop_coord[3]]
            if self.image2 is not None:
                self.image2 = self.image2[crop_coord[0]:crop_coord[1], crop_coord[2]:crop_coord[3], ...]
            if self.binary_image2 is not None:
                self.binary_image2 = self.binary_image2[crop_coord[0]:crop_coord[1], crop_coord[2]:crop_coord[3], ...]
            if self.subtract_background is not None:
                self.subtract_background = self.subtract_background[crop_coord[0]:crop_coord[1], crop_coord[2]:crop_coord[3], ...]
            if self.subtract_background2 is not None:
                self.subtract_background2 = self.subtract_background2[crop_coord[0]:crop_coord[1], crop_coord[2]:crop_coord[3], ...]
            self.validated_shapes = self.validated_shapes[crop_coord[0]:crop_coord[1], crop_coord[2]:crop_coord[3]]

            self.y_boundaries, y_max_sum = self.projection_to_get_peaks_boundaries(axis=1)
            self.x_boundaries, x_max_sum = self.projection_to_get_peaks_boundaries(axis=0)


