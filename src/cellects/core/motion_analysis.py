#!/usr/bin/env python3
"""
This script contains the MotionAnalysis class. This class, called by program_organizer,
 calls all methods used to read, process videos and save results.
1. load_images_and_videos: It starts by loading a video in .npy (which must have been written before, thanks to the one_video_per_blob file)
 and if it exists, the background used for background subtraction. Then, it uses a particular color space combination
 to convert the rgb video into greyscale.
 At this point, arenas have been delimited and each can be analyzed separately. The following describes what happens during the analysis of one arena. Also, while Cellects can work with either one or several cells in each arena, we will describe the algorithm for a single cell, making clarifications whenever anything changes for multiple cells.
Cellects starts by reading and converting the video of each arena into grayscale, using the selected color space combination. Then, it processes it through the following steps.
3. It validates the presence/absence of the specimen(s) in the first image of the video, named origin.
Cellects finds the frame in which the cell is visible for the first time in each arena. When the seed image is the first image, then all cells are visible from the beginning. Otherwise, it will apply the same segmentation as for the seed image to the first, second, third images, etc. until the cell appears in one of them.
4. It browses the first frames of the video to find the average covering duration of a pixel.
It does so using a very conservative method, to make sure that only pixels that really are covered by the specimen(s)
are used to do compute that covering duration.
5. It performs the main segmentation algorithm on the whole video.
This segmentation will consist in transforming the grayscale video resulting from the color space combination conversion
 into a binary video of presence/absence. To do this, Cellects provides several different options to detect specimen
 motion and growth throughout the video. The video segmentation transforms a grayscale video into a binary one.
 In simple datasets with strong contrast between specimens and background, Cellects can simply segment each image by
 thresholding. In more challenging conditions, the algorithm tracks the intensity of each pixel over time,
 using this dynamical information to determine when a pixel has been covered. This is done through an automatically
 determined threshold on the intensity or on its derivative. Additionally, Cellects can apply the logical operators
 AND or OR to these intensity and derivative thresholds. The default option is the dynamical intensity threshold and it
 works in many cases, but the user interface lets the user quickly check the results of different options and choose
 the best one by visual inspection of the segmentation result in different frames.
 For Cellects to be as versatile as possible, the user can select across five segmentation strategies.
The first option is the simplest: It starts at the frame in which the cell is visible for the first time and segments the video frame by frame, using the same method as when analyzing only one image (as described in sections 1 and 2). The only difference is an optional background subtraction algorithm, which subtracts the first image to all others.
The second option segments each frame by intensity thresholding. The threshold changes over time to adapt to changes in the background over time. To estimate the optimal threshold for each frame, Cellects proceeds as follows: It first estimates the typical background intensity of each frame as an intensity higher than the first decile of all pixels in the frame. Then, it defines an initial threshold for each frame at a fixed distance above this decile. This fixed distance is initially low, so that the initial segmentation is an overestimation of the actual area covered by the specimen. Then, it performs segmentation of all frames. If any frame presents a growth greater than a user_set threshold (whose default value is 5% of the area), all thresholds are diminished by 10%. Then the segmentation is performed again, and this process continues until no frame presents excessive growth. This description refers to cases in which the background is darker than the specimen. Cellects automatically detects if contrast is reversed, and adapts the method accordingly. Finally, Cellects segments the whole video with these adjusted intensity thresholds.
The third option uses the change in intensity over time: For each pixel, it considers the evolution in time of its intensity, and considers that the cell covers the pixel when the slope of this intensity over time exceeds a threshold (Fig 3d in the main text).  For each frame, Cellects computes each frame’s threshold with the similar procedure as in the second option, except for the following. As the value of the slope of a derivative is highly sensitive to noise, Cellects first smooths the intensity curves using a moving average with a window length adapted to the typical time it takes for the cell to cover each pixel. Cellects tries to compute this typical time using the dynamics of a subset of pixels whose intensity varies strongly at the beginning of the growth (see the code for further details), and uses a default value of 10 frames when this computation fails. Cellects also uses this subset of pixels to get the reference slope threshold. Finally, it progressively modifies this reference until the video segmentation matches the required growth ratio, as in the second step.
The two next options are combinations of the two first ones.
The fourth is a logical OR between the intensity value and the intensity slope segmentations. It provides a very permissive segmentation, which is useful when parts of the cells are very hard to detect.
The fifth is the logical AND between the intensity value and the intensity slope segmentations. It provides a more restrictive segmentation that can be useful when both the value and the slope segmentations detect areas that are not covered by the cell.
6. Video post-processing improves the resulting binary video obtained through segmentation.
The final step consists in improving the segmentation (see section S3.5 of the Supplementary Materials for more information). Cellects will first apply several filters that consistently improve the results, such as checking that each detected pixel was also detected at least twice in the three previous frames, omitting images containing too many detected pixels, and performing morphological opening and closing. Optionally, the user can activate the detection of areas left by the cell (See section S3.5.B of the Supplementary Materials for details).

Additionally, optional algorithms correct particular types of errors. The first algorithm is useful when the substrate on which the cells are at the first image is of a different color than the substrate on which they will grow, expand or move. This color difference may produce holes in the segmentation and we developed an optional algorithm to correct this kind of error around the initial shape. The second algorithm should be used when each arena contains a single specimen, which should generate a single connected component. We can use this information to correct mistakes in models such as P. polycephalum, whose strong heterogeneity produces large variations of opacity. In these cases, segmentation may fail in the most transparent parts of the specimens and identify two disconnected components. The correction algorithm merges these disconnecting components by finding the most likely pixels connecting them and the most likely times at which those pixels were covered during growth.
6.A Basic post-processing
This process improves the raw segmentation. It includes algorithms to filter out aberrant frames, remove small artifacts and holes, and to detect when the specimens leave pixels. First, it checks that every pixel was detected at least twice in the three previous frames. Second, it excludes frames containing too many newly detected pixels, according to the maximal growth ratio per frame (as defined in section 3B). For these frames, the previous segmentation is kept, making the analysis robust to events producing a sharp variation in the brightness of a few images in the video (for example, when an enclosed device is temporarily opened or a light is switched on or off). Third, it removes potential small artifacts and holes by performing morphological opening followed by morphological closing.

6.B Cell leaving detection
This optional algorithm detects when areas are left by the specimens. It is useful when the cells not only grow but also move, so they can leave pixels that were covered before. When a pixel is covered, Cellects saves the intensity it had before being covered, computed as the median of the pixel’s intensity over a time window before it was covered. The length of this time window matches the typical time it takes for the cell to cover each pixel (computed as described in section 4.B, third segmentation strategy). Then, pixels at the border of the cell whose intensity fall below the saved intensity, rescaled by a user-defined multiplier (set by default at 1) are considered to be left by the cell. When there should be only one cell in the arena, Cellects tries to remove each component one by one, accepting this removal only when it does not break the connectivity of all parts of the cell.

6.C Special error correction algorithms
At the time of writing, Cellects contains two post-processing algorithms adapted to two specific situations. The first one is useful when there should be only one specimen per arena and when Cellects fails to detect its distant parts because their connections are not sufficiently visible. The second one is useful when Cellects fails to detect small areas around the initial shape, for example due to reflections near the edges. The following explains how these optional algorithms work.

6.D Connect distant components:
This algorithm automatically and progressively adds distant shapes to the main one. This correcting process occurs in three steps. First, it selects which distant component should get connected to the main one. The user can adjust this selection process according to the distance of the distant components with the main shape, and the minimal and maximal size of these components. Second, for each distant component, it computes and creates the shortest connection with the main shape. The width of that connection depends on the size of the distant shape where the connection occurs. Third, it uses an algorithm similar to the one used to correct errors around initial shape to estimate how quickly the gaps should be filled. This algorithm uses distance and timing vectors to create a dynamic connection between these two shapes (Figure 3f-h in the main text).

6.E Correct errors around initial shape:
This correcting process occurs in two steps. The first one scans the formation of holes around the initial segmentation during the beginning of the growth. The second one finds out when and how these holes are to be filled. To determine how the holes should be covered, Cellects uses the same algorithm as the one used to connect distant components. Computing the speed at which growth occurs from the initial position allows Cellects to fill the holes at the same speed, and therefore to correct these errors.

7. Special algorithms for Physarum polycephalum
Although done for this organism, these methods can be used with other biological models, such as mycelia.
7.A. Oscillatory activity detection:
This algorithm analyzes grayscale video frames to detect whether pixel intensities increase or decrease over time. To prevent artifacts from arena-scale illumination fluctuations, pixel intensities are first standardized by the average intensity of the entire image. A pixel is considered to have increased (or decreased) in intensity if at least four of its eight neighboring pixels have also shown an increase (or decrease). Then, regions with adjacent pixels whose intensity is changing in the same direction are detected, keeping only those larger than a user-selected threshold. Each region is tracked throughout the video, recording its oscillatory period, phase, and coordinates until it dissipates or reaches the video’s end.

7.B. Network detection:
P. polycephalum cells are composed of two types of compartments: A tubular network that transports cytoplasmic materials, and a thinner compartment that covers the rest of the space. Cellects’ initial segmentation does not distinguish between these two compartments, detecting all pixels that have been covered by any of them. This step distinguishes them, in order to segment the tubular network, whose intensity is further from that of the background.
Cellects detects such a network using an algorithm that scores the segmentation results after using filters of
vesselness detection: sato and frangi. On top of testing these filters with around 10 variations of their parameters,
Cellects tries to segment the images adaptatively: segmenting each part of the image using a 2D rolling window.
Once the best segmentation strategy is found for the last image of the video, it is used to segment the network in all
other frames

8. Graph extraction:
Cellects can extract the graph of the specimen, or if detected, of its internal network.
To do so, Cellects does the following:
- Get the skeleton of the binary matrix of presence/absence of the specimen, as well as the specimen/network
 width at avery pixel of the skeleton.
If the original position from which the specimen started has not the same color as the rest of the arena, apply
a special algorithm to draw the skeleton at the border of that origin.
- Smooth the skeleton using an algorithm removing small loops of 3 pixels widths
- Keep only the largest connected component of the skeleton
- Use pixel connectivity and their neighborhood connectivity to detect all tips and branching vertices of the graph
summarizing the skeleton.
- Find and label all edges connecting tips and remove those that are shorter than the width of the skeleton arm it is connected to
- Find and label all edges connecting touching vertices
- Find and label all edges connected to the two previoussly mentioned vertices
- Find and label all edges forming loops and connected to only one vertex
- Remove all shapes of 1 or two pixels that are neither detected as vertices nor edges,
if and only if they do not break the skeleton into more than one connected component.
- Remove edge duplicates
- Remove vertices connectiong 2 edges
- Finally, create and save the tables storing edge and vertex coordinates and properties

9. Save
Once the image analysis is finished, the software determines the value of each morphological descriptor at each time frame (SI - Table 1). Finally, Cellects saves a new video for each arena with the original video next to the converted video displaying the segmentation result, so that the user can easily validate the result. If an arena shows a poor segmentation result, the user can re-analyze it, tuning all parameters for that specific arena.
- the final results of the segmentation and its contour (if applicable)
- descriptors summarizing the whole video
- validation images (efficiency tests) and videos

10. If this class has been used in the video_analysis_window only on one arena, the method
change_results_of_one_arena will open (or create if not existing) tables in the focal folder
and adjust every row corresponding to that particular arena to the current analysis results.

"""

import weakref
from gc import collect
from time import sleep
from numba.typed import Dict as TDict
from psutil import virtual_memory
import pandas as pd
from cellects.core.one_image_analysis import OneImageAnalysis
from cellects.image_analysis.cell_leaving_detection import cell_leaving_detection
from cellects.image_analysis.cluster_flux_study import ClusterFluxStudy
from cellects.image_analysis.image_segmentation import segment_with_lum_value, apply_filter
from cellects.image_analysis.morphological_operations import (find_major_incline, image_borders, draw_me_a_sun,
                                                              inverted_distance_transform, dynamically_expand_to_fill_holes,
                                                              box_counting_dimension, prepare_box_counting,
                                                              keep_one_connected_component, cc)
from cellects.image_analysis.network_functions import *
from cellects.image_analysis.progressively_add_distant_shapes import ProgressivelyAddDistantShapes
from cellects.image_analysis.shape_descriptors import ShapeDescriptors, from_shape_descriptors_class
from cellects.utils.utilitarian import PercentAndTimeTracker, smallest_memory_array


class MotionAnalysis:

    def __init__(self, l):

        """
        :param video_name: The name of the video to read
        :param convert_for_motion: The dict specifying the linear combination
                                   of color channels (rgb_hsv_lab) to use
        """
        self.one_descriptor_per_arena = {}
        self.one_descriptor_per_arena['arena'] = l[1]
        vars = l[2]
        detect_shape = l[3]
        analyse_shape = l[4]
        show_seg = l[5]
        videos_already_in_ram = l[6]
        self.visu = None
        self.binary = None
        self.origin_idx = None
        self.smoothing_flag: bool = False
        logging.info(f"Start the motion analysis of the arena n°{self.one_descriptor_per_arena['arena']}")

        self.vars = vars
        # self.origin = self.vars['first_image'][self.vars['top'][l[0]]:(
        #    self.vars['bot'][l[0]] + 1),
        #               self.vars['left'][l[0]]:(self.vars['right'][l[0]] + 1)]
        self.load_images_and_videos(videos_already_in_ram, l[0])

        self.dims = self.converted_video.shape
        self.segmentation = np.zeros(self.dims, dtype=np.uint8)

        self.covering_intensity = np.zeros(self.dims[1:], dtype=np.float64)
        self.mean_intensity_per_frame = np.mean(self.converted_video, (1, 2))

        self.borders = image_borders(self.dims[1:], shape=self.vars['arena_shape'])
        # if self.vars['arena_shape'] == "circle":
        #     self.borders = Ellipse(self.dims[1:]).create()
        #     img_contours = image_borders(self.dims[1:])
        #     self.borders = self.borders * img_contours
        # else:
        #     self.borders = image_borders(self.dims[1:])
        self.pixel_ring_depth = 9
        self.step = 10
        self.lost_frames = 10
        self.update_ring_width()

        self.start = None
        if detect_shape:
            #self=self.motion
            #self.drift_correction()
            self.start = None
            # Here to conditional layers allow to detect if an expansion/exploration occured
            self.get_origin_shape()
            # The first, user-defined is the 'first_move_threshold' and the second is the detection of the
            # substantial image: if any of them is not detected, the program considers there is not exp.
            if self.dims[0] >= 40:
                step = self.dims[0] // 20
            else:
                step = 1
            if self.start >= (self.dims[0] - step - 1):
                self.start = None
            else:
                self.get_covering_duration(step)
                if self.start is not None:
                    # self.vars['fading'] = -0.5
                    # self.vars['do_threshold_segmentation']: bool = False
                    # self.vars['do_slope_segmentation'] = True
                    # self.vars['true_if_use_light_AND_slope_else_OR']: bool = False
                    self.detection()
                    self.initialize_post_processing()
                    self.t = self.start
                    while self.t < self.binary.shape[0]:  #200: 
                        self.update_shape(show_seg)
                #
            if self.start is None:
                self.binary = np.repeat(np.expand_dims(self.origin, 0), self.converted_video.shape[0], axis=0)

            if analyse_shape:
                self.get_descriptors_from_binary()
                self.detect_growth_transitions()
                self.networks_detection(show_seg)
                self.study_cytoscillations(show_seg)
                self.fractal_descriptions()
                self.get_descriptors_summary()
                if videos_already_in_ram is None:
                    self.save_results()

    def load_images_and_videos(self, videos_already_in_ram, i):
        logging.info(f"Arena n°{self.one_descriptor_per_arena['arena']}. Load images and videos")
        self.origin = self.vars['origin_list'][i]# self.vars['origins_list'][i]
        if videos_already_in_ram is None:
            true_frame_width = self.origin.shape[1]
            vid_name = f"ind_{self.one_descriptor_per_arena['arena']}.npy"
            if len(self.vars['background_list']) == 0:
                self.background = None
            else:
                self.background = self.vars['background_list'][i]
            if len(self.vars['background_list2']) == 0:
                self.background2 = None
            else:
                self.background2 = self.vars['background_list2'][i]

            if self.vars['already_greyscale']:
                self.converted_video = video2numpy(
                    vid_name, None, self.background, true_frame_width)
                if len(self.converted_video.shape) == 4:
                    self.converted_video = self.converted_video[:, :, :, 0]
            else:
                self.visu = video2numpy(
                    vid_name, None, self.background, true_frame_width)
                self.get_converted_video()
        else:
            if self.vars['already_greyscale']:
                self.converted_video = videos_already_in_ram
            else:
                if self.vars['convert_for_motion']['logical'] == 'None':
                    self.visu, self.converted_video = videos_already_in_ram
                else:
                    (self.visu,
                        self.converted_video,
                        self.converted_video2) = videos_already_in_ram

    def get_converted_video(self):
        if not self.vars['already_greyscale']:
            logging.info(f"Arena n°{self.one_descriptor_per_arena['arena']}. Convert the RGB visu video into a greyscale image using the color space combination: {self.vars['convert_for_motion']}")
            first_dict = TDict()
            second_dict = TDict()
            c_spaces = []
            for k, v in self.vars['convert_for_motion'].items():
                if k != 'logical' and v.sum() > 0:
                    if k[-1] != '2':
                        first_dict[k] = v
                        c_spaces.append(k)
                    else:
                        second_dict[k[:-1]] = v
                        c_spaces.append(k[:-1])
            if self.vars['lose_accuracy_to_save_memory']:
                self.converted_video = np.zeros(self.visu.shape[:3], dtype=np.uint8)
            else:
                self.converted_video = np.zeros(self.visu.shape[:3], dtype=np.float64)
            if self.vars['convert_for_motion']['logical'] != 'None':
                if self.vars['lose_accuracy_to_save_memory']:
                    self.converted_video2 = np.zeros(self.visu.shape[:3], dtype=np.uint8)
                else:
                    self.converted_video2 = np.zeros(self.visu.shape[:3], dtype=np.float64)

            # Trying to subtract the first image to the first image is a nonsense so,
            # when doing background subtraction, the first and the second image are equal
            for counter in np.arange(self.visu.shape[0]):
                if self.vars['subtract_background'] and counter == 0:
                    img = self.visu[1, ...]
                else:
                    img = self.visu[counter, ...]
                greyscale_image, greyscale_image2 = generate_color_space_combination(img, c_spaces, first_dict,
                                                                                     second_dict, self.background,
                                                                                     self.background2,
                                                                                     self.vars['lose_accuracy_to_save_memory'])
                if self.vars['filter_spec'] is not None and self.vars['filter_spec']['filter1_type'] != "":
                    greyscale_image = apply_filter(greyscale_image, self.vars['filter_spec']['filter1_type'],
                                                   self.vars['filter_spec']['filter1_param'],
                                                   self.vars['lose_accuracy_to_save_memory'])
                    if greyscale_image2 is not None and self.vars['filter_spec']['filter2_type'] != "":
                        greyscale_image2 = apply_filter(greyscale_image2, self.vars['filter_spec']['filter2_type'],
                                                        self.vars['filter_spec']['filter2_param'],
                                                        self.vars['lose_accuracy_to_save_memory'])

                self.converted_video[counter, ...] = greyscale_image
                if self.vars['convert_for_motion']['logical'] != 'None':
                    self.converted_video2[counter, ...] = greyscale_image2

    def get_origin_shape(self):
        logging.info(f"Arena n°{self.one_descriptor_per_arena['arena']}. Make sure of origin shape")
        if self.vars['origin_state'] == "constant":
            self.start = 1
            self.origin_idx = np.nonzero(self.origin)
            if self.vars['lighter_background']:
                # Initialize the covering_intensity matrix as a reference for pixel fading
                self.covering_intensity[self.origin_idx[0], self.origin_idx[1]] = 200
            self.substantial_growth = 1.2 * self.origin.sum()
        else:
            self.start = 0
            analysisi = OneImageAnalysis(self.converted_video[0, :, :])
            analysisi.binary_image = 0
            if self.vars['drift_already_corrected']:
                mask_coord = np.zeros((self.dims[0], 4), dtype=np.uint32)
                for frame_i in np.arange(self.dims[0]):  # 100):#
                    true_pixels = np.nonzero(self.converted_video[frame_i, ...])
                    mask_coord[frame_i, :] = np.min(true_pixels[0]), np.max(true_pixels[0]), np.min(true_pixels[1]), np.max(
                        true_pixels[1])
            else:
                mask_coord = None
            while np.logical_and(np.sum(analysisi.binary_image) < self.vars['first_move_threshold'], self.start < self.dims[0]):
                analysisi = self.frame_by_frame_segmentation(self.start, mask_coord)
                self.start += 1

                # frame_i = OneImageAnalysis(self.converted_video[self.start, :, :])
                # frame_i.thresholding(self.vars['luminosity_threshold'], self.vars['lighter_background'])
                # frame_i.thresholding(self.vars['luminosity_threshold'], self.vars['lighter_background'])
                # self.start += 1

            # Use connected components to find which shape is the nearest from the image center.
            if self.vars['several_blob_per_arena']:
                self.origin = analysisi.binary_image
            else:
                nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(analysisi.binary_image,
                                                                                           connectivity=8)
                if self.vars['appearance_detection_method'] == 'most_central':
                    center = np.array((self.dims[2] // 2, self.dims[1] // 2))
                    stats = np.zeros(nb_components - 1)
                    for shape_i in np.arange(1, nb_components):
                        stats[shape_i - 1] = eudist(center, centroids[shape_i, :])
                    # The shape having the minimal euclidean distance from the center will be the original shape
                    self.origin = np.zeros((self.dims[1], self.dims[2]), dtype=np.uint8)
                    self.origin[output == (np.argmin(stats) + 1)] = 1
                elif self.vars['appearance_detection_method'] == 'largest':
                    self.origin = np.zeros((self.dims[1], self.dims[2]), dtype=np.uint8)
                    self.origin[output == np.argmax(stats[1:, 4])] = 1
            self.origin_idx = np.nonzero(self.origin)
            self.substantial_growth = self.origin.sum() + 250
        ##

    def get_covering_duration(self, step):
        logging.info(f"Arena n°{self.one_descriptor_per_arena['arena']}. Find a frame with a significant growth/motion and determine the number of frames necessary for a pixel to get covered")
        ## Find the time at which growth reached a substantial growth.
        self.substantial_time = self.start
        # To avoid noisy images to have deleterious effects, make sure that area area reaches the threshold thrice.
        occurrence = 0
        if self.vars['drift_already_corrected']:
            mask_coord = np.zeros((self.dims[0], 4), dtype=np.uint32)
            for frame_i in np.arange(self.dims[0]):  # 100):#
                true_pixels = np.nonzero(self.converted_video[frame_i, ...])
                mask_coord[frame_i, :] = np.min(true_pixels[0]), np.max(true_pixels[0]), np.min(true_pixels[1]), np.max(
                    true_pixels[1])
        else:
            mask_coord = None
        while np.logical_and(occurrence < 3, self.substantial_time < (self.dims[0] - step - 1)):
            self.substantial_time += step
            growth_vision = self.frame_by_frame_segmentation(self.substantial_time, mask_coord)

            # growth_vision = OneImageAnalysis(self.converted_video[self.substantial_time, :, :])
            # # growth_vision.thresholding()
            # if self.vars['convert_for_motion']['logical'] != 'None':
            #     growth_vision.image2 = self.converted_video2[self.substantial_time, ...]
            #
            # growth_vision.segmentation(self.vars['convert_for_motion']['logical'], self.vars['color_number'],
            #                            bio_label=self.vars["bio_label"], bio_label2=self.vars["bio_label2"],
            #                            grid_segmentation=self.vars['grid_segmentation'],
            #                            lighter_background=self.vars['lighter_background'])

            surfarea = np.sum(growth_vision.binary_image * self.borders)
            if surfarea > self.substantial_growth:
                occurrence += 1
        # get a rough idea of the area covered during this time
        if (self.substantial_time - self.start) > 20:
            if self.vars['lighter_background']:
                growth = (np.sum(self.converted_video[self.start:(self.start + 10), :, :], 0) / 10) - (np.sum(self.converted_video[(self.substantial_time - 10):self.substantial_time, :, :], 0) / 10)
            else:
                growth = (np.sum(self.converted_video[(self.substantial_time - 10):self.substantial_time, :, :], 0) / 10) - (
                            np.sum(self.converted_video[self.start:(self.start + 10), :, :], 0) / 10)
        else:
            if self.vars['lighter_background']:
                growth = self.converted_video[self.start, ...] - self.converted_video[self.substantial_time, ...]
            else:
                growth = self.converted_video[self.substantial_time, ...] - self.converted_video[self.start, ...]
        intensity_extent = np.ptp(self.converted_video[self.start:self.substantial_time, :, :], axis=0)
        growth[np.logical_or(growth < 0, intensity_extent < np.median(intensity_extent))] = 0
        growth = bracket_to_uint8_image_contrast(growth)
        growth *= self.borders
        growth_vision = OneImageAnalysis(growth)
        growth_vision.thresholding()
        self.substantial_image = cv2.erode(growth_vision.binary_image, cross_33, iterations=2)

        if np.any(self.substantial_image):
            natural_noise = np.nonzero(intensity_extent == np.min(intensity_extent))
            natural_noise = self.converted_video[self.start:self.substantial_time, natural_noise[0][0], natural_noise[1][0]]
            natural_noise = moving_average(natural_noise, 5)
            natural_noise = np.ptp(natural_noise)
            subst_idx = np.nonzero(self.substantial_image)
            cover_lengths = np.zeros(len(subst_idx[0]), dtype=np.uint32)
            for index in np.arange(len(subst_idx[0])):
                vector = self.converted_video[self.start:self.substantial_time, subst_idx[0][index], subst_idx[1][index]]
                left, right = find_major_incline(vector, natural_noise)
                # If find_major_incline did find a major incline: (otherwise it put 0 to left and 1 to right)
                if not np.logical_and(left == 0, right == 1):
                    cover_lengths[index] = len(vector[left:-right])
            # If this analysis fails put a deterministic step
            if len(cover_lengths[cover_lengths > 0]) > 0:
                self.step = (np.round(np.mean(cover_lengths[cover_lengths > 0])).astype(np.uint32) // 2) + 1
                logging.info(f"Arena n°{self.one_descriptor_per_arena['arena']}. Pre-processing detection: the time for a pixel to get covered is set to {self.step}")
            else:
                logging.info(f"Arena n°{self.one_descriptor_per_arena['arena']}. Pre-processing detection: could not automatically find the time for a pixel to get covered. Default value is 1 for video length < 40 and 10 otherwise")

            # Make sure to avoid a step overestimation
            if self.step > self.dims[0] // 20:
                self.step = self.dims[0] // 20

            if self.step == 0:
                self.step = 1
        # When the first_move_threshold is not stringent enough the program may detect a movement due to noise
        # In that case, the substantial_image is empty and there is no reason to proceed further
        else:
            self.start = None
        ##

    def detection(self, compute_all_possibilities=False):
        # self.lost_frames = (self.step - 1) * self.vars['repeat_video_smoothing'] # relevant when smoothing did not use padding.
        self.lost_frames = self.step
        # I/ Image by image segmentation algorithms
        # If images contain a drift correction (zeros at borders of the image,
        # Replace these 0 by normal background values before segmenting
        if self.vars['frame_by_frame_segmentation'] or compute_all_possibilities:
            logging.info(f"Arena n°{self.one_descriptor_per_arena['arena']}. Detect cell motion and growth using the frame by frame segmentation algorithm")
            self.segmentation = np.zeros(self.dims, dtype=np.uint8)
            if self.vars['drift_already_corrected']:
                logging.info(f"Arena n°{self.one_descriptor_per_arena['arena']}. Adjust images to drift correction and segment them")
                # 1. Get the mask valid for a number of images around it (step).
                mask_coord = np.zeros((self.dims[0], 4), dtype=np.uint32)
                for frame_i in np.arange(self.dims[0]):#100):#
                    true_pixels = np.nonzero(self.converted_video[frame_i, ...])
                    mask_coord[frame_i, :] = np.min(true_pixels[0]), np.max(true_pixels[0]), np.min(true_pixels[1]), np.max(true_pixels[1])
            else:
                mask_coord = None

            for t in np.arange(self.dims[0]):#20):#
                analysisi = self.frame_by_frame_segmentation(t, mask_coord)
                self.segmentation[t, ...] = analysisi.binary_image

                if self.vars['lose_accuracy_to_save_memory']:
                    self.converted_video[t, ...] = bracket_to_uint8_image_contrast(analysisi.image)
                else:
                    self.converted_video[t, ...] = analysisi.image
                if self.vars['convert_for_motion']['logical'] != 'None':
                    if self.vars['lose_accuracy_to_save_memory']:
                        self.converted_video2[t, ...] = bracket_to_uint8_image_contrast(analysisi.image2)
                    else:
                        self.converted_video2[t, ...] = analysisi.image2

        if self.vars['color_number'] == 2:
            luminosity_segmentation, l_threshold_over_time = self.lum_value_segmentation(self.converted_video, do_threshold_segmentation=self.vars['do_threshold_segmentation'] or compute_all_possibilities)
            self.converted_video = self.smooth_pixel_slopes(self.converted_video)
            if self.vars['do_slope_segmentation'] or compute_all_possibilities:
                gradient_segmentation = self.lum_slope_segmentation(self.converted_video)
                gradient_segmentation[-self.lost_frames:, ...] = np.repeat(gradient_segmentation[-self.lost_frames, :, :][np.newaxis, :, :], self.lost_frames, axis=0)
            if self.vars['convert_for_motion']['logical'] != 'None':
                if self.vars['do_threshold_segmentation'] or compute_all_possibilities:
                    luminosity_segmentation2, l_threshold_over_time2 = self.lum_value_segmentation(self.converted_video2, do_threshold_segmentation=True)
                    if self.vars['convert_for_motion']['logical'] == 'Or':
                        luminosity_segmentation = np.logical_or(luminosity_segmentation, luminosity_segmentation2)
                    elif self.vars['convert_for_motion']['logical'] == 'And':
                        luminosity_segmentation = np.logical_and(luminosity_segmentation, luminosity_segmentation2)
                    elif self.vars['convert_for_motion']['logical'] == 'Xor':
                        luminosity_segmentation = np.logical_xor(luminosity_segmentation, luminosity_segmentation2)
                self.converted_video2 = self.smooth_pixel_slopes(self.converted_video2)
                if self.vars['do_slope_segmentation'] or compute_all_possibilities:
                    gradient_segmentation2 = self.lum_slope_segmentation(self.converted_video2)
                    gradient_segmentation2[-self.lost_frames:, ...] = np.repeat(gradient_segmentation2[-self.lost_frames, :, :][np.newaxis, :, :], self.lost_frames, axis=0)
                    if self.vars['convert_for_motion']['logical'] == 'Or':
                        gradient_segmentation = np.logical_or(gradient_segmentation, gradient_segmentation2)
                    elif self.vars['convert_for_motion']['logical'] == 'And':
                        gradient_segmentation = np.logical_and(gradient_segmentation, gradient_segmentation2)
                    elif self.vars['convert_for_motion']['logical'] == 'Xor':
                        gradient_segmentation = np.logical_xor(gradient_segmentation, gradient_segmentation2)

            if compute_all_possibilities:
                logging.info(f"Arena n°{self.one_descriptor_per_arena['arena']}. Compute all options to detect cell motion and growth. Maximal growth per frame: {self.vars['maximal_growth_factor']}")
                self.luminosity_segmentation = np.nonzero(luminosity_segmentation)
                self.gradient_segmentation = np.nonzero(gradient_segmentation)
                self.logical_and = np.nonzero(np.logical_and(luminosity_segmentation, gradient_segmentation))
                self.logical_or = np.nonzero(np.logical_or(luminosity_segmentation, gradient_segmentation))
            elif not self.vars['frame_by_frame_segmentation']:
                if self.vars['do_threshold_segmentation'] and not self.vars['do_slope_segmentation']:
                    logging.info(f"Arena n°{self.one_descriptor_per_arena['arena']}. Detect with luminosity threshold segmentation algorithm")
                    self.segmentation = luminosity_segmentation
                if self.vars['do_slope_segmentation']:# and not self.vars['do_threshold_segmentation']: NEW
                    logging.info(f"Arena n°{self.one_descriptor_per_arena['arena']}. Detect with luminosity slope segmentation algorithm")
                    # gradient_segmentation[:(self.lost_frames + 1), ...] = luminosity_segmentation[:(self.lost_frames + 1), ...]
                    if not self.vars['do_threshold_segmentation']:# NEW
                        self.segmentation = gradient_segmentation
                if np.logical_and(self.vars['do_threshold_segmentation'], self.vars['do_slope_segmentation']):
                    if self.vars['true_if_use_light_AND_slope_else_OR']:
                        logging.info(f"Arena n°{self.one_descriptor_per_arena['arena']}. Detection resuts from threshold AND slope segmentation algorithms")
                        self.segmentation = np.logical_and(luminosity_segmentation, gradient_segmentation)
                    else:
                        logging.info(f"Arena n°{self.one_descriptor_per_arena['arena']}. Detection resuts from threshold OR slope segmentation algorithms")
                        self.segmentation = np.logical_or(luminosity_segmentation, gradient_segmentation)
                self.segmentation = self.segmentation.astype(np.uint8)
        self.converted_video2 = None


    def frame_by_frame_segmentation(self, t, mask_coord=None):

        contrasted_im = bracket_to_uint8_image_contrast(self.converted_video[t, :, :])
        if self.vars['convert_for_motion']['logical'] != 'None':
            contrasted_im2 = bracket_to_uint8_image_contrast(self.converted_video2[t, :, :])
        # 1. Get the mask valid for a number of images around it (step).
        if self.vars['drift_already_corrected']:
            if t < self.step // 2:
                t_start = 0
                t_end = self.step
            elif t > (self.dims[0] - self.step // 2):
                t_start = self.dims[0] - self.step
                t_end = self.dims[0]
            else:
                t_start = t - (self.step // 2)
                t_end = t + (self.step // 2)
            min_y, max_y = np.max(mask_coord[t_start:t_end, 0]), np.min(mask_coord[t_start:t_end, 1])
            min_x, max_x = np.max(mask_coord[t_start:t_end, 2]), np.min(mask_coord[t_start:t_end, 3])
            # 3. Bracket the focal image
            image_i = contrasted_im[min_y:(max_y + 1), min_x:(max_x + 1)].astype(np.float64)
            image_i /= np.mean(image_i)
            image_i = OneImageAnalysis(image_i)
            if self.vars['convert_for_motion']['logical'] != 'None':
                image_i2 = contrasted_im2[min_y:(max_y + 1), min_x:(max_x + 1)]
                image_i2 /= np.mean(image_i2)
                image_i.image2 = image_i2
            mask = (self.converted_video[t, ...] > 0).astype(np.uint8)
        else:
            mask = None
        # 3. Bracket the focal image
        if self.vars['grid_segmentation']:
            int_variation_thresh = 100 - (np.ptp(contrasted_im) * 90 / 255)
        else:
            int_variation_thresh = None
        analysisi = OneImageAnalysis(bracket_to_uint8_image_contrast(contrasted_im / np.mean(contrasted_im)))
        if self.vars['convert_for_motion']['logical'] != 'None':
            analysisi.image2 = bracket_to_uint8_image_contrast(contrasted_im2 / np.mean(contrasted_im2))

        if t == 0:
            analysisi.previous_binary_image = self.origin
        else:
            analysisi.previous_binary_image = deepcopy(self.segmentation[t - 1, ...])

        analysisi.segmentation(self.vars['convert_for_motion']['logical'], self.vars['color_number'],
                               bio_label=self.vars["bio_label"], bio_label2=self.vars["bio_label2"],
                               grid_segmentation=self.vars['grid_segmentation'],
                               lighter_background=self.vars['lighter_background'],
                               side_length=20, step=5, int_variation_thresh=int_variation_thresh, mask=mask,
                               filter_spec=None) # filtering already done when creating converted_video

        return analysisi

        # 1. Get the mask valid for a number of images around it (step).


    def lum_value_segmentation(self, converted_video, do_threshold_segmentation):
        shape_motion_failed: bool = False
        if self.vars['lighter_background']:
            covering_l_values = np.min(converted_video[:self.substantial_time, :, :],
                                             0) * self.substantial_image
        else:
            covering_l_values = np.max(converted_video[:self.substantial_time, :, :],
                                             0) * self.substantial_image
        # Avoid errors by checking whether the covering values are nonzero
        covering_l_values = covering_l_values[covering_l_values != 0]
        if len(covering_l_values) == 0:
            shape_motion_failed = True
        if not shape_motion_failed:
            value_segmentation_thresholds = np.arange(0.8, -0.7, -0.1)
            validated_thresholds = np.zeros(value_segmentation_thresholds.shape, dtype=bool)
            counter = 0
            while_condition = True
            max_motion_per_frame = (self.dims[1] * self.dims[2]) * self.vars['maximal_growth_factor'] * 2
            if self.vars['lighter_background']:
                basic_bckgrnd_values = np.quantile(converted_video[:(self.lost_frames + 1), ...], 0.9, axis=(1, 2))
            else:
                basic_bckgrnd_values = np.quantile(converted_video[:(self.lost_frames + 1), ...], 0.1, axis=(1, 2))
            # Try different values of do_threshold_segmentation and keep the one that does not
            # segment more than x percent of the image
            while counter <= 14:
                value_threshold = value_segmentation_thresholds[counter]
                if self.vars['lighter_background']:
                    l_threshold = (1 + value_threshold) * np.max(covering_l_values)
                else:
                    l_threshold = (1 - value_threshold) * np.min(covering_l_values)
                starting_segmentation, l_threshold_over_time = segment_with_lum_value(converted_video[:(self.lost_frames + 1), ...],
                                                               basic_bckgrnd_values, l_threshold,
                                                               self.vars['lighter_background'])

                changing_pixel_number = np.sum(np.absolute(np.diff(starting_segmentation.astype(np.int8), 1, 0)), (1, 2))
                validation = np.max(np.sum(starting_segmentation, (1, 2))) < max_motion_per_frame and (
                        np.max(changing_pixel_number) < max_motion_per_frame)
                validated_thresholds[counter] = validation
                if np.any(validated_thresholds):
                    if not validation:
                        break
                counter += 1
            # If any threshold is accepted, use their average to proceed the final thresholding
            valid_number = validated_thresholds.sum()
            if valid_number > 0:
                if valid_number > 2:
                    index_to_keep = 2
                else:
                    index_to_keep = valid_number - 1
                value_threshold = value_segmentation_thresholds[
                    np.uint8(np.floor(np.mean(np.nonzero(validated_thresholds)[0][index_to_keep])))]
            else:
                value_threshold = 0

            if self.vars['lighter_background']:
                l_threshold = (1 + value_threshold) * np.max(covering_l_values)
            else:
                l_threshold = (1 - value_threshold) * np.min(covering_l_values)
            if do_threshold_segmentation:
                if self.vars['lighter_background']:
                    basic_bckgrnd_values = np.quantile(converted_video, 0.9, axis=(1, 2))
                else:
                    basic_bckgrnd_values = np.quantile(converted_video, 0.1, axis=(1, 2))
                luminosity_segmentation, l_threshold_over_time = segment_with_lum_value(converted_video, basic_bckgrnd_values,
                                                                 l_threshold, self.vars['lighter_background'])
            else:
                luminosity_segmentation, l_threshold_over_time = segment_with_lum_value(converted_video[:(self.lost_frames + 1), ...],
                                                               basic_bckgrnd_values, l_threshold,
                                                               self.vars['lighter_background'])
        else:
            luminosity_segmentation = None

        return luminosity_segmentation, l_threshold_over_time

    def smooth_pixel_slopes(self, converted_video):
        # smoothed_video = np.zeros(
        #     (self.dims[0] - self.lost_frames, self.dims[1], self.dims[2]),
        #     dtype=np.float64)
        try:
            if self.vars['lose_accuracy_to_save_memory']:
                smoothed_video = np.zeros(self.dims, dtype=np.float16)
                smooth_kernel = np.ones(self.step) / self.step
                for i in np.arange(converted_video.shape[1]):
                    for j in np.arange(converted_video.shape[2]):
                        padded = np.pad(converted_video[:, i, j] / self.mean_intensity_per_frame,
                                     (self.step // 2, self.step - 1 - self.step // 2), mode='edge')
                        moving_average = np.convolve(padded, smooth_kernel, mode='valid')
                        if self.vars['repeat_video_smoothing'] > 1:
                            for it in np.arange(1, self.vars['repeat_video_smoothing']):
                                padded = np.pad(moving_average,
                                             (self.step // 2, self.step - 1 - self.step // 2), mode='edge')
                                moving_average = np.convolve(padded, smooth_kernel, mode='valid')
                        smoothed_video[:, i, j] = moving_average.astype(np.float16)
            else:
                smoothed_video = np.zeros(self.dims, dtype=np.float64)
                smooth_kernel = np.ones(self.step) / self.step
                for i in np.arange(converted_video.shape[1]):
                    for j in np.arange(converted_video.shape[2]):
                        padded = np.pad(converted_video[:, i, j] / self.mean_intensity_per_frame,
                                     (self.step // 2, self.step - 1 - self.step // 2), mode='edge')
                        moving_average = np.convolve(padded, smooth_kernel, mode='valid')
                        if self.vars['repeat_video_smoothing'] > 1:
                            for it in np.arange(1, self.vars['repeat_video_smoothing']):
                                padded = np.pad(moving_average,
                                             (self.step // 2, self.step - 1 - self.step // 2), mode='edge')
                                moving_average = np.convolve(padded, smooth_kernel, mode='valid')
                        smoothed_video[:, i, j] = moving_average
            return smoothed_video

        except MemoryError:
            logging.error("Not enough RAM available to smooth pixel curves. Detection may fail.")
            smoothed_video = converted_video
            return smoothed_video

    def lum_slope_segmentation(self, converted_video):
        shape_motion_failed : bool = False
        gradient_segmentation = np.zeros(self.dims, np.uint8)
        # 2) Contrast increase
        oridx = np.nonzero(self.origin)
        notoridx = np.nonzero(1 - self.origin)
        do_increase_contrast = np.mean(converted_video[0, oridx[0], oridx[1]]) * 10 > np.mean(
                converted_video[0, notoridx[0], notoridx[1]])
        necessary_memory = self.dims[0] * self.dims[1] * self.dims[2] * 64 * 2 * 1.16415e-10
        available_memory = (virtual_memory().available >> 30) - self.vars['min_ram_free']
        if self.vars['lose_accuracy_to_save_memory']:
            derive = converted_video.astype(np.float16)
        else:
            derive = converted_video.astype(np.float64)
        if necessary_memory > available_memory:
            converted_video = None

        if do_increase_contrast:
            derive = np.square(derive)

        # 3) Get the gradient
        necessary_memory = derive.size * 64 * 4 * 1.16415e-10
        available_memory = (virtual_memory().available >> 30) - self.vars['min_ram_free']
        if necessary_memory > available_memory:
            for cy in np.arange(self.dims[1]):
                for cx in np.arange(self.dims[2]):
                    if self.vars['lose_accuracy_to_save_memory']:
                        derive[:, cy, cx] = np.gradient(derive[:, cy, cx], self.step).astype(np.float16)
                    else:
                        derive[:, cy, cx] = np.gradient(derive[:, cy, cx], self.step)
        else:
            if self.vars['lose_accuracy_to_save_memory']:
                derive = np.gradient(derive, self.step, axis=0).astype(np.float16)
            else:
                derive = np.gradient(derive, self.step, axis=0)

        # 4) Segment
        if self.vars['lighter_background']:
            covering_slopes = np.min(derive[:self.substantial_time, :, :], 0) * self.substantial_image
        else:
            covering_slopes = np.max(derive[:self.substantial_time, :, :], 0) * self.substantial_image
        covering_slopes = covering_slopes[covering_slopes != 0]
        if len(covering_slopes) == 0:
            shape_motion_failed = True

        if not shape_motion_failed:
            ####
            # ease_slope_segmentation = 0.8
            value_segmentation_thresholds = np.arange(0.8, -0.7, -0.1)
            validated_thresholds = np.zeros(value_segmentation_thresholds.shape, dtype=bool)
            counter = 0
            while_condition = True
            max_motion_per_frame = (self.dims[1] * self.dims[2]) * self.vars['maximal_growth_factor']
            # Try different values of do_slope_segmentation and keep the one that does not
            # segment more than x percent of the image
            while counter <= 14:
                ease_slope_segmentation = value_segmentation_thresholds[counter]
                if self.vars['lighter_background']:
                    gradient_threshold = (1 + ease_slope_segmentation) * np.max(covering_slopes)
                    sample = np.less(derive[:self.substantial_time], gradient_threshold)
                else:
                    gradient_threshold = (1 - ease_slope_segmentation) * np.min(covering_slopes)
                    sample = np.greater(derive[:self.substantial_time], gradient_threshold)
                changing_pixel_number = np.sum(np.absolute(np.diff(sample.astype(np.int8), 1, 0)), (1, 2))
                validation = np.max(np.sum(sample, (1, 2))) < max_motion_per_frame and (
                        np.max(changing_pixel_number) < max_motion_per_frame)
                validated_thresholds[counter] = validation
                if np.any(validated_thresholds):
                    if not validation:
                        break
                counter += 1
                # If any threshold is accepted, use their average to proceed the final thresholding
            valid_number = validated_thresholds.sum()
            if valid_number > 0:
                if valid_number > 2:
                    index_to_keep = 2
                else:
                    index_to_keep = valid_number - 1
                ease_slope_segmentation = value_segmentation_thresholds[
                    np.uint8(np.floor(np.mean(np.nonzero(validated_thresholds)[0][index_to_keep])))]
            else:
                ease_slope_segmentation = 0

            if self.vars['lighter_background']:
                gradient_threshold = (1 - ease_slope_segmentation) * np.max(covering_slopes)
                gradient_segmentation[:-self.lost_frames, :, :] = np.less(derive, gradient_threshold)[self.lost_frames:, :, :]
            else:
                gradient_threshold = (1 - ease_slope_segmentation) * np.min(covering_slopes)
                gradient_segmentation[:-self.lost_frames, :, :] = np.greater(derive, gradient_threshold)[self.lost_frames:, :, :]
        else:
            gradient_segmentation = None
        return gradient_segmentation

    def update_ring_width(self):
        # Make sure that self.pixels_depths are odd and greater than 3
        if self.pixel_ring_depth <= 3:
            self.pixel_ring_depth = 3
        if self.pixel_ring_depth % 2 == 0:
            self.pixel_ring_depth = self.pixel_ring_depth + 1
        self.erodila_disk = Ellipse((self.pixel_ring_depth, self.pixel_ring_depth)).create().astype(np.uint8)
        self.max_distance = self.pixel_ring_depth * self.vars['detection_range_factor']

    def initialize_post_processing(self):
        ## Initialization
        logging.info(f"Arena n°{self.one_descriptor_per_arena['arena']}. Starting Post_processing. Fading detection: {self.vars['do_fading']}: {self.vars['fading']}, Subtract background: {self.vars['subtract_background']}, Correct errors around initial shape: {self.vars['correct_errors_around_initial']}, Connect distant shapes: {self.vars['detection_range_factor'] > 0}, How to select appearing cell(s): {self.vars['appearance_detection_method']}")

        self.binary = np.zeros(self.dims[:3], dtype=np.uint8)
        if self.origin.shape[0] != self.binary[self.start - 1, :, :].shape[0] or self.origin.shape[1] != self.binary[self.start - 1, :, :].shape[1]:
            logging.error("Unaltered videos deprecated, they have been created with different settings.\nDelete .npy videos and Data to run Cellects quickly.pkl and re-run")

        if self.vars['origin_state'] == "invisible":
            self.binary[self.start - 1, :, :] = deepcopy(self.origin)
            self.covering_intensity[self.origin_idx[0], self.origin_idx[1]] = self.converted_video[self.start, self.origin_idx[0], self.origin_idx[1]]
        else:
            if self.vars['origin_state'] == "fluctuating":
                self.covering_intensity[self.origin_idx[0], self.origin_idx[1]] = np.median(self.converted_video[:self.start, self.origin_idx[0], self.origin_idx[1]], axis=0)

            self.binary[:self.start, :, :] = np.repeat(np.expand_dims(self.origin, 0), self.start, axis=0)
            if self.start < self.step:
                frames_to_assess = self.step
                self.segmentation[self.start - 1, ...] = self.binary[self.start - 1, :, :]
                for t in np.arange(self.start, self.lost_frames):
                    # Only keep pixels that are always detected
                    always_found = np.sum(self.segmentation[t:(t + frames_to_assess), ...], 0)
                    always_found = always_found == frames_to_assess
                    # Remove too small shapes
                    without_small, stats, centro = cc(always_found.astype(np.uint8))
                    large_enough = np.nonzero(stats[1:, 4] > ((self.vars['first_move_threshold'] + 1) // 2))[0]
                    if len(large_enough) > 0:
                        always_found *= np.isin(always_found, large_enough + 1)
                        always_found = np.logical_or(always_found, self.segmentation[t - 1, ...])
                        self.segmentation[t, ...] *= always_found
                    else:
                        self.segmentation[t, ...] = 0
                    self.segmentation[t, ...] = np.logical_or(self.segmentation[t - 1, ...], self.segmentation[t, ...])
        self.mean_distance_per_frame = None
        self.surfarea = np.zeros(self.dims[0], dtype =np.uint64)
        self.surfarea[:self.start] = np.sum(self.binary[:self.start, :, :], (1, 2))
        self.gravity_field = inverted_distance_transform(self.binary[(self.start - 1), :, :],
                                           np.sqrt(np.sum(self.binary[(self.start - 1), :, :])))
        if self.vars['correct_errors_around_initial']:
            self.rays, self.sun = draw_me_a_sun(self.binary[(self.start - 1), :, :], ray_length_coef=1.25)  # plt.imshow(sun)
            self.holes = np.zeros(self.dims[1:], dtype=np.uint8)
            self.pixel_ring_depth += 2
            self.update_ring_width()

        if self.vars['prevent_fast_growth_near_periphery']:
            self.near_periphery = np.zeros(self.dims[1:])
            if self.vars['arena_shape'] == 'circle':
                periphery_width = self.vars['periphery_width'] * 2
                elliperiphery = Ellipse((self.dims[1] - periphery_width, self.dims[2] - periphery_width)).create()
                half_width = periphery_width // 2
                if periphery_width % 2 == 0:
                    self.near_periphery[half_width:-half_width, half_width:-half_width] = elliperiphery
                else:
                    self.near_periphery[half_width:-half_width - 1, half_width:-half_width - 1] = elliperiphery
                self.near_periphery = 1 - self.near_periphery
            else:
                self.near_periphery[:self.vars['periphery_width'], :] = 1
                self.near_periphery[-self.vars['periphery_width']:, :] = 1
                self.near_periphery[:, :self.vars['periphery_width']] = 1
                self.near_periphery[:, -self.vars['periphery_width']:] = 1
            self.near_periphery = np.nonzero(self.near_periphery)
            # near_periphery = np.zeros(self.dims[1:])
            # near_periphery[self.near_periphery] = 1

    def update_shape(self, show_seg):

        # Get from gradients, a 2D matrix of potentially covered pixels
        # I/ dilate the shape made with covered pixels to assess for covering

        # I/ 1) Only keep pixels that have been detected at least two times in the three previous frames
        if self.dims[0] < 100:
            new_potentials = self.segmentation[self.t, :, :]
        else:
            if self.t > 1:
                new_potentials = np.sum(self.segmentation[(self.t - 2): (self.t + 1), :, :], 0, dtype=np.uint8)
            else:
                new_potentials = np.sum(self.segmentation[: (self.t + 1), :, :], 0, dtype=np.uint8)
            new_potentials[new_potentials == 1] = 0
            new_potentials[new_potentials > 1] = 1

        # I/ 2) If an image displays more new potential pixels than 50% of image pixels,
        # one of these images is considered noisy and we try taking only one.
        frame_counter = -1
        maximal_size = 0.5 * new_potentials.size
        if (self.vars["do_threshold_segmentation"] or self.vars["frame_by_frame_segmentation"]) and self.t > np.max((self.start + self.step, 6)):
           maximal_size = np.min((np.max(self.binary[:self.t].sum((1, 2))) * (1 + self.vars['maximal_growth_factor']), self.borders.sum()))
        while np.logical_and(np.sum(new_potentials) > maximal_size,
                             frame_counter <= 5):  # np.logical_and(np.sum(new_potentials > 0) > 5 * np.sum(dila_ring), frame_counter <= 5):
            frame_counter += 1
            if frame_counter > self.t:
                break
            else:
                if frame_counter < 5:
                    new_potentials = self.segmentation[self.t - frame_counter, :, :]
                else:
                # If taking only one image is not enough, use the inverse of the fadinged matrix as new_potentials
                # Given it haven't been processed by any slope calculation, it should be less noisy
                    new_potentials = np.sum(self.segmentation[(self.t - 5): (self.t + 1), :, :], 0, dtype=np.uint8)
                    new_potentials[new_potentials < 6] = 0
                    new_potentials[new_potentials == 6] = 1


        new_shape = deepcopy(self.binary[self.t - 1, :, :])
        new_potentials = cv2.morphologyEx(new_potentials, cv2.MORPH_CLOSE, cross_33)
        new_potentials = cv2.morphologyEx(new_potentials, cv2.MORPH_OPEN, cross_33) * self.borders
        new_shape = np.logical_or(new_shape, new_potentials).astype(np.uint8)
        # Add distant shapes within a radius, score every added pixels according to their distance
        if not self.vars['several_blob_per_arena']:
            if new_shape.sum() == 0:
                new_shape = deepcopy(new_potentials)
            else:
                pads = ProgressivelyAddDistantShapes(new_potentials, new_shape, self.max_distance)
                r = weakref.ref(pads)
                # If max_distance is non nul look for distant shapes
                pads.consider_shapes_sizes(self.vars['min_size_for_connection'],
                                                     self.vars['max_size_for_connection'])
                pads.connect_shapes(only_keep_connected_shapes=True, rank_connecting_pixels=True)

                new_shape = deepcopy(pads.expanded_shape)
                new_shape[new_shape > 1] = 1
                if np.logical_and(self.t > self.step, self.t < self.dims[0]):
                    if np.any(pads.expanded_shape > 5):
                        # Add distant shapes back in time at the covering speed of neighbors
                        self.binary[self.t][np.nonzero(new_shape)] = 1
                        self.binary[(self.step):(self.t + 1), :, :] = \
                            pads.modify_past_analysis(self.binary[(self.step):(self.t + 1), :, :],
                                                      self.segmentation[(self.step):(self.t + 1), :, :])
                        new_shape = deepcopy(self.binary[self.t, :, :])
                pads = None

            # Fill holes
            new_shape = cv2.morphologyEx(new_shape, cv2.MORPH_CLOSE, cross_33)

        if self.vars['do_fading'] and (self.t > self.step + self.lost_frames):
            # Shape Erosion
            # I/ After a substantial growth, erode the shape made with covered pixels to assess for fading
            # Use the newly covered pixels to calculate their mean covering intensity
            new_idx = np.nonzero(np.logical_xor(new_shape, self.binary[self.t - 1, :, :]))
            start_intensity_monitoring = self.t - self.lost_frames - self.step
            end_intensity_monitoring = self.t - self.lost_frames
            self.covering_intensity[new_idx[0], new_idx[1]] = np.median(self.converted_video[start_intensity_monitoring:end_intensity_monitoring, new_idx[0], new_idx[1]], axis=0)
            previous_binary = self.binary[self.t - 1, :, :]
            greyscale_image = self.converted_video[self.t - self.lost_frames, :, :]
            protect_from_fading = None
            if self.vars['origin_state'] == 'constant':
                protect_from_fading = self.origin
            new_shape, self.covering_intensity = cell_leaving_detection(new_shape, self.covering_intensity, previous_binary, greyscale_image, self.vars['fading'], self.vars['lighter_background'], self.vars['several_blob_per_arena'], self.erodila_disk, protect_from_fading)

        self.covering_intensity *= new_shape
        self.binary[self.t, :, :] = new_shape * self.borders
        self.surfarea[self.t] = np.sum(self.binary[self.t, :, :])

        # Calculate the mean distance covered per frame and correct for a ring of not really fading pixels
        if self.mean_distance_per_frame is None:
            if self.vars['correct_errors_around_initial'] and not self.vars['several_blob_per_arena']:
                if np.logical_and((self.t % 20) == 0,
                                  np.logical_and(self.surfarea[self.t] > self.substantial_growth,
                                                 self.surfarea[self.t] < self.substantial_growth * 2)):
                    shape = self.binary[self.t, :, :] * self.sun
                    back = (1 - self.binary[self.t, :, :]) * self.sun
                    for ray in self.rays:
                        # For each sun's ray, see how they cross the shape/back and
                        # store the gravity_field value of these pixels (distance to the original shape).
                        ray_through_shape = (shape == ray) * self.gravity_field
                        ray_through_back = (back == ray) * self.gravity_field
                        if np.any(ray_through_shape):
                            if np.any(ray_through_back):
                                # If at least one back pixel is nearer to the original shape than a shape pixel,
                                # there is a hole to fill.
                                if np.any(ray_through_back > np.min(ray_through_shape[ray_through_shape > 0])):
                                    # Check if the nearest pixels are shape, if so, supress them until the nearest pixel
                                    # becomes back
                                    while np.max(ray_through_back) <= np.max(ray_through_shape):
                                        ray_through_shape[ray_through_shape == np.max(ray_through_shape)] = 0
                                    # Now, all back pixels that are nearer than the closest shape pixel should get filled
                                    # To do so, replace back pixels further than the nearest shape pixel by 0
                                    ray_through_back[ray_through_back < np.max(ray_through_shape)] = 0
                                    self.holes[np.nonzero(ray_through_back)] = 1
                            else:
                                self.rays = np.concatenate((self.rays[:(ray - 2)], self.rays[(ray - 1):]))
                        ray_through_shape = None
                        ray_through_back = None
            if np.any(self.surfarea[:self.t] > self.substantial_growth * 2):

                if self.vars['correct_errors_around_initial'] and not self.vars['several_blob_per_arena']:
                    # Apply the hole correction
                    self.holes = cv2.morphologyEx(self.holes, cv2.MORPH_CLOSE, cross_33, iterations=10)
                    # If some holes are not covered by now
                    if np.any(self.holes * (1 - self.binary[self.t, :, :])):
                        self.binary[:(self.t + 1), :, :], holes_time_end, distance_against_time = \
                            dynamically_expand_to_fill_holes(self.binary[:(self.t + 1), :, :], self.holes)
                        if holes_time_end is not None:
                            self.binary[holes_time_end:(self.t + 1), :, :] += self.binary[holes_time_end, :, :]
                            self.binary[holes_time_end:(self.t + 1), :, :][
                                self.binary[holes_time_end:(self.t + 1), :, :] > 1] = 1
                            self.surfarea[:(self.t + 1)] = np.sum(self.binary[:(self.t + 1), :, :], (1, 2))

                    else:
                        distance_against_time = [1, 2]
                else:
                    distance_against_time = [1, 2]
                distance_against_time = np.diff(distance_against_time)
                if len(distance_against_time) > 0:
                    self.mean_distance_per_frame = np.mean(- distance_against_time)
                else:
                    self.mean_distance_per_frame = 1

        if self.vars['prevent_fast_growth_near_periphery']:
            # growth_near_periphery = np.diff(self.binary[self.t-1:self.t+1, :, :] * self.near_periphery, axis=0)
            growth_near_periphery = np.diff(self.binary[self.t-1:self.t+1, self.near_periphery[0], self.near_periphery[1]], axis=0)
            if (growth_near_periphery == 1).sum() > self.vars['max_periphery_growth']:
                # self.binary[self.t, self.near_periphery[0], self.near_periphery[1]] = self.binary[self.t - 1, self.near_periphery[0], self.near_periphery[1]]
                periphery_to_remove = np.zeros(self.dims[1:], dtype=np.uint8)
                periphery_to_remove[self.near_periphery[0], self.near_periphery[1]] = self.binary[self.t, self.near_periphery[0], self.near_periphery[1]]
                shapes, stats, centers = cc(periphery_to_remove)
                periphery_to_remove = np.nonzero(np.isin(shapes, np.nonzero(stats[:, 4] > self.vars['max_periphery_growth'])[0][1:]))
                self.binary[self.t, periphery_to_remove[0], periphery_to_remove[1]] = self.binary[self.t - 1, periphery_to_remove[0], periphery_to_remove[1]]
                if not self.vars['several_blob_per_arena']:
                    shapes, stats, centers = cc(self.binary[self.t, ...])
                    shapes[shapes != 1] = 0
                    self.binary[self.t, ...] = shapes

        # Display

        if show_seg:
            if self.visu is not None:
                im_to_display = deepcopy(self.visu[self.t, ...])
                contours = np.nonzero(cv2.morphologyEx(self.binary[self.t, :, :], cv2.MORPH_GRADIENT, cross_33))
                if self.vars['lighter_background']:
                    im_to_display[contours[0], contours[1]] = 0
                else:
                    im_to_display[contours[0], contours[1]] = 255
            else:
                im_to_display = self.binary[self.t, :, :] * 255
            imtoshow = cv2.resize(im_to_display, (540, 540))
            cv2.imshow("shape_motion", imtoshow)
            cv2.waitKey(1)
        self.t += 1

    def save_coord_specimen_and_contour(self):
        if self.vars['save_coord_specimen']:
             np.save(f"coord_specimen{self.one_descriptor_per_arena['arena']}_t{self.dims[0]}_y{self.dims[1]}_x{self.dims[2]}.npy",
                 smallest_memory_array(np.nonzero(self.binary), "uint"))
        if self.vars['save_coord_contour']:
            contours = np.zeros(self.dims[:3], np.uint8)
            for frame in range(self.dims[0]):
                eroded_binary = cv2.erode(self.binary[frame, ...], cross_33, borderType=cv2.BORDER_CONSTANT, borderValue=0)
                contours[frame, ...] = self.binary[frame, ...] - eroded_binary
                np.save(f"coord_contour{self.one_descriptor_per_arena['arena']}_t{self.dims[0]}_y{self.dims[1]}_x{self.dims[2]}.npy",
                 smallest_memory_array(np.nonzero(contours), "uint"))

    def get_descriptors_from_binary(self, release_memory=True):
        ##
        if release_memory:
            self.substantial_image = None
            self.covering_intensity = None
            self.segmentation = None
            self.gravity_field = None
            self.sun = None
            self.rays = None
            self.holes = None
            collect()
        self.save_coord_specimen_and_contour()
        if self.vars['do_fading']:
            self.newly_explored_area = np.zeros(self.dims[0], dtype =np.uint64)
            self.already_explored_area = deepcopy(self.origin)
            for self.t in range(self.dims[0]):
                self.newly_explored_area[self.t] = ((self.binary[self.t, :, :] - self.already_explored_area) == 1).sum()
                self.already_explored_area = np.logical_or(self.already_explored_area, self.binary[self.t, :, :])

        self.surfarea = self.binary.sum((1, 2))
        timings = self.vars['exif']
        if len(timings) < self.dims[0]:
            timings = np.arange(self.dims[0])
        if np.any(timings > 0):
            self.time_interval = np.mean(np.diff(timings))
        timings = timings[:self.dims[0]]
        available_descriptors_in_sd = list(from_shape_descriptors_class.keys())
        # ["area", "perimeter", "circularity", "rectangularity", "total_hole_area", "solidity",
        #                          "convexity", "eccentricity", "euler_number", "standard_deviation_y",
        #                          "standard_deviation_x", "skewness_y", "skewness_x", "kurtosis_y", "kurtosis_x",
        #                          "major_axis_len", "minor_axis_len", "axes_orientation"]
        all_descriptors = []
        to_compute_from_sd = []
        for name, do_compute in self.vars['descriptors'].items():
            if do_compute:# and
                all_descriptors.append(name)
                if np.isin(name, available_descriptors_in_sd):
                    to_compute_from_sd.append(name)
        self.compute_solidity_separately: bool = self.vars['iso_digi_analysis'] and not self.vars['several_blob_per_arena'] and not np.isin("solidity", to_compute_from_sd)
        if self.compute_solidity_separately:
            self.solidity = np.zeros(self.dims[0], dtype=np.float64)
        if not self.vars['several_blob_per_arena']:
            self.one_row_per_frame = pd.DataFrame(np.zeros((self.dims[0], 2 + len(all_descriptors))),
                                              columns=['arena', 'time'] + all_descriptors)
            self.one_row_per_frame['arena'] = [self.one_descriptor_per_arena['arena']] * self.dims[0]
            self.one_row_per_frame['time'] = timings
            # solidity must be added if detect growth transition is computed
            origin = self.binary[0, :, :]
            self.one_descriptor_per_arena["first_move"] = pd.NA

            for t in np.arange(self.dims[0]):
                SD = ShapeDescriptors(self.binary[t, :, :], to_compute_from_sd)


                # NEW
                for descriptor in to_compute_from_sd:
                    self.one_row_per_frame.loc[t, descriptor] = SD.descriptors[descriptor]
                # Old
                # self.one_row_per_frame.iloc[t, 2: 2 + len(descriptors)] = SD.descriptors.values()


                if self.compute_solidity_separately:
                    solidity = ShapeDescriptors(self.binary[t, :, :], ["solidity"])
                    self.solidity[t] = solidity.descriptors["solidity"]
                    # self.solidity[t] = list(solidity.descriptors.values())[0]
                # I) Find a first pseudopod [aim: time]
                if pd.isna(self.one_descriptor_per_arena["first_move"]):
                    if self.surfarea[t] >= (origin.sum() + self.vars['first_move_threshold']):
                        self.one_descriptor_per_arena["first_move"] = t

            # Apply the scale to the variables
            if self.vars['output_in_mm']:
                if np.isin('area', to_compute_from_sd):
                    self.one_row_per_frame['area'] *= self.vars['average_pixel_size']
                if np.isin('total_hole_area', to_compute_from_sd):
                    self.one_row_per_frame['total_hole_area'] *= self.vars['average_pixel_size']
                if np.isin('perimeter', to_compute_from_sd):
                    self.one_row_per_frame['perimeter'] *= np.sqrt(self.vars['average_pixel_size'])
                if np.isin('major_axis_len', to_compute_from_sd):
                    self.one_row_per_frame['major_axis_len'] *= np.sqrt(self.vars['average_pixel_size'])
                if np.isin('minor_axis_len', to_compute_from_sd):
                    self.one_row_per_frame['minor_axis_len'] *= np.sqrt(self.vars['average_pixel_size'])
        else:
            # Objective: create a matrix with 4 columns (time, y, x, colony) containing the coordinates of all colonies
            # against time
            self.one_descriptor_per_arena["first_move"] = 1
            max_colonies = 0
            for t in np.arange(self.dims[0]):
                nb, shapes = cv2.connectedComponents(self.binary[t, :, :])
                max_colonies = np.max((max_colonies, nb))

            time_descriptor_colony = np.zeros((self.dims[0], len(to_compute_from_sd) * max_colonies * self.dims[0]),
                                              dtype=np.float32)  # Adjust max_colonies
            colony_number = 0
            colony_id_matrix = np.zeros(self.dims[1:], dtype =np.uint64)
            coord_colonies = []
            centroids = []

            pat_tracker = PercentAndTimeTracker(self.dims[0], compute_with_elements_number=True)
            for t in np.arange(self.dims[0]):  #21):#
                # t=0
                # t+=1
                # We rank colonies in increasing order to make sure that the larger colony issued from a colony division
                # keeps the previous colony name.
                shapes, stats, centers = cc(self.binary[t, :, :])

                # Consider that shapes bellow 3 pixels are noise. The loop will stop at nb and not compute them
                nb = stats[stats[:, 4] >= 4].shape[0]

                # nb = stats.shape[0]
                current_percentage, eta = pat_tracker.get_progress(t, element_number=nb)
                logging.info(f"Arena n°{self.one_descriptor_per_arena['arena']}, Colony descriptors computation: {current_percentage}%{eta}")

                updated_colony_names = np.zeros(1, dtype=np.uint32)
                for colony in (np.arange(nb - 1) + 1):  # 120)):# #92
                    # colony = 1
                    # colony+=1
                    # logging.info(f'Colony number {colony}')
                    current_colony_img = (shapes == colony).astype(np.uint8)

                    # I/ Find out which names the current colony had at t-1
                    colony_previous_names = np.unique(current_colony_img * colony_id_matrix)
                    colony_previous_names = colony_previous_names[colony_previous_names != 0]
                    # II/ Find out if the current colony name had already been analyzed at t
                    # If there no match with the saved colony_id_matrix, assign colony ID
                    if t == 0 or len(colony_previous_names) == 0:
                        # logging.info("New colony")
                        colony_number += 1
                        colony_names = [colony_number]
                    # If there is at least 1 match with the saved colony_id_matrix, we keep the colony_previous_name(s)
                    else:
                        colony_names = colony_previous_names.tolist()
                    # Handle colony division if necessary
                    if np.any(np.isin(updated_colony_names, colony_names)):
                        colony_number += 1
                        colony_names = [colony_number]

                    # Update colony ID matrix for the current frame
                    coords = np.nonzero(current_colony_img)
                    colony_id_matrix[coords[0], coords[1]] = colony_names[0]

                    # Add coordinates to coord_colonies
                    time_column = np.full(coords[0].shape, t, dtype=np.uint32)
                    colony_column = np.full(coords[0].shape, colony_names[0], dtype=np.uint32)
                    coord_colonies.append(np.column_stack((time_column, colony_column, coords[0], coords[1])))

                    # Calculate centroid and add to centroids list
                    centroid_x, centroid_y = centers[colony, :]
                    centroids.append((t, colony_names[0], centroid_y, centroid_x))

                    # Compute shape descriptors
                    SD = ShapeDescriptors(current_colony_img, to_compute_from_sd)
                    descriptors = list(SD.descriptors.values())
                    # Adjust descriptors if output_in_mm is specified
                    if self.vars['output_in_mm']:
                        if 'area' in to_compute_from_sd:
                            descriptors['area'] *= self.vars['average_pixel_size']
                        if 'total_hole_area' in to_compute_from_sd:
                            descriptors['total_hole_area'] *= self.vars['average_pixel_size']
                        if 'perimeter' in to_compute_from_sd:
                            descriptors['perimeter'] *= np.sqrt(self.vars['average_pixel_size'])
                        if 'major_axis_len' in to_compute_from_sd:
                            descriptors['major_axis_len'] *= np.sqrt(self.vars['average_pixel_size'])
                        if 'minor_axis_len' in to_compute_from_sd:
                            descriptors['minor_axis_len'] *= np.sqrt(self.vars['average_pixel_size'])

                    # Store descriptors in time_descriptor_colony
                    descriptor_index = (colony_names[0] - 1) * len(to_compute_from_sd)
                    time_descriptor_colony[t, descriptor_index:(descriptor_index + len(descriptors))] = descriptors

                    updated_colony_names = np.append(updated_colony_names, colony_names)

                # Reset colony_id_matrix for the next frame
                colony_id_matrix *= self.binary[t, :, :]

            coord_colonies = np.vstack(coord_colonies)
            centroids = np.array(centroids, dtype=np.float32)
            time_descriptor_colony = time_descriptor_colony[:, :(colony_number*len(to_compute_from_sd))]

            if self.vars['save_coord_specimen']:
                coord_colonies = pd.DataFrame(coord_colonies, columns=["time", "colony", "y", "x"])
                coord_colonies.to_csv(f"coord_colonies{self.one_descriptor_per_arena['arena']}_t{self.dims[0]}_col{colony_number}_y{self.dims[1]}_x{self.dims[2]}.csv", sep=';', index=False, lineterminator='\n')
                
            centroids = pd.DataFrame(centroids, columns=["time", "colony", "y", "x"])
            centroids.to_csv(f"colony_centroids{self.one_descriptor_per_arena['arena']}_t{self.dims[0]}_col{colony_number}_y{self.dims[1]}_x{self.dims[2]}.csv", sep=';', index=False, lineterminator='\n')

            # Format the final dataframe to have one row per time frame, and one column per descriptor_colony_name
            self.one_row_per_frame = pd.DataFrame({'arena': self.one_descriptor_per_arena['arena'], 'time': timings, 'area_total': self.surfarea.astype(np.float64)})
            if self.vars['output_in_mm']:
                self.one_row_per_frame['area_total'] *= self.vars['average_pixel_size']
            column_names = np.char.add(np.repeat(to_compute_from_sd, colony_number),
                                    np.tile((np.arange(colony_number) + 1).astype(str), len(to_compute_from_sd)))
            time_descriptor_colony = pd.DataFrame(time_descriptor_colony, columns=column_names)
            self.one_row_per_frame = pd.concat([self.one_row_per_frame, time_descriptor_colony], axis=1)


        if self.vars['do_fading']:
            self.one_row_per_frame['newly_explored_area'] = self.newly_explored_area
            if self.vars['output_in_mm']:
                self.one_row_per_frame['newly_explored_area'] *= self.vars['average_pixel_size']

    def detect_growth_transitions(self):
        ##
        if self.vars['iso_digi_analysis'] and not self.vars['several_blob_per_arena']:
            self.one_descriptor_per_arena["iso_digi_transi"] = pd.NA
            if not pd.isna(self.one_descriptor_per_arena["first_move"]):
                logging.info(f"Arena n°{self.one_descriptor_per_arena['arena']}. Starting growth transition analysis.")

                # II) Once a pseudopod is deployed, look for a disk/ around the original shape
                growth_begining = self.surfarea < ((self.surfarea[0] * 1.2) + ((self.dims[1] / 4) * (self.dims[2] / 4)))
                dilated_origin = cv2.dilate(self.binary[self.one_descriptor_per_arena["first_move"], :, :], kernel=cross_33, iterations=10, borderType=cv2.BORDER_CONSTANT, borderValue=0)
                isisotropic = np.sum(self.binary[:, :, :] * dilated_origin, (1, 2))
                isisotropic *= growth_begining
                # Ask if the dilated origin area is 90% covered during the growth beginning
                isisotropic = isisotropic > 0.9 * dilated_origin.sum()
                if np.any(isisotropic):
                    self.one_descriptor_per_arena["is_growth_isotropic"] = 1
                    # Determine a solidity reference to look for a potential breaking of the isotropic growth
                    if self.compute_solidity_separately:
                        solidity_reference = np.mean(self.solidity[:self.one_descriptor_per_arena["first_move"]])
                        different_solidity = self.solidity < (0.9 * solidity_reference)
                        del self.solidity
                    else:
                        solidity_reference = np.mean(
                            self.one_row_per_frame.iloc[:(self.one_descriptor_per_arena["first_move"]), :]["solidity"])
                        different_solidity = self.one_row_per_frame["solidity"].values < (0.9 * solidity_reference)
                    # Make sure that isotropic breaking not occur before isotropic growth
                    if np.any(different_solidity):
                        self.one_descriptor_per_arena["iso_digi_transi"] = np.nonzero(different_solidity)[0][0] * self.time_interval
                else:
                    self.one_descriptor_per_arena["is_growth_isotropic"] = 0
            else:
                self.one_descriptor_per_arena["is_growth_isotropic"] = pd.NA
                

    def check_converted_video_type(self):
        if self.converted_video.dtype != "uint8":
            self.converted_video -= np.min(self.converted_video)
            self.converted_video = np.round((255 * (self.converted_video / np.max(self.converted_video)))).astype(np.uint8)


    def networks_detection(self, show_seg=False):
        if not pd.isna(self.one_descriptor_per_arena["first_move"]) and not self.vars['several_blob_per_arena'] and (self.vars['save_coord_network'] or self.vars['network_analysis']):
            logging.info(f"Arena n°{self.one_descriptor_per_arena['arena']}. Starting network detection.")
            smooth_segmentation_over_time = True
            detect_pseudopods = True
            pseudopod_min_size = 50
            self.check_converted_video_type()
            if detect_pseudopods:
                pseudopod_vid = np.zeros_like(self.binary, dtype=bool)
            potential_network = np.zeros_like(self.binary, dtype=bool)
            self.network_dynamics = np.zeros_like(self.binary, dtype=np.uint8)
            greyscale = self.visu[-1, ...].mean(axis=-1)
            NetDet = NetworkDetection(greyscale, possibly_filled_pixels=self.binary[-1, ...],
                                      origin_to_add=self.origin)
            NetDet.get_best_network_detection_method()
            NetDet.change_greyscale(self.visu[-1, ...], c_space_dict=self.vars['convert_for_motion'])
            lighter_background = NetDet.greyscale_image[self.binary[-1, ...] > 0].mean() < NetDet.greyscale_image[self.binary[-1, ...] == 0].mean()


            for t in np.arange(self.one_descriptor_per_arena["first_move"], self.dims[0]):  # 20):#
                greyscale = self.visu[t, ...].mean(axis=-1)
                NetDet_fast = NetworkDetection(greyscale, possibly_filled_pixels=self.binary[t, ...],
                                          origin_to_add=self.origin, best_result=NetDet.best_result)
                NetDet_fast.detect_network()
                if detect_pseudopods:
                    NetDet_fast.detect_pseudopods(lighter_background, pseudopod_min_size=pseudopod_min_size)
                    NetDet_fast.merge_network_with_pseudopods()
                    pseudopod_vid[t, ...] = NetDet_fast.pseudopods
                potential_network[t, ...] = NetDet_fast.complete_network
            for t in np.arange(self.one_descriptor_per_arena["first_move"], self.dims[0]):  # 20):#
                if smooth_segmentation_over_time:
                    if 2 <= t <= (self.dims[0] - 2):
                        computed_network = potential_network[(t - 2):(t + 3), :, :].sum(axis=0)
                        computed_network[computed_network == 1] = 0
                        computed_network[computed_network > 1] = 1
                    else:
                        if t < 2:
                            computed_network = potential_network[:2, :, :].sum(axis=0)
                        else:
                            computed_network = potential_network[-2:, :, :].sum(axis=0)
                        computed_network[computed_network > 0] = 1
                else:
                    computed_network = computed_network[t, :, :].copy()

                if self.origin is not None:
                    computed_network = computed_network * (1 - self.origin)
                    origin_contours = get_contours(self.origin)
                    complete_network = np.logical_or(origin_contours, computed_network).astype(np.uint8)
                complete_network = keep_one_connected_component(complete_network)

                if detect_pseudopods:
                    # Make sure that removing pseudopods do not cut the network:
                    without_pseudopods = complete_network * (1 - pseudopod_vid[t])
                    only_connected_network = keep_one_connected_component(without_pseudopods)
                    # # Option A: To add these cutting regions to the pseudopods do:
                    pseudopods = (1 - only_connected_network) * complete_network
                    pseudopod_vid[t] = pseudopods
                self.network_dynamics[t] = complete_network

                # # Option B: To add these cutting regions to the network:
                # # Differentiate pseudopods that cut the network from the 'true ones'
                # # Dilate pseudopods and restrein them to the
                # pseudopods = cv2.dilate(pseudopod_vid[t], kernel=Ellipse((15, 15)).create().astype(np.uint8),
                #                         iterations=1) * self.binary[t, :, :]
                # nb, numbered_pseudopods = cv2.connectedComponents(pseudopods)
                # pseudopods = np.zeros_like(pseudopod_vid[t])
                # for p_i in range(1, nb + 1):
                #     pseudo_i = numbered_pseudopods == p_i
                #     nb_i, remainings, stats, centro = cv2.connectedComponentsWithStats(
                #         complete_network * (1 - pseudo_i.astype(np.uint8)))
                #     if (stats[:, 4] > pseudopod_min_size).sum() == 2:
                #         pseudopods[pseudo_i] = 1
                #         fragmented = np.nonzero(stats[:, 4] <= pseudopod_min_size)[0]
                #         pseudopods[np.isin(remainings, fragmented)] = 1
                # pseudopod_vid[t] = pseudopods
                # complete_network[pseudopods > 0] = 1
                # self.network_dynamics[t] = complete_network


                imtoshow = self.visu[t, ...]
                eroded_binary = cv2.erode(self.network_dynamics[t, ...], cross_33)
                net_coord = np.nonzero(self.network_dynamics[t, ...] - eroded_binary)
                imtoshow[net_coord[0], net_coord[1], :] = (34, 34, 158)
                if show_seg:
                    cv2.imshow("", cv2.resize(imtoshow, (1000, 1000)))
                    cv2.waitKey(1)
                else:
                    self.visu[t, ...] = imtoshow
                if show_seg:
                    cv2.destroyAllWindows()

            network_coord = smallest_memory_array(np.nonzero(self.network_dynamics), "uint")

            if detect_pseudopods:
                self.network_dynamics[pseudopod_vid > 0] = 2
                pseudopod_coord = smallest_memory_array(np.nonzero(pseudopod_vid), "uint")
            if self.vars['save_coord_network']:
                 np.save(
                    f"coord_tubular_network{self.one_descriptor_per_arena['arena']}_t{self.dims[0]}_y{self.dims[1]}_x{self.dims[2]}.npy",
                    network_coord)

                 if detect_pseudopods:
                     np.save(
                        f"coord_pseudopods{self.one_descriptor_per_arena['arena']}_t{self.dims[0]}_y{self.dims[1]}_x{self.dims[2]}.npy",
                        pseudopod_coord)

    def graph_extraction(self):
        if self.vars['graph_extraction'] and not self.vars['network_analysis'] and not self.vars['save_coord_network']:
            self.network_dynamics = self.binary
        _, _, _, origin_centroid = cv2.connectedComponentsWithStats(self.origin)
        origin_centroid = np.round((origin_centroid[1, 1], origin_centroid[1, 0])).astype(np.uint64)
        for t in np.arange(self.one_descriptor_per_arena["first_move"], self.dims[0]):  # 20):#


            if self.origin is not None:
                computed_network = self.network_dynamics[t, ...] * (1 - self.origin)
                origin_contours = get_contours(self.origin)
                computed_network = np.logical_or(origin_contours, computed_network).astype(np.uint8)
            else:
                origin_contours = None
                computed_network = self.network_dynamics[t, ...].astype(np.uint8)
            computed_network = keep_one_connected_component(computed_network)
            pad_network, pad_origin = add_padding([computed_network, self.origin])
            pad_origin_centroid = origin_centroid + 1
            pad_skeleton, pad_distances, pad_origin_contours = get_skeleton_and_widths(pad_network, pad_origin,
                                                                                       pad_origin_centroid)
            edge_id = EdgeIdentification(pad_skeleton, pad_distances)
            edge_id.run_edge_identification()
            if pad_origin_contours is not None:
                origin_contours = remove_padding([pad_origin_contours])[0]
            edge_id.make_vertex_table(origin_contours, self.network_dynamics[t, ...] == 2)
            edge_id.make_edge_table(self.converted_video[:, t])


            edge_id.vertex_table = np.hstack((np.repeat(t, edge_id.vertex_table.shape[0])[:, None], edge_id.vertex_table))
            edge_id.edge_table = np.hstack((np.repeat(t, edge_id.edge_table.shape[0])[:, None], edge_id.edge_table))
            if t == self.one_descriptor_per_arena["first_move"]:
                vertex_table = edge_id.vertex_table.copy()
                edge_table = edge_id.edge_table.copy()
            else:
                vertex_table = np.vstack((vertex_table, edge_id.vertex_table))
                edge_table = np.vstack((edge_table, edge_id.edge_table))

        vertex_table = pd.DataFrame(vertex_table, columns=["t", "y", "x", "vertex_id", "is_tip", "origin",
                                                 "vertex_connected"])
        edge_table = pd.DataFrame(edge_table,
                        columns=["t", "edge_id", "vertex1", "vertex2", "length", "average_width", "intensity", "betweenness_centrality"])
        vertex_table.to_csv(
            f"vertex_table{self.one_descriptor_per_arena['arena']}_t{self.dims[0]}_y{self.dims[1]}_x{self.dims[2]}.csv")
        edge_table.to_csv(
            f"edge_table{self.one_descriptor_per_arena['arena']}_t{self.dims[0]}_y{self.dims[1]}_x{self.dims[2]}.csv")


    def memory_allocation_for_cytoscillations(self):
        try:
            period_in_frame_nb = int(self.vars['expected_oscillation_period'] / self.time_interval)
            if period_in_frame_nb < 2:
                period_in_frame_nb = 2
            necessary_memory = self.converted_video.shape[0] * self.converted_video.shape[1] * \
                               self.converted_video.shape[2] * 64 * 4 * 1.16415e-10
            available_memory = (virtual_memory().available >> 30) - self.vars['min_ram_free']
            if len(self.converted_video.shape) == 4:
                self.converted_video = self.converted_video[:, :, :, 0]
            average_intensities = np.mean(self.converted_video, (1, 2))
            if self.vars['lose_accuracy_to_save_memory'] or (necessary_memory > available_memory):
                oscillations_video = np.zeros(self.converted_video.shape, dtype=np.float16)
                for cy in np.arange(self.converted_video.shape[1]):
                    for cx in np.arange(self.converted_video.shape[2]):
                        oscillations_video[:, cy, cx] = np.round(np.gradient(self.converted_video[:, cy, cx, ...]/average_intensities,
                                                                      period_in_frame_nb), 3).astype(np.float16)
            else:
                oscillations_video = np.gradient(self.converted_video/average_intensities, period_in_frame_nb, axis=0)
            # check if conv change here
            self.check_converted_video_type()
            if len(self.converted_video.shape) == 3:
                self.converted_video = np.stack((self.converted_video, self.converted_video, self.converted_video), axis=3)
            oscillations_video = np.sign(oscillations_video)
            return oscillations_video
        except Exception as exc:
            logging.error(f"{exc}. Retrying to allocate for 10 minutes before crashing. ")
            return None


    def study_cytoscillations(self, show_seg):
        if pd.isna(self.one_descriptor_per_arena["first_move"]):
            if not self.vars['lose_accuracy_to_save_memory']:
                self.check_converted_video_type()
            if self.vars['oscilacyto_analysis']:
                self.one_row_per_frame['mean_cluster_area'] = pd.NA
                self.one_row_per_frame['cluster_number'] = pd.NA
        else:
            if self.vars['save_coord_thickening_slimming'] or self.vars['oscilacyto_analysis']:
                logging.info(f"Arena n°{self.one_descriptor_per_arena['arena']}. Starting oscillation analysis.")
                oscillations_video = None
                staring_time = default_timer()
                current_time = staring_time
                while oscillations_video is None and (current_time - staring_time) < 600:
                    oscillations_video = self.memory_allocation_for_cytoscillations()
                    if oscillations_video is None:
                        sleep(30)
                        current_time = default_timer()

                within_range = (1 - self.binary[0, :, :]) * self.borders
                within_range = self.binary * within_range
                oscillations_video *= within_range
                del within_range
                oscillations_video += 1
                oscillations_video = oscillations_video.astype(np.uint8)

                dotted_image = np.ones(self.converted_video.shape[1:3], np.uint8)
                for cy in np.arange(dotted_image.shape[0]):
                    if cy % 2 != 0:
                        dotted_image[cy, :] = 0
                for cx in np.arange(dotted_image.shape[1]):
                    if cx % 2 != 0:
                        dotted_image[:, cx] = 0

                if self.start is None:
                    self.start = 0

                for t in np.arange(self.dims[0]):
                    eroded_binary = cv2.erode(self.binary[t, :, :], cross_33)
                    contours = self.binary[t, :, :] - eroded_binary
                    contours_idx = np.nonzero(contours)
                    imtoshow = deepcopy(self.converted_video[t, ...])
                    imtoshow[contours_idx[0], contours_idx[1], :] = self.vars['contour_color']
                    if self.vars['iso_digi_analysis'] and not self.vars['several_blob_per_arena'] and not pd.isna(self.one_descriptor_per_arena["iso_digi_transi"]):
                        if self.one_descriptor_per_arena["is_growth_isotropic"] == 1:
                            if t < self.one_descriptor_per_arena["iso_digi_transi"]:
                                imtoshow[contours_idx[0], contours_idx[1], 2] = 255
                    oscillations_image = np.zeros(self.dims[1:], np.uint8)
                    if t >= self.start:
                        # Add in or ef if a pixel has at least 4 neighbor in or ef
                        neigh_comp = CompareNeighborsWithValue(oscillations_video[t, :, :], connectivity=8, data_type=np.int8)
                        neigh_comp.is_inf(1, and_itself=False)
                        neigh_comp.is_sup(1, and_itself=False)
                        # Not verified if influx is really influx (resp efflux)
                        influx = neigh_comp.sup_neighbor_nb
                        efflux = neigh_comp.inf_neighbor_nb

                        # Only keep pixels having at least 4 positive (resp. negative) neighbors
                        influx[influx <= 4] = 0
                        efflux[efflux <= 4] = 0
                        influx[influx > 4] = 1
                        efflux[efflux > 4] = 1
                        if np.any(influx) or np.any(efflux):
                            influx, in_stats, in_centroids = cc(influx)
                            efflux, ef_stats, ef_centroids = cc(efflux)
                            # Only keep clusters larger than 'minimal_oscillating_cluster_size' pixels (smaller are considered as noise
                            in_smalls = np.nonzero(in_stats[:, 4] < self.vars['minimal_oscillating_cluster_size'])[0]
                            if len(in_smalls) > 0:
                                influx[np.isin(influx, in_smalls)] = 0
                                in_stats = in_stats[:in_smalls[0], :]
                                in_centroids = in_centroids[:in_smalls[0], :]
                            ef_smalls = np.nonzero(ef_stats[:, 4] < self.vars['minimal_oscillating_cluster_size'])[0]
                            if len(ef_smalls) > 0:
                                efflux[np.isin(efflux, ef_smalls)] = 0
                                ef_stats = ef_stats[:(ef_smalls[0]), :]
                                ef_centroids = ef_centroids[:(ef_smalls[0]), :]
                            in_idx = np.nonzero(influx)  # NEW
                            ef_idx = np.nonzero(efflux)  # NEW
                            oscillations_image[in_idx[0], in_idx[1]] = 1  # NEW
                            oscillations_image[ef_idx[0], ef_idx[1]] = 2  # NEW
                            # Prepare the image for display
                            influx *= dotted_image
                            efflux *= dotted_image
                            in_idx = np.nonzero(influx)
                            ef_idx = np.nonzero(efflux)
                            imtoshow[in_idx[0], in_idx[1], :2] = 153  # Green: influx, intensity increase
                            imtoshow[in_idx[0], in_idx[1], 2] = 0
                            imtoshow[ef_idx[0], ef_idx[1], 1:] = 0  # Blue: efflux, intensity decrease
                            imtoshow[ef_idx[0], ef_idx[1], 0] = 204
                    oscillations_video[t, :, :] = oscillations_image
                    self.converted_video[t, ...] = deepcopy(imtoshow)
                    if show_seg:
                        im_to_show = cv2.resize(imtoshow, (540, 540))
                        cv2.imshow("shape_motion", im_to_show)
                        cv2.waitKey(1)
                if show_seg:
                    cv2.destroyAllWindows()
                if self.vars['save_coord_thickening_slimming']:
                     np.save(f"coord_thickening{self.one_descriptor_per_arena['arena']}_t{self.dims[0]}_y{self.dims[1]}_x{self.dims[2]}.npy", smallest_memory_array(np.nonzero(oscillations_video == 1), "uint"))
                     np.save(f"coord_slimming{self.one_descriptor_per_arena['arena']}_t{self.dims[0]}_y{self.dims[1]}_x{self.dims[2]}.npy", smallest_memory_array(np.nonzero(oscillations_video == 2), "uint"))


                if self.vars['oscilacyto_analysis']:
                    # To get the median oscillatory period of each oscillating cluster,
                    # we create a dict containing two lists (for influx and efflux)
                    # Each list element correspond to a cluster and stores :
                    # All pixel coordinates of that cluster, their corresponding lifespan, their time of disappearing
                    # Row number will give the size. Euclidean distance between pix coord, the wave distance
                    self.clusters_final_data = np.empty((0, 6),
                                                     dtype=np.float32)  # ["mean_pixel_period", "phase", "total_size", "edge_distance", cy, cx]
                    period_tracking = np.zeros(self.converted_video.shape[1:3], dtype=np.uint32)
                    efflux_study = ClusterFluxStudy(self.converted_video.shape[:3])
                    influx_study = ClusterFluxStudy(self.converted_video.shape[:3])

                    if self.start is None:
                        self.start = 0
                    if self.vars['fractal_analysis']:
                        if os.path.exists(f"oscillating_clusters_temporal_dynamics.h5"):
                            remove_h5_key(f"oscillating_clusters_temporal_dynamics.h5",
                                          f"arena{self.one_descriptor_per_arena['arena']}")
                    cluster_id_matrix = np.zeros(self.dims[1:], dtype =np.uint64)
                    named_cluster_number = 0
                    mean_cluster_area = np.zeros(oscillations_video.shape[0])
                    pat_tracker = PercentAndTimeTracker(self.dims[0], compute_with_elements_number=True)
                    for t in np.arange(np.max((self.start, self.lost_frames)), self.dims[0]):  # np.arange(21): #
                        eroded_binary = cv2.erode(self.binary[t, :, :], cross_33)
                        contours = self.binary[t, :, :] - eroded_binary
                        oscillations_image = oscillations_video[t, ...]
                        influx = (oscillations_image == 1).astype(np.uint8)
                        efflux = (oscillations_image == 2).astype(np.uint8)
                        in_idx = np.nonzero(influx)  # NEW
                        ef_idx = np.nonzero(efflux)
                        influx, in_stats, in_centroids = cc(influx)
                        efflux, ef_stats, ef_centroids = cc(efflux)
                        in_stats = in_stats[1:]
                        in_centroids = in_centroids[1:]
                        ef_stats = ef_stats[1:]
                        ef_centroids = ef_centroids[1:]
                        # Sum the number of connected components minus the background to get the number of clusters
                        oscillating_cluster_number = in_stats.shape[0] + ef_stats.shape[0]
                        updated_cluster_names = [0]
                        if oscillating_cluster_number > 0:
                            current_percentage, eta = pat_tracker.get_progress(t, element_number=oscillating_cluster_number)
                            logging.info(
                                f"Arena n°{self.one_descriptor_per_arena['arena']}, Oscillatory cluster computation: {current_percentage}%{eta}")
                            if self.vars['fractal_analysis']:
                                # New analysis to get the surface dynamic of every oscillatory cluster: Part 2 openning:
                                network_at_t = np.zeros(self.dims[1:], dtype=np.uint8)
                                network_idx = self.network_dynamics[:, self.network_dynamics[0, :] == t]
                                network_at_t[network_idx[1, :], network_idx[2, :]] = 1
                                shapes = np.zeros(self.dims[1:], dtype=np.uint32)
                                shapes[in_idx[0], in_idx[1]] = influx[in_idx[0], in_idx[1]]
                                max_in = in_stats.shape[0]
                                shapes[ef_idx[0], ef_idx[1]] = max_in + efflux[ef_idx[0], ef_idx[1]]
                                centers = np.vstack((in_centroids, ef_centroids))
                                cluster_dynamic = np.zeros((int(oscillating_cluster_number) - 1, 13), dtype=np.float64)
                                for clust_i in np.arange(oscillating_cluster_number - 1, dtype=np.uint32):  # 120)):# #92
                                    cluster = clust_i + 1
                                    # cluster = 1
                                    # print(cluster)
                                    current_cluster_img = (shapes == cluster).astype(np.uint8)
                                    # I/ Find out which names the current cluster had at t-1
                                    cluster_previous_names = np.unique(current_cluster_img * cluster_id_matrix)
                                    cluster_previous_names = cluster_previous_names[cluster_previous_names != 0]
                                    # II/ Find out if the current cluster name had already been analyzed at t
                                    # If there no match with the saved cluster_id_matrix, assign cluster ID
                                    if t == 0 or len(cluster_previous_names) == 0:
                                        # logging.info("New cluster")
                                        named_cluster_number += 1
                                        cluster_names = [named_cluster_number]
                                    # If there is at least 1 match with the saved cluster_id_matrix, we keep the cluster_previous_name(s)
                                    else:
                                        cluster_names = cluster_previous_names.tolist()
                                    # Handle cluster division if necessary
                                    if np.any(np.isin(updated_cluster_names, cluster_names)):
                                        named_cluster_number += 1
                                        cluster_names = [named_cluster_number]

                                    # Get flow direction:
                                    if np.unique(oscillations_image * current_cluster_img)[1] == 1:
                                        flow = 1
                                    else:
                                        flow = - 1
                                    # Update cluster ID matrix for the current frame
                                    coords = np.nonzero(current_cluster_img)
                                    cluster_id_matrix[coords[0], coords[1]] = cluster_names[0]

                                    # Save the current cluster areas:
                                    inner_network = current_cluster_img * network_at_t
                                    inner_network_area = inner_network.sum()
                                    zoomed_binary, side_lengths = prepare_box_counting(current_cluster_img,
                                                                                       side_threshold=self.vars[
                                                                                           'fractal_box_side_threshold'],
                                                                                       zoom_step=self.vars[
                                                                                           'fractal_zoom_step'],
                                                                                       contours=True)
                                    box_count_dim, r_value, box_nb = box_counting_dimension(zoomed_binary, side_lengths)

                                    if np.any(inner_network):
                                        zoomed_binary, side_lengths = prepare_box_counting(inner_network,
                                                                                           side_threshold=self.vars[
                                                                                               'fractal_box_side_threshold'],
                                                                                           zoom_step=self.vars[
                                                                                               'fractal_zoom_step'],
                                                                                           contours=False)
                                        inner_network_box_count_dim, inner_net_r_value, inner_net_box_nb = box_counting_dimension(
                                            zoomed_binary, side_lengths)
                                    else:
                                        inner_network_box_count_dim, inner_net_r_value, inner_net_box_nb = 0, 0, 0
                                    # Calculate centroid and add to centroids list
                                    centroid_x, centroid_y = centers[cluster, :]
                                    if self.vars['output_in_mm']:
                                        cluster_dynamic[clust_i, :] = np.array(
                                            (t * self.time_interval, cluster_names[0], flow, centroid_y, centroid_x,
                                             current_cluster_img.sum() * self.vars['average_pixel_size'],
                                             inner_network_area * self.vars['average_pixel_size'], box_count_dim, r_value,
                                             box_nb, inner_network_box_count_dim, inner_net_r_value, inner_net_box_nb),
                                            dtype=np.float64)
                                    else:
                                        cluster_dynamic[clust_i, :] = np.array((t, cluster_names[0], flow, centroid_y,
                                                                             centroid_x, current_cluster_img.sum(),
                                                                             inner_network_area, box_count_dim, r_value,
                                                                             box_nb, inner_network_box_count_dim,
                                                                             inner_net_r_value, inner_net_box_nb),
                                                                            dtype=np.float64)

                                    updated_cluster_names = np.append(updated_cluster_names, cluster_names)
                                vstack_h5_array(f"oscillating_clusters_temporal_dynamics.h5",
                                                cluster_dynamic, key=f"arena{self.one_descriptor_per_arena['arena']}")

                                # Reset cluster_id_matrix for the next frame
                                cluster_id_matrix *= self.binary[t, :, :]

                            period_tracking, self.clusters_final_data = efflux_study.update_flux(t, contours, efflux,
                                                                                                 period_tracking,
                                                                                                 self.clusters_final_data)
                            period_tracking, self.clusters_final_data = influx_study.update_flux(t, contours, influx,
                                                                                                 period_tracking,
                                                                                                 self.clusters_final_data)
                            
                            mean_cluster_area[t] = np.mean(np.concatenate((in_stats[:, 4], ef_stats[:, 4])))
                    if self.vars['output_in_mm']:
                        self.clusters_final_data[:, 1] *= self.time_interval # phase
                        self.clusters_final_data[:, 2] *= self.vars['average_pixel_size']  # size
                        self.clusters_final_data[:, 3] *= np.sqrt(self.vars['average_pixel_size'])  # distance
                        self.one_row_per_frame['mean_cluster_area'] = mean_cluster_area * self.vars['average_pixel_size']
                    self.one_row_per_frame['cluster_number'] = named_cluster_number

                del oscillations_video


    def fractal_descriptions(self):
        if not pd.isna(self.one_descriptor_per_arena["first_move"]) and self.vars['fractal_analysis']:
            logging.info(f"Arena n°{self.one_descriptor_per_arena['arena']}. Starting fractal analysis.")

            if self.vars['network_analysis']:
                box_counting_dimensions = np.zeros((self.dims[0], 7), dtype=np.float64)
            else:
                box_counting_dimensions = np.zeros((self.dims[0], 3), dtype=np.float64)

            for t in np.arange(self.dims[0]):
                if self.vars['network_analysis']:
                    box_counting_dimensions[t, 0] = self.network_dynamics[t, ...].sum()
                    zoomed_binary, side_lengths = prepare_box_counting(self.binary[t, ...], side_threshold=self.vars[
                        'fractal_box_side_threshold'], zoom_step=self.vars['fractal_zoom_step'], contours=True)
                    box_counting_dimensions[t, 1], box_counting_dimensions[t, 2], box_counting_dimensions[
                        t, 3] = box_counting_dimension(zoomed_binary, side_lengths)
                    zoomed_binary, side_lengths = prepare_box_counting(self.network_dynamics[t, ...],
                                                                       side_threshold=self.vars[
                                                                           'fractal_box_side_threshold'],
                                                                       zoom_step=self.vars['fractal_zoom_step'],
                                                                       contours=False)
                    box_counting_dimensions[t, 4], box_counting_dimensions[t, 5], box_counting_dimensions[
                        t, 6] = box_counting_dimension(zoomed_binary, side_lengths)
                else:
                    zoomed_binary, side_lengths = prepare_box_counting(self.binary[t, ...],
                                                                       side_threshold=self.vars['fractal_box_side_threshold'],
                                                                       zoom_step=self.vars['fractal_zoom_step'], contours=True)
                    box_counting_dimensions[t, :] = box_counting_dimension(zoomed_binary, side_lengths)

            if self.vars['network_analysis']:
                self.one_row_per_frame["inner_network_size"] = box_counting_dimensions[:, 0]
                self.one_row_per_frame["fractal_dimension"] = box_counting_dimensions[:, 1]
                self.one_row_per_frame["fractal_r_value"] = box_counting_dimensions[:, 2]
                self.one_row_per_frame["fractal_box_nb"] = box_counting_dimensions[:, 3]
                self.one_row_per_frame["inner_network_fractal_dimension"] = box_counting_dimensions[:, 4]
                self.one_row_per_frame["inner_network_fractal_r_value"] = box_counting_dimensions[:, 5]
                self.one_row_per_frame["inner_network_fractal_box_nb"] = box_counting_dimensions[:, 6]
                if self.vars['output_in_mm']:
                    self.one_row_per_frame["inner_network_size"] *= self.vars['average_pixel_size']
            else:
                self.one_row_per_frame["fractal_dimension"] = box_counting_dimensions[:, 0]
                self.one_row_per_frame["fractal_box_nb"] = box_counting_dimensions[:, 1]
                self.one_row_per_frame["fractal_r_value"] = box_counting_dimensions[:, 2]

            if self.vars['network_analysis'] or self.vars['save_coord_network']:
                del self.network_dynamics

    def get_descriptors_summary(self):
        potential_descriptors = ["area", "perimeter", "circularity", "rectangularity", "total_hole_area", "solidity",
                                 "convexity", "eccentricity", "euler_number", "standard_deviation_y",
                                 "standard_deviation_x", "skewness_y", "skewness_x", "kurtosis_y", "kurtosis_x",
                                 "major_axis_len", "minor_axis_len", "axes_orientation"]

        self.one_descriptor_per_arena["final_area"] = self.binary[-1, :, :].sum()

    def save_efficiency_tests(self):
        # Provide images allowing to assess the analysis efficiency
        if self.dims[0] > 1:
            after_one_tenth_of_time = np.ceil(self.dims[0] / 10).astype(np.uint64)
        else:
            after_one_tenth_of_time = 0

        last_good_detection = self.dims[0] - 1
        if self.dims[0] > self.lost_frames:
            if self.vars['do_threshold_segmentation']:
                last_good_detection -= self.lost_frames
        else:
            last_good_detection = 0
        if self.visu is None:
            if len(self.converted_video.shape) == 3:
                self.converted_video = np.stack((self.converted_video, self.converted_video, self.converted_video),
                                             axis=3)
            self.efficiency_test_1 = deepcopy(self.converted_video[after_one_tenth_of_time, ...])
            self.efficiency_test_2 = deepcopy(self.converted_video[last_good_detection, ...])
        else:
            self.efficiency_test_1 = deepcopy(self.visu[after_one_tenth_of_time, :, :, :])
            self.efficiency_test_2 = deepcopy(self.visu[last_good_detection, :, :, :])

        position = (25, self.dims[1] // 2)
        text = str(self.one_descriptor_per_arena['arena'])
        eroded_binary = cv2.erode(self.binary[after_one_tenth_of_time, :, :], cross_33)
        contours = np.nonzero(self.binary[after_one_tenth_of_time, :, :] - eroded_binary)
        self.efficiency_test_1[contours[0], contours[1], :] = self.vars['contour_color']
        self.efficiency_test_1 = cv2.putText(self.efficiency_test_1, text, position, cv2.FONT_HERSHEY_SIMPLEX, 1,
                                        (self.vars["contour_color"], self.vars["contour_color"],
                                         self.vars["contour_color"], 255), 3)

        eroded_binary = cv2.erode(self.binary[last_good_detection, :, :], cross_33)
        contours = np.nonzero(self.binary[last_good_detection, :, :] - eroded_binary)
        self.efficiency_test_2[contours[0], contours[1], :] = self.vars['contour_color']
        self.efficiency_test_2 = cv2.putText(self.efficiency_test_2, text, position, cv2.FONT_HERSHEY_SIMPLEX, 1,
                                         (self.vars["contour_color"], self.vars["contour_color"],
                                          self.vars["contour_color"], 255), 3)

    def save_video(self):

        if self.vars['save_processed_videos']:
            self.check_converted_video_type()
            if len(self.converted_video.shape) == 3:
                self.converted_video = np.stack((self.converted_video, self.converted_video, self.converted_video),
                                                axis=3)
            for t in np.arange(self.dims[0]):

                eroded_binary = cv2.erode(self.binary[t, :, :], cross_33)
                contours = np.nonzero(self.binary[t, :, :] - eroded_binary)
                self.converted_video[t, contours[0], contours[1], :] = self.vars['contour_color']
                if "iso_digi_transi" in self.one_descriptor_per_arena.keys():
                    if self.vars['iso_digi_analysis']  and not self.vars['several_blob_per_arena'] and not pd.isna(self.one_descriptor_per_arena["iso_digi_transi"]):
                        if self.one_descriptor_per_arena["is_growth_isotropic"] == 1:
                            if t < self.one_descriptor_per_arena["iso_digi_transi"]:
                                self.converted_video[t, contours[0], contours[1], :] = 0, 0, 255
            del self.binary
            del self.surfarea
            del self.borders
            del self.origin
            del self.origin_idx
            del self.mean_intensity_per_frame
            del self.erodila_disk
            collect()
            if self.visu is None:
                true_frame_width = self.dims[2]
                if len(self.vars['background_list']) == 0:
                    self.background = None
                else:
                    self.background = self.vars['background_list'][self.one_descriptor_per_arena['arena'] - 1]
                self.visu = video2numpy(f"ind_{self.one_descriptor_per_arena['arena']}.npy", None, self.background, true_frame_width)
                if len(self.visu.shape) == 3:
                    self.visu = np.stack((self.visu, self.visu, self.visu), axis=3)
            self.converted_video = np.concatenate((self.visu, self.converted_video), axis=2)
            # self.visu = None

            if np.any(self.one_row_per_frame['time'] > 0):
                position = (5, self.dims[1] - 5)
                for t in np.arange(self.dims[0]):
                    image = self.converted_video[t, ...]
                    text = str(self.one_row_per_frame['time'][t]) + " min"
                    image = cv2.putText(image,  # numpy array on which text is written
                                    text,  # text
                                    position,  # position at which writing has to start
                                    cv2.FONT_HERSHEY_SIMPLEX,  # font family
                                    1,  # font size
                                    (self.vars["contour_color"], self.vars["contour_color"], self.vars["contour_color"], 255),  #(209, 80, 0, 255),  
                                    2)  # font stroke
                    self.converted_video[t, ...] = image
            vid_name = f"ind_{self.one_descriptor_per_arena['arena']}{self.vars['videos_extension']}"
            write_video(self.converted_video, vid_name, is_color=True, fps=self.vars['video_fps'])
            # self.converted_video = None

    def save_results(self):
        self.save_efficiency_tests()
        self.save_video()
        if self.vars['several_blob_per_arena']:
            try:
                with open(f"one_row_per_frame_arena{self.one_descriptor_per_arena['arena']}.csv", 'w') as file:
                    self.one_row_per_frame.to_csv(file, sep=';', index=False, lineterminator='\n')
            except PermissionError:
                logging.error(f"Never let one_row_per_frame_arena{self.one_descriptor_per_arena['arena']}.csv open when Cellects runs")

            create_new_csv: bool = False
            if os.path.isfile("one_row_per_arena.csv"):
                try:
                    with open(f"one_row_per_arena.csv", 'r') as file:
                        stats = pd.read_csv(file, header=0, sep=";")
                except PermissionError:
                    logging.error("Never let one_row_per_arena.csv open when Cellects runs")

                if len(self.one_descriptor_per_arena) == len(stats.columns) - 1:
                    try:
                        with open(f"one_row_per_arena.csv", 'w') as file:
                            stats.iloc[(self.one_descriptor_per_arena['arena'] - 1), 1:] = self.one_descriptor_per_arena.values()
                            # if len(self.vars['analyzed_individuals']) == 1:
                            #     stats = pd.DataFrame(self.one_descriptor_per_arena, index=[0])
                            # else:
                            #     stats = pd.DataFrame.from_dict(self.one_descriptor_per_arena)
                        # stats.to_csv("stats.csv", sep=';', index=False, lineterminator='\n')
                            stats.to_csv(file, sep=';', index=False, lineterminator='\n')
                    except PermissionError:
                        logging.error("Never let one_row_per_arena.csv open when Cellects runs")
                else:
                    create_new_csv = True
            else:
                create_new_csv = True
            if create_new_csv:
                with open(f"one_row_per_arena.csv", 'w') as file:
                    stats = pd.DataFrame(np.zeros((len(self.vars['analyzed_individuals']), len(self.one_descriptor_per_arena))),
                               columns=list(self.one_descriptor_per_arena.keys()))
                    stats.iloc[(self.one_descriptor_per_arena['arena'] - 1), :] = np.array(list(self.one_descriptor_per_arena.values()), dtype=np.uint32)
                    stats.to_csv(file, sep=';', index=False, lineterminator='\n')
        if not self.vars['keep_unaltered_videos'] and os.path.isfile(f"ind_{self.one_descriptor_per_arena['arena']}.npy"):
            os.remove(f"ind_{self.one_descriptor_per_arena['arena']}.npy")

    def change_results_of_one_arena(self):
        self.save_video()
        # I/ Update/Create one_row_per_arena.csv
        create_new_csv: bool = False
        if os.path.isfile("one_row_per_arena.csv"):
            try:
                with open(f"one_row_per_arena.csv", 'r') as file:
                    stats = pd.read_csv(file, header=0, sep=";")
                for stat_name, stat_value in self.one_descriptor_per_arena.items():
                    if stat_name in stats.columns:
                        stats.loc[(self.one_descriptor_per_arena['arena'] - 1), stat_name] = np.uint32(self.one_descriptor_per_arena[stat_name])
                with open(f"one_row_per_arena.csv", 'w') as file:
                    stats.to_csv(file, sep=';', index=False, lineterminator='\n')
            except PermissionError:
                logging.error("Never let one_row_per_arena.csv open when Cellects runs")
            except Exception as e:
                logging.error(f"{e}")
                create_new_csv = True
            # if len(self.one_descriptor_per_arena) == len(stats.columns):
            #     try:
            #         with open(f"one_row_per_arena.csv", 'w') as file:
            #             stats.iloc[(self.one_descriptor_per_arena['arena'] - 1), :] = self.one_descriptor_per_arena.values()
            #             # stats.to_csv("stats.csv", sep=';', index=False, lineterminator='\n')
            #             stats.to_csv(file, sep=';', index=False, lineterminator='\n')
            #     except PermissionError:
            #         logging.error("Never let one_row_per_arena.csv open when Cellects runs")
            # else:
            #     create_new_csv = True
        else:
            create_new_csv = True
        if create_new_csv:
            logging.info("Create a new one_row_per_arena.csv file")
            try:
                with open(f"one_row_per_arena.csv", 'w') as file:
                    stats = pd.DataFrame(np.zeros((len(self.vars['analyzed_individuals']), len(self.one_descriptor_per_arena))),
                               columns=list(self.one_descriptor_per_arena.keys()))
                    stats.iloc[(self.one_descriptor_per_arena['arena'] - 1), :] = np.array(list(self.one_descriptor_per_arena.values()), dtype=np.uint32)
                    stats.to_csv(file, sep=';', index=False, lineterminator='\n')
            except PermissionError:
                logging.error("Never let one_row_per_arena.csv open when Cellects runs")

        # II/ Update/Create one_row_per_frame.csv
        create_new_csv = False
        if os.path.isfile("one_row_per_frame.csv"):
            try:
                with open(f"one_row_per_frame.csv", 'r') as file:
                    descriptors = pd.read_csv(file, header=0, sep=";")
                for stat_name, stat_value in self.one_row_per_frame.items():
                    if stat_name in descriptors.columns:
                        descriptors.loc[((self.one_descriptor_per_arena['arena'] - 1) * self.dims[0]):((self.one_descriptor_per_arena['arena']) * self.dims[0] - 1), stat_name] = self.one_row_per_frame.loc[:, stat_name].values[:]
                with open(f"one_row_per_frame.csv", 'w') as file:
                    descriptors.to_csv(file, sep=';', index=False, lineterminator='\n')
                # with open(f"one_row_per_frame.csv", 'w') as file:
                #     for descriptor in descriptors.keys():
                #         descriptors.loc[
                #         ((self.one_descriptor_per_arena['arena'] - 1) * self.dims[0]):((self.one_descriptor_per_arena['arena']) * self.dims[0]),
                #         descriptor] = self.one_row_per_frame[descriptor]
                #     descriptors.to_csv(file, sep=';', index=False, lineterminator='\n')



                # if len(self.one_row_per_frame.columns) == len(descriptors.columns):
                #     with open(f"one_row_per_frame.csv", 'w') as file:
                #         # NEW
                #         for descriptor in descriptors.keys():
                #             descriptors.loc[((self.one_descriptor_per_arena['arena'] - 1) * self.dims[0]):((self.one_descriptor_per_arena['arena']) * self.dims[0]), descriptor] = self.one_row_per_frame[descriptor]
                #         # Old
                #         # descriptors.iloc[((self.one_descriptor_per_arena['arena'] - 1) * self.dims[0]):((self.one_descriptor_per_arena['arena']) * self.dims[0]), :] = self.one_row_per_frame
                #         descriptors.to_csv(file, sep=';', index=False, lineterminator='\n')
                # else:
                #     create_new_csv = True
            except PermissionError:
                logging.error("Never let one_row_per_frame.csv open when Cellects runs")
            except Exception as e:
                logging.error(f"{e}")
                create_new_csv = True
        else:
            create_new_csv = True
        if create_new_csv:
            logging.info("Create a new one_row_per_frame.csv file")
            try:
                with open(f"one_row_per_frame.csv", 'w') as file:
                    descriptors = pd.DataFrame(np.zeros((len(self.vars['analyzed_individuals']) * self.dims[0], len(self.one_row_per_frame.columns))),
                               columns=list(self.one_row_per_frame.keys()))
                    descriptors.iloc[((self.one_descriptor_per_arena['arena'] - 1) * self.dims[0]):((self.one_descriptor_per_arena['arena']) * self.dims[0]), :] = self.one_row_per_frame
                    descriptors.to_csv(file, sep=';', index=False, lineterminator='\n')
            except PermissionError:
                logging.error("Never let one_row_per_frame.csv open when Cellects runs")

        # III/ Update/Create one_row_per_oscillating_cluster.csv
        if not pd.isna(self.one_descriptor_per_arena["first_move"]) and self.vars['oscilacyto_analysis']:
            oscil_i = pd.DataFrame(
                np.c_[np.repeat(self.one_descriptor_per_arena['arena'], self.clusters_final_data.shape[0]), self.clusters_final_data],
                columns=['arena', 'mean_pixel_period', 'phase', 'cluster_size', 'edge_distance', 'coord_y', 'coord_x'])
            if os.path.isfile("one_row_per_oscillating_cluster.csv"):
                try:
                    with open(f"one_row_per_oscillating_cluster.csv", 'r') as file:
                        one_row_per_oscillating_cluster = pd.read_csv(file, header=0, sep=";")
                    with open(f"one_row_per_oscillating_cluster.csv", 'w') as file:
                        one_row_per_oscillating_cluster_before = one_row_per_oscillating_cluster[one_row_per_oscillating_cluster['arena'] < self.one_descriptor_per_arena['arena']]
                        one_row_per_oscillating_cluster_after = one_row_per_oscillating_cluster[one_row_per_oscillating_cluster['arena'] > self.one_descriptor_per_arena['arena']]
                        one_row_per_oscillating_cluster = pd.concat((one_row_per_oscillating_cluster_before, oscil_i, one_row_per_oscillating_cluster_after))
                        one_row_per_oscillating_cluster.to_csv(file, sep=';', index=False, lineterminator='\n')

                        # one_row_per_oscillating_cluster = one_row_per_oscillating_cluster[one_row_per_oscillating_cluster['arena'] != self.one_descriptor_per_arena['arena']]
                        # one_row_per_oscillating_cluster = pd.concat((one_row_per_oscillating_cluster, oscil_i))
                        # one_row_per_oscillating_cluster.to_csv(file, sep=';', index=False, lineterminator='\n')
                except PermissionError:
                    logging.error("Never let one_row_per_oscillating_cluster.csv open when Cellects runs")
            else:
                try:
                    with open(f"one_row_per_oscillating_cluster.csv", 'w') as file:
                        oscil_i.to_csv(file, sep=';', index=False, lineterminator='\n')
                except PermissionError:
                    logging.error("Never let one_row_per_oscillating_cluster.csv open when Cellects runs")

