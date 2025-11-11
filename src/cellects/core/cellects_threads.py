#!/usr/bin/env python3
"""
Cellects graphical user interface interacts with computational scripts through threads.
Especially, each thread calls one or several methods of the class named "program_organizer",
which regroup all available computation of the software.
These threads are started from a children of WindowType, run methods from program_organizer and send messages and
results to the corresponding children of WindowType, allowing, for instance, to display a result in the interface.



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


Arenas can be delimited automatically or manually. Cellects includes two automatic algorithms: A fast one to be used when arenas are symmetric around the initial position of the specimens or sufficiently far from each other, and a slower one to be used otherwise. These automatic algorithms work even if the arenas are not detectable in the images, but only work when there is a single individual in each arena. In the case of manual delimitation, the user draws each arena by holding down the mouse button. The following paragraphs describe the two automatic algorithms.
The fast algorithm computes each arena coordinate using the distances between the components detected in the seed image after step 1. For each component, Cellects finds its nearest neighbor and uses its distance as the side of the square, centered on the component, giving the x and y limits of the arena.
If the initial position of the cells do not provide good estimates of the center of each arena, Cellects can use the slower algorithm to find them. Because Cellects is intended to be very general, it cannot use specific characteristics of a particular arena to find its edges. Instead, it uses the motion and/or growth of the cell to infer the position of each arena. To do so, Cellects segments a sample of 5 images (equally spaced in time) using the same algorithm as for the seed image. Even if this segmentation is not accurate, the following algorithm finds the arenas robustly. First, it finds a rough estimate of the expected position of the cell. To do this, it dilates the cell in the first frame, until the edge of the dilated image is closer to the nearest centroid of other cells than to its own centroid. Then, it moves to the second image, and also dilates it in order to link together different disconnected components that may result from an inaccurate segmentation. Then, it performs an AND operation between these two dilated images and dilates the result so that it remains one component per arena. By doing this to all cells, we get an estimate of their shape in the second frame, and we can compute their centroids. We then repeat this procedure, for each pair of consecutive frames. Finally, Cellects computes the bounding boxes that contain the cells detected in the 5 frames for each arena, and uses them to estimate each arena coordinate.
In some experiments, all cells are located at one edge of the arena and move roughly in the same direction. Cellects includes an option to take advantage of this regularity and improve the accuracy of arena detection: Once the centroids of a frame have been estimated (as described above), Cellects finds the centroid with highest displacement with respect to the previous frame, and applies the same displacement to all centroids.

It also contains methods to write videos (as np arrays .npy files) corresponding to the pixels delimited by these arenas.

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

import logging
import weakref
from multiprocessing import Queue, Process, Manager
import os
import time
from glob import glob
from timeit import default_timer
from copy import deepcopy
import cv2
from numba.typed import Dict as TDict
import numpy as np
import pandas as pd
from PySide6 import QtCore
from cellects.image_analysis.morphological_operations import cross_33, Ellipse
from cellects.image_analysis.image_segmentation import generate_color_space_combination, apply_filter
from cellects.utils.load_display_save import read_and_rotate
from cellects.utils.formulas import bracket_to_uint8_image_contrast
from cellects.utils.utilitarian import PercentAndTimeTracker, reduce_path_len, split_dict
from cellects.core.one_video_per_blob import OneVideoPerBlob
from cellects.utils.load_display_save import write_video
from cellects.core.motion_analysis import MotionAnalysis


class LoadDataToRunCellectsQuicklyThread(QtCore.QThread):
    message_from_thread = QtCore.Signal(str)

    def __init__(self, parent=None):
        super(LoadDataToRunCellectsQuicklyThread, self).__init__(parent)
        self.setParent(parent)

    def run(self):
        self.parent().po.look_for_data()
        self.parent().po.load_data_to_run_cellects_quickly()
        if self.parent().po.first_exp_ready_to_run:
            self.message_from_thread.emit("Data found, Video tracking window and Run all directly are available")
        else:
            self.message_from_thread.emit("")


class LookForDataThreadInFirstW(QtCore.QThread):
    def __init__(self, parent=None):
        super(LookForDataThreadInFirstW, self).__init__(parent)
        self.setParent(parent)

    def run(self):
        self.parent().po.look_for_data()


class LoadFirstFolderIfSeveralThread(QtCore.QThread):
    message_when_thread_finished = QtCore.Signal(bool)
    def __init__(self, parent=None):
        super(LoadFirstFolderIfSeveralThread, self).__init__(parent)
        self.setParent(parent)

    def run(self):
        self.parent().po.load_data_to_run_cellects_quickly()
        if not self.parent().po.first_exp_ready_to_run:
            self.parent().po.get_first_image()
        self.message_when_thread_finished.emit(self.parent().po.first_exp_ready_to_run)


class GetFirstImThread(QtCore.QThread):
    message_when_thread_finished = QtCore.Signal(bool)
    def __init__(self, parent=None):
        """
        This class read the first image of the (first of the) selected analysis.
        According to the first_detection_frame value,it can be another image
        If this is the first time a first image is read, it also gather the following variables:
            - img_number
            - dims (video dimensions: time, y, x)
            - raw_images (whether images are in a raw format)
        If the selected analysis contains videos instead of images, it opens the first video
        and read the first_detection_frame th image.
        :param parent: An object containing all necessary variables.
        """
        super(GetFirstImThread, self).__init__(parent)
        self.setParent(parent)

    def run(self):
        self.parent().po.get_first_image()
        self.message_when_thread_finished.emit(True)


class GetLastImThread(QtCore.QThread):
    def __init__(self, parent=None):
        super(GetLastImThread, self).__init__(parent)
        self.setParent(parent)

    def run(self):
        self.parent().po.get_last_image()


class UpdateImageThread(QtCore.QThread):
    message_when_thread_finished = QtCore.Signal(bool)

    def __init__(self, parent=None):
        super(UpdateImageThread, self).__init__(parent)
        self.setParent(parent)

    def run(self):
        # I/ If this thread runs from user input, get the right coordinates
        # and convert them to fit the displayed image size
        user_input = len(self.parent().imageanalysiswindow.saved_coord) > 0 or len(self.parent().imageanalysiswindow.temporary_mask_coord) > 0
        if user_input:
            if len(self.parent().imageanalysiswindow.temporary_mask_coord) > 0:
                idx = self.parent().imageanalysiswindow.temporary_mask_coord
            else:
                idx = self.parent().imageanalysiswindow.saved_coord
            if len(idx) < 2:
                user_input = False
            else:
                # Convert coordinates:
                self.parent().imageanalysiswindow.display_image.update_image_scaling_factors()
                sf = self.parent().imageanalysiswindow.display_image.scaling_factors
                idx = np.array(((np.round(idx[0][0] * sf[0]), np.round(idx[0][1] * sf[1])), (np.round(idx[1][0] * sf[0]), np.round(idx[1][1] * sf[1]))), dtype=np.int64)
                min_y = np.min(idx[:, 0])
                max_y = np.max(idx[:, 0])
                min_x = np.min(idx[:, 1])
                max_x = np.max(idx[:, 1])
                if max_y > self.parent().imageanalysiswindow.drawn_image.shape[0]:
                    max_y = self.parent().imageanalysiswindow.drawn_image.shape[0] - 1
                if max_x > self.parent().imageanalysiswindow.drawn_image.shape[1]:
                    max_x = self.parent().imageanalysiswindow.drawn_image.shape[1] - 1
                if min_y < 0:
                    min_y = 0
                if min_x < 0:
                    min_x = 0

        if len(self.parent().imageanalysiswindow.temporary_mask_coord) == 0:
            # not_load
            # II/ If this thread aims at saving the last user input and displaying all user inputs:
            # Update the drawn_image according to every saved masks
            # 1) The segmentation mask
            # 2) The back_mask and bio_mask
            # 3) The automatically detected video contours
            # (re-)Initialize drawn image
            self.parent().imageanalysiswindow.drawn_image = deepcopy(self.parent().po.current_image)
            if self.parent().imageanalysiswindow.drawn_image.size < 1000000:
                contour_width = 3
            else:
                contour_width = 6
            # 1) The segmentation mask
            logging.info('Add the segmentation mask to the image')
            if self.parent().imageanalysiswindow.is_first_image_flag:
                im_combinations = self.parent().po.first_image.im_combinations
                im_mean = self.parent().po.first_image.image.mean()
            else:
                im_combinations = self.parent().po.last_image.im_combinations
                im_mean = self.parent().po.last_image.bgr.mean()
            # If there are image combinations, get the current corresponding binary image
            if im_combinations is not None and len(im_combinations) != 0:
                binary_idx = im_combinations[self.parent().po.current_combination_id]["binary_image"]
                # If it concerns the last image, only keep the contour coordinates

                cv2.eroded_binary = cv2.erode(binary_idx, cross_33)
                binary_idx = binary_idx - cv2.eroded_binary
                binary_idx = cv2.dilate(binary_idx, kernel=cross_33, iterations=contour_width)
                binary_idx = np.nonzero(binary_idx)
                # Color these coordinates in magenta on bright images, and in pink on dark images
                if im_mean > 126:
                    # logging.info('Color the segmentation mask in magenta')
                    self.parent().imageanalysiswindow.drawn_image[binary_idx[0], binary_idx[1], :] = np.array((20, 0, 150), dtype=np.uint8)
                else:
                    # logging.info('Color the segmentation mask in pink')
                    self.parent().imageanalysiswindow.drawn_image[binary_idx[0], binary_idx[1], :] = np.array((94, 0, 213), dtype=np.uint8)
            if user_input:# save
                mask = np.zeros(self.parent().imageanalysiswindow.drawn_image.shape[:2], dtype=np.uint8)
                if self.parent().imageanalysiswindow.back1_bio2 == 0:
                    logging.info("Save the user drawn mask of the current arena")
                    if self.parent().po.vars['arena_shape'] == 'circle':
                        ellipse = Ellipse((max_y - min_y, max_x - min_x)).create().astype(np.uint8)
                        mask[min_y:max_y, min_x:max_x, ...] = ellipse
                    else:
                        mask[min_y:max_y, min_x:max_x] = 1
                else:
                    logging.info("Save the user drawn mask of Cell or Back")

                    if self.parent().imageanalysiswindow.back1_bio2 == 2:
                        if self.parent().po.all['starting_blob_shape'] == 'circle':
                            ellipse = Ellipse((max_y - min_y, max_x - min_x)).create().astype(np.uint8)
                            mask[min_y:max_y, min_x:max_x, ...] = ellipse
                        else:
                            mask[min_y:max_y, min_x:max_x] = 1
                    else:
                        mask[min_y:max_y, min_x:max_x] = 1
                mask = np.nonzero(mask)

                if self.parent().imageanalysiswindow.back1_bio2 == 1:
                    self.parent().imageanalysiswindow.back_masks_number += 1
                    self.parent().imageanalysiswindow.back_mask[mask[0], mask[1]] = self.parent().imageanalysiswindow.available_back_names[0]
                elif self.parent().imageanalysiswindow.back1_bio2 == 2:
                    self.parent().imageanalysiswindow.bio_masks_number += 1
                    self.parent().imageanalysiswindow.bio_mask[mask[0], mask[1]] = self.parent().imageanalysiswindow.available_bio_names[0]
                elif self.parent().imageanalysiswindow.manual_delineation_flag:
                    self.parent().imageanalysiswindow.arena_masks_number += 1
                    self.parent().imageanalysiswindow.arena_mask[mask[0], mask[1]] = self.parent().imageanalysiswindow.available_arena_names[0]
                # 2)a) Apply all these masks to the drawn image:

            back_coord = np.nonzero(self.parent().imageanalysiswindow.back_mask)

            bio_coord = np.nonzero(self.parent().imageanalysiswindow.bio_mask)

            if self.parent().imageanalysiswindow.arena_mask is not None:
                arena_coord = np.nonzero(self.parent().imageanalysiswindow.arena_mask)
                self.parent().imageanalysiswindow.drawn_image[arena_coord[0], arena_coord[1], :] = np.repeat(self.parent().po.vars['contour_color'], 3).astype(np.uint8)

            self.parent().imageanalysiswindow.drawn_image[back_coord[0], back_coord[1], :] = np.array((224, 160, 81), dtype=np.uint8)

            self.parent().imageanalysiswindow.drawn_image[bio_coord[0], bio_coord[1], :] = np.array((17, 160, 212), dtype=np.uint8)

            image = self.parent().imageanalysiswindow.drawn_image
            # 3) The automatically detected video contours
            if self.parent().imageanalysiswindow.delineation_done:  # add a mask of the video contour
                # logging.info("Draw the delineation mask of each arena")
                for contour_i in range(len(self.parent().po.top)):
                    mask = np.zeros(self.parent().imageanalysiswindow.drawn_image.shape[:2], dtype=np.uint8)
                    min_cy = self.parent().po.top[contour_i]
                    max_cy = self.parent().po.bot[contour_i]
                    min_cx = self.parent().po.left[contour_i]
                    max_cx = self.parent().po.right[contour_i]
                    text = f"{contour_i + 1}"
                    position = (self.parent().po.left[contour_i] + 25, self.parent().po.top[contour_i] + (self.parent().po.bot[contour_i] - self.parent().po.top[contour_i]) // 2)
                    image = cv2.putText(image,  # numpy array on which text is written
                                    text,  # text
                                    position,  # position at which writing has to start
                                    cv2.FONT_HERSHEY_SIMPLEX,  # font family
                                    1,  # font size
                                    (138, 95, 18, 255),
                                    # (209, 80, 0, 255),  # font color
                                    2)  # font stroke
                    if (max_cy - min_cy) < 0 or (max_cx - min_cx) < 0:
                        self.parent().imageanalysiswindow.message.setText("Error: the shape number or the detection is wrong")
                    if self.parent().po.vars['arena_shape'] == 'circle':
                        ellipse = Ellipse((max_cy - min_cy, max_cx - min_cx)).create().astype(np.uint8)
                        ellipse = cv2.morphologyEx(ellipse, cv2.MORPH_GRADIENT, cross_33)
                        mask[min_cy:max_cy, min_cx:max_cx, ...] = ellipse
                    else:
                        mask[(min_cy, max_cy), min_cx:max_cx] = 1
                        mask[min_cy:max_cy, (min_cx, max_cx)] = 1
                    mask = cv2.dilate(mask, kernel=cross_33, iterations=contour_width)

                    mask = np.nonzero(mask)
                    image[mask[0], mask[1], :] = np.array((138, 95, 18), dtype=np.uint8)# self.parent().po.vars['contour_color']

        else: #load
            if user_input:
                # III/ If this thread runs from user input: update the drawn_image according to the current user input
                # Just add the mask to drawn_image as quick as possible
                # Add user defined masks
                # Take the drawn image and add the temporary mask to it
                image = deepcopy(self.parent().imageanalysiswindow.drawn_image)
                if self.parent().imageanalysiswindow.back1_bio2 == 0:
                    # logging.info("Dynamic drawing of the arena outline")
                    if self.parent().po.vars['arena_shape'] == 'circle':
                        ellipse = Ellipse((max_y - min_y, max_x - min_x)).create()
                        ellipse = np.stack((ellipse, ellipse, ellipse), axis=2).astype(np.uint8)
                        image[min_y:max_y, min_x:max_x, ...] *= (1 - ellipse)
                        image[min_y:max_y, min_x:max_x, ...] += ellipse
                    else:
                        mask = np.zeros(self.parent().imageanalysiswindow.drawn_image.shape[:2], dtype=np.uint8)
                        mask[min_y:max_y, min_x:max_x] = 1
                        mask = np.nonzero(mask)
                        image[mask[0], mask[1], :] = np.array((0, 0, 0), dtype=np.uint8)
                else:
                    # logging.info("Dynamic drawing of Cell or Back")
                    if self.parent().imageanalysiswindow.back1_bio2 == 2:
                        if self.parent().po.all['starting_blob_shape'] == 'circle':
                            ellipse = Ellipse((max_y - min_y, max_x - min_x)).create()
                            ellipse = np.stack((ellipse, ellipse, ellipse), axis=2).astype(np.uint8)
                            image[min_y:max_y, min_x:max_x, ...] *= (1 - ellipse)
                            ellipse[:, :, :] *= np.array((17, 160, 212), dtype=np.uint8)
                            image[min_y:max_y, min_x:max_x, ...] += ellipse
                        else:
                            mask = np.zeros(self.parent().imageanalysiswindow.drawn_image.shape[:2], dtype=np.uint8)
                            mask[min_y:max_y, min_x:max_x] = 1
                            mask = np.nonzero(mask)
                            image[mask[0], mask[1], :] = np.array((17, 160, 212), dtype=np.uint8)
                    else:
                        mask = np.zeros(self.parent().imageanalysiswindow.drawn_image.shape[:2], dtype=np.uint8)
                        mask[min_y:max_y, min_x:max_x] = 1
                        mask = np.nonzero(mask)
                        image[mask[0], mask[1], :] = np.array((224, 160, 81), dtype=np.uint8)

        self.parent().imageanalysiswindow.display_image.update_image(image)
        self.message_when_thread_finished.emit(True)


class FirstImageAnalysisThread(QtCore.QThread):
    message_from_thread = QtCore.Signal(str)
    message_when_thread_finished = QtCore.Signal(bool)

    def __init__(self, parent=None):
        super(FirstImageAnalysisThread, self).__init__(parent)
        self.setParent(parent)

    def run(self):
        tic = default_timer()
        biomask = None
        backmask = None
        if self.parent().imageanalysiswindow.bio_masks_number != 0:
            shape_nb, ordered_image = cv2.connectedComponents((self.parent().imageanalysiswindow.bio_mask > 0).astype(np.uint8))
            shape_nb -= 1
            biomask = np.nonzero(self.parent().imageanalysiswindow.bio_mask)
        else:
            shape_nb = 0
        if self.parent().imageanalysiswindow.back_masks_number != 0:
            backmask = np.nonzero(self.parent().imageanalysiswindow.back_mask)
        if self.parent().po.visualize or len(self.parent().po.first_im.shape) == 2 or shape_nb == self.parent().po.sample_number:
            self.message_from_thread.emit("Image segmentation, wait")
            if not self.parent().imageanalysiswindow.asking_first_im_parameters_flag and self.parent().po.all['scale_with_image_or_cells'] == 0 and self.parent().po.all["set_spot_size"]:
                self.parent().po.get_average_pixel_size()
                spot_size = self.parent().po.starting_blob_hsize_in_pixels
            else:
                spot_size = None
            self.parent().po.all["bio_mask"] = biomask
            self.parent().po.all["back_mask"] = backmask
            self.parent().po.fast_image_segmentation(is_first_image=True, biomask=biomask, backmask=backmask, spot_size=spot_size)
            if shape_nb == self.parent().po.sample_number and self.parent().po.first_image.im_combinations[self.parent().po.current_combination_id]['shape_number'] != self.parent().po.sample_number:
                self.parent().po.first_image.im_combinations[self.parent().po.current_combination_id]['shape_number'] = shape_nb
                self.parent().po.first_image.shape_number = shape_nb
                self.parent().po.first_image.validated_shapes = (self.parent().imageanalysiswindow.bio_mask > 0).astype(np.uint8)
                self.parent().po.first_image.im_combinations[self.parent().po.current_combination_id]['binary_image'] = self.parent().po.first_image.validated_shapes
        else:
            self.message_from_thread.emit("Generating analysis options, wait...")
            if self.parent().po.vars["color_number"] > 2:
                kmeans_clust_nb = self.parent().po.vars["color_number"]
                if self.parent().po.carefully:
                    self.message_from_thread.emit("Generating analysis options, wait less than 30 minutes")
                else:
                    self.message_from_thread.emit("Generating analysis options, a few minutes")
            else:
                kmeans_clust_nb = None
                if self.parent().po.carefully:
                    self.message_from_thread.emit("Generating analysis options, wait a few minutes")
                else:
                    self.message_from_thread.emit("Generating analysis options, around 1 minute")
            if self.parent().imageanalysiswindow.asking_first_im_parameters_flag:
                self.parent().po.first_image.find_first_im_csc(sample_number=self.parent().po.sample_number,
                                                               several_blob_per_arena=None,
                                                               spot_shape=None, spot_size=None,
                                                               kmeans_clust_nb=kmeans_clust_nb,
                                                               biomask=self.parent().po.all["bio_mask"],
                                                               backmask=self.parent().po.all["back_mask"],
                                                               color_space_dictionaries=None)
            else:
                if self.parent().po.all['scale_with_image_or_cells'] == 0:
                    self.parent().po.get_average_pixel_size()
                else:
                    self.parent().po.starting_blob_hsize_in_pixels = None
                self.parent().po.first_image.find_first_im_csc(sample_number=self.parent().po.sample_number,
                                                                                   several_blob_per_arena=self.parent().po.vars['several_blob_per_arena'],
                                                                                   spot_shape=self.parent().po.all['starting_blob_shape'],
                                                               spot_size=self.parent().po.starting_blob_hsize_in_pixels,
                                                                                   kmeans_clust_nb=kmeans_clust_nb,
                                                                                   biomask=self.parent().po.all["bio_mask"],
                                                                                   backmask=self.parent().po.all["back_mask"],
                                                                                   color_space_dictionaries=None)

        logging.info(f" image analysis lasted {default_timer() - tic} secondes")
        logging.info(f" image analysis lasted {np.round((default_timer() - tic) / 60)} minutes")
        self.message_when_thread_finished.emit(True)


class LastImageAnalysisThread(QtCore.QThread):
    message_from_thread = QtCore.Signal(str)
    message_when_thread_finished = QtCore.Signal(bool)

    def __init__(self, parent=None):
        super(LastImageAnalysisThread, self).__init__(parent)
        self.setParent(parent)

    def run(self):
        self.parent().po.cropping(False)
        self.parent().po.get_background_to_subtract()
        biomask = None
        backmask = None
        if self.parent().imageanalysiswindow.bio_masks_number != 0:
            biomask = np.nonzero(self.parent().imageanalysiswindow.bio_mask)
        if self.parent().imageanalysiswindow.back_masks_number != 0:
            backmask = np.nonzero(self.parent().imageanalysiswindow.back_mask)
        if self.parent().po.visualize or len(self.parent().po.first_im.shape) == 2:
            self.message_from_thread.emit("Image segmentation, wait...")
            self.parent().po.fast_image_segmentation(is_first_image=False, biomask=biomask, backmask=backmask)
        else:
            self.message_from_thread.emit("Generating analysis options, wait...")
            if self.parent().po.vars['several_blob_per_arena']:
                concomp_nb = [self.parent().po.sample_number, self.parent().po.first_image.size // 50]
                max_shape_size = .75 * self.parent().po.first_image.size
                total_surfarea = .99 * self.parent().po.first_image.size
            else:
                concomp_nb = [self.parent().po.sample_number, self.parent().po.sample_number * 200]
                if self.parent().po.all['are_zigzag'] == "columns":
                    inter_dist = np.mean(np.diff(np.nonzero(self.parent().po.videos.first_image.y_boundaries)))
                elif self.parent().po.all['are_zigzag'] == "rows":
                    inter_dist = np.mean(np.diff(np.nonzero(self.parent().po.videos.first_image.x_boundaries)))
                else:
                    dist1 = np.mean(np.diff(np.nonzero(self.parent().po.videos.first_image.y_boundaries)))
                    dist2 = np.mean(np.diff(np.nonzero(self.parent().po.videos.first_image.x_boundaries)))
                    inter_dist = np.max(dist1, dist2)
                if self.parent().po.all['starting_blob_shape'] == "circle":
                    max_shape_size = np.pi * np.square(inter_dist)
                else:
                    max_shape_size = np.square(2 * inter_dist)
                total_surfarea = max_shape_size * self.parent().po.sample_number
            out_of_arenas = None
            if self.parent().po.all['are_gravity_centers_moving'] != 1:
                out_of_arenas = np.ones_like(self.parent().po.videos.first_image.validated_shapes)
                for blob_i in np.arange(len(self.parent().po.vars['analyzed_individuals'])):
                    out_of_arenas[self.parent().po.top[blob_i]: (self.parent().po.bot[blob_i] + 1),
                    self.parent().po.left[blob_i]: (self.parent().po.right[blob_i] + 1)] = 0
            ref_image = self.parent().po.first_image.validated_shapes
            self.parent().po.first_image.generate_subtract_background(self.parent().po.vars['convert_for_motion'])
            kmeans_clust_nb = None
            self.parent().po.last_image.find_last_im_csc(concomp_nb, total_surfarea, max_shape_size, out_of_arenas,
                                                         ref_image, self.parent().po.first_image.subtract_background,
                                                         kmeans_clust_nb, biomask, backmask, color_space_dictionaries=None,
                                                         carefully=self.parent().po.carefully)
        self.message_when_thread_finished.emit(True)


class CropScaleSubtractDelineateThread(QtCore.QThread):
    message_from_thread = QtCore.Signal(str)
    message_when_thread_finished = QtCore.Signal(str)

    def __init__(self, parent=None):
        super(CropScaleSubtractDelineateThread, self).__init__(parent)
        self.setParent(parent)

    def run(self):
        logging.info("Start cropping if required")
        self.parent().po.cropping(is_first_image=True)
        self.parent().po.cropping(is_first_image=False)
        self.parent().po.get_average_pixel_size()
        if os.path.isfile('Data to run Cellects quickly.pkl'):
            os.remove('Data to run Cellects quickly.pkl')
        logging.info("Save data to run Cellects quickly")
        self.parent().po.data_to_save['first_image'] = True
        self.parent().po.save_data_to_run_cellects_quickly()
        self.parent().po.data_to_save['first_image'] = False
        if not self.parent().po.vars['several_blob_per_arena']:
            logging.info("Check whether the detected shape number is ok")
            nb, shapes, stats, centroids = cv2.connectedComponentsWithStats(self.parent().po.first_image.validated_shapes)
            y_lim = self.parent().po.first_image.y_boundaries
            if ((nb - 1) != self.parent().po.sample_number or np.any(stats[:, 4] == 1)):
                self.message_from_thread.emit("Image analysis failed to detect the right cell(s) number: restart the analysis.")
            elif len(np.nonzero(y_lim == - 1)) != len(np.nonzero(y_lim == 1)):
                self.message_from_thread.emit("Automatic arena delineation cannot work if one cell touches the image border.")
                self.parent().po.first_image.y_boundaries = None
            else:
                logging.info("Start automatic video delineation")
                analysis_status = self.parent().po.delineate_each_arena()
                self.message_when_thread_finished.emit(analysis_status["message"])
        else:
            logging.info("Start automatic video delineation")
            analysis_status = self.parent().po.delineate_each_arena()
            self.message_when_thread_finished.emit(analysis_status["message"])


class SaveManualDelineationThread(QtCore.QThread):

    def __init__(self, parent=None):
        super(SaveManualDelineationThread, self).__init__(parent)
        self.setParent(parent)

    def run(self):
        self.parent().po.left = np.arange(self.parent().po.sample_number)
        self.parent().po.right = np.arange(self.parent().po.sample_number)
        self.parent().po.top = np.arange(self.parent().po.sample_number)
        self.parent().po.bot = np.arange(self.parent().po.sample_number)
        for arena in np.arange(1, self.parent().po.sample_number + 1):
            y, x = np.nonzero(self.parent().imageanalysiswindow.arena_mask == arena)
            self.parent().po.left[arena - 1] = np.min(x)
            self.parent().po.right[arena - 1] = np.max(x)
            self.parent().po.top[arena - 1] = np.min(y)
            self.parent().po.bot[arena - 1] = np.max(y)

        logging.info("Save data to run Cellects quickly")
        self.parent().po.data_to_save['coordinates'] = True
        self.parent().po.save_data_to_run_cellects_quickly()
        self.parent().po.data_to_save['coordinates'] = False

        logging.info("Save manual video delineation")
        self.parent().po.vars['analyzed_individuals'] = np.arange(self.parent().po.sample_number) + 1
        self.parent().po.videos = OneVideoPerBlob(self.parent().po.first_image, self.parent().po.starting_blob_hsize_in_pixels, self.parent().po.all['raw_images'])
        self.parent().po.videos.left = self.parent().po.left
        self.parent().po.videos.right = self.parent().po.right
        self.parent().po.videos.top = self.parent().po.top
        self.parent().po.videos.bot = self.parent().po.bot


class GetExifDataThread(QtCore.QThread):

    def __init__(self, parent=None):
        super(GetExifDataThread, self).__init__(parent)
        self.setParent(parent)

    def run(self):
        self.parent().po.extract_exif()


class FinalizeImageAnalysisThread(QtCore.QThread):

    def __init__(self, parent=None):
        super(FinalizeImageAnalysisThread, self).__init__(parent)
        self.setParent(parent)

    def run(self):
        self.parent().po.get_background_to_subtract()

        self.parent().po.get_origins_and_backgrounds_lists()

        if self.parent().po.last_image is None:
            self.parent().po.get_last_image()
            self.parent().po.fast_image_segmentation(False)
        self.parent().po.find_if_lighter_background()
        logging.info("The current (or the first) folder is ready to run")
        self.parent().po.first_exp_ready_to_run = True
        self.parent().po.data_to_save['coordinates'] = True
        self.parent().po.data_to_save['exif'] = True
        self.parent().po.save_data_to_run_cellects_quickly()
        self.parent().po.data_to_save['coordinates'] = False
        self.parent().po.data_to_save['exif'] = False


class SaveAllVarsThread(QtCore.QThread):

    def __init__(self, parent=None):
        super(SaveAllVarsThread, self).__init__(parent)
        self.setParent(parent)

    def run(self):
        self.parent().po.save_variable_dict()

        #self.parent().po.all['global_pathway']
        #os.getcwd()

        self.set_current_folder()
        self.parent().po.save_data_to_run_cellects_quickly(new_one_if_does_not_exist=False)
        #if os.access(f"", os.R_OK):
        #    self.parent().po.save_data_to_run_cellects_quickly()
        #else:
        #    logging.error(f"No permission access to write in {os.getcwd()}")

    def set_current_folder(self):
        if self.parent().po.all['folder_number'] > 1: # len(self.parent().po.all['folder_list']) > 1:  # len(self.parent().po.all['folder_list']) > 0:
            logging.info(f"Use {self.parent().po.all['folder_list'][0]} folder")
            self.parent().po.update_folder_id(self.parent().po.all['sample_number_per_folder'][0],
                                              self.parent().po.all['folder_list'][0])
        else:
            curr_path = reduce_path_len(self.parent().po.all['global_pathway'], 6, 10)
            logging.info(f"Use {curr_path} folder")
            self.parent().po.update_folder_id(self.parent().po.all['first_folder_sample_number'])


class OneArenaThread(QtCore.QThread):
    message_from_thread_starting = QtCore.Signal(str)
    image_from_thread = QtCore.Signal(dict)
    when_loading_finished = QtCore.Signal(bool)
    when_detection_finished = QtCore.Signal(str)

    def __init__(self, parent=None):
        super(OneArenaThread, self).__init__(parent)
        self.setParent(parent)
        self._isRunning = False

    def run(self):
        continue_analysis = True
        self._isRunning = True
        self.message_from_thread_starting.emit("Video loading, wait...")

        self.set_current_folder()
        print(self.parent().po.vars['convert_for_motion'])
        if not self.parent().po.first_exp_ready_to_run:
            self.parent().po.load_data_to_run_cellects_quickly()
            if not self.parent().po.first_exp_ready_to_run:
                #Need a look for data when Data to run Cellects quickly.pkl and 1 folder selected amon several
                continue_analysis = self.pre_processing()
        if continue_analysis:
            print(self.parent().po.vars['convert_for_motion'])
            memory_diff = self.parent().po.update_available_core_nb()
            if self.parent().po.cores == 0:
                self.message_from_thread_starting.emit(f"Analyzing one arena requires {memory_diff}GB of additional RAM to run")
            else:
                if self.parent().po.motion is None or self.parent().po.load_quick_full == 0:
                    self.load_one_arena()
                if self.parent().po.load_quick_full > 0:
                    if self.parent().po.motion.start is not None:
                        logging.info("One arena detection has started")
                        self.detection()
                        if self.parent().po.load_quick_full > 1:
                            logging.info("One arena post-processing has started")
                            self.post_processing()
                        else:
                            self.when_detection_finished.emit("Detection done, read to see the result")
                    else:
                        self.message_from_thread_starting.emit(f"The current parameters failed to detect the cell(s) motion")

    def stop(self):
        self._isRunning = False

    def set_current_folder(self):
        if self.parent().po.all['folder_number'] > 1:
            logging.info(f"Use {self.parent().po.all['folder_list'][0]} folder")
            self.parent().po.update_folder_id(self.parent().po.all['sample_number_per_folder'][0],
                                              self.parent().po.all['folder_list'][0])
        else:
            curr_path = reduce_path_len(self.parent().po.all['global_pathway'], 6, 10)
            logging.info(f"Use {curr_path} folder")
            self.parent().po.update_folder_id(self.parent().po.all['first_folder_sample_number'])

    def pre_processing(self):
        logging.info("Pre-processing has started")
        analysis_status = {"continue": True, "message": ""}

        self.parent().po.get_first_image()
        self.parent().po.fast_image_segmentation(is_first_image=True)
        if len(self.parent().po.vars['analyzed_individuals']) != self.parent().po.first_image.shape_number:
            self.message_from_thread_starting.emit(f"Wrong specimen number: (re)do the complete analysis.")
            analysis_status["continue"] = False
        else:
            self.parent().po.cropping(is_first_image=True)
            self.parent().po.get_average_pixel_size()
            analysis_status = self.parent().po.delineate_each_arena()
            if not analysis_status["continue"]:
                self.message_from_thread_starting.emit(analysis_status["message"])
                logging.error(analysis_status['message'])
            else:
                self.parent().po.data_to_save['exif'] = True
                self.parent().po.save_data_to_run_cellects_quickly()
                self.parent().po.data_to_save['exif'] = False
                self.parent().po.get_background_to_subtract()
                if len(self.parent().po.vars['analyzed_individuals']) != len(self.parent().po.top):
                    self.message_from_thread_starting.emit(f"Wrong specimen number: (re)do the complete analysis.")
                    analysis_status["continue"] = False
                else:
                    self.parent().po.get_origins_and_backgrounds_lists()
                    self.parent().po.get_last_image()
                    self.parent().po.fast_image_segmentation(False)
                    self.parent().po.find_if_lighter_backgnp.round()
                    logging.info("The current (or the first) folder is ready to run")
                    self.parent().po.first_exp_ready_to_run = True
        return analysis_status["continue"]

    def load_one_arena(self):
        arena = self.parent().po.all['arena']
        i = np.nonzero(self.parent().po.vars['analyzed_individuals'] == arena)[0][0]
        save_loaded_video: bool = False
        if not os.path.isfile(f'ind_{arena}.npy') or self.parent().po.all['overwrite_unaltered_videos']:
            logging.info(f"Starting to load arena n°{arena} from images")
            add_to_c = 1
            self.parent().po.one_arenate_done = True
            i = np.nonzero(self.parent().po.vars['analyzed_individuals'] == arena)[0][0]
            if self.parent().po.vars['lose_accuracy_to_save_memory']:
                self.parent().po.converted_video = np.zeros(
                    (len(self.parent().po.data_list), self.parent().po.bot[i] - self.parent().po.top[i] + add_to_c, self.parent().po.right[i] - self.parent().po.left[i] + add_to_c),
                    dtype=np.uint8)
            else:
                self.parent().po.converted_video = np.zeros(
                    (len(self.parent().po.data_list), self.parent().po.bot[i] - self.parent().po.top[i] + add_to_c, self.parent().po.right[i] - self.parent().po.left[i] + add_to_c),
                    dtype=float)
            if not self.parent().po.vars['already_greyscale']:
                self.parent().po.visu = np.zeros((len(self.parent().po.data_list), self.parent().po.bot[i] - self.parent().po.top[i] + add_to_c,
                                   self.parent().po.right[i] - self.parent().po.left[i] + add_to_c, 3), dtype=np.uint8)
                if self.parent().po.vars['convert_for_motion']['logical'] != 'None':
                    if self.parent().po.vars['lose_accuracy_to_save_memory']:
                        self.parent().po.converted_video2 = np.zeros((len(self.parent().po.data_list), self.parent().po.bot[i] - self.parent().po.top[i] + add_to_c,
                                                       self.parent().po.right[i] - self.parent().po.left[i] + add_to_c), dtype=np.uint8)
                    else:
                        self.parent().po.converted_video2 = np.zeros((len(self.parent().po.data_list), self.parent().po.bot[i] - self.parent().po.top[i] + add_to_c,
                                                       self.parent().po.right[i] - self.parent().po.left[i] + add_to_c), dtype=float)
                first_dict, second_dict, c_spaces = split_dict(self.parent().po.vars['convert_for_motion'])
            prev_img = None
            background = None
            background2 = None
            pat_tracker = PercentAndTimeTracker(self.parent().po.vars['img_number'])
            for image_i, image_name in enumerate(self.parent().po.data_list):
                current_percentage, eta = pat_tracker.get_progress()
                is_landscape = self.parent().po.first_image.image.shape[0] < self.parent().po.first_image.image.shape[1]
                img = read_and_rotate(image_name, prev_img, self.parent().po.all['raw_images'], is_landscape)
                # img = self.parent().po.videos.read_and_rotate(image_name, prev_img)
                prev_img = deepcopy(img)
                if self.parent().po.first_image.cropped:
                    img = img[self.parent().po.first_image.crop_coord[0]:self.parent().po.first_image.crop_coord[1],
                          self.parent().po.first_image.crop_coord[2]:self.parent().po.first_image.crop_coord[3], :]
                img = img[self.parent().po.top[arena - 1]: (self.parent().po.bot[arena - 1] + add_to_c),
                      self.parent().po.left[arena - 1]: (self.parent().po.right[arena - 1] + add_to_c), :]

                self.image_from_thread.emit({"message": f"Video loading: {current_percentage}%{eta}", "current_image": img})
                if self.parent().po.vars['already_greyscale']:
                    if self.parent().po.reduce_image_dim:
                        self.parent().po.converted_video[image_i, ...] = img[:, :, 0]
                    else:
                        self.parent().po.converted_video[image_i, ...] = img
                else:
                    self.parent().po.visu[image_i, ...] = img

                    if self.parent().po.vars['subtract_background']:
                        background = self.parent().po.vars['background_list'][i]
                        if self.parent().po.vars['convert_for_motion']['logical'] != 'None':
                            background2 = self.parent().po.vars['background_list2'][i]
                    greyscale_image, greyscale_image2, all_c_spaces, first_pc_vector = generate_color_space_combination(img, c_spaces,
                                                                                         first_dict,
                                                                                         second_dict,background,background2,
                                                                                         self.parent().po.vars[
                                                                                             'lose_accuracy_to_save_memory'])

                    if self.parent().po.vars['filter_spec'] is not None and self.parent().po.vars['filter_spec']['filter1_type'] != "":
                        greyscale_image = apply_filter(greyscale_image,
                                                       self.parent().po.vars['filter_spec']['filter1_type'],
                                                       self.parent().po.vars['filter_spec']['filter1_param'],
                                                       self.parent().po.vars['lose_accuracy_to_save_memory'])
                        if greyscale_image2 is not None and self.parent().po.vars['filter_spec']['filter2_type'] != "":
                            greyscale_image2 = apply_filter(greyscale_image2,
                                                            self.parent().po.vars['filter_spec']['filter2_type'],
                                                            self.parent().po.vars['filter_spec']['filter2_param'],
                                                            self.parent().po.vars['lose_accuracy_to_save_memory'])
                    self.parent().po.converted_video[image_i, ...] = greyscale_image
                    if self.parent().po.vars['convert_for_motion']['logical'] != 'None':
                        self.parent().po.converted_video2[image_i, ...] = greyscale_image2

            save_loaded_video = True
            if self.parent().po.vars['already_greyscale']:
                self.videos_in_ram = self.parent().po.converted_video
            else:
                if self.parent().po.vars['convert_for_motion']['logical'] == 'None':
                    self.videos_in_ram = [self.parent().po.visu, deepcopy(self.parent().po.converted_video)]
                else:
                    self.videos_in_ram = [self.parent().po.visu, deepcopy(self.parent().po.converted_video), deepcopy(self.parent().po.converted_video2)]

            # videos = [self.parent().po.video.copy(), self.parent().po.converted_video.copy()]
        else:
            logging.info(f"Starting to load arena n°{arena} from .npy saved file")
            self.videos_in_ram = None
        l = [i, arena, self.parent().po.vars, False, False, False, self.videos_in_ram]
        self.parent().po.motion = MotionAnalysis(l)
        r = weakref.ref(self.parent().po.motion)

        if self.videos_in_ram is None:
            self.parent().po.converted_video = deepcopy(self.parent().po.motion.converted_video)
            if self.parent().po.vars['convert_for_motion']['logical'] != 'None':
                self.parent().po.converted_video2 = deepcopy(self.parent().po.motion.converted_video2)
        self.parent().po.motion.get_origin_shape()

        if self.parent().po.motion.dims[0] >= 40:
            step = self.parent().po.motion.dims[0] // 20
        else:
            step = 1
        if self.parent().po.motion.start >= (self.parent().po.motion.dims[0] - step - 1):
            self.parent().po.motion.start = None
        else:
            self.parent().po.motion.get_covering_duration(step)
        self.when_loading_finished.emit(save_loaded_video)

        if self.parent().po.motion.visu is None:
            visu = self.parent().po.motion.converted_video
            visu -= np.min(visu)
            visu = 255 * (visu / np.max(visu))
            visu = np.round(visu).astype(np.uint8)
            if len(visu.shape) == 3:
                visu = np.stack((visu, visu, visu), axis=3)
            self.parent().po.motion.visu = visu

    def detection(self):
        self.message_from_thread_starting.emit(f"Quick video segmentation")
        self.parent().po.motion.converted_video = deepcopy(self.parent().po.converted_video)
        if self.parent().po.vars['convert_for_motion']['logical'] != 'None':
            self.parent().po.motion.converted_video2 = deepcopy(self.parent().po.converted_video2)
        # self.parent().po.motion.detection(compute_all_possibilities=True)
        self.parent().po.motion.detection(compute_all_possibilities=self.parent().po.all['compute_all_options'])
        if self.parent().po.all['compute_all_options']:
            self.parent().po.computed_video_options = np.ones(5, bool)
        else:
            self.parent().po.computed_video_options = np.zeros(5, bool)
            self.parent().po.computed_video_options[self.parent().po.all['video_option']] = True
        # if self.parent().po.vars['color_number'] > 2:

    def post_processing(self):
        self.parent().po.motion.smoothed_video = None
        # if self.parent().po.vars['already_greyscale']:
        #     if self.parent().po.vars['convert_for_motion']['logical'] == 'None':
        #         self.videos_in_ram = self.parent().po.converted_video
        #     else:
        #         self.videos_in_ram = self.parent().po.converted_video, self.parent().po.converted_video2
        # else:
        #     if self.parent().po.vars['convert_for_motion']['logical'] == 'None':
        #         videos_in_ram = self.parent().po.visu, self.parent().po.converted_video
        #     else:
        #         videos_in_ram = self.parent().po.visu, self.parent().po.converted_video, \
        #                         self.parent().po.converted_video2

        if self.parent().po.vars['color_number'] > 2:
            analyses_to_compute = [0]
        else:
            if self.parent().po.all['compute_all_options']:
                analyses_to_compute = np.arange(5)
            else:
                logging.info(f"option: {self.parent().po.all['video_option']}")
                analyses_to_compute = [self.parent().po.all['video_option']]
        time_parameters = [self.parent().po.motion.start, self.parent().po.motion.step,
                           self.parent().po.motion.lost_frames, self.parent().po.motion.substantial_growth]

        args = [self.parent().po.all['arena'] - 1, self.parent().po.all['arena'], self.parent().po.vars,
                False, False, False, self.videos_in_ram]
        if self.parent().po.vars['do_fading']:
            self.parent().po.newly_explored_area = np.zeros((self.parent().po.motion.dims[0], 5), np.unp.int64)
        for seg_i in analyses_to_compute:
            analysis_i = MotionAnalysis(args)
            r = weakref.ref(analysis_i)
            analysis_i.segmentation = np.zeros(analysis_i.converted_video.shape[:3], dtype=np.uint8)
            if self.parent().po.all['compute_all_options']:
                if seg_i == 0:
                    analysis_i.segmentation = self.parent().po.motion.segmentation
                else:
                    if seg_i == 1:
                        mask = self.parent().po.motion.luminosity_segmentation
                    elif seg_i == 2:
                        mask = self.parent().po.motion.gradient_segmentation
                    elif seg_i == 3:
                        mask = self.parent().po.motion.logical_and
                    elif seg_i == 4:
                        mask = self.parent().po.motion.logical_or
                    analysis_i.segmentation[mask[0], mask[1], mask[2]] = 1
            else:
                if self.parent().po.computed_video_options[self.parent().po.all['video_option']]:
                    analysis_i.segmentation = self.parent().po.motion.segmentation

            analysis_i.start = time_parameters[0]
            analysis_i.step = time_parameters[1]
            analysis_i.lost_frames = time_parameters[2]
            analysis_i.substantial_growth = time_parameters[3]
            analysis_i.origin_idx = self.parent().po.motion.origin_idx
            analysis_i.initialize_post_processing()
            analysis_i.t = analysis_i.start
            # print_progress = ForLoopCounter(self.start)

            while self._isRunning and analysis_i.t < analysis_i.binary.shape[0]:
                # analysis_i.update_shape(True)
                analysis_i.update_shape(False)
                contours = np.nonzero(
                    cv2.morphologyEx(analysis_i.binary[analysis_i.t - 1, :, :], cv2.MORPH_GRADIENT, cross_33))
                current_image = deepcopy(self.parent().po.motion.visu[analysis_i.t - 1, :, :, :])
                current_image[contours[0], contours[1], :] = self.parent().po.vars['contour_color']
                self.image_from_thread.emit(
                    {"message": f"Tracking option n°{seg_i + 1}. Image number: {analysis_i.t - 1}",
                     "current_image": current_image})
            if analysis_i.start is None:
                analysis_i.binary = np.repeat(np.expand_dims(analysis_i.origin, 0),
                                           analysis_i.converted_video.shape[0], axis=0)
                if self.parent().po.vars['color_number'] > 2:
                    self.message_from_thread_starting.emit(
                        f"Failed to detect motion. Redo image analysis (with only 2 colors?)")
                else:
                    self.message_from_thread_starting.emit(f"Tracking option n°{seg_i + 1} failed to detect motion")

            if self.parent().po.all['compute_all_options']:
                if seg_i == 0:
                    self.parent().po.motion.segmentation = analysis_i.binary
                elif seg_i == 1:
                    self.parent().po.motion.luminosity_segmentation = np.nonzero(analysis_i.binary)
                elif seg_i == 2:
                    self.parent().po.motion.gradient_segmentation = np.nonzero(analysis_i.binary)
                elif seg_i == 3:
                    self.parent().po.motion.logical_and = np.nonzero(analysis_i.binary)
                elif seg_i == 4:
                    self.parent().po.motion.logical_or = np.nonzero(analysis_i.binary)
            else:
                self.parent().po.motion.segmentation = analysis_i.binary

        # self.message_from_thread_starting.emit("If there are problems, change some parameters and try again")
        self.when_detection_finished.emit("Post processing done, read to see the result")



class VideoReaderThread(QtCore.QThread):
    message_from_thread = QtCore.Signal(dict)

    def __init__(self, parent=None):
        super(VideoReaderThread, self).__init__(parent)
        self.setParent(parent)

    def run(self):
        video_analysis = deepcopy(self.parent().po.motion.visu)
        self.message_from_thread.emit(
            {"current_image": video_analysis[0, ...], "message": f"Video preparation, wait..."})
        if self.parent().po.load_quick_full > 0:

            if self.parent().po.all['compute_all_options']:
                if self.parent().po.all['video_option'] == 0:
                    video_mask = self.parent().po.motion.segmentation
                else:
                    if self.parent().po.all['video_option'] == 1:
                        mask = self.parent().po.motion.luminosity_segmentation
                    elif self.parent().po.all['video_option'] == 2:
                        mask = self.parent().po.motion.gradient_segmentation
                    elif self.parent().po.all['video_option'] == 3:
                        mask = self.parent().po.motion.logical_and
                    elif self.parent().po.all['video_option'] == 4:
                        mask = self.parent().po.motion.logical_or
                    video_mask = np.zeros(self.parent().po.motion.dims[:3], dtype=np.uint8)
                    video_mask[mask[0], mask[1], mask[2]] = 1
            else:
                video_mask = np.zeros(self.parent().po.motion.dims[:3], dtype=np.uint8)
                if self.parent().po.computed_video_options[self.parent().po.all['video_option']]:
                    video_mask = self.parent().po.motion.segmentation

            if self.parent().po.load_quick_full == 1:
                video_mask = np.cumsum(video_mask.astype(np.uint32), axis=0)
                video_mask[video_mask > 0] = 1
                video_mask = video_mask.astype(np.uint8)
        logging.info(f"sum: {video_mask.sum()}")
        # timings = genfromtxt("timings.csv")
        for t in np.arange(self.parent().po.motion.dims[0]):
            mask = cv2.morphologyEx(video_mask[t, ...], cv2.MORPH_GRADIENT, cross_33)
            mask = np.stack((mask, mask, mask), axis=2)
            # current_image[current_image > 0] = self.parent().po.vars['contour_color']
            current_image = deepcopy(video_analysis[t, ...])
            current_image[mask > 0] = self.parent().po.vars['contour_color']
            self.message_from_thread.emit(
                {"current_image": current_image, "message": f"Reading in progress... Image number: {t}"}) #, "time": timings[t]
            time.sleep(1 / 50)
        self.message_from_thread.emit({"current_image": current_image, "message": ""})#, "time": timings[t]


class ChangeOneRepResultThread(QtCore.QThread):
    message_from_thread = QtCore.Signal(str)

    def __init__(self, parent=None):
        super(ChangeOneRepResultThread, self).__init__(parent)
        self.setParent(parent)

    def run(self):
        self.message_from_thread.emit(
            f"Arena n°{self.parent().po.all['arena']}: modifying its results...")
        # self.parent().po.motion2 = deepcopy(self.parent().po.motion)
        if self.parent().po.motion.start is None:
            self.parent().po.motion.binary = np.repeat(np.expand_dims(self.parent().po.motion.origin, 0),
                                                     self.parent().po.motion.converted_video.shape[0], axis=0).astype(np.uint8)
        else:
            if self.parent().po.all['compute_all_options']:
                if self.parent().po.all['video_option'] == 0:
                    self.parent().po.motion.binary = self.parent().po.motion.segmentation
                else:
                    if self.parent().po.all['video_option'] == 1:
                        mask = self.parent().po.motion.luminosity_segmentation
                    elif self.parent().po.all['video_option'] == 2:
                        mask = self.parent().po.motion.gradient_segmentation
                    elif self.parent().po.all['video_option'] == 3:
                        mask = self.parent().po.motion.logical_and
                    elif self.parent().po.all['video_option'] == 4:
                        mask = self.parent().po.motion.logical_or
                    self.parent().po.motion.binary = np.zeros(self.parent().po.motion.dims, dtype=np.uint8)
                    self.parent().po.motion.binary[mask[0], mask[1], mask[2]] = 1
            else:
                self.parent().po.motion.binary = np.zeros(self.parent().po.motion.dims[:3], dtype=np.uint8)
                if self.parent().po.computed_video_options[self.parent().po.all['video_option']]:
                    self.parent().po.motion.binary = self.parent().po.motion.segmentation

        if self.parent().po.vars['do_fading']:
            self.parent().po.motion.newly_explored_area = self.parent().po.newly_explored_area[:, self.parent().po.all['video_option']]
        self.parent().po.motion.max_distance = 9 * self.parent().po.vars['detection_range_factor']
        self.parent().po.motion.get_descriptors_from_binary(release_memory=False)
        self.parent().po.motion.detect_growth_transitions()
        self.parent().po.motion.networks_detection(False)
        self.parent().po.motion.study_cytoscillations(False)
        self.parent().po.motion.fractal_descriptions()
        self.parent().po.motion.change_results_of_one_arena()
        self.parent().po.motion = None
        # self.parent().po.motion = None
        self.message_from_thread.emit("")


class WriteVideoThread(QtCore.QThread):
    # message_from_thread_in_thread = QtCore.Signal(bool)
    def __init__(self, parent=None):
        super(WriteVideoThread, self).__init__(parent)
        self.setParent(parent)

    def run(self):
        # self.message_from_thread_in_thread.emit({True})
        arena = self.parent().po.all['arena']
        if not self.parent().po.vars['already_greyscale']:
            write_video(self.parent().po.visu, f'ind_{arena}.npy')
        else:
            write_video(self.parent().po.converted_video, f'ind_{arena}.npy')


class RunAllThread(QtCore.QThread):
    message_from_thread = QtCore.Signal(str)
    image_from_thread = QtCore.Signal(dict)

    def __init__(self, parent=None):
        super(RunAllThread, self).__init__(parent)
        self.setParent(parent)

    def run(self):
        analysis_status = {"continue": True, "message": ""}
        message = self.set_current_folder(0)

        if self.parent().po.first_exp_ready_to_run:

            self.message_from_thread.emit(message + ": Write videos...")
            if not self.parent().po.vars['several_blob_per_arena'] and self.parent().po.sample_number != len(self.parent().po.bot):
                analysis_status["continue"] = False
                analysis_status["message"] = f"Wrong specimen number: redo the first image analysis."
                self.message_from_thread.emit(f"Wrong specimen number: restart Cellects and do another analysis.")
            else:
                analysis_status = self.run_video_writing(message)
                if analysis_status["continue"]:
                    self.message_from_thread.emit(message + ": Analyse all videos...")
                    analysis_status = self.run_motion_analysis(message)
                if analysis_status["continue"]:
                    if self.parent().po.all['folder_number'] > 1:
                        self.parent().po.all['folder_list'] = self.parent().po.all['folder_list'][1:]
                        self.parent().po.all['sample_number_per_folder'] = self.parent().po.all['sample_number_per_folder'][1:]
        else:
            self.parent().po.look_for_data()

        if analysis_status["continue"] and (not self.parent().po.first_exp_ready_to_run or self.parent().po.all['folder_number'] > 1):
            folder_number = np.max((len(self.parent().po.all['folder_list']), 1))

            for exp_i in np.arange(folder_number):
                if len(self.parent().po.all['folder_list']) > 0:
                    logging.info(self.parent().po.all['folder_list'][exp_i])
                self.parent().po.first_im = None
                self.parent().po.first_image = None
                self.parent().po.last_im = None
                self.parent().po.last_image = None
                self.parent().po.videos = None
                self.parent().po.top = None

                message = self.set_current_folder(exp_i)
                self.message_from_thread.emit(f'{message}, pre-processing...')
                self.parent().po.load_data_to_run_cellects_quickly()
                if not self.parent().po.first_exp_ready_to_run:
                    analysis_status = self.pre_processing()
                if analysis_status["continue"]:
                    self.message_from_thread.emit(message + ": Write videos from images before analysis...")
                    if not self.parent().po.vars['several_blob_per_arena'] and self.parent().po.sample_number != len(self.parent().po.bot):
                        self.message_from_thread.emit(f"Wrong specimen number: first image analysis is mandatory.")
                        analysis_status["continue"] = False
                        analysis_status["message"] = f"Wrong specimen number: first image analysis is mandatory."
                    else:
                        analysis_status = self.run_video_writing(message)
                        if analysis_status["continue"]:
                            self.message_from_thread.emit(message + ": Starting analysis...")
                            analysis_status = self.run_motion_analysis(message)

                if not analysis_status["continue"]:
                    # self.message_from_thread.emit(analysis_status["message"])
                    break
                # if not continue_analysis:
                #     self.message_from_thread.emit(f"Error: wrong folder or parameters")
                #     break
                # if not enough_memory:
                #     self.message_from_thread.emit(f"Error: not enough memory")
                #     break
                print(self.parent().po.vars['convert_for_motion'])
        if analysis_status["continue"]:
            if self.parent().po.all['folder_number'] > 1:
                self.message_from_thread.emit(f"Exp {self.parent().po.all['folder_list'][0]} to {self.parent().po.all['folder_list'][-1]} analyzed.")
            else:
                curr_path = reduce_path_len(self.parent().po.all['global_pathway'], 6, 10)
                self.message_from_thread.emit(f'Exp {curr_path}, analyzed.')
        else:
            logging.error(message + " " + analysis_status["message"])
            self.message_from_thread.emit(message + " " + analysis_status["message"])

    def set_current_folder(self, exp_i):
        if self.parent().po.all['folder_number'] > 1:
            logging.info(f"Use {self.parent().po.all['folder_list'][exp_i]} folder")

            message = f"{str(self.parent().po.all['global_pathway'])[:6]} ... {self.parent().po.all['folder_list'][exp_i]}"
            self.parent().po.update_folder_id(self.parent().po.all['sample_number_per_folder'][exp_i],
                                              self.parent().po.all['folder_list'][exp_i])
        else:
            message = reduce_path_len(self.parent().po.all['global_pathway'], 6, 10)
            logging.info(f"Use {message} folder")
            self.parent().po.update_folder_id(self.parent().po.all['first_folder_sample_number'])
        return message

    def pre_processing(self):
        analysis_status = {"continue": True, "message": ""}
        logging.info("Pre-processing has started")
        if len(self.parent().po.data_list) > 0:
            self.parent().po.get_first_image()
            self.parent().po.fast_image_segmentation(True)
            self.parent().po.cropping(is_first_image=True)
            self.parent().po.get_average_pixel_size()
            try:
                analysis_status = self.parent().po.delineate_each_arena()
            except ValueError:
                analysis_status[
                    "message"] = f"Failed to detect the right cell(s) number: the first image analysis is mandatory."
                analysis_status["continue"] = False

            if analysis_status["continue"]:
                self.parent().po.data_to_save['exif'] = True
                self.parent().po.save_data_to_run_cellects_quickly()
                self.parent().po.data_to_save['exif'] = False
                # self.parent().po.extract_exif()
                self.parent().po.get_background_to_subtract()
                if len(self.parent().po.vars['analyzed_individuals']) != len(self.parent().po.top):
                    analysis_status["message"] = f"Failed to detect the right cell(s) number: the first image analysis is mandatory."
                    analysis_status["continue"] = False
                elif self.parent().po.top is None and self.parent().imageanalysiswindow.manual_delineation_flag:
                    analysis_status["message"] = f"Auto video delineation failed, use manual delineation tool"
                    analysis_status["continue"] = False
                else:
                    self.parent().po.get_origins_and_backgrounds_lists()
                    self.parent().po.get_last_image()
                    self.parent().po.fast_image_segmentation(is_first_image=False)
                    self.parent().po.find_if_lighter_backgnp.round()
            return analysis_status
        else:
            analysis_status["message"] = f"Wrong folder or parameters"
            analysis_status["continue"] = False
            return analysis_status

    def run_video_writing(self, message):
        analysis_status = {"continue": True, "message": ""}
        look_for_existing_videos = glob('ind_' + '*' + '.npy')
        there_already_are_videos = len(look_for_existing_videos) == len(self.parent().po.vars['analyzed_individuals'])
        logging.info(f"{len(look_for_existing_videos)} .npy video files found for {len(self.parent().po.vars['analyzed_individuals'])} arenas to analyze")
        do_write_videos = not there_already_are_videos or (
                there_already_are_videos and self.parent().po.all['overwrite_unaltered_videos'])
        if do_write_videos:
            logging.info(f"Starting video writing")
            # self.videos.write_videos_as_np_arrays(self.data_list, self.vars['convert_for_motion'], in_colors=self.vars['save_in_colors'])
            in_colors = not self.parent().po.vars['already_greyscale']
            self.parent().po.videos = OneVideoPerBlob(self.parent().po.first_image,
                                                      self.parent().po.starting_blob_hsize_in_pixels,
                                                      self.parent().po.all['raw_images'])
            self.parent().po.videos.left = self.parent().po.left
            self.parent().po.videos.right = self.parent().po.right
            self.parent().po.videos.top = self.parent().po.top
            self.parent().po.videos.bot = self.parent().po.bot
            self.parent().po.videos.first_image.shape_number = self.parent().po.sample_number
            bunch_nb, video_nb_per_bunch, sizes, video_bunch, vid_names, rom_memory_required, analysis_status, remaining = self.parent().po.videos.prepare_video_writing(
                self.parent().po.data_list, self.parent().po.vars['min_ram_free'], in_colors)
            if analysis_status["continue"]:
                # Check that there is enough available RAM for one video par bunch and ROM for all videos
                if video_nb_per_bunch > 0 and rom_memory_required is None:
                    pat_tracker1 = PercentAndTimeTracker(bunch_nb * self.parent().po.vars['img_number'])
                    pat_tracker2 = PercentAndTimeTracker(len(self.parent().po.vars['analyzed_individuals']))
                    arena_percentage = 0
                    is_landscape = self.parent().po.first_image.image.shape[0] < self.parent().po.first_image.image.shape[1]
                    for bunch in np.arange(bunch_nb):
                        # Update the labels of arenas and the video_bunch to write
                        if bunch == (bunch_nb - 1) and remaining > 0:
                            arena = np.arange(bunch * video_nb_per_bunch, bunch * video_nb_per_bunch + remaining)
                        else:
                            arena = np.arange(bunch * video_nb_per_bunch, (bunch + 1) * video_nb_per_bunch)
                        if self.parent().po.videos.use_list_of_vid:
                            video_bunch = [np.zeros(sizes[i, :], dtype=np.uint8) for i in arena]
                        else:
                            video_bunch = np.zeros(np.append(sizes[0, :], len(arena)), dtype=np.uint8)
                        prev_img = None
                        images_done = bunch * self.parent().po.vars['img_number']
                        for image_i, image_name in enumerate(self.parent().po.data_list):
                            image_percentage, remaining_time = pat_tracker1.get_progress(image_i + images_done)
                            self.message_from_thread.emit(message + f" Step 1/2: Video writing ({np.round((image_percentage + arena_percentage) / 2, 2)}%)")
                            if not os.path.exists(image_name):
                                raise FileNotFoundError(image_name)
                            img = read_and_rotate(image_name, prev_img, self.parent().po.all['raw_images'], is_landscape, self.parent().po.first_image.crop_coord)
                            prev_img = deepcopy(img)
                            if self.parent().po.vars['already_greyscale'] and self.parent().po.reduce_image_dim:
                                img = img[:, :, 0]

                            for arena_i, arena_name in enumerate(arena):
                                try:
                                    sub_img = img[self.parent().po.top[arena_name]: (self.parent().po.bot[arena_name] + 1),
                                              self.parent().po.left[arena_name]: (self.parent().po.right[arena_name] + 1), ...]
                                    if self.parent().po.videos.use_list_of_vid:
                                        video_bunch[arena_i][image_i, ...] = sub_img
                                    else:
                                        if len(video_bunch.shape) == 5:
                                            video_bunch[image_i, :, :, :, arena_i] = sub_img
                                        else:
                                            video_bunch[image_i, :, :, arena_i] = sub_img
                                except ValueError:
                                    analysis_status["message"] = f"One (or more) image has a different size (restart)"
                                    analysis_status["continue"] = False
                                    logging.info(f"In the {message} folder: one (or more) image has a different size (restart)")
                                    break
                            if not analysis_status["continue"]:
                                break
                        if not analysis_status["continue"]:
                            break
                        if analysis_status["continue"]:
                            for arena_i, arena_name in enumerate(arena):
                                try:
                                    arena_percentage, eta = pat_tracker2.get_progress()
                                    self.message_from_thread.emit(message + f" Step 1/2: Video writing ({np.round((image_percentage + arena_percentage) / 2, 2)}%)")# , ETA {remaining_time}
                                    if self.parent().po.videos.use_list_of_vid:
                                        np.save(vid_names[arena_name], video_bunch[arena_i])
                                    else:
                                        if len(video_bunch.shape) == 5:
                                            np.save(vid_names[arena_name], video_bunch[:, :, :, :, arena_i])
                                        else:
                                            np.save(vid_names[arena_name], video_bunch[:, :, :, arena_i])
                                except OSError:
                                    self.message_from_thread.emit(message + f"full disk memory, clear space and retry")
                        logging.info(f"Bunch n°{bunch + 1} over {bunch_nb} saved.")
                    logging.info("When they exist, do not overwrite unaltered video")
                    self.parent().po.all['overwrite_unaltered_videos'] = False
                    self.parent().po.save_variable_dict()
                    self.parent().po.save_data_to_run_cellects_quickly()
                    analysis_status["message"] = f"Video writing complete."
                    if self.parent().po.videos is not None:
                        del self.parent().po.videos
                    return analysis_status
                else:
                    analysis_status["continue"] = False
                    if video_nb_per_bunch == 0:
                        memory_diff = self.parent().po.update_available_core_nb()
                        ram_message = f"{memory_diff}GB of additional RAM"
                    if rom_memory_required is not None:
                        rom_message = f"at least {rom_memory_required}GB of free ROM"

                    if video_nb_per_bunch == 0 and rom_memory_required is not None:
                        analysis_status["message"] = f"Requires {ram_message} and {rom_message} to run"
                        # self.message_from_thread.emit(f"Analyzing {message} requires {ram_message} and {rom_message} to run")
                    elif video_nb_per_bunch == 0:
                        analysis_status["message"] = f"Requires {ram_message} to run"
                        # self.message_from_thread.emit(f"Analyzing {message} requires {ram_message} to run")
                    elif rom_memory_required is not None:
                        analysis_status["message"] = f"Requires {rom_message} to run"
                        # self.message_from_thread.emit(f"Analyzing {message} requires {rom_message} to run")
                    logging.info(f"Cellects is not writing videos: insufficient memory")
                    return analysis_status
            else:
                return analysis_status


        else:
            logging.info(f"Cellects is not writing videos: unnecessary")
            analysis_status["message"] = f"Cellects is not writing videos: unnecessary"
            return analysis_status

    def run_motion_analysis(self, message):
        analysis_status = {"continue": True, "message": ""}
        logging.info(f"Starting motion analysis with the detection method n°{self.parent().po.all['video_option']}")
        self.parent().po.instantiate_tables()
        try:
            memory_diff = self.parent().po.update_available_core_nb()
            if self.parent().po.cores > 0: # i.e. enough memory
                if not self.parent().po.all['do_multiprocessing'] or self.parent().po.cores == 1:
                    self.message_from_thread.emit(f"{message} Step 2/2: Video analysis")
                    logging.info("fStarting sequential analysis")
                    tiii = default_timer()
                    pat_tracker = PercentAndTimeTracker(len(self.parent().po.vars['analyzed_individuals']))
                    for i, arena in enumerate(self.parent().po.vars['analyzed_individuals']):

                        l = [i, arena, self.parent().po.vars, True, True, False, None]
                        # l = [0, 1, self.parent().po.vars, True, False, False, None]
                        analysis_i = MotionAnalysis(l)
                        r = weakref.ref(analysis_i)
                        if not self.parent().po.vars['several_blob_per_arena']:
                            # Save basic statistics
                            self.parent().po.update_one_row_per_arena(i, analysis_i.one_descriptor_per_arena)


                            # Save descriptors in long_format
                            self.parent().po.update_one_row_per_frame(i * self.parent().po.vars['img_number'], arena * self.parent().po.vars['img_number'], analysis_i.one_row_per_frame)
                            
                            # Save cytosol_oscillations
                        if not pd.isna(analysis_i.one_descriptor_per_arena["first_move"]):
                            if self.parent().po.vars['oscilacyto_analysis']:
                                oscil_i = pd.DataFrame(
                                    np.c_[np.repeat(arena,
                                              analysis_i.clusters_final_data.shape[0]), analysis_i.clusters_final_data],
                                    columns=['arena', 'mean_pixel_period', 'phase', 'cluster_size', 'edge_distance', 'coord_y', 'coord_x'])
                                if self.parent().po.one_row_per_oscillating_cluster is None:
                                    self.parent().po.one_row_per_oscillating_cluster = oscil_i
                                else:
                                    self.parent().po.one_row_per_oscillating_cluster = pd.concat((self.parent().po.one_row_per_oscillating_cluster, oscil_i))
                                
                        # Save efficiency visualization
                        self.parent().po.add_analysis_visualization_to_first_and_last_images(i, analysis_i.efficiency_test_1,
                                                                                 analysis_i.efficiency_test_2)
                        # Emit message to the interface
                        current_percentage, eta = pat_tracker.get_progress()
                        self.image_from_thread.emit({"current_image": self.parent().po.last_image.bgr,
                                                     "message": f"{message} Step 2/2: analyzed {arena} out of {len(self.parent().po.vars['analyzed_individuals'])} arenas ({current_percentage}%){eta}"})
                        del analysis_i
                    logging.info(f"Sequential analysis lasted {(default_timer() - tiii)/ 60} minutes")
                else:
                    self.message_from_thread.emit(
                        f"{message}, Step 2/2:  Analyse all videos using {self.parent().po.cores} cores...")

                    logging.info("fStarting analysis in parallel")

                    # new
                    tiii = default_timer()
                    arena_number = len(self.parent().po.vars['analyzed_individuals'])
                    self.advance = 0
                    self.pat_tracker = PercentAndTimeTracker(len(self.parent().po.vars['analyzed_individuals']),
                                                        core_number=self.parent().po.cores)

                    fair_core_workload = arena_number // self.parent().po.cores
                    cores_with_1_more = arena_number % self.parent().po.cores
                    EXTENTS_OF_SUBRANGES = []
                    bound = 0
                    parallel_organization = [fair_core_workload + 1 for _ in range(cores_with_1_more)] + [fair_core_workload for _ in range(self.parent().po.cores - cores_with_1_more)]
                    # Emit message to the interface
                    self.image_from_thread.emit({"current_image": self.parent().po.last_image.bgr,
                                                 "message": f"{message} Step 2/2: Analysis running on {self.parent().po.cores} CPU cores"})
                    for i, extent_size in enumerate(parallel_organization):
                        EXTENTS_OF_SUBRANGES.append((bound, bound := bound + extent_size))

                    try:
                        PROCESSES = []
                        subtotals = Manager().Queue()# Queue()
                        for extent in EXTENTS_OF_SUBRANGES:
                            # print(extent)
                            p = Process(target=motion_analysis_process, args=(extent[0], extent[1], self.parent().po.vars, subtotals))
                            p.start()
                            PROCESSES.append(p)

                        for p in PROCESSES:
                            p.join()

                        self.message_from_thread.emit(f"{message}, Step 2/2:  Saving all results...")
                        for i in range(subtotals.qsize()):
                            grouped_results = subtotals.get()
                            for j, results_i in enumerate(grouped_results):
                                if not self.parent().po.vars['several_blob_per_arena']:
                                    # Save basic statistics
                                    self.parent().po.update_one_row_per_arena(results_i['i'], results_i['one_row_per_arena'])
                                    # Save descriptors in long_format
                                    self.parent().po.update_one_row_per_frame(results_i['i'] * self.parent().po.vars['img_number'],
                                                                              results_i['arena'] * self.parent().po.vars['img_number'],
                                                                              results_i['one_row_per_frame'])
                                if not pd.isna(results_i['first_move']):
                                    # Save cytosol_oscillations
                                    if self.parent().po.vars['oscilacyto_analysis']:
                                        if self.parent().po.one_row_per_oscillating_cluster is None:
                                            self.parent().po.one_row_per_oscillating_cluster = results_i['one_row_per_oscillating_cluster']
                                        else:
                                            self.parent().po.one_row_per_oscillating_cluster = pd.concat((self.parent().po.one_row_per_oscillating_cluster, results_i['one_row_per_oscillating_cluster']))
                                        
                                # Save efficiency visualization
                                self.parent().po.add_analysis_visualization_to_first_and_last_images(results_i['i'], results_i['efficiency_test_1'],
                                                                                         results_i['efficiency_test_2'])
                        self.image_from_thread.emit(
                            {"current_image": self.parent().po.last_image.bgr,
                             "message": f"{message} Step 2/2: analyzed {len(self.parent().po.vars['analyzed_individuals'])} out of {len(self.parent().po.vars['analyzed_individuals'])} arenas ({100}%)"})

                        logging.info(f"Parallel analysis lasted {(default_timer() - tiii)/ 60} minutes")
                    except MemoryError:
                        analysis_status["continue"] = False
                        analysis_status["message"] = f"Not enough memory, reduce the core number for parallel analysis"
                        self.message_from_thread.emit(f"Analyzing {message} requires to reduce the core number for parallel analysis")
                        return analysis_status
                self.parent().po.save_tables()
                return analysis_status
            else:
                analysis_status["continue"] = False
                analysis_status["message"] = f"Requires an additional {memory_diff}GB of RAM to run"
                self.message_from_thread.emit(f"Analyzing {message} requires an additional {memory_diff}GB of RAM to run")
                return analysis_status
        except MemoryError:
            analysis_status["continue"] = False
            analysis_status["message"] = f"Requires additional memory to run"
            self.message_from_thread.emit(f"Analyzing {message} requires additional memory to run")
            return analysis_status


def motion_analysis_process(lower_bound: int, upper_bound: int, vars: dict, subtotals: Queue) -> None:
    grouped_results = []
    for i in range(lower_bound, upper_bound):
        analysis_i = MotionAnalysis([i, i + 1, vars, True, True, False, None])
        r = weakref.ref(analysis_i)
        results_i = dict()
        results_i['arena'] = analysis_i.one_descriptor_per_arena['arena']
        results_i['i'] = analysis_i.one_descriptor_per_arena['arena'] - 1
        arena = results_i['arena']
        i = arena - 1
        if not vars['several_blob_per_arena']:
            # Save basic statistics
            results_i['one_row_per_arena'] = analysis_i.one_descriptor_per_arena
            # Save descriptors in long_format
            results_i['one_row_per_frame'] = analysis_i.one_row_per_frame
            # Save cytosol_oscillations

        results_i['first_move'] = analysis_i.one_descriptor_per_arena["first_move"]
        if not pd.isna(analysis_i.one_descriptor_per_arena["first_move"]):
            if vars['oscilacyto_analysis']:
                results_i['clusters_final_data'] = analysis_i.clusters_final_data
                results_i['one_row_per_oscillating_cluster'] = pd.DataFrame(
                    np.c_[np.repeat(arena, analysis_i.clusters_final_data.shape[0]), analysis_i.clusters_final_data],
                    columns=['arena', 'mean_pixel_period', 'phase', 'cluster_size', 'edge_distance', 'coord_y', 'coord_x'])
            if vars['fractal_analysis']:
                results_i['fractal_box_sizes'] = pd.DataFrame(analysis_i.fractal_boxes,
                               columns=['arena', 'time', 'fractal_box_lengths', 'fractal_box_widths'])

        # Save efficiency visualization
        results_i['efficiency_test_1'] = analysis_i.efficiency_test_1
        results_i['efficiency_test_2'] = analysis_i.efficiency_test_2
        grouped_results.append(results_i)

    subtotals.put(grouped_results)