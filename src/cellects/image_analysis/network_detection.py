#!/usr/bin/env python3
"""
This script contains the class for detecting networks out of a grayscale image of Physarum polycephalum
"""

# A completely different strategy could be to segment the network by layers of luminosity.
# The first layer captures the brightest veins and replace their pixels by background pixels.
# The second layer captures other veins, (make sure that they are connected to the first?) and replace their pixels too.
# During one layer segmentation, the algorithm make sure that all detected veins are as long as possible
# but less long than and connected to the previous.

import random
from copy import deepcopy
from scipy import ndimage
import cv2
import numpy as np
from cellects.image_analysis.image_segmentation import generate_color_space_combination, otsu_thresholding
from cellects.image_analysis.morphological_operations import make_gravity_field, cross_33, cc, CompareNeighborsWithValue, get_rolling_window_coordinates_list
from cellects.image_analysis.shape_descriptors import ShapeDescriptors
from cellects.utils.load_display_save import See
from cellects.utils.formulas import max_cum_sum_from_rolling_window
import os
from numba.typed import Dict as TDict
from cellects.utils.formulas import bracket_to_uint8_image_contrast


def get_vertices_from_skeleton(skeleton):
    """
    :return:
    """
    im_shape = skeleton.shape
    cnv = CompareNeighborsWithValue(skeleton, 8)
    cnv.is_equal(1, and_itself=True)

    # All pixels having only one neighbor, and containing the value 1, is a termination for sure
    sure_terminations = np.zeros(im_shape, dtype=np.uint8)
    sure_terminations[cnv.equal_neighbor_nb == 1] = 1

    # Create a kernel to dilate properly the known vertices.
    square_33 = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=np.uint8)
    # Initiate the vertices final matrix as a copy of the sure_terminations
    vertices = deepcopy(sure_terminations)
    # All pixels that have neighbor_nb neighbors, none of which is already detected as a vertex.
    for i, neighbor_nb in enumerate([8, 7, 6, 5, 4, 3]):
        # All pixels having neighbor_nb neighbor are potential vertices
        potential_vertices = np.zeros(im_shape, dtype=np.uint8)
        potential_vertices[cnv.equal_neighbor_nb == neighbor_nb] = 1
        # remove the false intersections that are a neighbor of a previously detected intersection
        # Dilate vertices to make sure that no neighbors of the current potential vertices are already vertices.
        dilated_previous_intersections = cv2.dilate(vertices, square_33)
        potential_vertices *= (1 - dilated_previous_intersections)
        vertices[np.nonzero(potential_vertices)] = 1

    labeled_vertices, num_labels = ndimage.label(vertices, structure=np.ones((3, 3), dtype=np.uint8))
    vertices_positions = ndimage.center_of_mass(vertices, labeled_vertices, range(1, num_labels + 1))
    vertices_positions = np.round(np.asarray(vertices_positions), 0).astype(np.uint64)
    vertices_positions = np.column_stack((vertices_positions, np.arange(1, num_labels + 1, dtype=np.uint64)))
    vertices_number = (labeled_vertices > 0).sum()
    return vertices_positions, vertices_number


def visualize_vertices(skeleton, vertices_coord):
    # skeleton = test_skel; vertices_coord = vertices_positions
    img = np.stack((skeleton, skeleton, skeleton), axis=2, dtype=np.uint8)
    img *= 255
    img[vertices_coord[:, 0], vertices_coord[:, 1], :2] = 0
    image = cv2.resize(img, (1000, 1000))
    cv2.imshow("Vertices", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def get_segments_from_vertices_skeleton(skeleton, vertices_coord):
    # skeleton = test_skel; vertices_coord = vertices_positions
    im_shape = skeleton.shape
    vertices = np.zeros(im_shape, dtype=np.uint8)
    vertices[vertices_coord[:, 0], vertices_coord[:, 1]] = 1
    vertices = cv2.dilate(vertices, np.ones((3, 3), np.uint8))
    segments = (1 - vertices) * skeleton
    return segments


def find_image_best_threshold(greyscale_image, descriptor): # greyscale_image=win_conv; descriptor='perimeter'
    # Find out what score maximizes the descriptor
    max_score = np.max(greyscale_image)
    decreasing_score_step = (max_score // 10).astype(np.uint64)
    score_min = np.max((decreasing_score_step, np.min(greyscale_image)))
    score_max = max_score
    while decreasing_score_step > 1:
        score_scan = np.arange(score_min, score_max, decreasing_score_step)
        if score_scan[-1] != score_max:
            score_scan = np.concatenate((score_scan, [score_max]))
        # print(score_scan)
        descriptor_values = np.zeros(len(score_scan))
        # loop over these values to find a threshold that maximizes the descriptor
        for s_i, score in enumerate(score_scan): # score = 770
            binary_image = (greyscale_image > score).astype(np.uint8)
            if 1 < binary_image.sum() < binary_image.size * 0.6:
                # contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                # perimeters = [cv2.arcLength(cnt, closed=True) for cnt in contours]
                nb, ordered_image, stats, centers = cv2.connectedComponentsWithStats(binary_image)
                if nb > 2:
                    binary_image[ordered_image != np.argsort(stats[:, 4])[-2]] = 0
                SD = ShapeDescriptors(binary_image, [descriptor])
                descriptor_values[s_i] = SD.descriptors[descriptor]
        best_idx = np.argmax(descriptor_values)
        score_min = score_scan[np.max((0, best_idx - 1))]
        score_max = score_scan[np.min((len(score_scan) - 1, best_idx + 1))]
        score_diff = score_max - score_min
        decreasing_score_step = score_diff // 10
        # decreasing_score_step = (decreasing_score_step * 0.75).astype(np.uint64)
    threshold = score_scan[best_idx]
    final_binary_image = (greyscale_image > threshold).astype(np.uint8)
    nb, ordered_image, stats, centers = cv2.connectedComponentsWithStats(final_binary_image)
    if nb > 2:
        final_binary_image[ordered_image != np.argsort(stats[:, 4])[-2]] = 0
    SD = ShapeDescriptors(final_binary_image, [descriptor])
    descriptor_value = SD.descriptors[descriptor]

    return final_binary_image, threshold, descriptor_value


def get_best_side_length_window_step(greyscale_image, allowed_pixels, set_step_value=None, sample_number=50):
    height, width = greyscale_image.shape
    # 3.a. Find the best side length for a rolling window to segment the network

    # Find the best side length and window step to minimize the number of square fully segmented/unsegmented in the image. Do it only for the last image.
    min_im_shape = np.min((height, width))
    min_side_length = np.min((5, min_im_shape))
    # side_lengths = np.arange(min_side_length, 100, 5)
    side_lengths = []
    window_steps = []
    for side_length in np.arange(min_side_length, 100, 5):  # side_length=side_lengths[10];window_step=8
        if set_step_value is None:
            for window_step in np.arange(2, np.min((20, side_length)), 2, dtype=np.uint64):
                side_lengths.append(side_length)
                window_steps.append(window_step)
        else:
            window_step = set_step_value
            side_lengths.append(side_length)
            window_steps.append(window_step)
    side_step_scores = np.column_stack(
        (np.array(side_lengths, np.uint64), np.array(window_steps, np.uint64), np.zeros(len(side_lengths), np.uint64)))
    for sss_i in range(len(side_lengths)):  # sss_i = 100
        side_length, window_step = side_step_scores[sss_i, :2]
        window_coords = get_rolling_window_coordinates_list(height, width, side_length, window_step, allowed_pixels)

        if len(window_coords) > sample_number:
            window_coords = random.sample(window_coords, sample_number)

        # inflated_network = np.zeros(greyscale_image.shape, dtype=np.uint32)
        for w_i, win in enumerate(window_coords):  # w_i = 100; win = window_coords[w_i]
            win_grey = greyscale_image[win[0]:win[1], win[2]:win[3]]
            win_binary, best_score, hole_area = find_image_best_threshold(win_grey, 'total_hole_area')
            if np.any(win_binary) and not np.all(win_binary):
                side_step_scores[sss_i, 2] += 1
    if np.all(side_step_scores[:, 2] == np.max(side_step_scores[:, 2])):
        if set_step_value is None:
            side_length, window_step = side_step_scores[-1, 0], 2
        else:
            side_length, window_step = side_step_scores[-1, 0], set_step_value
    else:
        best_side_step_idx = np.nonzero(side_step_scores[:, 2] != np.max(side_step_scores[:, 2]))[0][-1] + 1
        if best_side_step_idx == len(side_lengths):
            best_side_step_idx -= 1
        side_length, window_step = side_step_scores[best_side_step_idx, :2]
    return side_length, window_step


def tracking_noise_correction(array_in_1, step):
    half_step = step // 2
    array_out = np.zeros_like(array_in_1)
    array_out[:half_step] = np.all(array_in_1[:half_step], axis=0)
    array_out[-half_step:] = np.all(array_in_1[-half_step:], axis=0)
    for i in np.arange(half_step, array_in_1.shape[0] - half_step):
        result = np.all(array_in_1[(i-half_step):(i+half_step), ...], axis=0)
        if result.sum() <= (array_out[i - 1, ...].sum() * 0.75):
            result = np.any(array_in_1[(i-half_step):(i+half_step), ...], axis=0)
            while result.sum() <= (array_out[i - 1, ...].sum() * 0.75) and (i-half_step) >= 0 and (i+half_step) < array_in_1.shape[0] and half_step <= (step * 2):
                half_step += 1
                result = np.any(array_in_1[(i-half_step):(i+half_step), ...], axis=0)
            half_step = step // 2
        array_out[i, ...] = result
    return array_out


def network_detection(binary_video, converted_video, origin, origin_state, lighter_background, sliding_sum_step=10, int_variation_thresh=20, side_length=40, window_step=4, display=False):
    dims = binary_video.shape
    if lighter_background:
        converted_video = 255 - converted_video
    sliding_cumulated_sum = np.zeros((dims[0], dims[1], dims[2]), dtype=np.uint64)
    half_step = sliding_sum_step // 2
    sliding_cumulated_sum[:half_step] = np.sum(converted_video[:sliding_sum_step, ...], axis=0)
    sliding_cumulated_sum[-half_step:] = np.sum(converted_video[-sliding_sum_step:, ...], axis=0)
    for f_i in np.arange(half_step, dims[0] - half_step):  # f_i = half_step
        sliding_cumulated_sum[f_i] = np.sum(converted_video[(f_i - half_step):(f_i + half_step), ...], axis=0)


    # 0. get_best_side_length_window_step using the last image
    last_binary_image = binary_video[-1, ...]
    allowed_pixels = deepcopy(last_binary_image)
    # Put 0 to not put the origin in the windows to scan
    origin_idx = np.nonzero(origin)
    if origin_state == "constant":
        allowed_pixels[origin_idx[0], origin_idx[1]] = 0
    # last_images_sum = sliding_cumulated_sum[-1] * last_binary_image
    # # The following function needs a crop to not have the maximal side_length value: 95
    # side_length, window_step = get_best_side_length_window_step(last_images_sum, allowed_pixels, set_step_value=None, sample_number=50)
    max_value = max_cum_sum_from_rolling_window(side_length, window_step)

    # 1. Start looping over each frames
    frames = np.arange(1, dims[0], 1)
    networks = np.zeros((len(frames) + 1, dims[1], dims[2]), dtype=np.uint8)
    networks[0, ...] = origin
    for frame in frames:# frame=1; networks[frame - 1, ...] = inflated_binary_network
        print(frame)
        # 1.b. Crop images to the size of the binary image

        binary_image = binary_video[frame, ...]
        y, x = np.nonzero(binary_image)
        min_y = min(y)
        if (min_y - 20) >= 0:
            min_y -= 20
        else:
            min_y = 0
        max_y = max(y)
        if (max_y + 20) < dims[1]:
            max_y += 20
        else:
            max_y = dims[1] - 1
        min_x = min(x)
        if (min_x - 20) >= 0:
            min_x -= 20
        else:
            min_x = 0
        max_x = max(x)
        if (max_x + 20) < dims[2]:
            max_x += 20
        else:
            max_x = dims[2] - 1

        images_sum = sliding_cumulated_sum[frame]
        cropped_binary_image = binary_image[min_y:max_y, min_x:max_x].copy()
        cropped_images_sum = images_sum[min_y:max_y, min_x:max_x].copy() * cropped_binary_image
        cropped_converted = converted_video[frame, ...][min_y:max_y, min_x:max_x].copy()
        cropped_prev_net = networks[frame - 1][min_y:max_y, min_x:max_x].copy()
        cropped_origin = origin[min_y:max_y, min_x:max_x]
        origin_idx = np.nonzero(cropped_origin)
        # cropped_previous_network = previous_network[min_y:max_y, min_x:max_x].copy()

        height, width = cropped_images_sum.shape


        # 2. Segment the network using the previous one and the converted image
        # 2.a. Create a gradient around the previous network and multiply it with the cropped_converted_image
        # Do a gravity field around the skeleton of that network
        grad_around = make_gravity_field(cropped_prev_net, max_distance=50, with_erosion=1)
        # grad_around = np.square(grad_around)
        grad_around[np.nonzero(cropped_prev_net)] = np.max(grad_around)
        # Make a scoring combining the distance to that skeleton (gravity) and the current intensity values.

        scoring = cropped_converted.astype(np.uint64)
        scoring *= grad_around
        scoring *= cropped_binary_image


        # 3. Segment the network using a rolling window on the sliding_cumulated_sum image
        allowed_pixels = deepcopy(cropped_binary_image)
        # Put 0 to not put the origin in the windows to scan
        if origin_state == "constant":
            allowed_pixels[origin_idx[0], origin_idx[1]] = 0

        # 3.b Create a list of coordinates corresponding to a rolling window on y and x.
        # 1. Get an inflated complete network
        window_coords = get_rolling_window_coordinates_list(height, width, side_length, window_step, allowed_pixels)

        # from scipy.ndimage import gaussian_filter
        # cropped_images_sum = gaussian_filter(images_sum[min_y:max_y, min_x:max_x], sigma=2)

        # 3.c. Use the rolling window to score each time each pixel is segmented to optimize the hole to area ratio
        # 4. Make the final network by combining the two methods 2. and 3.
        tested_descriptor = 'perimeter' # 'euler_number' # 'perimeter' # 'circularity'
        putative_network = np.zeros(cropped_images_sum.shape, dtype=np.uint32)
        for w_i, win in enumerate(window_coords):  # w_i = 0; win = window_coords[w_i]
            # See(cropped_converted[450:700, 750:1000])
            # See(cropped_converted[500:640, 810:950]) # win=[
            # See(cropped_converted[500:540, 860:900]) # win=[500, 540, 860, 900]
            win_score = scoring[win[0]:win[1], win[2]:win[3]]
            if np.any(win_score):
                win_conv = cropped_converted[win[0]:win[1], win[2]:win[3]]
                if np.ptp(win_conv[np.nonzero(win_conv)]) > int_variation_thresh:
                    # Find the threshold for which the binary image present one shape with the largest perimeter
                    conv_net, conv_thresh, conv_descriptor = find_image_best_threshold(win_conv, tested_descriptor)
                    if np.ptp(win_score[np.nonzero(win_score)]) > int_variation_thresh * 2:
                        score_net, score_thresh, score_descriptor = find_image_best_threshold(win_score, tested_descriptor)
                    else:
                        score_net, score_thresh, score_descriptor = np.zeros_like(win_score), 0., 0.

                    win_im_sum = cropped_images_sum[win[0]:win[1], win[2]:win[3]]
                    if np.ptp(win_im_sum[np.nonzero(win_im_sum)]) > int_variation_thresh * sliding_sum_step:
                        sum_net, sum_thresh, sum_descriptor = find_image_best_threshold(win_im_sum, tested_descriptor)
                    else:
                        sum_net, sum_thresh, sum_descriptor = np.zeros_like(win_im_sum), 0., 0.

                    if conv_descriptor > 0 and score_descriptor > 0 and sum_descriptor > 0:
                        otsu_net = otsu_thresholding(win_conv)
                        nb, ordered_image, stats, centers = cv2.connectedComponentsWithStats(otsu_net)
                        if nb > 2:
                            otsu_net[ordered_image != np.argsort(stats[:, 4])[-2]] = 0
                        SD = ShapeDescriptors(otsu_net, [tested_descriptor])
                        otsu_descriptor = SD.descriptors[tested_descriptor]

                        potential_networks = [conv_net, score_net, sum_net, otsu_net]
                        net_sizes = np.array((conv_net.sum(), score_net.sum(), sum_net.sum(), otsu_net.sum()))
                        descriptors = np.array((conv_descriptor, score_descriptor, sum_descriptor, otsu_descriptor))
                        size_filter = (net_sizes / win_conv.size) < 0.3
                        if np.any(size_filter):
                            descriptor_order = np.argsort(descriptors)
                            for i in np.arange(3, -1, -1):
                                if size_filter[descriptor_order[i]]:
                                    putative_network[win[0]:win[1], win[2]:win[3]] += potential_networks[descriptor_order[i]]
                                    break
                        else:
                            putative_network[win[0]:win[1], win[2]:win[3]] += potential_networks[np.argmin(net_sizes / win_conv.size)]

        # See(putative_network)
        # See((perimeter_network >= np.sqrt(max_value)).astype(np.uint8))

        binary_net = (putative_network >= np.sqrt(max_value)).astype(np.uint8)
        # binary_net, score_thresh, score_descriptor = find_image_best_threshold(putative_network,
        #                                                                       'total_hole_area')
        # See(binary_net)
        if origin_state == "constant":
            binary_net[origin_idx[0], origin_idx[1]] = 1

        binary_net = cv2.morphologyEx(binary_net, cv2.MORPH_OPEN, cross_33)
        ordered_image, stats, centers = cc(binary_net)
        binary_net[ordered_image != 1] = 0

        # if origin_state == "constant":
        #     binary_net[origin_idx[0], origin_idx[1]] = 0

        # network = np.zeros(cropped_images_sum.shape, dtype=np.uint32)
        # # Use the inflated_network skeleton to separate each segment
        # from skimage import morphology
        # skel = morphology.skeletonize(binary_net)
        # network[np.nonzero(skel)] = 1
        # vertices_coord, vertices_number = get_vertices_from_skeleton(skel)
        # # visualize_vertices(skel, vertices_coord)
        # segments = get_segments_from_vertices_skeleton(skel, vertices_coord)
        # nb, segments = cv2.connectedComponents(segments, connectivity=8)
        # # For each segment, only keep the pixels having a value above the median value of the skeleton:
        # for segment_id in range(1, nb):  # segment_id = 1
        #     # 3. Crop each segment until its skeleton breaks.
        #     current_segment = segments == segment_id
        #     segment_coord = np.nonzero(current_segment)
        #     segment_skeleton_scores = cropped_converted[segment_coord[0], segment_coord[1]]
        #     # segment_skeleton_scores = cropped_images_sum[segment_coord[0], segment_coord[1]]
        #     # segment_skeleton_scores = putative_network[segment_coord[0], segment_coord[1]]
        #
        #     if lighter_background:
        #         score_to_keep_the_skeleton = np.max(segment_skeleton_scores)
        #     else:
        #         score_to_keep_the_skeleton = np.min(segment_skeleton_scores)
        #
        #     lower_bound, higher_bound, lefter_bound, righter_bound = np.min(segment_coord[0]), np.max(
        #         segment_coord[0]), np.min(
        #         segment_coord[1]), np.max(
        #         segment_coord[1])
        #     v_diff, h_diff = higher_bound - lower_bound, righter_bound - lefter_bound
        #     lower_bound = np.max((lower_bound - v_diff, 0))
        #     higher_bound = np.min((higher_bound + v_diff, skel.shape[0] - 1))
        #     lefter_bound = np.max((lefter_bound - h_diff, 0))
        #     righter_bound = np.min((righter_bound + h_diff, skel.shape[1] - 1))
        #
        #     if lighter_background:
        #         true_segment = (cropped_converted[lower_bound:higher_bound + 1, lefter_bound:righter_bound + 1] <= score_to_keep_the_skeleton).astype(np.uint8)
        #     else:
        #         true_segment = (cropped_converted[lower_bound:higher_bound + 1, lefter_bound:righter_bound + 1] >= score_to_keep_the_skeleton).astype(np.uint8)
        #     # See(true_segment)
        #     # See(network[lower_bound:higher_bound + 1, lefter_bound:righter_bound + 1])
        #     # true_segment = (cropped_images_sum[min_y:max_y+1, min_x:max_x+1] >= score_to_keep_the_skeleton).astype(np.uint8)
        #     # true_segment = putative_network[lower_bound:higher_bound + 1, lefter_bound:righter_bound + 1]
        #     # true_segment = (true_segment >= score_to_keep_the_skeleton).astype(np.uint8)
        #     # 4. Remove noise around the segment:
        #     # true_segment = cv2.morphologyEx(true_segment, cv2.MORPH_OPEN, cross_33)
        #     if true_segment.sum() < 0.9 * true_segment.size:
        #         network[lower_bound:higher_bound + 1, lefter_bound:righter_bound + 1] += true_segment

        #     network *= cropped_binary_image
        #     if origin_state == "constant":
        #         network[origin_idx[0], origin_idx[1]] = 1
        #     ordered_image, stats, centers = cc(network)
        #     network[ordered_image != 1] = 0
        network = binary_net
        network *= cropped_binary_image
        # See(network > 1)
        # See(network > np.sqrt(network.max()))
        # See(network * inflated_binary_network)

        networks[frame, ...][min_y:max_y, min_x:max_x] = network
        if display:
            disp_im = np.zeros((dims[1], dims[2] * 2), dtype=np.uint8)
            disp_im[:, dims[2]:] = converted_video[frame, ...]
            disp_im[:, :dims[2]][min_y:max_y, min_x:max_x] = network * 255
            See(disp_im, size=(2000, 1000), keep_display=1)
    if display:
        cv2.destroyAllWindows()
    networks = tracking_noise_correction(networks, 2)
    return networks


if __name__ == "__main__":
    # 1. Prepare the data
    # 1.a. Generate converted_video and sliding_cumulated_sum
    video_nb = 3
    for video_nb in np.arange(1, 7): #  np.array((5, 6), int):
        os.chdir("D:\Directory\Data\Audrey")
        visu = np.load(f"ind_{video_nb}.npy")
        dims = visu.shape[:3]
        binary_coord = np.load(f"coord_specimen{video_nb}_t720_y1475_x1477.npy")
        binary_video = np.zeros((720, 1475, 1477), np.uint8)
        binary_video[binary_coord[0, :],binary_coord[1, :], binary_coord[2, :]] = 1
        origin = binary_video[0, ...]
        first_dict = TDict()
        first_dict["lab"] = np.array((0, 0, 1))
        first_dict["luv"] = np.array((0, 0, 1))
        origin_state = "constant"
        lighter_background = False
        int_variation_thresh = 20
        side_length, window_step = 40, 4
        sliding_sum_step = 10

        converted_video = np.zeros((dims[0], dims[1], dims[2]), dtype=np.uint8)
        for f_i in np.arange(dims[0]):
            bgr_image = visu[f_i, ...]
            converted_video[f_i, ...], _ = generate_color_space_combination(bgr_image, list(first_dict.keys()), first_dict,
                                                                            convert_to_uint8=True)
        networks = network_detection(binary_video, converted_video, origin, origin_state, lighter_background,
                                     sliding_sum_step=10, int_variation_thresh=int_variation_thresh,
                                     side_length=side_length, window_step=window_step)
        np.save(f"new_net_detection{video_nb}.npy", networks)

    # video_nb = 2
    # for video_nb in np.array((2, 4), int):
    #     os.chdir("/Users/Directory/Data/dossier1")
    #     visu = np.load(f"ind_{video_nb}.npy")
    #     write_video(visu, f"simple_video{video_nb}.mp4", is_color=True, fps=40)
