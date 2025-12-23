# tests/_base.py
from __future__ import annotations
import cv2
import numpy as np
from numpy.typing import NDArray
from cellects.image_analysis.morphological_operations import rhombus_55, create_ellipse, cross_33, get_contours
import itertools
import os
from pathlib import Path
import unittest

class CellectsUnitTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.d = Path(__file__).resolve().parents[1] / "data" # set up data path for the tests
        cls.path_input = cls.d / "input"
        cls.path_output = cls.d / "output"
        cls.path_experiment = cls.d / "experiment"

        if not os.path.isdir(cls.path_output):
            os.mkdir(cls.path_output)

np.random.seed(1234)
# AIM: Make one test for each condition:
# - One large noisy-circle in the middle
# - One large noisy-circle on a side
# - One small noisy-circle in the middle
# - One small noisy-circle on a side
# - Many small noisy-circles
# - Many large noisy-circles
# - Many differing sizes noisy-circles
# Multiply by n the analysis with:
# n=1: All the same with huge background variations
# n=2: All the same with huge specimen variations

def make_noisy_circle(size):
    noisy_circle = create_ellipse(size, size).astype(np.uint8)
    contour = get_contours(noisy_circle)
    contour_coord = np.nonzero(contour)
    sample = np.random.choice(np.arange(len(contour_coord[0])), len(contour_coord[0]) // 3)
    noisy_circle[contour_coord[0][sample], contour_coord[1][sample]] = 0
    if size > 15:
        coef = int(round(np.log10(size))) * 2 // 2 + 3
        noisy_circle = cv2.erode(noisy_circle, np.ones((coef , coef), dtype=np.uint8), iterations=1)
        if size > 30:
            noisy_circle = cv2.erode(noisy_circle, np.ones((coef + 1, 1), dtype=np.uint8), iterations=1)
            noisy_circle = cv2.erode(noisy_circle, np.ones((1, coef + 1), dtype=np.uint8), iterations=1)
    return noisy_circle

def get_random_blobs_coord(im_size, blob_size, blob_nb):
    possible_coord = np.arange(0, im_size, blob_size)
    possible_coord = possible_coord[possible_coord < im_size - blob_size]
    possible_coord = np.array(list(itertools.combinations(possible_coord, 2)))
    all_coord = np.zeros((possible_coord.shape[0] * 2, possible_coord.shape[1]), np.uint32)
    all_coord[:possible_coord.shape[0], :] = possible_coord
    all_coord[possible_coord.shape[0]:, :] = possible_coord[:, ::-1]
    sample_coord = np.random.choice(np.arange(len(all_coord)), blob_nb, replace=False)
    sample_coord = all_coord[sample_coord, :]
    return sample_coord

def make_rgb_from_mask(mask, blob_to_back_diff: int=100, blob_extent: int=20, back_extent: int=20) -> NDArray[np.uint8]:
    # blob_to_back_diff = np.uint8(100)
    blob_diff = blob_extent // 2
    back_diff = back_extent // 2
    blob_c = []
    blob_c_min = []
    blob_c_max = []
    back_c = []
    back_c_min = []
    back_c_max = []
    for i in range(3):
        pot_bc = np.random.choice((np.random.randint(0, 255 - blob_to_back_diff), np.random.randint(blob_to_back_diff + 1, 256)))
        blob_c.append(pot_bc)
        if blob_c[i] <= blob_to_back_diff:
            # print(blob_c[i])
            back_c.append(np.random.randint(blob_c[i] + blob_to_back_diff, 256))
        elif blob_c[i] > 255 - blob_to_back_diff:
            back_c.append(np.random.randint(0 , blob_c[i] - blob_to_back_diff))
        else:
            back_c.append(np.random.choice((np.random.randint(blob_c[i] + blob_to_back_diff , 256), np.random.randint(0, blob_c[i] - blob_to_back_diff))))
        blob_c_min.append(np.max((blob_c[i] - blob_diff, 0)))
        blob_c_max.append(np.min((blob_c[i] + blob_diff, 255)))
        back_c_min.append(np.max((back_c[i] - back_diff, 0)))
        back_c_max.append(np.min((back_c[i] + back_diff, 255)))
        # print(f'back: {(back_c_min[i], back_c_max[i])}, blob: {(blob_c_min[i], blob_c_max[i])}')
    if len(mask.shape) == 2:
        rgb_from_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    elif len(mask.shape) == 3:
        rgb_from_mask = np.zeros((mask.shape[0], mask.shape[1], mask.shape[2], 3), dtype=np.uint8)
    for c_, (blob_min, blob_max, back_min, back_max) in enumerate(zip(blob_c_min, blob_c_max, back_c_min, back_c_max)):
        if len(mask.shape) == 2:
            rgb_from_mask[:, :, c_][mask > 0] = np.random.randint(blob_min, blob_max, mask.sum())
            rgb_from_mask[:, :, c_][mask == 0] = np.random.randint(back_min, back_max, mask.size - mask.sum())
        elif len(mask.shape) == 3:
            rgb_from_mask[:, :, :, c_][mask > 0] = np.random.randint(blob_min, blob_max, mask.sum())
            rgb_from_mask[:, :, :, c_][mask == 0] = np.random.randint(back_min, back_max, mask.size - mask.sum())
    return rgb_from_mask


# Make the binary images
im_size = 1000
large_size = 500
large = make_noisy_circle(large_size)
medium_size = 150
medium = make_noisy_circle(medium_size)
small_size = 50
small = make_noisy_circle(small_size)
one_large_central_blob = np.zeros((im_size, im_size), dtype=np.uint8)
one_large_central_blob[250:750,250:750] = large

one_large_side_blob = np.zeros((im_size, im_size), dtype=np.uint8)
one_large_side_blob[500:,500:] = large

one_small_central_blob = np.zeros((im_size, im_size), dtype=np.uint8)
one_small_central_blob[475:(475 + small_size),475:(475 + small_size)] = small

one_small_side_blob = np.zeros((im_size, im_size), dtype=np.uint8)
one_small_side_blob[950:,950:] = small

small_blob_nb = 50
many_small_blobs = np.zeros((im_size, im_size), dtype=np.uint8)
sample_coord = get_random_blobs_coord(im_size, small_size, small_blob_nb)
for coord in sample_coord:
    many_small_blobs[coord[0]:(coord[0] + small_size),coord[1]:(coord[1] + small_size)] = small

medium_blob_nb = 20
many_medium_blobs = np.zeros((im_size, im_size), dtype=np.uint8)
sample_coord = get_random_blobs_coord(im_size, medium_size, medium_blob_nb)
for coord in sample_coord:
    many_medium_blobs[coord[0]:(coord[0] + medium_size),coord[1]:(coord[1] + medium_size)] = medium

varying_blob_nb = 30
many_varying_blobs = np.zeros((im_size, im_size), dtype=np.uint8)
sample_coord = get_random_blobs_coord(im_size, medium_size, medium_blob_nb)
for coord in sample_coord:
    varying_size = np.random.randint(medium_size - 1) + 1
    varying = make_noisy_circle(varying_size)
    many_varying_blobs[coord[0]:(coord[0] + varying_size),coord[1]:(coord[1] + varying_size)] = varying

# Make the rgb images
blob_variation = 80
blob_vary_rgb_one_large_central_blob = make_rgb_from_mask(one_large_central_blob, blob_extent=blob_variation)
blob_vary_rgb_one_large_side_blob = make_rgb_from_mask(one_large_side_blob, blob_extent=blob_variation)
blob_vary_rgb_one_small_central_blob = make_rgb_from_mask(one_small_central_blob, blob_extent=blob_variation)
blob_vary_gb_one_small_side_blob = make_rgb_from_mask(one_small_side_blob, blob_extent=blob_variation)
blob_vary_rgb_many_small_blobs = make_rgb_from_mask(many_small_blobs, blob_extent=blob_variation)
blob_vary_rgb_many_medium_blobs = make_rgb_from_mask(many_medium_blobs, blob_extent=blob_variation)
blob_vary_rgb_many_varying_blobs = make_rgb_from_mask(many_varying_blobs, blob_extent=blob_variation)

back_variation = 80
back_vary_rgb_one_large_central_blob = make_rgb_from_mask(one_large_central_blob, back_extent=back_variation)
back_vary_rgb_one_large_side_blob = make_rgb_from_mask(one_large_side_blob, back_extent=back_variation)
back_vary_rgb_one_small_central_blob = make_rgb_from_mask(one_small_central_blob, back_extent=back_variation)
back_vary_rgb_one_small_side_blob = make_rgb_from_mask(one_small_side_blob, back_extent=back_variation)
back_vary_rgb_many_small_blobs = make_rgb_from_mask(many_small_blobs, back_extent=back_variation)
back_vary_rgb_many_medium_blobs = make_rgb_from_mask(many_medium_blobs, back_extent=back_variation)
back_vary_rgb_many_varying_blobs = make_rgb_from_mask(many_varying_blobs, back_extent=back_variation)
# show(blob_vary_rgb_one_large_central_blob)
# show(blob_vary_rgb_one_large_side_blob)
# show(blob_vary_rgb_one_small_central_blob)
# show(blob_vary_gb_one_small_side_blob)
# show(blob_vary_rgb_many_small_blobs)
# show(blob_vary_rgb_many_medium_blobs)
# show(blob_vary_rgb_many_varying_blobs)



binary_video_test = np.zeros((6, 10, 10), dtype=np.uint8)
binary_video_test[0, :5, :5] = rhombus_55
binary_video_test[1, :6, :6] = create_ellipse(6, 6)
binary_video_test[2, :7, :7] = create_ellipse(7, 7)
binary_video_test[3, :, :] = np.array([[0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                            [0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
                            [0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                            [0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                            [1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                            [0, 1, 1, 1, 1, 1, 1, 0, 0, 0],
                            [0, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                            [0, 0, 1, 1, 1, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.uint8)
binary_video_test[4, :, :] = np.array([[0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                            [0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
                            [0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                            [0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                            [0, 1, 1, 1, 1, 1, 1, 0, 0, 0],
                            [0, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                            [0, 0, 1, 1, 1, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.uint8)
binary_video_test[5, :, :] = np.array([[0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                            [0, 0, 1, 1, 1, 1, 1, 1, 1, 0],
                            [0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
                            [0, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                            [0, 1, 1, 1, 1, 0, 0, 0, 1, 1],
                            [0, 1, 1, 1, 1, 0, 0, 1, 1, 1],
                            [0, 0, 1, 1, 1, 0, 0, 1, 1, 0]], dtype=np.uint8)
rgb_video_test = make_rgb_from_mask(binary_video_test, blob_to_back_diff=150, blob_extent=50, back_extent=50)

several_arenas_bin_img = np.array([[0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                                        [0,  0,  1,  0,  0,  0,  1,  0,  0,  1,  0],
                                        [0,  1,  1,  1,  0,  0,  1,  0,  1,  0,  0],
                                        [0,  0,  1,  0,  0,  1,  1,  0,  1,  1,  0],
                                        [0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0],
                                        [0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                                        [0,  1,  0,  0,  0,  0,  0,  0,  0,  1,  0],
                                        [0,  0,  1,  0,  0,  1,  0,  0,  1,  0,  0],
                                        [0,  0,  0,  1,  0,  0,  1,  0,  0,  1,  0],
                                        [0,  0,  0,  1,  0,  0,  0,  0,  0,  1,  0],
                                        [0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0]], dtype=np.uint8)
rgb_several_arenas_img = np.zeros((several_arenas_bin_img.shape[0], several_arenas_bin_img.shape[1], 3), dtype=np.uint8)
rgb_several_arenas_img[:, :, :][several_arenas_bin_img > 0] = 235, 235, 235
rgb_several_arenas_img[:, :, :][several_arenas_bin_img == 0] = 20, 20, 20

patches_video = np.random.choice([0, 1, 2], 500, p=[0.9, 0.05, 0.05]).reshape((5, 10, 10)).astype(np.uint8)
for t in range(5):
    patches_video[t, ...] = cv2.dilate(patches_video[t, ...], np.ones((3, 3), dtype=np.uint8), iterations=2)

several_arenas_bin_vid = np.zeros((8, 20, 20), dtype=np.uint8)
several_arenas_bin_vid[0:3, 4:7, 4:7] += cross_33
several_arenas_bin_vid[0:3, 13:16, 4:7] += cross_33
several_arenas_bin_vid[1:4, 4:7, 6:9] += cross_33
several_arenas_bin_vid[1:4, 13:16, 6:9] += cross_33
several_arenas_bin_vid[2:6, 4:7, 7:10] += cross_33
several_arenas_bin_vid[2:6, 13:16, 7:10] += cross_33
several_arenas_bin_vid[3:5, 4:7, 8:11] += cross_33
several_arenas_bin_vid[3:5, 13:16, 8:11] += cross_33
several_arenas_bin_vid[4:6, 4:7, 9:12] += cross_33
several_arenas_bin_vid[4:6, 13:16, 9:12] += cross_33
several_arenas_bin_vid[5:8, 4:7, 10:13] += cross_33
several_arenas_bin_vid[5:8, 13:16, 10:13] += cross_33
several_arenas_bin_vid[7, 4:7, 11:14] += cross_33
several_arenas_bin_vid[7, 13:16, 11:14] += cross_33
several_arenas_bin_vid[several_arenas_bin_vid > 0] = 1

several_arenas_vid = np.zeros((several_arenas_bin_vid.shape[0],several_arenas_bin_vid.shape[1],several_arenas_bin_vid.shape[2],3), dtype=np.uint8)
several_arenas_vid[:, :, :, :][several_arenas_bin_vid > 0] = np.random.randint(120, 130, several_arenas_bin_vid.sum() * 3).reshape(-1, 3)
several_arenas_vid[:, :, :, :][several_arenas_bin_vid == 0] = np.random.randint(235, 255, (several_arenas_bin_vid.size - several_arenas_bin_vid.sum()) * 3).reshape(-1, 3)


