# tests/_base.py
from __future__ import annotations
import cv2
import numpy as np
from cellects.image_analysis.morphological_operations import rhombus_55, Ellipse, cross_33

import os
from pathlib import Path
import unittest

class CellectsUnitTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        d = Path(__file__).resolve().parents[1] / "data" # set up data path for the tests
        cls.path_input = d / "input"
        cls.path_output = d / "output"
        cls.path_experiment = d / "experiment"

        if not os.path.isdir(cls.path_output):
            os.mkdir(cls.path_output)

binary_video_test = np.zeros((6, 10, 10), dtype=np.uint8)
binary_video_test[0, :5, :5] = rhombus_55
binary_video_test[1, :6, :6] = Ellipse((6, 6)).create()
binary_video_test[2, :7, :7] = Ellipse((7, 7)).create()
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
video_test = np.zeros((binary_video_test.shape[0],binary_video_test.shape[1],binary_video_test.shape[2],3), dtype=np.uint8)
video_test[:, :, :, 0][binary_video_test > 0] = np.random.randint(205, 255, binary_video_test.sum())
video_test[:, :, :, 0][binary_video_test == 0] = np.random.randint(0, 50, binary_video_test.size - binary_video_test.sum())
video_test[:, :, :, 1][binary_video_test > 0] = np.random.randint(205, 255, binary_video_test.sum())
video_test[:, :, :, 1][binary_video_test == 0] = np.random.randint(0, 50, binary_video_test.size - binary_video_test.sum())
video_test[:, :, :, 2][binary_video_test > 0] = np.random.randint(205, 255, binary_video_test.sum())
video_test[:, :, :, 2][binary_video_test == 0] = np.random.randint(0, 50, binary_video_test.size - binary_video_test.sum())


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
several_arenas_img = np.zeros((several_arenas_bin_img.shape[0], several_arenas_bin_img.shape[1], 3), dtype=np.uint8)
several_arenas_img[:, :, 0][several_arenas_bin_img > 0] = np.random.randint(205, 255, several_arenas_bin_img.sum())
several_arenas_img[:, :, 0][several_arenas_bin_img == 0] = np.random.randint(0, 50, several_arenas_bin_img.size - several_arenas_bin_img.sum())
several_arenas_img[:, :, 1][several_arenas_bin_img > 0] = np.random.randint(205, 255, several_arenas_bin_img.sum())
several_arenas_img[:, :, 1][several_arenas_bin_img == 0] = np.random.randint(0, 50, several_arenas_bin_img.size - several_arenas_bin_img.sum())
several_arenas_img[:, :, 2][several_arenas_bin_img > 0] = np.random.randint(205, 255, several_arenas_bin_img.sum())
several_arenas_img[:, :, 2][several_arenas_bin_img == 0] = np.random.randint(0, 50, several_arenas_bin_img.size - several_arenas_bin_img.sum())

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
several_arenas_vid[:, :, :, 0][several_arenas_bin_vid > 0] = np.random.randint(205, 255, several_arenas_bin_vid.sum())
several_arenas_vid[:, :, :, 0][several_arenas_bin_vid == 0] = np.random.randint(0, 50, several_arenas_bin_vid.size - several_arenas_bin_vid.sum())
several_arenas_vid[:, :, :, 1][several_arenas_bin_vid > 0] = np.random.randint(205, 255, several_arenas_bin_vid.sum())
several_arenas_vid[:, :, :, 1][several_arenas_bin_vid == 0] = np.random.randint(0, 50, several_arenas_bin_vid.size - several_arenas_bin_vid.sum())
several_arenas_vid[:, :, :, 2][several_arenas_bin_vid > 0] = np.random.randint(205, 255, several_arenas_bin_vid.sum())
several_arenas_vid[:, :, :, 2][several_arenas_bin_vid == 0] = np.random.randint(0, 50, several_arenas_bin_vid.size - several_arenas_bin_vid.sum())
