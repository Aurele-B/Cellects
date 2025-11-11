# tests/_base.py
from __future__ import annotations
import cv2
import numpy as np
from cellects.image_analysis.morphological_operations import rhombus_55, Ellipse

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