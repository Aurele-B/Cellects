#!/usr/bin/env python3
"""
Test module for the `load` module in Cellects.

This test suite covers file loading operations. Test cases include success scenarios,
edge case handling for different file extensions (.mp4, .h5, etc.), and dimensional/property verification
for videos and images.

Notes
-----
All test classes inherit from CellectsUnitTest, providing shared setup/teardown for temporary files.
Test files are automatically cleaned up in tearDown methods.
"""
import unittest
from tests._base import CellectsUnitTest
from cellects.io.load import *


class TestReadH5Array(CellectsUnitTest):
    """Test suite for read_h5 function."""

    def test_read_h5_file(self):
        """Test that read_h5 manage to read."""
        file_name = self.path_input + '/' + f"test_vstack.h5"
        key_name = "first_key"
        # Execute function
        read_file = read_h5(file_name, key_name)
        self.assertTrue(isinstance(read_file, np.ndarray))


class TestGetH5Keys(CellectsUnitTest):
    """Test suite for get_h5_keys function."""

    def test_get_h5_keys(self):
        """Test that read_h5 manage to read."""
        file_name = self.path_input + '/' + f"test_vstack.h5"
        key_name = "first_key"
        read_key = get_h5_keys(file_name)
        self.assertTrue(key_name == read_key[0])


class TestVideo2Numpy(CellectsUnitTest):
    """Test suite for video2numpy function."""

    def test_video2numpy_with_h5_extension(self):
        """Test with h5 extension."""
        # Create a temporary h5 file
        video = video2numpy(self.path_input + '/' + 'test_read_video.h5')
        array_shape = (10, 5, 5, 3)
        # Verify the dimensions of the video
        self.assertTrue(np.sum(video.shape) == np.sum(array_shape))

    def test_video2numpy_with_video_file(self):
        """Test with mp4 extension."""
        # Create a temporary mp4 file
        video = video2numpy(self.path_input + '/' + 'test_read_video.mp4')
        self.assertTrue(isinstance(video, np.ndarray))

    def test_video2numpy_with_true_frame_width(self):
        """Test with specified frame width."""
        true_frame_width = 2
        video = video2numpy(self.path_input + '/' + 'test_read_video.mp4', true_frame_width=true_frame_width)
        self.assertTrue(isinstance(video, np.ndarray))

    def test_video2numpy_h5_with_conversion(self):
        """Test h5_with_conversion."""
        true_frame_width = 2
        conversion_dict = {'bgr': [1, 1, 1], 'logical': "Or", 'hsv2': [0, 1, 0]}
        video, converted_video = video2numpy(self.path_input + '/' + 'test_read_video.h5', conversion_dict, true_frame_width=true_frame_width)
        self.assertIsInstance(video, np.ndarray)
        self.assertIsInstance(converted_video, np.ndarray)

    def test_video2numpy_mp4_with_conversion(self):
        """Test mp4."""
        true_frame_width = 2
        conversion_dict = {'bgr': [1, 1, 1], 'logical': "Or", 'hsv2': [0, 1, 0]}
        video, converted_video = video2numpy(self.path_input + '/' + 'test_read_video.mp4', conversion_dict, true_frame_width=true_frame_width)
        self.assertIsInstance(video, np.ndarray)
        self.assertIsInstance(converted_video, np.ndarray)


class TestIsRawImage(CellectsUnitTest):
    """Test suite for is_raw_image function."""

    def test_is_raw_image_with_non_raw_format(self):
        """Test with non-raw file."""
        img = np.random.randint(0, 255, size=(10, 10, 3), dtype=np.uint8)
        image_path = self.path_output + '/' + "img.jpg"
        cv2.imwrite(image_path, img)

        is_raw = is_raw_image(str(image_path))

        # Verify the expected result
        self.assertFalse(is_raw)

    def tearDown(self):
        """Remove all written files."""
        if os.path.isfile(self.path_output + '/' + 'img.jpg'):
            os.remove(self.path_output + '/' + 'img.jpg')

class TestReadImage(CellectsUnitTest):
    """Test suite for readim function."""
    def test_readim_with_regular_image(self):
        """Test basic functionality."""
        image_path = self.path_experiment + '/' + "image1.tif"
        raw_image = False

        # Call the function
        result = readim(str(image_path), raw_image)

        ref_size = (245, 300, 3)
        # Verify the expected result
        self.assertTrue(np.array_equal(ref_size, result.shape))


class TestReadAndRotate(CellectsUnitTest):
    """Test suite for read_and_rotate function."""
    def test_read_and_rotate_orientation_correction(self):
        """Test basic functionality."""
        image_1 = self.path_experiment + '/' + "image1.tif"
        image_2 = self.path_experiment + '/' + "image2.tif"
        raw_images = False
        is_landscape = True

        # Call the function
        im1 = read_and_rotate(str(image_1), None, raw_images, is_landscape)
        im2 = read_and_rotate(str(image_2), im1, raw_images, is_landscape)
        im2a = cv2.rotate(im2, cv2.ROTATE_90_CLOCKWISE)
        im2b = cv2.rotate(im2, cv2.ROTATE_90_COUNTERCLOCKWISE)

        im2a_path = str(self.path_output + '/' + 'im2a.jpg')
        im2b_path = str(self.path_output + '/' + 'im2b.jpg')
        cv2.imwrite(im2a_path, im2a)
        cv2.imwrite(im2b_path, im2b)
        im2a = read_and_rotate(im2a_path, im1, raw_images, is_landscape)
        im2b = read_and_rotate(im2b_path, im1, raw_images, is_landscape)

        self.assertTrue(np.allclose(im2, im2a, atol=15))
        self.assertTrue(np.allclose(im2, im2b, atol=15))

    def tearDown(self):
        """Remove all written files."""
        if os.path.isfile(self.path_output + '/' + f"im2a.jpg"):
            os.remove(self.path_output + '/' + f"im2a.jpg")
        if os.path.isfile(self.path_output + '/' + f"im2b.jpg"):
            os.remove(self.path_output + '/' + f"im2b.jpg")


class TestExtractTime(CellectsUnitTest):
    """Test suite for extract_time function."""
    def test_extract_time_basic_behavior(self):
        """Test extract_time basic behavior."""
        # Execute function
        result = extract_time(self.d + '/multiple_experiments/f1')
        # Verify result
        self.assertTrue(isinstance(result, np.ndarray))

    def test_extract_time_with_valid_images_same_timestamp(self):
        """Test extract_time with valid images having same timestamp."""
        # Setup test data - would create actual image files in real tests
        image_list = ["image1.tif", "image2.tif"]
        expected_time = np.array([0, 0])  # Assuming all timestamps are same

        # Execute function
        result = extract_time(self.path_experiment, image_list)
        # Verify result
        self.assertTrue(np.array_equal(result, expected_time))


class TestReadArena(CellectsUnitTest):
    """Test suite for read_one_arena function."""
    def test_read_one_arena_with_videos_in_ram(self):
        """Test read_one_arena with videos in ram."""
        csc_dict = {'bgr': [0, 0, 1], 'logical': 'None'}
        videos_already_in_ram = [np.zeros(0), np.zeros(0)]
        visu, converted_video, converted_video2 = read_one_arena(arena_label=1, already_greyscale=True,
                                                                 csc_dict=csc_dict,
                                                                 videos_already_in_ram=videos_already_in_ram)
        visu, converted_video, converted_video2 = read_one_arena(arena_label=1, already_greyscale=False,
                                                                 csc_dict=csc_dict,
                                                                 videos_already_in_ram=videos_already_in_ram)
        self.assertTrue(converted_video2 is None)
        csc_dict = {'bgr': [0, 0, 1], 'logical': 'Or'}
        videos_already_in_ram = [np.zeros(0), np.zeros(0), np.zeros(0)]
        visu, converted_video, converted_video2 = read_one_arena(arena_label=1, already_greyscale=False,
                                                                 csc_dict=csc_dict,
                                                                 videos_already_in_ram=videos_already_in_ram)
        self.assertTrue(converted_video2 is not None)

    def test_read_one_arena_with_vid_name(self):
        """Test read_one_arena with a video on the disk."""
        os.chdir(self.path_input)
        csc_dict = {'bgr': [0, 0, 1], 'logical': 'None'}
        vid_name = "test_read_video.mp4"
        visu, converted_video, converted_video2 = read_one_arena(arena_label=1, already_greyscale=False,
                                                                 csc_dict=csc_dict, vid_name=vid_name)
        self.assertIsInstance(visu, np.ndarray)
        self.assertTrue(converted_video is None)
        csc_dict = {'bgr': [0, 0, 1], 'logical': 'Or', 'hsv2': [0, 0, 1]}
        visu, converted_video, converted_video2 = read_one_arena(arena_label=1, already_greyscale=True,
                                                                 csc_dict=csc_dict, vid_name=vid_name)
        self.assertIsInstance(converted_video, np.ndarray)
        self.assertTrue(visu is None)


if __name__ == '__main__':
    unittest.main()
