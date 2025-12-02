#!/usr/bin/env python3
"""
Test module for the `load_display_save` module in Cellects, containing 18 unit tests.

This test suite covers file I/O operations (pickle), video handling routines, image format validation,
and utility functions for loading/storing multimedia data. Test cases include success scenarios,
edge case handling for different file extensions (.mp4, .npy, etc.), and dimensional/property verification
for videos and images.

Notes
-----
All test classes inherit from CellectsUnitTest, providing shared setup/teardown for temporary files.
Test files are automatically cleaned up in tearDown methods.
"""
import unittest

import cv2

from tests._base import CellectsUnitTest
from cellects.utils.load_display_save import *
from matplotlib.figure import Figure


class TestPickleRick(CellectsUnitTest):
    """Test suite for PickleRick class."""
    def test_write_file_success(self):
        """Test write_file."""
        # Create an instance of PickleRick
        pickle_rick = PickleRick(10)

        # Define test data
        file_content = {"Test": "This is some test content"}
        file_name = self.path_output / "test_file.pkl"

        # Call the write_file method
        pickle_rick.write_file(file_content, file_name)

        # Assert that the file was written successfully
        self.assertTrue(os.path.isfile(file_name))

        # Clean up the test files
        if os.path.isfile("PickleRick10.pkl"):
            os.remove("PickleRick10.pkl")
        os.remove(file_name)

    def test_read_file_success(self):
        """Test read_file."""
        # Create an instance of PickleRick
        pickle_rick = PickleRick(10)

        # Define test data
        file_content = "This is some test content"
        file_name = self.path_input / "test_file.pkl"

        # Write the test file
        with open(file_name, 'wb') as file_to_write:
            pickle.dump(file_content, file_to_write)

        # Call the read_file method
        result = pickle_rick.read_file(file_name)

        # Assert that the file was read successfully
        self.assertEqual(result, file_content)

        # Clean up the test files
        if os.path.isfile("PickleRick10.pkl"):
            os.remove("PickleRick10.pkl")
        os.remove(file_name)


class TestWriteVideo(CellectsUnitTest):
    """Test suite for write_video function."""
    def test_write_video_with_mp4_extension(self):
        """Test as .mp4 file."""
        # Create a temporary mp4 file
        with open(self.path_output / 'test_write_video.mp4', 'wb') as temp_file:
            # Generate a sample numpy array
            np_array = np.random.randint(0, 255, size=(5, 10, 10, 3), dtype=np.uint8)

            # Write the video
            write_video(np_array, temp_file.name)

            # Verify that the file exists
            self.assertTrue(os.path.exists(temp_file.name))

    def test_write_video_with_unknown_extension(self):
        """Test with unknown extension."""
        # Create a temporary file without a recognized extension
        with open(self.path_output / 'test_write_video.xyz', 'wb') as temp_file:
            # Generate a sample numpy array
            np_array = np.random.randint(0, 255, size=(5, 10, 10, 3), dtype=np.uint8)

            # Write the video
            write_video(np_array, temp_file.name)
            new_name = temp_file.name[:-4]
            new_name += '.mp4'
            # Verify that the file exists
            self.assertTrue(os.path.exists(new_name))

    def test_write_video_with_npy_extension(self):
        """Test with .npy extension."""
        # Create a temporary mp4 file
        with open(self.path_output / 'test_write_video.npy', 'wb') as temp_file:
            # Generate a sample numpy array
            np_array = np.random.randint(0, 255, size=(5, 10, 10, 3), dtype=np.uint8)

            # Write the video
            write_video(np_array, temp_file.name)

            # Verify that the file exists
            self.assertTrue(os.path.exists(temp_file.name))


    def test_write_video_dimensions_and_fps(self):
        """Test with dim and fps."""
        # Create a temporary mp4 file
        with open(self.path_output / 'test_write_video.mp4', 'wb') as temp_file:
            # Generate a sample numpy array
            np_array = np.random.randint(0, 255, size=(5, 10, 10, 3), dtype=np.uint8)
            fps = 30

            # Write the video
            write_video(np_array, temp_file.name, fps=fps)

            # Read the video back
            vid = cv2.VideoCapture(temp_file.name)
            
            # Verify the video dimensions
            self.assertEqual(vid.get(cv2.CAP_PROP_FRAME_WIDTH), np_array.shape[2])
            self.assertEqual(vid.get(cv2.CAP_PROP_FRAME_HEIGHT), np_array.shape[1])
            self.assertEqual(vid.get(cv2.CAP_PROP_FPS), fps)

    def tearDown(self):
        """Remove all written files."""
        if os.path.isfile(self.path_output / 'test_write_video.mp4'):
            os.remove(self.path_output / 'test_write_video.mp4')
        if os.path.isfile(self.path_output / 'test_write_video.npy'):
            os.remove(self.path_output / 'test_write_video.npy')
        if os.path.isfile(self.path_output / 'test_write_video.xyz'):
            os.remove(self.path_output / 'test_write_video.xyz')

# np_array = np.random.randint(0, 255, size=(10, 5, 5, 3), dtype=np.uint8)
# np.save('/Users/Directory/Scripts/python/Cellects/data/input/test_read_video.npy', np_array)
#
# write_video(np_array,'/Users/Directory/Scripts/python/Cellects/data/input/test_read_video.npy', is_color=True, fps=1)
#
# video = video2numpy('/Users/Directory/Scripts/python/Cellects/data/input/test_read_video.npy')
# video = video2numpy('/Users/Directory/Scripts/python/Cellects/data/input/test_read_video.mp4')


class TestVideo2Numpy(CellectsUnitTest):
    """Test suite for video2numpy function."""
    def test_video2numpy_with_npy_extension(self):
        """Test with npy extension."""
        # Create a temporary npy file
        video = video2numpy(str(self.path_input / 'test_read_video.npy'))
        array_shape = (10, 5, 5, 3)
        # Verify the dimensions of the video
        self.assertTrue(np.sum(video.shape) == np.sum(array_shape))

    def test_video2numpy_with_video_file(self):
        """Test with mp4 extension."""
        # Create a temporary mp4 file
        video = video2numpy(str(self.path_input / 'test_read_video.mp4'))
        self.assertTrue(isinstance(video, np.ndarray))

    def test_video2numpy_with_true_frame_width(self):
        """Test with specified frame width."""
        true_frame_width = 2
        video = video2numpy(str(self.path_input / 'test_read_video.mp4'), true_frame_width=true_frame_width)
        self.assertTrue(isinstance(video, np.ndarray))

    def test_video2numpy_npy_with_conversion(self):
        """Test npy_with_conversion."""
        true_frame_width = 2
        conversion_dict = {'bgr': np.array((1, 1, 1)), 'logical': "Or", 'hsv2': np.array((0, 1, 0))}
        video, converted_video = video2numpy(str(self.path_input / 'test_read_video.npy'), conversion_dict, true_frame_width=true_frame_width)
        self.assertIsInstance(video, np.ndarray)
        self.assertIsInstance(converted_video, np.ndarray)

    def test_video2numpy_mp4_with_conversion(self):
        """Test mp4."""
        true_frame_width = 2
        conversion_dict = {'bgr': np.array((1, 1, 1)), 'logical': "Or", 'hsv2': np.array((0, 1, 0))}
        video, converted_video = video2numpy(str(self.path_input / 'test_read_video.mp4'), conversion_dict, true_frame_width=true_frame_width)
        self.assertIsInstance(video, np.ndarray)
        self.assertIsInstance(converted_video, np.ndarray)


class TestIsRawImage(CellectsUnitTest):
    """Test suite for is_raw_image function."""

    def test_is_raw_image_with_non_raw_format(self):
        """Test with non-raw file."""
        img = np.random.randint(0, 255, size=(10, 10, 3), dtype=np.uint8)
        image_path = self.path_output / "img.jpg"
        cv2.imwrite(image_path, img)

        is_raw = is_raw_image(str(image_path))

        # Verify the expected result
        self.assertFalse(is_raw)

    def tearDown(self):
        """Remove all written files."""
        if os.path.isfile(self.path_output / 'img.jpg'):
            os.remove(self.path_output / 'img.jpg')

class TestReadImage(CellectsUnitTest):
    """Test suite for readim function."""
    def test_readim_with_regular_image(self):
        """Test basic functionality."""
        image_path = self.path_experiment / "image1.tif"
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
        image_1 = self.path_experiment / "image1.tif"
        image_2 = self.path_experiment / "image2.tif"
        raw_images = False
        is_landscape = True

        # Call the function
        im1 = read_and_rotate(str(image_1), None, raw_images, is_landscape)
        im2 = read_and_rotate(str(image_2), im1, raw_images, is_landscape)
        im2a = cv2.rotate(im2, cv2.ROTATE_90_CLOCKWISE)
        im2b = cv2.rotate(im2, cv2.ROTATE_90_COUNTERCLOCKWISE)

        im2a_path = str(self.path_output / 'im2a.jpg')
        im2b_path = str(self.path_output / 'im2b.jpg')
        cv2.imwrite(im2a_path, im2a)
        cv2.imwrite(im2b_path, im2b)
        im2a = read_and_rotate(im2a_path, im1, raw_images, is_landscape)
        im2b = read_and_rotate(im2b_path, im1, raw_images, is_landscape)

        self.assertTrue(np.allclose(im2, im2a, atol=15))
        self.assertTrue(np.allclose(im2, im2b, atol=15))

    def tearDown(self):
        """Remove all written files."""
        if os.path.isfile(self.path_output / f"im2a.jpg"):
            os.remove(self.path_output / f"im2a.jpg")
        if os.path.isfile(self.path_output / f"im2b.jpg"):
            os.remove(self.path_output / f"im2b.jpg")


class TestVStackH5Array(CellectsUnitTest):
    """Test suite for vstack_h5_array function."""

    def test_create_new_file_with_valid_input(self):
        """Test that a new HDF5 file and dataset is created with valid input."""
        # Setup unique filename in output directory
        file_name = self.path_output / f"test_vstack.h5"
        table = np.array([[1, 2], [3, 4]])

        # Execute function
        vstack_h5_array(file_name, table)

        # Verify result
        with h5py.File(file_name, 'r') as f:
            self.assertIn("data", f)  # Check dataset exists
            np.testing.assert_equal(f["data"][...], table)  # Check data matches

    def test_append_existing_dataset(self):
        """Test that new table is appended to existing dataset."""
        file_name = self.path_output / f"test_vstack.h5"

        # Initial write
        initial_data = np.array([[1, 2], [3, 4]])
        vstack_h5_array(file_name, initial_data)

        # Append new data
        append_data = np.array([[5, 6], [7, 8]])
        vstack_h5_array(file_name, append_data)

        with h5py.File(file_name, 'r') as f:
            expected = np.vstack((initial_data, append_data))
            self.assertEqual(f["data"].shape[0], expected.shape[0])  # Check shape matches
            np.testing.assert_equal(f["data"][...], expected)  # Check data content

    def test_empty_table_input(self):
        """Test that function handles empty input table without errors."""
        file_name = self.path_output / f"test_vstack.h5"
        empty_table = np.empty((0, 2))

        vstack_h5_array(file_name, empty_table)

        with h5py.File(file_name, 'r') as f:
            self.assertEqual(f["data"].shape[0], 0)  # Verify dataset is empty

    def tearDown(self):
        """Remove all written files."""
        if os.path.isfile(self.path_output / f"test_vstack.h5"):
            os.remove(self.path_output / f"test_vstack.h5")


class TestReadH5Array(CellectsUnitTest):
    """Test suite for read_h5_array function."""

    def test_read_h5_file(self):
        """Test that read_h5_array manage to read."""
        file_name = self.path_input / f"test_vstack.h5"
        key_name = "first_key"
        # Execute function
        read_file = read_h5_array(file_name, key_name)
        self.assertTrue(isinstance(read_file, np.ndarray))


class TestGetH5Keys(CellectsUnitTest):
    """Test suite for get_h5_keys function."""

    def test_get_h5_keys(self):
        """Test that read_h5_array manage to read."""
        file_name = self.path_input / f"test_vstack.h5"
        key_name = "first_key"
        read_key = get_h5_keys(file_name)
        self.assertTrue(key_name == read_key[0])


class TestRemoveH5Keys(CellectsUnitTest):
    """Test suite for remove_h5_key function."""

    def test_remove_h5_key(self):
        """Test that read_h5_array manage to read."""
        file_name = self.path_output / f"test_vstack.h5"
        table1 = np.array([[1, 2], [3, 4]])
        key1 = "first_key"
        table2 = table1[::-1]
        key2 = "second_key"
        # Execute function
        vstack_h5_array(file_name, table1, key1)
        vstack_h5_array(file_name, table2, key2)
        remove_h5_key(file_name, key=key2)
        read_keys = get_h5_keys(file_name)
        self.assertTrue(read_keys[0] == key1)

    def tearDown(self):
        """Remove all written files."""
        if os.path.isfile(self.path_output / f"test_vstack.h5"):
            os.remove(self.path_output / f"test_vstack.h5")

class TestGetMplColormap(CellectsUnitTest):
    """Test suite for get_mpl_colormap."""

    def test_get_mpl_colormap_valid_input(self):
        """Verify correct output shape and data type with valid cmap name."""
        result = get_mpl_colormap("viridis")

        # Validate array structure
        self.assertEqual(result.shape, (256, 1, 3))
        self.assertTrue(np.issubdtype(result.dtype, np.integer))

        # Ensure all values are within byte range [0-255]
        self.assertTrue(np.all((result >= 0) & (result <= 255)))

    def test_get_mpl_colormap_invalid_cmap_name(self):
        """Ensure ValueError is raised for non-existent colormap names."""
        with self.assertRaises(ValueError):
            get_mpl_colormap("invalid_cmap_name")

    def test_get_mpl_colormap_alpha_exclusion(self):
        """Confirm RGBA -> RGB conversion in output array."""
        result = get_mpl_colormap("gray")

        # Test shape after alpha channel removal
        self.assertEqual(result.shape, (256, 1, 3))


class TestShow(CellectsUnitTest):
    """Test suite for the `show` function."""
    def test_show_with_interactive_mode_on(self):
        """Test if returns valid object."""
        img = np.random.rand(100, 100)
        fig, ax = show(img, interactive=False, show=False)
        self.assertTrue(isinstance(fig, Figure))

    def tearDown(self):
        """Close all figures."""
        plt.close("all")

class TestSaveFig(CellectsUnitTest):
    """Test suite for save_fig function."""

    def test_create_new_file_with_valid_input(self):
        """Test that a new HDF5 file and dataset is created with valid input."""
        # Setup unique filename in output directory
        file_name = self.path_output / f"test_save_fig.jpg"
        img = np.random.rand(10, 10)
        save_fig(img, file_name)
        self.assertTrue(os.path.isfile(file_name))

    def tearDown(self):
        """Remove all written files."""
        if os.path.isfile(self.path_output / f"test_save_fig.jpg"):
            os.remove(self.path_output / f"test_save_fig.jpg")


class TestDisplayBoxes(CellectsUnitTest):
    """Test suite for display_boxes function."""

    def test_display_boxes(self):
        """Test if returns valid object."""
        binary_image = np.random.rand(10, 10)
        line_nb = display_boxes(binary_image, box_diameter=2, show=False)
        self.assertTrue(line_nb == 12)

    def tearDown(self):
        """Close all figures."""
        plt.close("all")


class TestExtractTime(CellectsUnitTest):
    """Test suite for extract_time function."""
    def test_extract_time_with_valid_images_same_timestamp(self):
        """Test extract_time with valid images having same timestamp."""
        # Setup test data - would create actual image files in real tests
        image_list = ["image1.tif", "image2.tif"]
        expected_time = np.array([0, 0])  # Assuming all timestamps are same

        # Execute function
        result = extract_time(image_list, self.path_experiment)
        # Verify result
        self.assertTrue(np.array_equal(result, expected_time))


class TestReadOneArena(CellectsUnitTest):
    """Test suite for read_one_arena function."""
    def test_read_one_arena_with_videos_in_ram(self):
        """Test read_one_arena with videos in ram."""
        csc_dict = {'bgr': np.array([0, 0, 1]), 'logical': 'None'}
        videos_already_in_ram = [np.zeros(0), np.zeros(0)]
        visu, converted_video, converted_video2 = read_one_arena(arena_label=1, already_greyscale=True,
                                                                 csc_dict=csc_dict,
                                                                 videos_already_in_ram=videos_already_in_ram)
        visu, converted_video, converted_video2 = read_one_arena(arena_label=1, already_greyscale=False,
                                                                 csc_dict=csc_dict,
                                                                 videos_already_in_ram=videos_already_in_ram)
        self.assertTrue(converted_video2 is None)
        csc_dict = {'bgr': np.array([0, 0, 1]), 'logical': 'Or'}
        videos_already_in_ram = [np.zeros(0), np.zeros(0), np.zeros(0)]
        visu, converted_video, converted_video2 = read_one_arena(arena_label=1, already_greyscale=False,
                                                                 csc_dict=csc_dict,
                                                                 videos_already_in_ram=videos_already_in_ram)
        self.assertTrue(converted_video2 is not None)

    def test_read_one_arena_with_vid_name(self):
        """Test read_one_arena with a video on the disk."""
        os.chdir(self.path_input)
        csc_dict = {'bgr': np.array([0, 0, 1]), 'logical': 'None'}
        vid_name = "test_read_video.mp4"
        visu, converted_video, converted_video2 = read_one_arena(arena_label=1, already_greyscale=False,
                                                                 csc_dict=csc_dict, vid_name=vid_name)
        self.assertIsInstance(visu, np.ndarray)
        self.assertTrue(converted_video is None)
        csc_dict = {'bgr': np.array([0, 0, 1]), 'logical': 'Or', 'hsv2': np.array([0, 1, 0])}
        visu, converted_video, converted_video2 = read_one_arena(arena_label=1, already_greyscale=True,
                                                                 csc_dict=csc_dict, vid_name=vid_name)
        self.assertIsInstance(converted_video, np.ndarray)
        self.assertTrue(visu is None)




if __name__ == '__main__':
    unittest.main()

