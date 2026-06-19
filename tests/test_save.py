#!/usr/bin/env python3
"""
Test module for the `io.save` module in Cellects.

This test suite covers file saving operations. Test cases include success scenarios,
edge case handling for different file extensions (.mp4, .h5, etc.), and dimensional/property verification
for videos and images.

Notes
-----
All test classes inherit from CellectsUnitTest, providing shared setup/teardown for temporary files.
Test files are automatically cleaned up in tearDown methods.
"""
import unittest
from tests._base import CellectsUnitTest
from cellects.io.save import *


class TestWriteVideo(CellectsUnitTest):
    """Test suite for write_video function."""

    def test_write_video_with_mp4_extension(self):
        """Test as .mp4 file."""
        # Create a temporary mp4 file
        with open(self.path_output + '/' + 'test_write_video.mp4', 'wb') as temp_file:
            # Generate a sample numpy array
            np_array = np.random.randint(0, 255, size=(5, 10, 10, 3), dtype=np.uint8)

            # Write the video
            write_video(np_array, temp_file.name)

            # Verify that the file exists
            self.assertTrue(os.path.exists(temp_file.name))

    def test_write_video_with_unknown_extension(self):
        """Test with unknown extension."""
        # Create a temporary file without a recognized extension
        with open(self.path_output + '/' + 'test_write_video.xyz', 'wb') as temp_file:
            # Generate a sample numpy array
            np_array = np.random.randint(0, 255, size=(5, 10, 10, 3), dtype=np.uint8)

            # Write the video
            write_video(np_array, temp_file.name)
            new_name = temp_file.name[:-4]
            new_name += '.mp4'
            # Verify that the file exists
            self.assertTrue(os.path.exists(new_name))

    def test_write_video_with_h5_extension(self):
        """Test with .h5 extension."""
        # Create a temporary h5 file
        np_array = np.random.randint(0, 255, size=(5, 10, 10, 3), dtype=np.uint8)
        write_h5(self.path_output + '/' + 'test_write_video.h5', np_array)
        self.assertTrue(os.path.exists(self.path_output + '/' + 'test_write_video.h5'))

    def test_write_video_dimensions_and_fps(self):
        """Test with dim and fps."""
        # Create a temporary mp4 file
        with open(self.path_output + '/' + 'test_write_video.mp4', 'wb') as temp_file:
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
        if os.path.isfile(self.path_output + '/' + 'test_write_video.mp4'):
            os.remove(self.path_output + '/' + 'test_write_video.mp4')
        if os.path.isfile(self.path_output + '/' + 'test_write_video.h5'):
            os.remove(self.path_output + '/' + 'test_write_video.h5')
        if os.path.isfile(self.path_output + '/' + 'test_write_video.xyz'):
            os.remove(self.path_output + '/' + 'test_write_video.xyz')


class TestVStackH5Array(CellectsUnitTest):
    """Test suite for vstack_h5_array function."""

    def test_create_new_file_with_valid_input(self):
        """Test that a new HDF5 file and dataset is created with valid input."""
        # Setup unique filename in output directory
        file_name = self.path_output + '/' + f"test_vstack.h5"
        table = np.array([[1, 2], [3, 4]])

        # Execute function
        vstack_h5_array(file_name, table)

        # Verify result
        with h5py.File(file_name, 'r') as f:
            self.assertIn("data", f)  # Check dataset exists
            np.testing.assert_equal(f["data"][...], table)  # Check data matches

    def test_append_existing_dataset(self):
        """Test that new table is appended to existing dataset."""
        file_name = self.path_output + '/' + f"test_vstack.h5"

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
        file_name = self.path_output + '/' + f"test_vstack.h5"
        empty_table = np.empty((0, 2))

        vstack_h5_array(file_name, empty_table)

        with h5py.File(file_name, 'r') as f:
            self.assertEqual(f["data"].shape[0], 0)  # Verify dataset is empty

    def tearDown(self):
        """Remove all written files."""
        if os.path.isfile(self.path_output + '/' + f"test_vstack.h5"):
            os.remove(self.path_output + '/' + f"test_vstack.h5")


class TestRemoveH5Keys(CellectsUnitTest):
    """Test suite for remove_h5_key function."""

    def test_remove_h5_key(self):
        """Test that read_h5 manage to read."""
        file_name = self.path_output + '/' + f"test_vstack.h5"
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
        if os.path.isfile(self.path_output + '/' + f"test_vstack.h5"):
            os.remove(self.path_output + '/' + f"test_vstack.h5")


class TestWriteVideoFromImages(CellectsUnitTest):
    """Test suite for write_video_from_images function."""
    def test_write_video_from_images(self):
        """Test write_video_from_images basic behavior."""
        write_video_from_images(self.path_experiment)

    def tearDown(self):
        """Remove all written files."""
        if os.path.isfile(self.path_experiment + '/' + f"timelapse.mp4"):
            os.remove(self.path_experiment + '/' + f"timelapse.mp4")


class TestSaveFig(CellectsUnitTest):
    """Test suite for save_fig function."""

    def test_create_new_file_with_valid_input(self):
        """Test that a new HDF5 file and dataset is created with valid input."""
        # Setup unique filename in output directory
        file_name = self.path_output + '/' + f"test_save_fig.jpg"
        img = np.random.rand(10, 10)
        save_im(img, file_name)
        self.assertTrue(os.path.isfile(file_name))

    def tearDown(self):
        """Remove all written files."""
        if os.path.isfile(self.path_output + '/' + f"test_save_fig.jpg"):
            os.remove(self.path_output + '/' + f"test_save_fig.jpg")


if __name__ == '__main__':
    unittest.main()
