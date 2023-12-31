#!/usr/bin/env python3
"""
This script contains all unit tests of the load_display_save script
18 tests
"""

import unittest
from cellects.test.cellects_unit_test import CellectsUnitTest
from cellects.utils.load_display_save import *
from numpy import zeros, uint8, float32, random, array, testing, array_equal, allclose
from cv2 import imwrite, rotate, ROTATE_90_COUNTERCLOCKWISE, ROTATE_90_CLOCKWISE, VideoCapture, CAP_PROP_FRAME_WIDTH, CAP_PROP_FRAME_HEIGHT, CAP_PROP_FPS


class TestPickleRick(CellectsUnitTest):

    def test_write_file_success(self):
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
        # Create an instance of PickleRick
        pickle_rick = PickleRick(10)

        # Define test data
        file_content = "This is some test content"
        file_name = self.path_output / "test_file.pkl"

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


class TestSee(CellectsUnitTest):

    def test_see_with_uint8_image(self):
        # Create a uint8 image for testing
        image = zeros((100, 100), dtype=uint8)
        img_name = "test_image"
        size = (500, 500)

        # Call the See function
        See(image, img_name=img_name, size=size, keep_display=1)

        # Add assertions to verify the expected behavior, if applicable
        # ...

    def test_see_with_non_uint8_image(self):
        # Create a non-uint8 image for testing
        image = zeros((100, 100), dtype=float32)
        img_name = "test_image"
        size = (500, 500)

        # Call the See function
        See(image, img_name=img_name, size=size, keep_display=1)


class TestWriteVideo(CellectsUnitTest):

    def test_write_video_with_mp4_extension(self):
        # Create a temporary mp4 file
        with open(self.path_output / 'test_write_video.mp4', 'wb') as temp_file:
            # Generate a sample numpy array
            np_array = random.randint(0, 255, size=(10, 480, 640, 3), dtype=uint8)

            # Write the video
            write_video(np_array, temp_file.name)

            # Verify that the file exists
            self.assertTrue(os.path.exists(temp_file.name))

    def test_write_video_with_unknown_extension(self):
        # Create a temporary file without a recognized extension
        with open(self.path_output / 'test_write_video.xyz', 'wb') as temp_file:
            # Generate a sample numpy array
            np_array = random.randint(0, 255, size=(10, 480, 640, 3), dtype=uint8)

            # Write the video
            write_video(np_array, temp_file.name)
            new_name = temp_file.name[:-4]
            new_name += '.mp4'
            # Verify that the file exists
            self.assertTrue(os.path.exists(new_name))


    def test_write_video_with_npy_extension(self):
        # Create a temporary mp4 file
        with open(self.path_output / 'test_write_video.npy', 'wb') as temp_file:
            # Generate a sample numpy array
            np_array = random.randint(0, 255, size=(10, 480, 640, 3), dtype=uint8)

            # Write the video
            write_video(np_array, temp_file.name)

            # Verify that the file exists
            self.assertTrue(os.path.exists(temp_file.name))


    def test_write_video_dimensions_and_fps(self):
        # Create a temporary mp4 file
        with open(self.path_output / 'test_write_video.mp4', 'wb') as temp_file:
            # Generate a sample numpy array
            np_array = random.randint(0, 255, size=(10, 480, 640, 3), dtype=uint8)
            fps = 30

            # Write the video
            write_video(np_array, temp_file.name, fps=fps)

            # Read the video back
            vid = VideoCapture(temp_file.name)
            
            # Verify the video dimensions
            self.assertEqual(vid.get(CAP_PROP_FRAME_WIDTH), np_array.shape[2])
            self.assertEqual(vid.get(CAP_PROP_FRAME_HEIGHT), np_array.shape[1])
            self.assertEqual(vid.get(CAP_PROP_FPS), fps)

    def tearDown(self):
        if os.path.isfile(self.path_output / 'test_write_video.mp4'):
            os.remove(self.path_output / 'test_write_video.mp4')
        if os.path.isfile(self.path_output / 'test_write_video.npy'):
            os.remove(self.path_output / 'test_write_video.npy')
        if os.path.isfile(self.path_output / 'test_write_video.xyz'):
            os.remove(self.path_output / 'test_write_video.xyz')


class TestVideo2Numpy(CellectsUnitTest):

    def test_video2numpy_with_npy_extension(self):
        # Create a temporary npy file
        with open(self.path_output / 'test_write_video.npy', 'wb') as temp_file:
            # Generate a sample numpy array
            np_array = random.randint(0, 255, size=(10, 480, 640, 3), dtype=uint8)
            save(temp_file.name, np_array)

            # Read the video
            video = video2numpy(temp_file.name)

            # Verify the dimensions of the video
            self.assertEqual(video.shape, np_array.shape)

    def test_video2numpy_with_video_file(self):
        # Create a temporary mp4 file
        with open(self.path_output / 'test_write_video.mp4', 'wb') as temp_file:
            # Generate a sample numpy array
            np_array = random.randint(0, 255, size=(10, 480, 640, 3), dtype=uint8)

            # Write the video
            write_video(np_array, temp_file.name)

            # Read the video
            video = video2numpy(temp_file.name)

            # Verify the dimensions of the video
            self.assertEqual(video.shape, np_array.shape)

    def test_video2numpy_with_conversion_dict(self):
        # Create a temporary mp4 file
        with open(self.path_output / 'test_write_video.mp4', 'wb') as temp_file:
            # Generate a sample numpy array
            np_array = random.randint(0, 255, size=(10, 480, 640, 3), dtype=uint8)

            # Write the video
            write_video(np_array, temp_file.name)

            # Define the conversion dictionary
            conversion_dict = {'hsv': array((0, 1, 0))}

            # Read the video with conversion
            video, converted_video = video2numpy('test_write_video.mp4', conversion_dict)
            video, converted_video = video2numpy(temp_file.name, conversion_dict)

            # Verify the dimensions of the videos
            self.assertEqual(video.shape, np_array.shape)
            self.assertEqual(converted_video.shape, (10, 480, 640))

    def test_video2numpy_with_true_frame_width(self):
        # Create a temporary mp4 file with double width
        with open(self.path_output / 'test_write_video.mp4', 'wb') as temp_file:
            # Generate a sample numpy array with double width
            np_array = random.randint(0, 255, size=(10, 480, 1280, 3), dtype=uint8)

            # Write the video
            write_video(np_array, temp_file.name)

            # Specify the true frame width
            true_frame_width = 640

            # Read the video with true frame width
            video = video2numpy(temp_file.name, true_frame_width=true_frame_width)

            # Verify the dimensions of the video
            self.assertEqual(video.shape, (10, 480, 640, 3))

    def tearDown(self):
        if os.path.isfile(self.path_output / 'test_write_video.mp4'):
            os.remove(self.path_output / 'test_write_video.mp4')
        if os.path.isfile(self.path_output / 'test_write_video.npy'):
            os.remove(self.path_output / 'test_write_video.npy')


class TestMovie(CellectsUnitTest):

    def test_movie(self):
        np_array = random.randint(0, 255, size=(10, 480, 640), dtype=uint8)
        movie(np_array, keyboard=1)



class TestIsRawImage(CellectsUnitTest):

    def test_is_raw_image_with_raw_format(self):
        # Mock the image_path with a raw format extension
        image_path = self.path_input / "IMG_9731.cr2"

        is_raw = is_raw_image(str(image_path))

        # Verify the expected result
        self.assertTrue(is_raw)

    def test_is_raw_image_with_non_raw_format(self):
        # Mock the image_path with a non-raw format extension
        image_path = self.path_input / "last_original_img.jpg"

        is_raw = is_raw_image(str(image_path))

        # Verify the expected result
        self.assertFalse(is_raw)


class TestReadImage(CellectsUnitTest):

    def test_readim_with_raw_image(self):

        image_path = self.path_input / "IMG_9731.cr2"
        raw_image = True

        # Call the function
        result = readim(str(image_path), raw_image)

        ref_size = array((3670, 5496, 3))
        # Verify the expected result
        self.assertTrue(array_equal(ref_size, result.shape))


    def test_readim_with_regular_image(self):

        image_path = self.path_input / "last_original_img.tif"
        raw_image = False

        # Call the function
        result = readim(str(image_path), raw_image)

        ref_size = array((995, 1003, 3))
        # Verify the expected result
        self.assertTrue(array_equal(ref_size, result.shape))



class TestReadAndRotate(CellectsUnitTest):

    def test_read_and_rotate_orientation_correction(self):

        image_1 = self.path_experiment / "IMG_7653.jpg"
        image_2 = self.path_experiment / "IMG_7665.jpg"
        raw_images = False
        is_landscape = True

        # Call the function
        im1 = read_and_rotate(str(image_1), None, raw_images, is_landscape)
        im2 = read_and_rotate(str(image_2), im1, raw_images, is_landscape)
        im2a = rotate(im2, ROTATE_90_CLOCKWISE)
        im2b = rotate(im2, ROTATE_90_COUNTERCLOCKWISE)

        im2a_path = str(self.path_output / 'im2a.jpg')
        im2b_path = str(self.path_output / 'im2b.jpg')
        imwrite(im2a_path, im2a)
        imwrite(im2b_path, im2b)
        im2a = read_and_rotate(im2a_path, im1, raw_images, is_landscape)
        im2b = read_and_rotate(im2b_path, im1, raw_images, is_landscape)

        self.assertTrue(allclose(im2, im2a, atol=15))
        self.assertTrue(allclose(im2, im2b, atol=15))


if __name__ == '__main__':
    unittest.main()

