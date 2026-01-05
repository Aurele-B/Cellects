
import unittest
from cellects.image_analysis.network_functions import *
from tests._base import CellectsUnitTest

# --- Tests -------------------------------------------------------------------

class TestNetworkDetection(CellectsUnitTest):
    """Test suite for get_best_network_detection_method() method"""
    @classmethod
    def setUpClass(cls):
        """Setup test fixtures."""
        super().setUpClass()
        cls.dims = (100, 100)
        cls.possibly_filled_pixels = np.random.randint(0, 2, cls.dims, dtype=np.uint8)
        cls.possibly_filled_pixels = keep_one_connected_component(cls.possibly_filled_pixels)
        cls.origin_to_add = np.zeros(cls.dims, dtype=np.uint8)
        mid = cls.dims[0] // 2
        ite = 2
        while not cls.origin_to_add.any():
            ite += 1
            cls.origin_to_add[mid - ite: mid + ite, mid - ite: mid + ite] = cls.possibly_filled_pixels[mid - ite: mid + ite, mid - ite: mid + ite]
        cls.greyscale_image = cls.possibly_filled_pixels.copy()
        cls.greyscale_image[cls.greyscale_image > 0] = np.random.randint(170, 255, cls.possibly_filled_pixels.sum())
        cls.greyscale_image[cls.greyscale_image == 0] = np.random.randint(0, 50, cls.possibly_filled_pixels.size - cls.possibly_filled_pixels.sum())
        cls.add_rolling_window=True
        cls.NetDet = NetworkDetection(cls.greyscale_image, cls.possibly_filled_pixels, cls.add_rolling_window,
                                      cls.origin_to_add)
        cls.NetDet.get_best_network_detection_method()

    def test_get_best_network_detection_method_outputs_proper_method(self):
        """Check that best network detection method outputs proper method"""
        self.assertTrue(isinstance(self.NetDet.best_result['method'], str))

    def test_get_best_network_detection_method_outputs_proper_binary_image(self):
        """Check that best network detection method outputs proper binary image"""
        self.assertTrue(isinstance(self.NetDet.best_result['binary'], np.ndarray))

    def test_get_best_network_detection_method_outputs_proper_quality(self):
        """Check that best network detection method outputs proper quality"""
        self.assertTrue(isinstance(self.NetDet.best_result['quality'], np.float64))

    def test_get_best_network_detection_method_outputs_proper_filtered_image(self):
        """Check that best network detection method outputs proper filtered image"""
        self.assertTrue(isinstance(self.NetDet.best_result['filtered'], np.ndarray))

    def test_get_best_network_detection_method_outputs_proper_rolling_window_option(self):
        """Check that best network detection method outputs proper rolling window option"""
        self.assertTrue(isinstance(self.NetDet.best_result['rolling_window'], bool))

    def test_get_best_network_detection_method_outputs_proper_sigmas(self):
        """Check that best network detection method outputs proper sigmas"""
        self.assertTrue(isinstance(self.NetDet.best_result['sigmas'], list))

    def test_get_best_network_detection_method_finds_consistent_result(self):
        """Check that best network detection method finds a result that contains more ones than the origin image
        and less than the maximum"""
        self.assertTrue(self.origin_to_add.sum() <= self.NetDet.best_result['binary'].sum() < self.possibly_filled_pixels.size)

    def test_detect_network_frangi(self):
        """Check that detect network finds an identical result using the best_result argument"""
        self.NetDet.best_result['filter'] = "Frangi"
        self.NetDet.best_result['rolling_window'] = False
        NetDet_fast = NetworkDetection(self.greyscale_image, possibly_filled_pixels=self.possibly_filled_pixels,
                                       add_rolling_window=self.add_rolling_window,
                                       origin_to_add=self.origin_to_add, best_result=self.NetDet.best_result)
        NetDet_fast.detect_network()
        self.assertTrue(isinstance(NetDet_fast.incomplete_network, np.ndarray))

    def test_detect_network_frangi_rolling(self):
        """Check that detect network finds an identical result using the best_result argument"""
        self.NetDet.best_result['filter'] = "Frangi"
        self.NetDet.best_result['rolling_window'] = True
        NetDet_fast = NetworkDetection(self.greyscale_image, possibly_filled_pixels=self.possibly_filled_pixels,
                                       add_rolling_window=self.add_rolling_window,
                                       origin_to_add=self.origin_to_add, best_result=self.NetDet.best_result)
        NetDet_fast.detect_network()
        self.assertTrue(isinstance(NetDet_fast.incomplete_network, np.ndarray))

    def test_detect_network_frangi(self):
        """Check that detect network finds an identical result using the best_result argument"""
        self.NetDet.best_result['filter'] = "sato"
        self.NetDet.best_result['rolling_window'] = False
        NetDet_fast = NetworkDetection(self.greyscale_image, possibly_filled_pixels=self.possibly_filled_pixels,
                                       add_rolling_window=self.add_rolling_window,
                                       origin_to_add=self.origin_to_add, best_result=self.NetDet.best_result)
        NetDet_fast.detect_network()
        self.assertTrue(isinstance(NetDet_fast.incomplete_network, np.ndarray))

    def test_detect_network_frangi_rolling(self):
        """Check that detect network finds an identical result using the best_result argument"""
        self.NetDet.best_result['filter'] = "sato"
        self.NetDet.best_result['rolling_window'] = True
        NetDet_fast = NetworkDetection(self.greyscale_image, possibly_filled_pixels=self.possibly_filled_pixels,
                                       add_rolling_window=self.add_rolling_window,
                                       origin_to_add=self.origin_to_add, best_result=self.NetDet.best_result)
        NetDet_fast.detect_network()
        self.assertTrue(isinstance(NetDet_fast.incomplete_network, np.ndarray))

    def test_change_greyscale(self):
        """Check that change greyscale function works"""
        img = np.random.randint(255, size=(self.dims[0], self.dims[1], 3), dtype=np.uint8)
        first_dict = {"hsv": np.array([0, 1, 0])}
        previous_greyscale = self.greyscale_image.copy()
        self.NetDet.change_greyscale(img, first_dict)
        self.assertFalse(np.array_equal(previous_greyscale, self.NetDet.greyscale_image))

    def test_detect_pseudopods(self):
        """Check that change greyscale function works"""
        lighter_background = True
        pseudopod_min_width = 1
        pseudopod_min_size = 3
        self.NetDet.detect_pseudopods(lighter_background, pseudopod_min_width, pseudopod_min_size)
        self.assertTrue(isinstance(self.NetDet.pseudopods, np.ndarray))

    def test_detect_pseudopods_dark_back_no_ori(self):
        """Check that change greyscale function works"""
        lighter_background = False
        pseudopod_min_width = 1
        pseudopod_min_size = 3
        self.NetDet.origin_to_add = None
        self.NetDet.detect_pseudopods(lighter_background, pseudopod_min_width, pseudopod_min_size)
        self.assertTrue(isinstance(self.NetDet.pseudopods, np.ndarray))

    def test_merge_network_with_pseudopods(self):
        """Check that change greyscale function works"""
        self.NetDet.merge_network_with_pseudopods()
        expected_complete_network = np.logical_or(self.NetDet.incomplete_network, self.NetDet.pseudopods).astype(np.uint8)
        expected_incomplete_network = self.NetDet.incomplete_network * (1 - self.NetDet.pseudopods)
        self.assertTrue(np.array_equal(self.NetDet.complete_network, expected_complete_network))
        self.assertTrue(np.array_equal(self.NetDet.incomplete_network, expected_incomplete_network))


class TestGetSkeletonAndWidths(CellectsUnitTest):
    """Test suite for get_skeleton_and_widths() skeletonization functionality"""

    def test_get_skeleton_valid_input(self):
        """Test basic skeletonization with valid input array"""
        # Setup: Create a simple 3x3 binary network pattern
        pad_network = np.array([[0, 1, 0],
                                [1, 1, 1],
                                [0, 1, 0]], dtype=np.uint8)

        # Execute function with default parameters
        skeleton, distances, contours = get_skeleton_and_widths(pad_network)

        # Verify output structure and types
        self.assertIsInstance(skeleton, np.ndarray)
        self.assertEqual(skeleton.dtype, np.uint8)
        self.assertIsInstance(distances, np.ndarray)
        self.assertIsNone(contours)

        # Basic shape validation (should maintain original dimensions)
        self.assertEqual(skeleton.shape, pad_network.shape)

    def test_skeleton_with_pad_origin(self):
        """Test skeletonization with optional origin parameter"""
        # Test pattern where we can verify origin contour handling
        network = np.ones((5, 5), dtype=np.uint8)
        pad_network = ad_pad(network)
        pad_origin = np.zeros_like(pad_network)
        pad_origin_centroid = np.array((2, 2))
        pad_origin[pad_origin_centroid[0], pad_origin_centroid[1]] = 1

        skeleton, distances, contours = get_skeleton_and_widths(
            pad_network, pad_origin, pad_origin_centroid)

        self.assertIsNotNone(contours)
        # Expect the origin contour to be a subset of the original network
        self.assertTrue(np.all(contours <= pad_origin))

    def test_skeleton_with_not_overlapping_origin(self):
        """Test skeletonization when origin does not overlap the network"""
        # Test pattern where we can verify origin contour handling
        network = np.ones((5, 5), dtype=np.uint8)
        pad_network = ad_pad(network)
        pad_origin = np.zeros_like(pad_network)
        pad_origin_centroid = np.array((2, 2))
        pad_origin[pad_origin_centroid[0], pad_origin_centroid[1]] = 1
        pad_network[pad_origin_centroid[0], pad_origin_centroid[1]] = 0

        skeleton, distances, contours = get_skeleton_and_widths(
            pad_network, pad_origin, pad_origin_centroid)
        self.assertTrue(skeleton.any())

    def test_skeleton_with_disconnected_origin(self):
        """Test skeletonization with origin containing two shapes"""

        # Test pattern where we can verify origin contour handling
        network = np.ones((5, 5), dtype=np.uint8)
        pad_network = ad_pad(network)
        pad_origin = np.zeros_like(pad_network)
        pad_origin_centroid = np.array((2, 2))
        pad_origin[pad_origin_centroid[0], pad_origin_centroid[1]] = 1
        pad_origin[pad_origin_centroid[0]+3, pad_origin_centroid[1]] = 1

        skeleton, distances, contours = get_skeleton_and_widths(
            pad_network, pad_origin, pad_origin_centroid)

        self.assertIsNotNone(contours)
        self.assertTrue(np.any(np.isnan(distances)))

    def test_empty_network(self):
        """Test skeletonization with empty input array"""
        empty_array = np.zeros((10, 10), dtype=np.uint8)

        # Should return all-zero arrays without errors
        skeleton, distances, contours = get_skeleton_and_widths(empty_array)

        self.assertTrue(np.all(skeleton == 0))
        self.assertTrue(np.all(distances == 0))
        self.assertIsNone(contours)

    def test_single_pixel_network(self):
        """Test edge case with single pixel network"""
        single_pixel = np.zeros((3, 3), dtype=np.uint8)
        single_pixel[1, 1] = 1

        skeleton, distances, contours = get_skeleton_and_widths(single_pixel)

        self.assertEqual(np.sum(skeleton), 1)  # Should preserve the single pixel
        self.assertTrue(np.all(distances[skeleton == 0] == 0))

    def test_skeleton_connectivity(self):
        """Test that only one connected component is preserved"""
        # Create a pattern with two separate components
        pad_network = np.zeros((6, 6), dtype=np.uint8)
        pad_network[1:3, 1:3] = 1  # Top-left quadrant
        pad_network[2:4, 2:4] = 1  # Center quadrant
        pad_network[3:5, 3:5] = 1    # Bottom-right quadrant

        skeleton, _, _ = get_skeleton_and_widths(pad_network)

        expected = np.array([[0, 0, 0, 0, 0, 0],
                                   [0, 1, 0, 0, 0, 0],
                                   [0, 0, 1, 0, 0, 0],
                                   [0, 0, 0, 1, 0, 0],
                                   [0, 0, 0, 0, 1, 0],
                                   [0, 0, 0, 0, 0, 0]], dtype=np.uint8)

        self.assertTrue(np.array_equal(skeleton, expected))  # background + single component


class TestGetTerminationsAndConnectedNodes(unittest.TestCase):
    def test_various_patterns(self):
        cases = [
            ("plus / cross_33", cross_33.copy(), cross_33.copy()),

            ("asterisk",
             np.array([[1,0,1],[0,1,0],[1,0,1]], dtype=np.uint8),
             np.array([[1,0,1],[0,0,0],[1,0,1]], dtype=np.uint8)),

            ("asymmetric plus",
             np.array([[0,0,0,1,0],
                       [0,0,0,1,0],
                       [1,1,1,1,1],
                       [0,0,0,1,0]], dtype=np.uint8),
             np.array([[0,0,0,1,0],
                       [0,0,0,0,0],
                       [1,0,0,1,1],
                       [0,0,0,1,0]], dtype=np.uint8)),

            ("short tripod",
             np.array([[0,1,0],
                       [1,1,0],
                       [0,0,1]], dtype=np.uint8),
             np.array([[0,1,0],
                       [1,0,0],
                       [0,0,1]], dtype=np.uint8)),

            ("long tripod",
             np.array([[0,0,0,1,0,0],
                       [0,0,0,1,0,0],
                       [1,1,1,1,0,0],
                       [0,0,0,0,1,0],
                       [0,0,0,0,0,1]], dtype=np.uint8),
             np.array([[0,0,0,1,0,0],
                       [0,0,0,0,0,0],
                       [1,0,0,0,0,0],
                       [0,0,0,0,0,0],
                       [0,0,0,0,0,1]], dtype=np.uint8)),

            ("twisted branch",
             np.array([[0,1,0],
                       [0,1,0],
                       [1,1,1],
                       [0,0,1]], dtype=np.uint8),
             np.array([[0,1,0],
                       [0,0,0],
                       [1,1,0],
                       [0,0,1]], dtype=np.uint8)),

            ("long twisted branch",
             np.array([[1,0,1,0],
                       [0,1,0,0],
                       [0,1,0,0],
                       [0,1,1,0],
                       [1,0,0,1]], dtype=np.uint8),
             np.array([[1,0,1,0],
                       [0,0,0,0],
                       [0,0,0,0],
                       [0,0,0,0],
                       [1,0,0,1]], dtype=np.uint8)),

            ("strange line",
             np.array([[0,0,0,1],
                       [0,0,1,0],
                       [0,1,0,0],
                       [1,1,0,0],
                       [1,0,0,0],
                       [1,0,0,0]], dtype=np.uint8),
             np.array([[0,0,0,1],
                       [0,0,0,0],
                       [0,0,0,0],
                       [0,0,0,0],
                       [0,0,0,0],
                       [1,0,0,0]], dtype=np.uint8)),

            ("long cross with strange lines 1",
             np.array([[1,0,0,1,0],
                               [1,1,0,1,0],
                               [0,0,1,0,0],
                               [0,0,1,1,1],
                               [1,1,0,0,0]], dtype=np.uint8),
             np.array([[1,0,0,1,0],
                           [0,0,0,0,0],
                           [0,0,0,0,0],
                           [0,0,0,0,1],
                           [1,0,0,0,0]], dtype=np.uint8)),

            ("long cross with strange lines 2",
             np.array([[0,0,1,0,0],
                       [1,1,0,0,0],
                       [0,1,0,0,0],
                       [0,0,1,0,0],
                       [0,0,0,1,0],
                       [0,0,1,0,1]], dtype=np.uint8),
             np.array([[0,0,1,0,0],
                       [1,0,0,0,0],
                       [0,0,0,0,0],
                       [0,0,0,0,0],
                       [0,0,0,0,0],
                       [0,0,1,0,1]], dtype=np.uint8)),

            ("loop (no terminations)",
             np.array([[0,1,1,1,1,0],
                       [1,0,0,0,0,1],
                       [0,1,0,0,0,1],
                       [0,0,1,1,1,0]], dtype=np.uint8),
             np.zeros((4,6), dtype=np.uint8)),
            ("bigger network",
             np.array([[1,0,1,1,1,0,0,1,0,0],
                       [0,0,0,1,0,0,0,1,0,0],
                       [0,1,1,1,1,1,1,1,0,0],
                       [0,0,0,1,0,1,0,1,0,0],
                       [0,0,0,0,1,0,0,1,0,0],
                       [0,0,0,1,0,1,0,1,0,0],
                       [0,0,1,0,0,0,0,1,0,0]], dtype=np.uint8),
             np.array([[0,0,1,0,1,0,0,1,0,0],
                 [0,0,0,0,0,0,0,0,0,0],
                 [0,1,0,0,0,0,0,0,0,0],
                 [0,0,0,0,0,0,0,0,0,0],
                 [0,0,0,0,0,0,0,0,0,0],
                 [0,0,0,0,0,1,0,0,0,0],
                 [0,0,1,0,0,0,0,1,0,0]], dtype=np.uint8)),
        ]

        for name, skeleton, target in cases:
            with self.subTest(name=name):
                pad_skeleton = ad_pad(skeleton)
                cnv4, cnv8 = get_neighbor_comparisons(pad_skeleton)
                pad_terminations = get_terminations_and_their_connected_nodes(pad_skeleton, cnv4, cnv8)
                vertices = un_pad(pad_terminations)
                self.assertTrue(np.array_equal(vertices, target))

    def test_thick_node_is_not_equal_to_given_target(self):
        skeleton = np.array([[0,0,1,0,0],
                             [0,0,1,1,0],
                             [1,1,1,1,1]], dtype=np.uint8)
        target   = np.array([[0,0,1,0,0],
                             [0,0,0,0,0],
                             [1,0,0,0,1]], dtype=np.uint8)
        pad_skeleton = ad_pad(skeleton)
        cnv4, cnv8 = get_neighbor_comparisons(pad_skeleton)
        pad_terminations = get_terminations_and_their_connected_nodes(pad_skeleton, cnv4, cnv8)
        vertices = un_pad(pad_terminations)
        self.assertFalse(np.array_equal(vertices, target))

    def test_non_connectivity_consistency(self):
        skeleton = np.array([
            [1,0,1,1,1,0,0,1,0,0],
            [0,0,0,1,0,0,0,1,0,0],
            [0,1,1,1,1,1,1,1,0,0],
            [0,0,0,1,0,1,0,1,0,0],
            [0,0,0,0,1,0,0,1,0,0],
            [0,0,0,1,0,1,0,1,0,0],
            [0,0,1,0,0,0,0,1,0,0]], dtype=np.uint8)
        pad_skeleton = ad_pad(skeleton)
        cnv4, cnv8 = get_neighbor_comparisons(pad_skeleton)
        pad_tips = get_terminations_and_their_connected_nodes(pad_skeleton, cnv4, cnv8)
        nb, labels = cv2.connectedComponents(pad_tips)
        self.assertEqual(int(pad_tips.sum()), nb - 1)


class TestGetInnerVertices(unittest.TestCase):
    def test_various_patterns_basic(self):
        """Cases where we only compare vertices output."""
        cases = [
            ("plus / cross_33",
             cross_33.copy(),
             cross_33.copy()),

            ("asterisk",
             np.array([[1,0,1],
                       [0,1,0],
                       [1,0,1]], dtype=np.uint8),
             np.array([[1,0,1],
                       [0,1,0],
                       [1,0,1]], dtype=np.uint8)),

            ("short tripod",
             np.array([[0,1,0],[1,1,0],[0,0,1]], dtype=np.uint8),
             np.array([[0,1,0],[1,1,0],[0,0,1]], dtype=np.uint8)),

            ("long tripod",
             np.array([[0,0,0,1,0,0],
                       [0,0,0,1,0,0],
                       [1,1,1,1,0,0],
                       [0,0,0,0,1,0],
                       [0,0,0,0,0,1]], dtype=np.uint8),
             np.array([[0,0,0,1,0,0],
                       [0,0,0,0,0,0],
                       [1,0,0,1,0,0],
                       [0,0,0,0,0,0],
                       [0,0,0,0,0,1]], dtype=np.uint8)),

            ("twisted branch",
             np.array([[0,1,0],
                       [0,1,0],
                       [1,1,1],
                       [0,0,1]], dtype=np.uint8),
             np.array([[0,1,0],
                       [0,0,0],
                       [1,1,0],
                       [0,0,1]], dtype=np.uint8)),

            ("long twisted branch",
             np.array([[1,0,1,0],
                       [0,1,0,0],
                       [0,1,0,0],
                       [0,1,1,0],
                       [1,0,0,1]], dtype=np.uint8),
             np.array([[1,0,1,0],
                       [0,1,0,0],
                       [0,0,0,0],
                       [0,1,0,0],
                       [1,0,0,1]], dtype=np.uint8)),

            ("strange line",
             np.array([[0,0,0,1],
                       [0,0,1,0],
                       [0,1,0,0],
                       [1,1,0,0],
                       [1,0,0,0],
                       [1,0,0,0]], dtype=np.uint8),
             np.array([[0,0,0,1],
                       [0,0,0,0],
                       [0,0,0,0],
                       [0,0,0,0],
                       [0,0,0,0],
                       [1,0,0,0]], dtype=np.uint8)),

            ("long cross with strange lines 1",
             np.array([[1,0,0,1,0],
                       [1,1,0,1,0],
                       [0,0,1,0,0],
                       [0,0,1,1,1],
                       [1,1,0,0,0]], dtype=np.uint8),
             np.array([[1,0,0,1,0],
                       [0,0,0,0,0],
                       [0,0,1,0,0],
                       [0,0,0,0,1],
                       [1,0,0,0,0]], dtype=np.uint8)),

            ("long cross with strange lines 2",
             np.array([[0,0,1,0,0],
                       [1,1,0,0,0],
                       [0,1,0,0,0],
                       [0,0,1,0,0],
                       [0,0,0,1,0],
                       [0,0,1,0,1]], dtype=np.uint8),
             np.array([[0,0,1,0,0],
                       [1,1,0,0,0],
                       [0,0,0,0,0],
                       [0,0,0,0,0],
                       [0,0,0,1,0],
                       [0,0,1,0,1]], dtype=np.uint8)),

            ("loop (no vertices)",
             np.array([[0,1,1,1,1,0],
                       [1,0,0,0,0,1],
                       [0,1,0,0,0,1],
                       [0,0,1,1,1,0]], dtype=np.uint8),
             np.zeros((4,6), dtype=np.uint8)),

            ("bigger network",
             np.array([[0,0,1,1,1,0,0,1,0,0],
                       [0,0,0,1,0,0,0,1,0,0],
                       [0,1,1,1,1,1,1,1,0,0],
                       [0,0,0,1,0,1,0,1,0,0],
                       [0,0,0,0,1,0,0,1,0,0],
                       [0,0,0,1,0,1,0,1,0,0],
                       [0,0,1,0,0,0,0,1,0,0]], dtype=np.uint8),
             np.array([ [0,0,1,1,1,0,0,1,0,0],
                               [0,0,0,1,0,0,0,0,0,0],
                               [0,1,0,0,1,0,1,0,0,0],
                               [0,0,0,1,0,1,0,0,0,0],
                               [0,0,0,0,1,0,0,0,0,0],
                               [0,0,0,0,0,1,0,0,0,0],
                               [0,0,1,0,0,0,0,1,0,0]], dtype=np.uint8)),
        ]

        for name, skeleton, v_target in cases:
            with self.subTest(name=name):
                pad = ad_pad(skeleton)
                cnv4, cnv8 = get_neighbor_comparisons(pad)
                potential_tips = get_terminations_and_their_connected_nodes(pad, cnv4, cnv8)
                pad_vertices, _ = get_inner_vertices(pad, potential_tips, cnv4, cnv8)
                vertices = un_pad(pad_vertices)
                self.assertTrue(np.array_equal(vertices, v_target), msg=name)

    def test_asymmetric_plus_checks_vertices_and_tips(self):
        """Original script checked both vertices and tips for this case."""
        skeleton = np.array([[0,0,0,1,0],
                             [0,0,0,1,0],
                             [1,1,1,1,1],
                             [0,0,0,1,0]], dtype=np.uint8)

        v_target = np.array([[0,0,0,1,0],
                             [0,0,0,0,0],
                             [1,0,0,1,1],
                             [0,0,0,1,0]], dtype=np.uint8)

        t_target = np.array([[0,0,0,1,0],
                             [0,0,0,0,0],
                             [1,0,0,0,1],
                             [0,0,0,1,0]], dtype=np.uint8)

        pad = ad_pad(skeleton)
        cnv4, cnv8 = get_neighbor_comparisons(pad)
        potential_tips = get_terminations_and_their_connected_nodes(pad, cnv4, cnv8)
        pad_vertices, pad_tips = get_inner_vertices(pad, potential_tips, cnv4, cnv8)

        vertices = un_pad(pad_vertices)
        tips = un_pad(pad_tips)

        self.assertTrue(np.array_equal(tips, t_target), msg="tips")
        self.assertTrue(np.array_equal(vertices, v_target), msg="vertices")

    def test_thick_node_is_not_equal_to_given_target(self):
        """Original printed 'not np.array_equal(vertices, target)' for this case."""
        skeleton = np.array([[0,0,1,0,0],
                             [0,0,1,1,0],
                             [1,1,1,1,1]], dtype=np.uint8)
        target = np.array([[0,0,1,0,0],
                           [0,0,0,0,0],
                           [1,0,1,0,1]], dtype=np.uint8)

        pad = ad_pad(skeleton)
        cnv4, cnv8 = get_neighbor_comparisons(pad)
        potential_tips = get_terminations_and_their_connected_nodes(pad, cnv4, cnv8)
        pad_vertices, _ = get_inner_vertices(pad, potential_tips, cnv4, cnv8)
        vertices = un_pad(pad_vertices)

        self.assertFalse(np.array_equal(vertices, target))


class TestRemoveSmallLoops(unittest.TestCase):
    def test_remove_loops_and_keep_distances(self):
        # case 1: small loop removal
        skeleton = np.array([
            [0,1,0,1,0],
            [0,0,1,0,0],
            [1,1,0,1,1],
            [0,0,1,0,0],
        ], dtype=np.uint8)
        pad = ad_pad(skeleton)
        pad = remove_small_loops(pad)
        new_skeleton = remove_padding([pad])[0]
        target = np.array([
            [0,1,0,1,0],
            [0,0,1,0,0],
            [1,1,0,1,1],
            [0,0,0,0,0],
        ], dtype=np.uint8)
        self.assertTrue(np.array_equal(new_skeleton, target))

        # case 2: bow loop
        skeleton = np.array([
            [1,0,0,0,0,0,0],
            [0,1,0,0,0,0,0],
            [0,0,1,0,0,0,0],
            [0,1,0,1,1,1,1],
            [0,0,1,0,0,0,0],
            [0,1,0,0,0,0,0],
            [1,0,0,0,0,0,0],
        ], dtype=np.uint8)
        pad = ad_pad(skeleton)
        pad = remove_small_loops(pad)
        new_skeleton = remove_padding([pad])[0]
        target = np.array([
            [1,0,0,0,0,0,0],
            [0,1,0,0,0,0,0],
            [0,0,1,0,0,0,0],
            [0,0,0,1,1,1,1],
            [0,0,1,0,0,0,0],
            [0,1,0,0,0,0,0],
            [1,0,0,0,0,0,0],
        ], dtype=np.uint8)
        self.assertTrue(np.array_equal(new_skeleton, target))

        # case 3: distances modified alongside loop removal
        filled_loops = np.array([
            [1,1,1,0,0,0,0],
            [0,1,1,1,1,0,0],
            [0,1,1,1,1,1,0],
            [0,0,1,1,1,1,0],
            [0,0,0,1,1,1,1],
            [0,0,0,0,0,1,0],
        ], dtype=np.uint8)
        skeleton = np.array([
            [1,0,1,0,0,0,0],
            [0,1,0,1,0,0,0],
            [0,0,1,0,1,0,0],
            [0,0,0,1,0,1,0],
            [0,0,0,0,1,0,1],
            [0,0,0,0,0,1,0],
        ], dtype=np.uint8)
        distances = np.array([
            [2,0,2,0,0,0,0],
            [0,2,0,2,0,0,0],
            [0,0,2,0,2,0,0],
            [0,0,0,2,0,2,0],
            [0,0,0,0,2,0,2],
            [0,0,0,0,0,2,0],
        ], dtype=np.float64)

        pad_skel = ad_pad(skeleton)
        pad_dist = ad_pad(distances)
        pad_skel, pad_dist = remove_small_loops(pad_skel, pad_dist)

        target_dist = np.array([
            [0.,0.,0.,0.,0.,0.,0.,0.,0.],
            [0.,2.,0.,0.,0.,0.,0.,0.,0.],
            [0.,0.,2.,0.,0.,0.,0.,0.,0.],
            [0.,0.,0.,2.,np.nan,0.,0.,0.,0.],
            [0.,0.,0.,0.,0.,np.nan,0.,0.,0.],
            [0.,0.,0.,0.,0.,0.,0.,0.,0.],
            [0.,0.,0.,0.,0.,0.,0.,0.,0.],
            [0.,0.,0.,0.,0.,0.,0.,0.,0.],
        ], dtype=float)
        self.assertTrue(np.array_equal(pad_dist, target_dist, equal_nan=True))

    def test_followup_vertices_after_loop_removal(self):
        """Replicates the script’s post-removal vertex checks."""
        # weird line tips -> vertices target
        skeleton = np.array([
            [1,0,1,0,0,0,0],
            [0,1,0,1,0,0,0],
            [0,0,1,0,1,0,0],
            [0,0,0,1,0,1,0],
            [0,0,0,0,1,0,1],
            [0,0,0,0,0,1,0],
        ], dtype=np.uint8)
        pad_skel = ad_pad(skeleton)
        # distances not needed for this check; rely on remove_small_loops default
        pad_skel = remove_small_loops(pad_skel)
        cnv4, cnv8 = get_neighbor_comparisons(pad_skel)
        potential = get_terminations_and_their_connected_nodes(pad_skel, cnv4, cnv8)
        pad_vertices, _ = get_inner_vertices(pad_skel, potential, cnv4, cnv8)
        vertices = un_pad(pad_vertices)
        target = np.array([
            [1,0,0,0,0,0,0],
            [0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0],
            [0,0,0,0,1,0,0],
            [0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0],
        ], dtype=np.uint8)
        self.assertTrue(np.array_equal(vertices, target))

        # another bow loop, then vertex check
        skeleton = np.array([
            [0,0,0,0,0,0,1],
            [1,1,0,1,0,1,0],
            [0,0,1,0,1,0,0],
            [0,0,0,1,0,0,0],
            [0,0,0,0,1,0,0],
            [0,0,0,0,0,1,0],
        ], dtype=np.uint8)
        pad_skel = ad_pad(skeleton)
        pad_skel = remove_small_loops(pad_skel)
        cnv4, cnv8 = get_neighbor_comparisons(pad_skel)
        potential = get_terminations_and_their_connected_nodes(pad_skel, cnv4, cnv8)
        pad_vertices, _ = get_inner_vertices(pad_skel, potential, cnv4, cnv8)
        vertices = un_pad(pad_vertices)
        target = np.array([
            [0,0,0,0,0,0,1],
            [1,0,0,0,0,0,0],
            [0,0,0,0,0,0,0],
            [0,0,0,1,0,0,0],
            [0,0,0,0,0,0,0],
            [0,0,0,0,0,1,0],
        ], dtype=np.uint8)
        self.assertTrue(np.array_equal(vertices, target))

        # small loop that comes back with medial axis
        skeleton = np.array([
            [0,0,1,0,0,0],
            [0,0,1,0,1,1],
            [1,1,0,1,0,0],
            [0,0,1,0,0,0],
            [0,0,0,1,0,0],
            [0,0,0,1,0,0],
        ], dtype=np.uint8)
        pad_skel = ad_pad(skeleton)
        pad_skel = remove_small_loops(pad_skel)
        new_skeleton = remove_padding([pad_skel])[0]
        target = np.array([
            [0,0,1,0,0,0],
            [0,0,1,0,1,1],
            [1,1,1,1,0,0],
            [0,0,1,0,0,0],
            [0,0,0,1,0,0],
            [0,0,0,1,0,0],
        ], dtype=np.uint8)
        self.assertTrue(np.array_equal(new_skeleton, target))

        # holed cross with large side
        skeleton = np.array([
            [0,0,0,1,0,0,0],
            [0,0,0,1,0,0,0],
            [0,0,0,1,0,0,0],
            [0,0,1,0,1,1,1],
            [0,1,0,1,1,0,0],
            [1,0,0,0,0,1,0],
            [0,0,0,0,0,0,1],
        ], dtype=np.uint8)
        pad_skel = ad_pad(skeleton)
        pad_skel = remove_small_loops(pad_skel)
        new_skeleton = remove_padding([pad_skel])[0]
        target = np.array([
            [0,0,0,1,0,0,0],
            [0,0,0,1,0,0,0],
            [0,0,0,1,0,0,0],
            [0,0,1,1,0,1,1],
            [0,1,0,0,1,0,0],
            [1,0,0,0,0,1,0],
            [0,0,0,0,0,0,1],
        ], dtype=np.uint8)
        self.assertTrue(np.array_equal(new_skeleton, target))

class TestVerticesAndTipsFromSkeleton(unittest.TestCase):
    def test_case_1_grid_tips_isolated(self):
        skeleton = np.array([
            [1,0,1,1,1,0,0,1,0,0],
            [0,0,0,1,0,0,0,1,0,0],
            [0,1,1,1,1,1,1,1,0,0],
            [0,0,0,1,0,1,0,1,0,0],
            [0,0,0,0,1,0,0,1,0,0],
            [0,0,0,1,0,1,0,1,0,0],
            [0,0,1,0,0,0,0,1,0,0],
        ], dtype=np.uint8)
        pad_skeleton = ad_pad(skeleton)
        pad_vertices, pad_tips = get_vertices_and_tips_from_skeleton(pad_skeleton)
        nb, labels = cv2.connectedComponents(pad_tips)
        self.assertEqual((labels > 0).sum(), nb - 1)

    def test_case_2_other_grid_tips_isolated(self):
        skeleton = np.array([
            [0,0,0,0,1,0,0],
            [0,0,0,1,0,0,1],
            [1,0,0,1,0,1,0],
            [0,1,1,1,1,0,0],
            [0,1,0,1,0,0,0],
            [1,0,0,0,1,0,0],
            [0,0,0,0,0,1,1],
        ], dtype=np.uint8)
        pad_skeleton = ad_pad(skeleton)
        pad_vertices, pad_tips = get_vertices_and_tips_from_skeleton(pad_skeleton)
        nb, labels = cv2.connectedComponents(pad_tips)
        self.assertEqual((labels > 0).sum(), nb - 1)

    def test_case_3_variant_grid_tips_isolated(self):
        skeleton = np.array([
            [0,0,0,1,0,0,0],
            [0,0,1,0,0,0,0],
            [0,0,0,1,0,0,0],
            [1,0,1,1,1,0,0],
            [0,1,0,1,0,1,1],
            [0,0,0,1,0,0,0],
            [0,0,0,1,0,0,0],
        ], dtype=np.uint8)
        pad_skeleton = ad_pad(skeleton)
        pad_vertices, pad_tips = get_vertices_and_tips_from_skeleton(pad_skeleton)
        nb, labels = cv2.connectedComponents(pad_tips)
        self.assertEqual((labels > 0).sum(), nb - 1)

    def test_swimming_thing_vertices_target(self):
        skeleton = np.array([
            [0,0,0,0,1,0,0],
            [0,0,0,1,0,0,1],
            [1,1,0,1,0,1,0],
            [0,0,1,1,1,0,0],
            [0,0,0,1,0,0,0],
            [0,0,0,0,1,0,0],
            [0,0,0,0,1,0,0],
        ], dtype=np.uint8)
        pad_skeleton = ad_pad(skeleton)
        pad_skeleton = keep_one_connected_component(pad_skeleton)
        pad_vertices, pad_tips = get_vertices_and_tips_from_skeleton(pad_skeleton)

        vertices = remove_padding([pad_vertices])[0]
        target = np.array([
            [0,0,0,0,1,0,0],
            [0,0,0,0,0,0,1],
            [1,0,0,0,0,0,0],
            [0,0,0,1,0,0,0],
            [0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0],
            [0,0,0,0,1,0,0],
        ], dtype=np.uint8)
        self.assertTrue(np.array_equal(vertices, target))

    def test_false_tip_case_with_target(self):
        skeleton = np.array([
            [0,0,0,1,0,0,0],
            [0,0,1,0,0,0,0],
            [0,0,1,0,1,1,1],
            [1,1,1,1,0,0,0],
            [0,0,1,0,0,0,0],
            [0,0,0,1,0,0,0],
            [0,0,0,1,0,0,0],
        ], dtype=np.uint8)
        pad = ad_pad(skeleton)
        pad = keep_one_connected_component(pad)
        pad_vertices, pad_tips = get_vertices_and_tips_from_skeleton(pad)

        vertices = remove_padding([pad_vertices])[0]
        target = np.array([
            [0,0,0,1,0,0,0],
            [0,0,0,0,0,0,0],
            [0,0,0,0,0,0,1],
            [1,0,1,0,0,0,0],
            [0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0],
            [0,0,0,1,0,0,0],
        ], dtype=np.uint8)
        self.assertTrue(np.array_equal(vertices, target))

    def test_false_tip_misc_1(self):
        skeleton = np.array([
            [1,0,0,0,0,1,0],
            [0,1,0,0,0,1,0],
            [0,0,1,0,1,0,0],
            [0,0,0,1,1,0,0],
            [0,0,1,0,0,1,0],
            [1,1,0,0,0,0,1],
        ], dtype=np.uint8)
        pad_skeleton = ad_pad(skeleton)
        pad_vertices, pad_tips = get_vertices_and_tips_from_skeleton(pad_skeleton)
        vertices = remove_padding([pad_vertices])[0]
        target = np.array([
            [1, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 1],
        ], dtype=np.uint8)
        self.assertTrue(np.array_equal(vertices, target))

    def test_special_cross(self):
        skeleton = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0],
            [1, 1, 1, 1, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 1, 0, 0],
            [0, 0, 1, 1, 0, 0, 0, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
        ], dtype=np.uint8)
        pad_skeleton = ad_pad(skeleton)
        pad_vertices, pad_tips = get_vertices_and_tips_from_skeleton(pad_skeleton)
        vertices = remove_padding([pad_vertices])[0]
        target = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
        ], dtype=np.uint8)
        self.assertTrue(np.array_equal(vertices, target))

    def test_false_tip_misc_3(self):
        skeleton = np.array([
            [0,0,1,0],
            [0,0,1,0],
            [1,1,1,0],
            [0,0,1,0],
            [0,1,1,0],
            [1,0,0,1],
        ], dtype=np.uint8)
        pad_skeleton = ad_pad(skeleton)
        pad_vertices, pad_tips = get_vertices_and_tips_from_skeleton(pad_skeleton)
        vertices = remove_padding([pad_vertices])[0]
        target = np.array([
            [0, 0, 1, 0],
            [0, 0, 0, 0],
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 0],
            [1, 0, 0, 1],
        ], dtype=np.uint8)
        self.assertTrue(np.array_equal(vertices, target))

    def test_false_tip_misc_4(self):
        skeleton = np.array([
            [0,1,0,0,1,0,0],
            [0,0,1,1,0,0,0],
            [0,0,0,1,0,1,1],
            [0,1,1,1,1,0,0],
            [1,0,0,0,0,1,0],
        ], dtype=np.uint8)
        pad = keep_one_connected_component(ad_pad(skeleton))
        pad_vertices, pad_tips = get_vertices_and_tips_from_skeleton(pad)
        vertices = remove_padding([pad_vertices])[0]
        target = np.array([
            [0, 1, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 1],
            [0, 0, 0, 0, 1, 0, 0],
            [1, 0, 0, 0, 0, 1, 0],
        ], dtype=np.uint8)
        self.assertTrue(np.array_equal(vertices, target))

    def test_false_tip_misc_2(self):
        skeleton = np.array([
            [1,0,0,0,1,0,0],
            [0,1,0,1,0,0,0],
            [0,0,1,0,0,1,1],
            [0,1,1,1,1,0,0],
            [1,0,1,0,0,1,0],
            [0,0,1,0,0,0,1],
        ], dtype=np.uint8)
        pad_skeleton = ad_pad(skeleton)
        pad_vertices, pad_tips = get_vertices_and_tips_from_skeleton(pad_skeleton)
        vertices = remove_padding([pad_vertices])[0]
        target = np.array([
            [1, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 1],
            [0, 1, 0, 1, 1, 0, 0],
            [1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 1],
        ], dtype=np.uint8)
        self.assertTrue(np.array_equal(vertices, target))

    def test_false_tip_misc_5(self):
        skeleton = np.array([
            [0,1,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,1,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,1,0,0,0,0,0,0,0,0,0,0,0,0,1],
            [0,1,0,1,0,0,0,1,0,0,0,0,0,1,0],
            [1,0,1,0,1,1,1,0,1,1,1,0,0,1,0],
            [0,0,1,0,0,0,0,0,0,0,0,1,0,1,0],
            [0,0,0,1,0,0,0,0,0,0,0,0,1,0,0],
            [0,0,0,0,1,0,0,0,0,0,0,0,0,1,0],
            [0,0,0,0,0,1,0,0,0,0,0,0,1,0,1],
        ], dtype=np.uint8)
        pad = keep_one_connected_component(ad_pad(skeleton))
        pad_vertices, pad_tips = get_vertices_and_tips_from_skeleton(pad)
        vertices = remove_padding([pad_vertices])[0]
        target = np.array([
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1],
        ], dtype=np.uint8)
        self.assertTrue(np.array_equal(vertices, target))

    def test_false_tip_misc_6(self):
        skeleton = np.array([
            [0,0,1,0,0,0,1,1,1,0,1,0,0],
            [0,0,1,0,0,1,0,0,0,1,0,0,0],
            [0,1,0,0,1,0,0,0,0,1,0,0,0],
            [1,0,0,1,0,0,0,0,0,1,0,0,0],
            [0,1,1,0,1,1,0,0,0,1,1,0,1],
            [0,1,0,0,0,0,1,0,1,0,0,1,0],
            [1,0,0,0,0,0,0,1,0,0,0,0,0],
            [1,0,0,0,0,0,0,1,0,0,0,0,0],
            [1,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,1,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,1,0,0,0,0,0,0,0,0,0,0],
        ], dtype=np.uint8)
        pad = keep_one_connected_component(ad_pad(skeleton))
        pad_vertices, pad_tips = get_vertices_and_tips_from_skeleton(pad)
        vertices = remove_padding([pad_vertices])[0]
        target = np.array([
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ], dtype=np.uint8)
        self.assertTrue(np.array_equal(vertices, target))

    def test_cross_two_edges_branch_1(self):
        skeleton = np.array([
            [0,0,0,1,0,0,0,1,1,1,0,1,0,0],
            [0,0,1,0,0,0,1,0,0,0,1,0,0,0],
            [1,0,0,1,0,1,0,0,0,0,1,0,0,0],
            [0,1,1,1,1,0,0,0,0,0,1,0,0,0],
            [0,0,0,1,0,1,0,0,0,0,1,1,0,1],
            [0,0,1,0,0,0,1,1,0,1,0,0,1,0],
            [1,1,0,0,0,0,0,0,1,0,0,0,0,0],
        ], dtype=np.uint8)
        pad = keep_one_connected_component(ad_pad(skeleton))
        pad_vertices, pad_tips = get_vertices_and_tips_from_skeleton(pad)
        vertices = remove_padding([pad_vertices])[0]
        target = np.array([
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ], dtype=np.uint8)
        self.assertTrue(np.array_equal(vertices, target))

    def test_cross_two_edges_branch_3_large(self):
        skeleton = np.array([
            [0,1,0,0,0,0,0,0,0,0,1,0,0,0,0],
            [0,0,1,0,0,0,0,0,0,1,0,0,0,0,0],
            [0,0,0,1,0,0,0,0,1,0,0,0,0,0,0],
            [0,0,0,1,0,0,0,1,1,0,0,0,0,0,0],
            [1,1,1,1,0,0,1,0,0,1,0,0,0,0,0],
            [0,0,0,0,1,1,0,0,0,1,0,0,0,0,0],
            [0,0,0,0,1,0,0,0,0,0,1,0,0,0,0],
            [0,0,0,1,0,0,0,0,0,0,0,1,0,0,1],
            [0,0,0,1,0,0,0,0,0,0,0,0,1,1,0],
            [0,0,1,0,0,0,0,0,0,0,0,0,1,0,0],
            [0,0,1,0,0,0,0,0,0,0,0,1,0,0,0],
            [0,0,1,0,0,0,0,0,0,0,0,0,1,0,0],
            [0,1,0,1,0,1,0,0,0,0,0,1,1,0,0],
            [1,0,0,0,1,0,1,1,1,1,1,0,1,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,1,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
        ], dtype=np.uint8)
        pad = keep_one_connected_component(ad_pad(skeleton))
        pad_vertices, pad_tips = get_vertices_and_tips_from_skeleton(pad)
        vertices = remove_padding([pad_vertices])[0]
        target = np.array([
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        ], dtype=np.uint8)
        self.assertTrue(np.array_equal(vertices, target))

    def test_cross_two_edges_branch_2(self):
        """ This is the last known case for which the algorithm is not perfect"""
        skeleton = np.array([
            [1,0,0,0,0,0],
            [0,1,0,0,0,0],
            [0,1,0,0,0,1],
            [0,0,1,0,1,0],
            [1,1,1,1,0,0],
            [0,0,0,0,1,1],
        ], dtype=np.uint8)
        pad_skeleton = ad_pad(skeleton)
        pad_vertices, pad_tips = get_vertices_and_tips_from_skeleton(pad_skeleton)
        vertices = un_pad(pad_vertices)
        target = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0],
            [1, 0, 1, 1, 0, 0],
            [0, 0, 0, 0, 0, 1],
        ], dtype=np.uint8)
        self.assertTrue(np.array_equal(vertices, target))

def _coords_to_mask(shape, coords):
    """coords: (N,2) array of [row, col]."""
    m = np.zeros(shape, dtype=np.uint8)
    if coords.size:
        m[coords[:, 0], coords[:, 1]] = 1
    return m

class TestGetBranchesAndTipsCoord(unittest.TestCase):
    def test_tip_and_branching_vertex_coordinates_map(self):
        skeleton = np.array([
            [1,0,1,1,1,0,0,1,0,0],
            [0,0,0,1,0,0,0,1,0,0],
            [0,1,1,1,1,1,1,1,0,0],
            [0,0,0,1,0,1,0,1,0,0],
            [0,0,0,0,1,0,0,1,0,0],
            [0,0,0,1,0,1,0,1,0,0],
            [0,0,1,0,0,0,0,1,0,0],
        ], dtype=np.uint8)

        pad = ad_pad(skeleton)
        pad_vertices, pad_tips = get_vertices_and_tips_from_skeleton(pad)
        non_tip_vertices, tips_coord = get_branches_and_tips_coord(pad_vertices, pad_tips)

        # Rebuild the map (1 for tips, 2 for non-tip/branching vertices)
        vt_map = np.zeros_like(pad)
        if tips_coord.size:
            vt_map[tips_coord[:, 0], tips_coord[:, 1]] = 1
        if non_tip_vertices.size:
            vt_map[non_tip_vertices[:, 0], non_tip_vertices[:, 1]] = 2

        target = np.array([
            [0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,1,2,1,0,0,1,0,0,0],
            [0,0,0,0,2,0,0,0,0,0,0,0],
            [0,0,1,0,0,2,0,2,0,0,0,0],
            [0,0,0,0,2,0,2,0,0,0,0,0],
            [0,0,0,0,0,2,0,0,0,0,0,0],
            [0,0,0,0,0,0,1,0,0,0,0,0],
            [0,0,0,1,0,0,0,0,1,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0],
        ], dtype=np.uint8)

        self.assertTrue(np.array_equal(vt_map, target))


class TestEdgeIdentification(CellectsUnitTest):
    """Test suite for EdgeIdentification class."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        # Create minimal valid input data
        self.valid_skeleton = ad_pad(np.array([[0, 0, 1, 0, 1],
                                                    [1, 1, 1, 1, 0],
                                                    [0, 0, 1, 0, 1],
                                                    [0, 0, 1, 0, 1]], dtype=np.uint8))
        self.valid_distances = ad_pad(np.array([[0.0, 0.0, 1.0, 0.0, 1.0],
                                                     [1.0, 1.0, 1.0, 1.0, 0.0],
                                                     [0.0, 0.0, 1.0, 0.0, 1.0],
                                                     [0.0, 0.0, 1.0, 0.0, 1.0]], dtype=np.float64))
        self.dims = self.valid_skeleton.shape

        # Create empty skeleton
        self.empty_skeleton = np.zeros(self.dims, dtype=np.uint8)
        self.empty_distances = np.zeros(self.dims, dtype=np.float64)

    def test_init_with_valid_inputs(self):
        """Test that EdgeIdentification initializes with valid inputs."""
        edge_id = EdgeIdentification(self.valid_skeleton, self.valid_distances)
        self.assertEqual(edge_id.pad_skeleton.shape, self.dims)
        self.assertEqual(edge_id.pad_distances.shape, self.dims)
        self.assertIsNone(edge_id.remaining_vertices)
        self.assertIsNone(edge_id.vertices)
        self.assertIsNone(edge_id.growing_vertices)

    def test_init_with_empty_skeleton(self):
        """Test initialization with empty skeleton."""
        edge_id = EdgeIdentification(self.empty_skeleton, self.empty_distances)
        self.assertEqual(edge_id.pad_skeleton.sum(), 0)  # All zeros
        self.assertEqual(edge_id.pad_distances.sum(), 0)  # All zeros

    def test_get_vertices_and_tips_coord_with_valid_skeleton(self):
        """Test vertex and tip extraction with valid skeleton."""
        edge_id = EdgeIdentification(self.valid_skeleton, self.valid_distances)
        edge_id.get_vertices_and_tips_coord()
        # Should have some vertices and tips
        self.assertIsNotNone(edge_id.non_tip_vertices)
        self.assertIsNotNone(edge_id.tips_coord)

    def test_get_tipped_edges_with_valid_skeleton(self):
        """Test tipped edge extraction with valid skeleton."""
        edge_id = EdgeIdentification(self.valid_skeleton, self.valid_distances)
        # First get vertices and tips
        edge_id.get_vertices_and_tips_coord()
        self.assertTrue(np.array_equal(edge_id.non_tip_vertices, np.array([[2, 3]], dtype=np.int64)))
        self.assertTrue(np.array_equal(edge_id.tips_coord, np.array([[1, 3], [1, 5], [2, 1], [4, 3], [4, 5]], dtype=np.int64)))

        # Then get tipped edges
        edge_id.get_tipped_edges()
        self.assertTrue(np.array_equal(edge_id.edge_lengths, np.array([1., 2., 2., 2., 3.], dtype=np.float64)))

    def test_remove_tipped_edge_smaller_than_branch_width(self):
        """Test removal of short tipped edges."""
        edge_id = EdgeIdentification(self.valid_skeleton, self.valid_distances)
        edge_id.get_vertices_and_tips_coord()
        edge_id.get_tipped_edges()

        # Remove short edges
        edge_id.remove_tipped_edge_smaller_than_branch_width()
        expected = np.array([[0, 0, 0, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 1, 0],
                                   [0, 1, 1, 1, 1, 0, 0],
                                   [0, 0, 0, 1, 0, 1, 0],
                                   [0, 0, 0, 1, 0, 1, 0],
                                   [0, 0, 0, 0, 0, 0, 0]], dtype=np.uint8)

        self.assertTrue(np.array_equal(edge_id.pad_skeleton, expected))

    def test_remove_tipped_edge_smaller_than_branch_width_when_new_tip_is_connected_to_a_vertex(self):
        """Test removal of short tipped edges in the particular case where removing tips create a tip that is
        connected to another vertex."""
        valid_skeleton = ad_pad(np.array([[0, 1, 0, 1, 0],
                                               [0, 0, 1, 0, 0],
                                               [0, 0, 1, 0, 0],
                                               [0, 1, 0, 1, 0],
                                               [1, 0, 0, 0, 1],
                                               [0, 1, 0, 1, 0]], dtype=np.uint8))
        valid_distances = ad_pad(np.array([[0, 1, 0, 1, 0],
                                               [0, 0, 1, 0, 0],
                                               [0, 0, 1, 0, 0],
                                               [0, 1, 0, 1, 0],
                                               [1, 0, 0, 0, 1],
                                               [0, 1, 0, 1, 0]], dtype=np.float64))
        edge_id = EdgeIdentification(valid_skeleton, valid_distances)
        edge_id.get_vertices_and_tips_coord()
        edge_id.get_tipped_edges()

        # Remove short edges + the newly created tip, that is only one pixel long
        edge_id.remove_tipped_edge_smaller_than_branch_width()
        expected = ad_pad(np.array([[0, 0, 0, 0, 0],
                                         [0, 0, 0, 0, 0],
                                         [0, 0, 1, 0, 0],
                                         [0, 1, 0, 1, 0],
                                         [1, 0, 0, 0, 1],
                                         [0, 1, 0, 1, 0]], dtype=np.uint8))

        self.assertTrue(np.array_equal(edge_id.pad_skeleton, expected))


    def test_remove_tipped_edge_smaller_than_branch_width_with_no_edge_longer_than_1(self):
        """Test remove_tipped_edge_smaller_than_branch_width with no edge longer than 1."""
        valid_skeleton = ad_pad(np.array([[0, 1, 0, 1, 0],
                                               [0, 0, 1, 0, 0],
                                               [0, 0, 1, 0, 0],
                                               [0, 1, 0, 1, 0],
                                               [1, 0, 1, 0, 1],
                                               [0, 1, 0, 1, 0]], dtype=np.uint8))
        valid_distances = ad_pad(np.array([[0, 1, 0, 1, 0],
                                               [0, 0, 1, 0, 0],
                                               [0, 0, 1, 0, 0],
                                               [0, 1, 0, 1, 0],
                                               [1, 0, 1, 0, 1],
                                               [0, 1, 0, 1, 0]], dtype=np.float64))
        edge_id = EdgeIdentification(valid_skeleton, valid_distances)
        edge_id.get_vertices_and_tips_coord()
        edge_id.get_tipped_edges()
        edge_id.remove_tipped_edge_smaller_than_branch_width()
        expected = np.array([[0, 0, 0, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 0, 0],
                                   [0, 0, 0, 1, 0, 0, 0],
                                   [0, 0, 1, 0, 1, 0, 0],
                                   [0, 1, 0, 1, 0, 1, 0],
                                   [0, 0, 1, 0, 1, 0, 0],
                                   [0, 0, 0, 0, 0, 0, 0]], dtype=np.uint8)
        self.assertTrue(np.array_equal(edge_id.pad_skeleton, expected))

    def test_label_tipped_edges_and_their_vertices(self):
        """Test that label_tipped_edges_and_their_vertices completes without errors."""
        edge_id = EdgeIdentification(self.valid_skeleton, self.valid_distances)
        # This should complete all steps without raising exceptions
        edge_id.get_vertices_and_tips_coord()
        edge_id.get_tipped_edges()
        edge_id.remove_tipped_edge_smaller_than_branch_width()
        edge_id.label_tipped_edges_and_their_vertices()

        expected = np.array([[2, 4]], dtype=np.uint32)

        self.assertTrue(np.array_equal(edge_id.vertices_branching_tips, expected))


    def test_label_edges_connected_with_vertex_clusters(self):
        """Test that label_edges_connected_with_vertex_clusters completes without errors."""
        valid_skeleton = ad_pad(np.array([[1, 1, 1, 0, 1],
                                               [1, 0, 1, 0, 1],
                                               [1, 1, 0, 0, 1],
                                               [1, 0, 0, 1, 0],
                                               [0, 1, 1, 0, 1],
                                               [0, 0, 0, 1, 0]], dtype=np.uint8))

        valid_distances = np.ones_like(valid_skeleton, dtype=np.float64)
        edge_id = EdgeIdentification(valid_skeleton, valid_distances)
        # This should complete all steps without raising exceptions
        edge_id.get_vertices_and_tips_coord()
        edge_id.get_tipped_edges()
        edge_id.remove_tipped_edge_smaller_than_branch_width()
        edge_id.label_tipped_edges_and_their_vertices()

        self.assertEqual(edge_id.edge_pix_coord.shape[0], 2)
        edge_id.label_edges_connected_with_vertex_clusters()
        self.assertEqual(edge_id.edge_pix_coord.shape[0], 13)

    def test_label_edges_connecting_vertex_clusters(self):
        """Test that label_edges_connecting_vertex_clusters completes without errors."""
        valid_skeleton = ad_pad(np.array([[1, 1, 1, 0, 1],
                                               [1, 0, 1, 0, 1],
                                               [1, 1, 0, 0, 1],
                                               [1, 0, 0, 1, 0],
                                               [0, 1, 1, 0, 1],
                                               [0, 0, 0, 1, 0]], dtype=np.uint8))

        valid_distances = np.ones_like(valid_skeleton, dtype=np.float64)
        edge_id = EdgeIdentification(valid_skeleton, valid_distances)
        # This should complete all steps without raising exceptions
        edge_id.get_vertices_and_tips_coord()
        edge_id.get_tipped_edges()
        edge_id.remove_tipped_edge_smaller_than_branch_width()
        edge_id.label_tipped_edges_and_their_vertices()

        edge_id.label_edges_connected_with_vertex_clusters()
        self.assertEqual(len(edge_id.edge_lengths), 5)
        edge_id.label_edges_connecting_vertex_clusters()
        self.assertEqual(len(edge_id.edge_lengths), 8)

    def test_label_edges_from_known_vertices_iteratively_without_vertex_clusters(self):
        """Test that label_edges_from_known_vertices_iteratively works without vertex clusters."""
        valid_skeleton = ad_pad(np.array([[0, 1, 0, 0, 1, 1, 1, 0, 0],
                                               [1, 0, 1, 0, 1, 0, 0, 1, 1],
                                               [0, 1, 0, 0, 1, 0, 1, 0, 0],
                                               [1, 0, 0, 0, 1, 0, 0, 1, 1],
                                               [0, 1, 1, 0, 1, 0, 1, 0, 0],
                                               [1, 0, 0, 1, 0, 0, 0, 1, 0]], dtype=np.uint8))

        valid_distances = np.ones_like(valid_skeleton, dtype=np.float64)
        edge_id = EdgeIdentification(valid_skeleton, valid_distances)
        # This should complete all steps without raising exceptions
        edge_id.get_vertices_and_tips_coord()
        edge_id.get_tipped_edges()
        edge_id.remove_tipped_edge_smaller_than_branch_width()
        edge_id.label_tipped_edges_and_their_vertices()
        edge_id.label_edges_connected_with_vertex_clusters()
        edge_id.label_edges_connecting_vertex_clusters()
        edge_id.label_edges_from_known_vertices_iteratively()
        self.assertEqual(len(edge_id.edge_lengths), 1)

    def test_label_edges_from_known_vertices_iteratively_with_vertex_clusters(self):
        """Test that label_edges_from_known_vertices_iteratively works with vertex clusters."""
        valid_skeleton = ad_pad(np.array([[1, 1, 1, 0, 1, 1, 1, 0, 0],
                                               [1, 0, 1, 0, 1, 0, 1, 1, 1],
                                               [0, 1, 0, 0, 1, 0, 1, 0, 0],
                                               [1, 0, 0, 1, 0, 0, 1, 1, 1],
                                               [0, 1, 1, 0, 1, 0, 1, 0, 1],
                                               [0, 0, 0, 1, 0, 0, 1, 1, 1]], dtype=np.uint8))

        valid_distances = np.ones_like(valid_skeleton, dtype=np.float64)
        edge_id = EdgeIdentification(valid_skeleton, valid_distances)
        # This should complete all steps without raising exceptions
        edge_id.get_vertices_and_tips_coord()
        edge_id.get_tipped_edges()
        edge_id.remove_tipped_edge_smaller_than_branch_width()
        edge_id.label_tipped_edges_and_their_vertices()
        edge_id.label_edges_connected_with_vertex_clusters()
        edge_id.label_edges_connecting_vertex_clusters()
        edge_id.label_edges_from_known_vertices_iteratively()
        self.assertEqual(len(edge_id.edge_lengths), 11)

    def test_label_edges_looping_on_1_vertex(self):
        """Test that label_edges_looping_on_1_vertex completes without errors."""
        valid_skeleton = ad_pad(np.array([[1, 0, 1, 0, 1, 0, 1, 0, 1],
                                               [1, 0, 1, 0, 1, 0, 1, 0, 1],
                                               [1, 1, 1, 1, 1, 0, 1, 1, 1],
                                               [0, 1, 0, 0, 0, 1, 0, 0, 1],
                                               [1, 1, 1, 1, 0, 0, 0, 0, 1],
                                               [1, 0, 1, 0, 1, 0, 1, 1, 1],
                                               [0, 1, 0, 1, 1, 0, 0, 0, 1],
                                               [1, 0, 0, 1, 0, 0, 1, 0, 0],
                                               [0, 0, 1, 0, 1, 1, 0, 1, 0],
                                               [0, 1, 0, 0, 0, 0, 0, 0, 1],
                                               [0, 1, 1, 1, 1, 1, 1, 1, 1]], dtype=np.uint8))

        valid_distances = np.ones_like(valid_skeleton, dtype=np.float64)
        edge_id = EdgeIdentification(valid_skeleton, valid_distances)
        # This should complete all steps without raising exceptions
        edge_id.get_vertices_and_tips_coord()
        edge_id.get_tipped_edges()
        edge_id.remove_tipped_edge_smaller_than_branch_width()
        edge_id.label_tipped_edges_and_their_vertices()
        edge_id.label_edges_connected_with_vertex_clusters()
        edge_id.label_edges_connecting_vertex_clusters()
        edge_id.label_edges_from_known_vertices_iteratively()
        self.assertEqual(len(edge_id.edge_lengths), 16)
        edge_id.label_edges_looping_on_1_vertex()
        self.assertEqual(len(edge_id.edge_lengths), 17)
        self.assertEqual(edge_id.edge_lengths[-1], 15)

    def test_clear_areas_of_1_or_2_unidentified_pixels(self):
        """Test that clear_areas_of_1_or_2_unidentified_pixels completes without errors."""
        valid_skeleton = ad_pad(np.array([[1, 0, 1, 0, 1, 0, 1, 0, 1],
                                               [1, 0, 1, 0, 1, 0, 1, 0, 1],
                                               [1, 1, 1, 1, 1, 0, 1, 1, 1],
                                               [0, 1, 0, 0, 0, 1, 0, 0, 1],
                                               [1, 1, 1, 1, 0, 0, 0, 0, 1],
                                               [1, 0, 1, 0, 1, 0, 1, 1, 1],
                                               [0, 1, 0, 1, 1, 0, 0, 0, 1],
                                               [1, 0, 0, 1, 0, 0, 1, 0, 0],
                                               [0, 0, 1, 0, 1, 1, 0, 1, 0],
                                               [0, 1, 0, 0, 0, 0, 0, 0, 1],
                                               [0, 1, 1, 1, 1, 1, 1, 1, 1]], dtype=np.uint8))

        valid_distances = np.ones_like(valid_skeleton, dtype=np.float64)
        edge_id = EdgeIdentification(valid_skeleton, valid_distances)
        # This should complete all steps without raising exceptions
        edge_id.get_vertices_and_tips_coord()
        edge_id.get_tipped_edges()
        edge_id.remove_tipped_edge_smaller_than_branch_width()
        edge_id.label_tipped_edges_and_their_vertices()
        edge_id.label_edges_connected_with_vertex_clusters()
        edge_id.label_edges_connecting_vertex_clusters()
        edge_id.label_edges_from_known_vertices_iteratively()
        edge_id.label_edges_looping_on_1_vertex()
        self.assertEqual(edge_id.pad_skeleton.sum(), 51)
        edge_id.clear_areas_of_1_or_2_unidentified_pixels()
        self.assertLess(edge_id.pad_skeleton.sum(), 51)

    def test_clear_edge_duplicates(self):
        """Test that clear_edge_duplicates completes without errors."""
        valid_skeleton = ad_pad(np.array([[1, 0, 1, 0, 1, 0, 1, 0, 1],
                                               [1, 0, 1, 0, 1, 0, 1, 0, 1],
                                               [1, 1, 1, 1, 1, 0, 1, 1, 1],
                                               [0, 1, 0, 0, 0, 1, 0, 0, 1],
                                               [1, 1, 1, 1, 0, 0, 0, 0, 1],
                                               [1, 0, 1, 0, 1, 0, 1, 1, 1],
                                               [0, 1, 0, 1, 1, 0, 0, 0, 1],
                                               [1, 0, 0, 1, 0, 0, 1, 0, 0],
                                               [0, 0, 1, 0, 1, 1, 0, 1, 0],
                                               [0, 1, 0, 0, 0, 0, 0, 0, 1],
                                               [0, 1, 1, 1, 1, 1, 1, 1, 1]], dtype=np.uint8))

        valid_distances = np.ones_like(valid_skeleton, dtype=np.float64)
        edge_id = EdgeIdentification(valid_skeleton, valid_distances)
        # This should complete all steps without raising exceptions
        edge_id.get_vertices_and_tips_coord()
        edge_id.get_tipped_edges()
        edge_id.remove_tipped_edge_smaller_than_branch_width()
        edge_id.label_tipped_edges_and_their_vertices()
        edge_id.label_edges_connected_with_vertex_clusters()
        edge_id.label_edges_connecting_vertex_clusters()
        edge_id.label_edges_from_known_vertices_iteratively()
        edge_id.label_edges_looping_on_1_vertex()
        edge_id.clear_areas_of_1_or_2_unidentified_pixels()
        self.assertEqual(edge_id.edges_labels.shape[0], 17)
        edge_id.clear_edge_duplicates()
        self.assertEqual(edge_id.edges_labels.shape[0], 17)

    def test_clear_vertices_connecting_2_edges(self):
        """Test that clear_vertices_connecting_2_edges completes without errors."""
        valid_skeleton = ad_pad(np.array([[1, 0, 1, 0, 1, 0, 1, 0, 1],
                                               [1, 0, 1, 0, 1, 0, 1, 0, 1],
                                               [1, 1, 1, 1, 1, 0, 1, 1, 1],
                                               [0, 1, 0, 0, 0, 1, 0, 0, 1],
                                               [1, 1, 1, 1, 0, 0, 0, 0, 1],
                                               [1, 0, 1, 0, 1, 0, 1, 1, 1],
                                               [0, 1, 0, 1, 1, 0, 0, 0, 1],
                                               [1, 0, 0, 1, 0, 0, 1, 0, 0],
                                               [0, 0, 1, 0, 1, 1, 0, 1, 0],
                                               [0, 1, 0, 0, 0, 0, 0, 0, 1],
                                               [0, 1, 1, 1, 1, 1, 1, 1, 1]], dtype=np.uint8))

        valid_distances = np.ones_like(valid_skeleton, dtype=np.float64)
        edge_id = EdgeIdentification(valid_skeleton, valid_distances)
        # This should complete all steps without raising exceptions
        edge_id.get_vertices_and_tips_coord()
        edge_id.get_tipped_edges()
        edge_id.remove_tipped_edge_smaller_than_branch_width()
        edge_id.label_tipped_edges_and_their_vertices()
        edge_id.label_edges_connected_with_vertex_clusters()
        edge_id.label_edges_connecting_vertex_clusters()
        edge_id.label_edges_from_known_vertices_iteratively()
        edge_id.label_edges_looping_on_1_vertex()
        edge_id.clear_areas_of_1_or_2_unidentified_pixels()
        edge_id.clear_edge_duplicates()
        self.assertEqual(edge_id.non_tip_vertices.shape[0], 7)
        edge_id.clear_vertices_connecting_2_edges()
        self.assertEqual(edge_id.non_tip_vertices.shape[0], 6)

    def test_run_edge_identification_completes(self):
        """Test that run_edge_identification completes without errors."""
        edge_id = EdgeIdentification(self.valid_skeleton, self.valid_distances)
        # This should complete all steps without raising exceptions
        edge_id.run_edge_identification()

        res = np.zeros(self.dims, dtype=np.uint8)
        res[edge_id.edge_pix_coord[:, 0], edge_id.edge_pix_coord[:, 1]] = 1
        res[edge_id.non_tip_vertices[:, 0], edge_id.non_tip_vertices[:, 1]] = 1
        res[edge_id.tips_coord[:, 0], edge_id.tips_coord[:, 1]] = 1
        self.assertTrue(np.array_equal(res, edge_id.pad_skeleton))

    def test_make_vertex_table(self):
        """Test that make_vertex_table completes without errors."""
        edge_id = EdgeIdentification(self.valid_skeleton, self.valid_distances)
        # This should complete all steps without raising exceptions
        edge_id.run_edge_identification()
        origin_contours = np.zeros((self.dims[0] - 2, self.dims[1] - 2), dtype=np.uint8)
        y_ori, x_ori = 1, 3
        origin_contours[y_ori, x_ori] = 1
        growing_areas = np.array([[3], [4]])
        edge_id.make_vertex_table(origin_contours, growing_areas)

        # Check that there are 4 tips
        self.assertEqual(edge_id.vertex_table[:, 3].sum(), 4)

        # Check that the origin region is well located
        self.assertTrue(np.array_equal(edge_id.vertex_table[edge_id.vertex_table[:, 4] == 1, :2], np.array([[y_ori, x_ori]], dtype=np.uint32)))

        # Check that the growing region is well located
        self.assertTrue(np.array_equal(edge_id.vertex_table[edge_id.vertex_table[:, 4] == 2, :2], np.transpose(growing_areas).astype(np.uint32)))

        # Check that the connected vertices are well located
        self.assertTrue(np.array_equal(edge_id.vertex_table[edge_id.vertex_table[:, 5] == 1, :2], np.array([[0, 4], [1, 3]], dtype=np.uint32)))

    def test_make_edge_table(self):
        """Test that make_edge_table completes without errors."""
        edge_id = EdgeIdentification(self.valid_skeleton, self.valid_distances)
        # This should complete all steps without raising exceptions
        edge_id.run_edge_identification()
        origin_contours = np.zeros((self.dims[0] - 2, self.dims[1] - 2), dtype=np.uint8)
        y_ori, x_ori = 1, 3
        origin_contours[y_ori, x_ori] = 1
        growing_areas = np.array([[3], [4]])
        edge_id.make_vertex_table(origin_contours, growing_areas)
        greyscale = self.valid_skeleton.copy()
        greyscale[greyscale > 0] = np.random.randint(170, 255, self.valid_skeleton.sum())
        greyscale[greyscale == 0] = np.random.randint(0, 50, self.valid_skeleton.size - self.valid_skeleton.sum())
        greyscale = un_pad(greyscale)
        edge_id.make_edge_table(greyscale)

        # Check 5 edges are documented
        self.assertEqual(edge_id.edge_table.shape[0], 4)

        res = np.zeros((self.dims[0] - 2, self.dims[1] - 2), dtype=np.uint8)
        res[edge_id.edge_pix_coord[:, 0], edge_id.edge_pix_coord[:, 1]] = 1
        res[edge_id.vertex_table[:, 0], edge_id.vertex_table[:, 1]] = 1

        expected = un_pad(self.valid_skeleton)
        expected[0, 2] = 0
        self.assertTrue(np.array_equal(res, expected))

if __name__ == "__main__":
    unittest.main()