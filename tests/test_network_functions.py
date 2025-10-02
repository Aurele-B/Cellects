# tests/test_network_functions_converted.py
import unittest
import numpy as np
import cv2

from cellects.image_analysis.network_functions import (
    cross_33,
    get_neighbor_comparisons,
    get_terminations_and_their_connected_nodes,
    get_inner_vertices,
    get_vertices_and_tips_from_skeleton,
    get_branches_and_tips_coord,
    remove_small_loops,
    remove_padding,
    find_closest_vertices,
    keep_one_connected_component,
)

# --- Small helpers -----------------------------------------------------------

def _pad(arr):
    return np.pad(arr, [(1, ), (1, )], mode='constant')

def _unpad(arr):
    return arr[1:-1, 1:-1]

# --- Tests -------------------------------------------------------------------

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
                pad_skeleton = _pad(skeleton)
                cnv4, cnv8 = get_neighbor_comparisons(pad_skeleton)
                pad_terminations = get_terminations_and_their_connected_nodes(pad_skeleton, cnv4, cnv8)
                vertices = _unpad(pad_terminations)
                self.assertTrue(np.array_equal(vertices, target))

    def test_thick_node_is_not_equal_to_given_target(self):
        skeleton = np.array([[0,0,1,0,0],
                             [0,0,1,1,0],
                             [1,1,1,1,1]], dtype=np.uint8)
        target   = np.array([[0,0,1,0,0],
                             [0,0,0,0,0],
                             [1,0,0,0,1]], dtype=np.uint8)
        pad_skeleton = _pad(skeleton)
        cnv4, cnv8 = get_neighbor_comparisons(pad_skeleton)
        pad_terminations = get_terminations_and_their_connected_nodes(pad_skeleton, cnv4, cnv8)
        vertices = _unpad(pad_terminations)
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
        pad_skeleton = _pad(skeleton)
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
             np.array([[0,0,1,1,1,0,0,1,0,0],
                       [0,0,0,1,0,0,0,0,0,0],
                       [0,1,0,0,1,0,1,1,0,0],
                       [0,0,0,1,0,1,0,0,0,0],
                       [0,0,0,0,1,0,0,0,0,0],
                       [0,0,0,0,0,1,0,0,0,0],
                       [0,0,1,0,0,0,0,1,0,0]], dtype=np.uint8)),
        ]

        for name, skeleton, v_target in cases:
            with self.subTest(name=name):
                pad = _pad(skeleton)
                cnv4, cnv8 = get_neighbor_comparisons(pad)
                potential_tips = get_terminations_and_their_connected_nodes(pad, cnv4, cnv8)
                pad_vertices, _ = get_inner_vertices(pad, potential_tips, cnv4, cnv8)
                vertices = _unpad(pad_vertices)
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

        pad = _pad(skeleton)
        cnv4, cnv8 = get_neighbor_comparisons(pad)
        potential_tips = get_terminations_and_their_connected_nodes(pad, cnv4, cnv8)
        pad_vertices, pad_tips = get_inner_vertices(pad, potential_tips, cnv4, cnv8)

        vertices = _unpad(pad_vertices)
        tips = _unpad(pad_tips)

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

        pad = _pad(skeleton)
        cnv4, cnv8 = get_neighbor_comparisons(pad)
        potential_tips = get_terminations_and_their_connected_nodes(pad, cnv4, cnv8)
        pad_vertices, _ = get_inner_vertices(pad, potential_tips, cnv4, cnv8)
        vertices = _unpad(pad_vertices)

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
        pad = _pad(skeleton)
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
        pad = _pad(skeleton)
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

        pad_skel = _pad(skeleton)
        pad_dist = _pad(distances)
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
        pad_skel = _pad(skeleton)
        # distances not needed for this check; rely on remove_small_loops default
        pad_skel = remove_small_loops(pad_skel)
        cnv4, cnv8 = get_neighbor_comparisons(pad_skel)
        potential = get_terminations_and_their_connected_nodes(pad_skel, cnv4, cnv8)
        pad_vertices, _ = get_inner_vertices(pad_skel, potential, cnv4, cnv8)
        vertices = _unpad(pad_vertices)
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
        pad_skel = _pad(skeleton)
        pad_skel = remove_small_loops(pad_skel)
        cnv4, cnv8 = get_neighbor_comparisons(pad_skel)
        potential = get_terminations_and_their_connected_nodes(pad_skel, cnv4, cnv8)
        pad_vertices, _ = get_inner_vertices(pad_skel, potential, cnv4, cnv8)
        vertices = _unpad(pad_vertices)
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
        pad_skel = _pad(skeleton)
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
        pad_skel = _pad(skeleton)
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
        pad = _pad(skeleton)
        pad_vertices, pad_tips = get_vertices_and_tips_from_skeleton(pad)
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
        pad = _pad(skeleton)
        pad_vertices, pad_tips = get_vertices_and_tips_from_skeleton(pad)
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
        pad = _pad(skeleton)
        pad_vertices, pad_tips = get_vertices_and_tips_from_skeleton(pad)
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
        pad = _pad(skeleton)
        pad = keep_one_connected_component(pad)
        pad_vertices, pad_tips = get_vertices_and_tips_from_skeleton(pad)

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
        pad = _pad(skeleton)
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
        pad = keep_one_connected_component(_pad(skeleton))
        pad_vertices, pad_tips = get_vertices_and_tips_from_skeleton(pad)

    def test_false_tip_misc_2(self):
        skeleton = np.array([
            [1,0,0,0,1,0,0],
            [0,1,0,1,0,0,0],
            [0,0,1,0,0,1,1],
            [0,1,1,1,1,0,0],
            [1,0,1,0,0,1,0],
            [0,0,1,0,0,0,1],
        ], dtype=np.uint8)
        pad = keep_one_connected_component(_pad(skeleton))
        pad_vertices, pad_tips = get_vertices_and_tips_from_skeleton(pad)

    def test_false_tip_misc_3(self):
        skeleton = np.array([
            [0,0,1,0],
            [0,0,1,0],
            [1,1,1,0],
            [0,0,1,0],
            [0,1,1,0],
            [1,0,0,1],
        ], dtype=np.uint8)
        pad = keep_one_connected_component(_pad(skeleton))
        pad_vertices, pad_tips = get_vertices_and_tips_from_skeleton(pad)

    def test_false_tip_misc_4(self):
        skeleton = np.array([
            [0,1,0,0,1,0,0],
            [0,0,1,1,0,0,0],
            [0,0,0,1,0,1,1],
            [0,1,1,1,1,0,0],
            [1,0,0,0,0,1,0],
        ], dtype=np.uint8)
        pad = keep_one_connected_component(_pad(skeleton))
        pad_vertices, pad_tips = get_vertices_and_tips_from_skeleton(pad)

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
        pad = keep_one_connected_component(_pad(skeleton))
        pad_vertices, pad_tips = get_vertices_and_tips_from_skeleton(pad)

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
        pad = keep_one_connected_component(_pad(skeleton))
        pad_vertices, pad_tips = get_vertices_and_tips_from_skeleton(pad)

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
        pad = keep_one_connected_component(_pad(skeleton))
        pad_vertices, pad_tips = get_vertices_and_tips_from_skeleton(pad)

    def test_cross_two_edges_branch_2(self):
        skeleton = np.array([
            [1,0,0,0,0,0],
            [0,1,0,0,0,0],
            [0,1,0,0,0,1],
            [0,0,1,0,1,0],
            [1,1,1,1,0,0],
            [0,0,0,0,1,1],
        ], dtype=np.uint8)
        pad = keep_one_connected_component(_pad(skeleton))
        pad_vertices, pad_tips = get_vertices_and_tips_from_skeleton(pad)

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
        pad = keep_one_connected_component(_pad(skeleton))
        pad_vertices, pad_tips = get_vertices_and_tips_from_skeleton(pad)

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

        pad = _pad(skeleton)
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
            [0,0,1,0,0,2,0,2,2,0,0,0],
            [0,0,0,0,2,0,2,0,0,0,0,0],
            [0,0,0,0,0,2,0,0,0,0,0,0],
            [0,0,0,0,0,0,1,0,0,0,0,0],
            [0,0,0,1,0,0,0,0,1,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0],
        ], dtype=np.uint8)

        self.assertTrue(np.array_equal(vt_map, target))

if __name__ == "__main__":
    unittest.main()