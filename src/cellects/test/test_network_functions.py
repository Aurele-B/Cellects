import cv2
import numpy as np
from cellects.image_analysis.network_functions import *


def test_get_terminations_and_their_connected_nodes():
    test_nb = 1
    test_name = f"Test {test_nb}, +: "
    skeleton = cross_33
    pad_skeleton = np.pad(skeleton, [(1, ), (1, )], mode='constant')
    cnv4, cnv8 = get_neighbor_comparisons(pad_skeleton)
    pad_terminations = get_terminations_and_their_connected_nodes(pad_skeleton, cnv4, cnv8)
    vertices = pad_terminations[1:-1, 1:-1]
    target = cross_33
    print(f"{test_name}{np.array_equal(vertices, target)}")

    test_nb += 1
    test_name = f"Test {test_nb}, *: "
    skeleton = np.array([
        [1, 0, 1],
        [0, 1, 0],
        [1, 0, 1],])
    pad_skeleton = np.pad(skeleton, [(1, ), (1, )], mode='constant')
    cnv4, cnv8 = get_neighbor_comparisons(pad_skeleton)
    pad_terminations = get_terminations_and_their_connected_nodes(pad_skeleton, cnv4, cnv8)
    vertices = pad_terminations[1:-1, 1:-1]
    target = np.array([
        [1, 0, 1],
        [0, 0, 0],
        [1, 0, 1],])
    print(f"{test_name}{np.array_equal(vertices, target)}")

    test_nb += 1
    test_name = f"Test {test_nb}, asymmetric +: "
    skeleton = np.array([
       [0, 0, 0, 1, 0],
       [0, 0, 0, 1, 0],
       [1, 1, 1, 1, 1],
       [0, 0, 0, 1, 0]], dtype=np.uint8)
    pad_skeleton = np.pad(skeleton, [(1, ), (1, )], mode='constant')
    cnv4, cnv8 = get_neighbor_comparisons(pad_skeleton)
    pad_terminations = get_terminations_and_their_connected_nodes(pad_skeleton, cnv4, cnv8)
    vertices = pad_terminations[1:-1, 1:-1]
    target = np.array([
       [0, 0, 0, 1, 0],
       [0, 0, 0, 0, 0],
       [1, 0, 0, 1, 1],
       [0, 0, 0, 1, 0]], dtype=np.uint8)
    print(f"{test_name}{np.array_equal(vertices, target)}")


    test_nb += 1
    test_name = f"Test {test_nb} short tripod: "
    skeleton = np.array([
       [0, 1, 0],
       [1, 1, 0],
       [0, 0, 1]], dtype=np.uint8)
    pad_skeleton = np.pad(skeleton, [(1, ), (1, )], mode='constant')
    cnv4, cnv8 = get_neighbor_comparisons(pad_skeleton)
    pad_terminations = get_terminations_and_their_connected_nodes(pad_skeleton, cnv4, cnv8)
    vertices = pad_terminations[1:-1, 1:-1]
    target = np.array([
        [0, 1, 0],
        [1, 0, 0],
        [0, 0, 1]], dtype=np.uint8)
    print(f"{test_name}{np.array_equal(vertices, target)}")


    test_nb += 1
    test_name = f"Test {test_nb} long tripod: "
    skeleton = np.array([
       [0, 0, 0, 1, 0, 0],
       [0, 0, 0, 1, 0, 0],
       [1, 1, 1, 1, 0, 0],
       [0, 0, 0, 0, 1, 0],
       [0, 0, 0, 0, 0, 1]], dtype=np.uint8)
    pad_skeleton = np.pad(skeleton, [(1, ), (1, )], mode='constant')
    cnv4, cnv8 = get_neighbor_comparisons(pad_skeleton)
    pad_terminations = get_terminations_and_their_connected_nodes(pad_skeleton, cnv4, cnv8)
    vertices = pad_terminations[1:-1, 1:-1]
    target = np.array([
       [0, 0, 0, 1, 0, 0],
       [0, 0, 0, 0, 0, 0],
       [1, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 1]], dtype=np.uint8)
    print(f"{test_name}{np.array_equal(vertices, target)}")


    test_nb += 1
    test_name = f"Test {test_nb}, twisted branch: "
    skeleton = np.array([
        [0, 1, 0],
        [0, 1, 0],
        [1, 1, 1],
        [0, 0, 1]], dtype=np.uint8)
    pad_skeleton = np.pad(skeleton, [(1, ), (1, )], mode='constant')
    cnv4, cnv8 = get_neighbor_comparisons(pad_skeleton)
    pad_terminations = get_terminations_and_their_connected_nodes(pad_skeleton, cnv4, cnv8)
    vertices = pad_terminations[1:-1, 1:-1]
    target = np.array([
        [0, 1, 0],
        [0, 0, 0],
        [1, 1, 0],
        [0, 0, 1]], dtype=np.uint8)
    print(f"{test_name}{np.array_equal(vertices, target)}")


    test_nb += 1
    test_name = f"Test {test_nb}, long twisted branch: "
    skeleton = np.array([
       [1, 0, 1, 0],
       [0, 1, 0, 0],
       [0, 1, 0, 0],
       [0, 1, 1, 0],
       [1, 0, 0, 1,]], dtype=np.uint8)
    pad_skeleton = np.pad(skeleton, [(1, ), (1, )], mode='constant')
    cnv4, cnv8 = get_neighbor_comparisons(pad_skeleton)
    pad_terminations = get_terminations_and_their_connected_nodes(pad_skeleton, cnv4, cnv8)
    vertices = pad_terminations[1:-1, 1:-1]
    target = np.array([
        [1, 0, 1, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [1, 0, 0, 1]], dtype=np.uint8)
    print(f"{test_name}{np.array_equal(vertices, target)}")


    test_nb += 1
    test_name = f"Test {test_nb}, strange line: "
    skeleton = np.array([
       [0, 0, 0, 1],
       [0, 0, 1, 0],
       [0, 1, 0, 0],
       [1, 1, 0, 0],
       [1, 0, 0, 0],
       [1, 0, 0, 0]], dtype=np.uint8)
    pad_skeleton = np.pad(skeleton, [(1,), (1,)], mode='constant')
    cnv4, cnv8 = get_neighbor_comparisons(pad_skeleton)
    pad_terminations = get_terminations_and_their_connected_nodes(pad_skeleton, cnv4, cnv8)
    vertices = pad_terminations[1:-1, 1:-1]
    target = np.array([
        [0, 0, 0, 1],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [1, 0, 0, 0]], dtype=np.uint8)
    print(f"{test_name}{np.array_equal(vertices, target)}")


    test_nb += 1
    test_name = f"Test {test_nb}, long cross with strange lines1: "
    skeleton = np.array([
        [1, 0, 0, 1, 0],
        [1, 1, 0, 1, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 1, 1],
        [1, 1, 0, 0, 0]], dtype=np.uint8)
    pad_skeleton = np.pad(skeleton, [(1,), (1,)], mode='constant')
    cnv4, cnv8 = get_neighbor_comparisons(pad_skeleton)
    pad_terminations = get_terminations_and_their_connected_nodes(pad_skeleton, cnv4, cnv8)
    vertices = pad_terminations[1:-1, 1:-1]
    target = np.array([
        [1, 0, 0, 1, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0]], dtype=np.uint8)
    print(f"{test_name}{np.array_equal(vertices, target)}")
    
    
    test_nb += 1
    test_name = f"Test {test_nb}, long cross with strange lines2: "
    skeleton = np.array([
        [0, 0, 1, 0, 0],
        [1, 1, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 1, 0, 1]], dtype=np.uint8)
    pad_skeleton = np.pad(skeleton, [(1,), (1,)], mode='constant')
    cnv4, cnv8 = get_neighbor_comparisons(pad_skeleton)
    pad_terminations = get_terminations_and_their_connected_nodes(pad_skeleton, cnv4, cnv8)
    vertices = pad_terminations[1:-1, 1:-1]
    target = np.array([
        [0, 0, 1, 0, 0],
        [1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 1, 0, 1]], dtype=np.uint8)
    print(f"{test_name}{np.array_equal(vertices, target)}")
    
    test_nb += 1
    test_name = f"Test {test_nb}, loop: "
    skeleton = np.array([
       [0, 1, 1, 1, 1, 0],
       [1, 0, 0, 0, 0, 1],
       [0, 1, 0, 0, 0, 1],
       [0, 0, 1, 1, 1, 0]], dtype=np.uint8)
    pad_skeleton = np.pad(skeleton, [(1,), (1,)], mode='constant')
    cnv4, cnv8 = get_neighbor_comparisons(pad_skeleton)
    pad_terminations = get_terminations_and_their_connected_nodes(pad_skeleton, cnv4, cnv8)
    vertices = pad_terminations[1:-1, 1:-1]
    target = np.array([
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0]], dtype=np.uint8)
    print(f"{test_name}{np.array_equal(vertices, target)}")

    test_nb += 1
    test_name = f"Test {test_nb}, bigger network: "
    skeleton = np.array([
        [1, 0, 1, 1, 1, 0, 0, 1, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 1, 0, 0],
        [0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
        [0, 0, 0, 1, 0, 1, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 1, 0, 0],
        [0, 0, 0, 1, 0, 1, 0, 1, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 1, 0, 0]], dtype=np.uint8)
    pad_skeleton = np.pad(skeleton, [(1,), (1,)], mode='constant')
    cnv4, cnv8 = get_neighbor_comparisons(pad_skeleton)
    pad_terminations = get_terminations_and_their_connected_nodes(pad_skeleton, cnv4, cnv8)
    vertices = pad_terminations[1:-1, 1:-1]
    target = np.array([
        [0, 0, 1, 0, 1, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 1, 0, 0]], dtype=np.uint8)
    print(f"{test_name}{np.array_equal(vertices, target)}")

    test_nb += 1
    test_name = f"Test {test_nb}, thick node: "
    skeleton = np.array([
        [0, 0, 1, 0, 0],
        [0, 0, 1, 1, 0],
        [1, 1, 1, 1, 1]], dtype=np.uint8)
    pad_skeleton = np.pad(skeleton, [(1,), (1,)], mode='constant')
    cnv4, cnv8 = get_neighbor_comparisons(pad_skeleton)
    pad_terminations = get_terminations_and_their_connected_nodes(pad_skeleton, cnv4, cnv8)
    vertices = pad_terminations[1:-1, 1:-1]
    target = np.array([
        [0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0],
        [1, 0, 0, 0, 1]], dtype=np.uint8)
    print(f"{test_name}{not np.array_equal(vertices, target)}")









def test_get_inner_vertices():
    test_nb = 1
    test_name = f"Test {test_nb}, +: "
    skeleton = cross_33
    pad_skeleton = np.pad(skeleton, [(1, ), (1, )], mode='constant')
    pad_skeleton = remove_small_loops(pad_skeleton)
    cnv4, cnv8 = get_neighbor_comparisons(pad_skeleton)
    sure_terminations = get_terminations_and_their_connected_nodes(pad_skeleton, cnv4, cnv8)
    pad_vertices = get_inner_vertices(sure_terminations, cnv8)
    vertices = pad_vertices[1:-1, 1:-1]
    target = cross_33
    print(f"{test_name}{np.array_equal(vertices, target)}")

    test_nb += 1
    test_name = f"Test {test_nb}, *: "
    skeleton = np.array([
        [1, 0, 1],
        [0, 1, 0],
        [1, 0, 1],])
    pad_skeleton = np.pad(skeleton, [(1, ), (1, )], mode='constant')
    pad_skeleton = remove_small_loops(pad_skeleton)
    cnv4, cnv8 = get_neighbor_comparisons(pad_skeleton)
    sure_terminations = get_terminations_and_their_connected_nodes(pad_skeleton, cnv4, cnv8)
    pad_vertices = get_inner_vertices(sure_terminations, cnv8)
    vertices = pad_vertices[1:-1, 1:-1]
    target = np.array([
        [1, 0, 1],
        [0, 1, 0],
        [1, 0, 1],])
    print(f"{test_name}{np.array_equal(vertices, target)}")

    test_nb += 1
    test_name = f"Test {test_nb}, asymmetric +: "
    skeleton = np.array([
       [0, 0, 0, 1, 0],
       [0, 0, 0, 1, 0],
       [1, 1, 1, 1, 1],
       [0, 0, 0, 1, 0]], dtype=np.uint8)
    pad_skeleton = np.pad(skeleton, [(1, ), (1, )], mode='constant')
    pad_skeleton = remove_small_loops(pad_skeleton)
    cnv4, cnv8 = get_neighbor_comparisons(pad_skeleton)
    sure_terminations = get_terminations_and_their_connected_nodes(pad_skeleton, cnv4, cnv8)
    pad_vertices = get_inner_vertices(sure_terminations, cnv8)
    vertices = pad_vertices[1:-1, 1:-1]
    target = np.array([
       [0, 0, 0, 1, 0],
       [0, 0, 0, 0, 0],
       [1, 0, 0, 1, 1],
       [0, 0, 0, 1, 0]], dtype=np.uint8)
    print(f"{test_name}{np.array_equal(vertices, target)}")


    test_nb += 1
    test_name = f"Test {test_nb} short tripod: "
    skeleton = np.array([
       [0, 1, 0],
       [1, 1, 0],
       [0, 0, 1]], dtype=np.uint8)
    pad_skeleton = np.pad(skeleton, [(1, ), (1, )], mode='constant')
    pad_skeleton = remove_small_loops(pad_skeleton)
    cnv4, cnv8 = get_neighbor_comparisons(pad_skeleton)
    sure_terminations = get_terminations_and_their_connected_nodes(pad_skeleton, cnv4, cnv8)
    pad_vertices = get_inner_vertices(sure_terminations, cnv8)
    vertices = pad_vertices[1:-1, 1:-1]
    target = np.array([
        [0, 1, 0],
        [1, 1, 0],
        [0, 0, 1]], dtype=np.uint8)
    print(f"{test_name}{np.array_equal(vertices, target)}")


    test_nb += 1
    test_name = f"Test {test_nb} long tripod: "
    skeleton = np.array([
       [0, 0, 0, 1, 0, 0],
       [0, 0, 0, 1, 0, 0],
       [1, 1, 1, 1, 0, 0],
       [0, 0, 0, 0, 1, 0],
       [0, 0, 0, 0, 0, 1]], dtype=np.uint8)
    pad_skeleton = np.pad(skeleton, [(1, ), (1, )], mode='constant')
    pad_skeleton = remove_small_loops(pad_skeleton)
    cnv4, cnv8 = get_neighbor_comparisons(pad_skeleton)
    sure_terminations = get_terminations_and_their_connected_nodes(pad_skeleton, cnv4, cnv8)
    pad_vertices = get_inner_vertices(sure_terminations, cnv8)
    vertices = pad_vertices[1:-1, 1:-1]
    target = np.array([
       [0, 0, 0, 1, 0, 0],
       [0, 0, 0, 0, 0, 0],
       [1, 0, 0, 1, 0, 0],
       [0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 1]], dtype=np.uint8)
    print(f"{test_name}{np.array_equal(vertices, target)}")


    test_nb += 1
    test_name = f"Test {test_nb}, twisted branch: "
    skeleton = np.array([
        [0, 1, 0],
        [0, 1, 0],
        [1, 1, 1],
        [0, 0, 1]], dtype=np.uint8)
    pad_skeleton = np.pad(skeleton, [(1, ), (1, )], mode='constant')
    pad_skeleton = remove_small_loops(pad_skeleton)
    cnv4, cnv8 = get_neighbor_comparisons(pad_skeleton)
    sure_terminations = get_terminations_and_their_connected_nodes(pad_skeleton, cnv4, cnv8)
    pad_vertices = get_inner_vertices(sure_terminations, cnv8)
    vertices = pad_vertices[1:-1, 1:-1]
    target = np.array([
        [0, 1, 0],
        [0, 0, 0],
        [1, 1, 0],
        [0, 0, 1]], dtype=np.uint8)
    print(f"{test_name}{np.array_equal(vertices, target)}")


    test_nb += 1
    test_name = f"Test {test_nb}, long twisted branch: "
    skeleton = np.array([
       [1, 0, 1, 0],
       [0, 1, 0, 0],
       [0, 1, 0, 0],
       [0, 1, 1, 0],
       [1, 0, 0, 1,]], dtype=np.uint8)
    pad_skeleton = np.pad(skeleton, [(1, ), (1, )], mode='constant')
    pad_skeleton = remove_small_loops(pad_skeleton)
    cnv4, cnv8 = get_neighbor_comparisons(pad_skeleton)
    sure_terminations = get_terminations_and_their_connected_nodes(pad_skeleton, cnv4, cnv8)
    pad_vertices = get_inner_vertices(sure_terminations, cnv8)
    vertices = pad_vertices[1:-1, 1:-1]
    target = np.array([
        [1, 0, 1, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 0],
        [0, 1, 0, 0],
        [1, 0, 0, 1]], dtype=np.uint8)
    print(f"{test_name}{np.array_equal(vertices, target)}")


    test_nb += 1
    test_name = f"Test {test_nb}, strange line: "
    skeleton = np.array([
       [0, 0, 0, 1],
       [0, 0, 1, 0],
       [0, 1, 0, 0],
       [1, 1, 0, 0],
       [1, 0, 0, 0],
       [1, 0, 0, 0]], dtype=np.uint8)
    pad_skeleton = np.pad(skeleton, [(1,), (1,)], mode='constant')
    pad_skeleton = remove_small_loops(pad_skeleton)
    cnv4, cnv8 = get_neighbor_comparisons(pad_skeleton)
    sure_terminations = get_terminations_and_their_connected_nodes(pad_skeleton, cnv4, cnv8)
    pad_vertices = get_inner_vertices(sure_terminations, cnv8)
    vertices = pad_vertices[1:-1, 1:-1]
    target = np.array([
        [0, 0, 0, 1],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [1, 0, 0, 0]], dtype=np.uint8)
    print(f"{test_name}{np.array_equal(vertices, target)}")


    test_nb += 1
    test_name = f"Test {test_nb}, long cross with strange lines1: "
    skeleton = np.array([
        [1, 0, 0, 1, 0],
        [1, 1, 0, 1, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 1, 1],
        [1, 1, 0, 0, 0]], dtype=np.uint8)
    pad_skeleton = np.pad(skeleton, [(1,), (1,)], mode='constant')
    pad_skeleton = remove_small_loops(pad_skeleton)
    cnv4, cnv8 = get_neighbor_comparisons(pad_skeleton)
    sure_terminations = get_terminations_and_their_connected_nodes(pad_skeleton, cnv4, cnv8)
    pad_vertices = get_inner_vertices(sure_terminations, cnv8)
    vertices = pad_vertices[1:-1, 1:-1]
    target = np.array([
        [1, 0, 0, 1, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 1],
        [1, 0, 0, 0, 0]], dtype=np.uint8)
    print(f"{test_name}{np.array_equal(vertices, target)}")
    
    
    test_nb += 1
    test_name = f"Test {test_nb}, long cross with strange lines2: "
    skeleton = np.array([
        [0, 0, 1, 0, 0],
        [1, 1, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 1, 0, 1]], dtype=np.uint8)
    pad_skeleton = np.pad(skeleton, [(1,), (1,)], mode='constant')
    pad_skeleton = remove_small_loops(pad_skeleton)
    cnv4, cnv8 = get_neighbor_comparisons(pad_skeleton)
    sure_terminations = get_terminations_and_their_connected_nodes(pad_skeleton, cnv4, cnv8)
    pad_vertices = get_inner_vertices(sure_terminations, cnv8)
    vertices = pad_vertices[1:-1, 1:-1]
    target = np.array([
        [0, 0, 1, 0, 0],
        [1, 1, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 1, 0, 1]], dtype=np.uint8)
    print(f"{test_name}{np.array_equal(vertices, target)}")
    
    test_nb += 1
    test_name = f"Test {test_nb}, loop: "
    skeleton = np.array([
       [0, 1, 1, 1, 1, 0],
       [1, 0, 0, 0, 0, 1],
       [0, 1, 0, 0, 0, 1],
       [0, 0, 1, 1, 1, 0]], dtype=np.uint8)
    pad_skeleton = np.pad(skeleton, [(1,), (1,)], mode='constant')
    pad_skeleton = remove_small_loops(pad_skeleton)
    cnv4, cnv8 = get_neighbor_comparisons(pad_skeleton)
    sure_terminations = get_terminations_and_their_connected_nodes(pad_skeleton, cnv4, cnv8)
    pad_vertices = get_inner_vertices(sure_terminations, cnv8)
    vertices = pad_vertices[1:-1, 1:-1]
    target = np.array([
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0]], dtype=np.uint8)
    print(f"{test_name}{np.array_equal(vertices, target)}")

    test_nb += 1
    test_name = f"Test {test_nb}, bigger network: "
    skeleton = np.array([
        [1, 0, 1, 1, 1, 0, 0, 1, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 1, 0, 0],
        [0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
        [0, 0, 0, 1, 0, 1, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 1, 0, 0],
        [0, 0, 0, 1, 0, 1, 0, 1, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 1, 0, 0]], dtype=np.uint8)
    pad_skeleton = np.pad(skeleton, [(1,), (1,)], mode='constant')
    pad_skeleton = remove_small_loops(pad_skeleton)
    cnv4, cnv8 = get_neighbor_comparisons(pad_skeleton)
    sure_terminations = get_terminations_and_their_connected_nodes(pad_skeleton, cnv4, cnv8)
    pad_vertices = get_inner_vertices(sure_terminations, cnv8)
    vertices = pad_vertices[1:-1, 1:-1]
    target = np.array([
        [0, 0, 1, 1, 1, 0, 0, 1, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 1, 0, 1, 1, 0, 0],
        [0, 0, 0, 1, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 1, 0, 0]], dtype=np.uint8)
    print(f"{test_name}{np.array_equal(vertices, target)}")


    test_nb += 1
    test_name = f"Test {test_nb}, thick node: "
    skeleton = np.array([
        [0, 0, 1, 0, 0],
        [0, 0, 1, 1, 0],
        [1, 1, 1, 1, 1]], dtype=np.uint8)
    pad_skeleton = np.pad(skeleton, [(1,), (1,)], mode='constant')
    pad_skeleton = remove_small_loops(pad_skeleton)
    cnv4, cnv8 = get_neighbor_comparisons(pad_skeleton)
    sure_terminations = get_terminations_and_their_connected_nodes(pad_skeleton, cnv4, cnv8)
    pad_vertices = get_inner_vertices(sure_terminations, cnv8)
    vertices = pad_vertices[1:-1, 1:-1]
    target = np.array([
        [0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0],
        [1, 0, 1, 0, 1]], dtype=np.uint8)
    print(f"{test_name}{not np.array_equal(vertices, target)}")


    # NEW
    test_nb += 1
    test_name = f"Test {test_nb}, weird line: "
    skeleton = np.array([
        [1, 0, 1, 0, 0, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 0, 0, 1, 0, 1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 0, 0, 0, 0, 1, 0]], dtype=np.uint8)
    pad_skeleton = np.pad(skeleton, [(1,), (1,)], mode='constant')
    pad_skeleton = remove_small_loops(pad_skeleton)
    cnv4, cnv8 = get_neighbor_comparisons(pad_skeleton)
    sure_terminations = get_terminations_and_their_connected_nodes(pad_skeleton, cnv4, cnv8)
    pad_vertices = get_inner_vertices(sure_terminations, cnv8)
    vertices = pad_vertices[1:-1, 1:-1]
    target = np.array([
        [1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0]], dtype=np.uint8)
    print(f"{test_name}{np.array_equal(vertices, target)}")

    test_nb += 1
    test_name = f"Test {test_nb}, small loop: "
    skeleton = np.array([
        [0, 0, 0, 0, 0, 0, 1],
        [1, 1, 0, 1, 0, 1, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 1, 0]], dtype=np.uint8)
    pad_skeleton = np.pad(skeleton, [(1,), (1,)], mode='constant')
    pad_skeleton = remove_small_loops(pad_skeleton)
    cnv4, cnv8 = get_neighbor_comparisons(pad_skeleton)
    sure_terminations = get_terminations_and_their_connected_nodes(pad_skeleton, cnv4, cnv8)
    pad_vertices = get_inner_vertices(sure_terminations, cnv8)
    vertices = pad_vertices[1:-1, 1:-1]
    target = np.array([
        [0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0]], dtype=np.uint8)
    print(f"{test_name}{np.array_equal(vertices, target)}")


if __name__ == '__main__':
    test_get_terminations_and_their_connected_nodes()
    test_get_inner_vertices()