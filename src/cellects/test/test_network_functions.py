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
    pad_tips = get_terminations_and_their_connected_nodes(pad_skeleton, cnv4, cnv8)
    vertices = pad_tips[1:-1, 1:-1]
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


    test_nb += 1
    test_name = f"Test {test_nb}, non connectivity: "
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
    pad_tips = get_terminations_and_their_connected_nodes(pad_skeleton, cnv4, cnv8)
    nb, shape = cv2.connectedComponents(pad_tips)
    print(f"{test_name}{pad_tips.sum() == nb - 1}")


def test_get_inner_vertices():
    test_nb = 1
    test_name = f"Test {test_nb}, +: "
    skeleton = cross_33
    pad_skeleton = np.pad(skeleton, [(1, ), (1, )], mode='constant')

    cnv4, cnv8 = get_neighbor_comparisons(pad_skeleton)
    potential_tips = get_terminations_and_their_connected_nodes(pad_skeleton, cnv4, cnv8)
    pad_vertices, potential_tips = get_inner_vertices(pad_skeleton, potential_tips, cnv4, cnv8)
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

    cnv4, cnv8 = get_neighbor_comparisons(pad_skeleton)
    potential_tips = get_terminations_and_their_connected_nodes(pad_skeleton, cnv4, cnv8)
    pad_vertices, potential_tips = get_inner_vertices(pad_skeleton, potential_tips, cnv4, cnv8)
    vertices = pad_vertices[1:-1, 1:-1]
    target = np.array([
        [1, 0, 1],
        [0, 1, 0],
        [1, 0, 1],])
    print(f"{test_name}{np.array_equal(vertices, target)}")

    test_nb += 1
    test_name = f"Test {test_nb}, asymmetric + ,tips test: "
    skeleton = np.array([
       [0, 0, 0, 1, 0],
       [0, 0, 0, 1, 0],
       [1, 1, 1, 1, 1],
       [0, 0, 0, 1, 0]], dtype=np.uint8)
    pad_skeleton = np.pad(skeleton, [(1, ), (1, )], mode='constant')
    cnv4, cnv8 = get_neighbor_comparisons(pad_skeleton)
    potential_tips = get_terminations_and_their_connected_nodes(pad_skeleton, cnv4, cnv8)
    pad_vertices, pad_tips = get_inner_vertices(pad_skeleton, potential_tips, cnv4, cnv8)
    vertices = pad_vertices[1:-1, 1:-1]
    tips = pad_tips[1:-1, 1:-1]
    v_target = np.array([
       [0, 0, 0, 1, 0],
       [0, 0, 0, 0, 0],
       [1, 0, 0, 1, 1],
       [0, 0, 0, 1, 0]], dtype=np.uint8)
    t_target = np.array([
       [0, 0, 0, 1, 0],
       [0, 0, 0, 0, 0],
       [1, 0, 0, 0, 1],
       [0, 0, 0, 1, 0]], dtype=np.uint8)
    print(f"{test_name}{np.array_equal(tips, t_target)}")
    test_nb += 1
    test_name = f"Test {test_nb}, asymmetric + ,vertex test: "
    print(f"{test_name}{np.array_equal(vertices, v_target)}")


    test_nb += 1
    test_name = f"Test {test_nb} short tripod: "
    skeleton = np.array([
       [0, 1, 0],
       [1, 1, 0],
       [0, 0, 1]], dtype=np.uint8)
    pad_skeleton = np.pad(skeleton, [(1, ), (1, )], mode='constant')
    cnv4, cnv8 = get_neighbor_comparisons(pad_skeleton)
    potential_tips = get_terminations_and_their_connected_nodes(pad_skeleton, cnv4, cnv8)
    pad_vertices, potential_tips = get_inner_vertices(pad_skeleton, potential_tips, cnv4, cnv8)
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
    cnv4, cnv8 = get_neighbor_comparisons(pad_skeleton)
    potential_tips = get_terminations_and_their_connected_nodes(pad_skeleton, cnv4, cnv8)
    pad_vertices, potential_tips = get_inner_vertices(pad_skeleton, potential_tips, cnv4, cnv8)
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
    cnv4, cnv8 = get_neighbor_comparisons(pad_skeleton)
    potential_tips = get_terminations_and_their_connected_nodes(pad_skeleton, cnv4, cnv8)
    pad_vertices, potential_tips = get_inner_vertices(pad_skeleton, potential_tips, cnv4, cnv8)
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
    cnv4, cnv8 = get_neighbor_comparisons(pad_skeleton)
    potential_tips = get_terminations_and_their_connected_nodes(pad_skeleton, cnv4, cnv8)
    pad_vertices, potential_tips = get_inner_vertices(pad_skeleton, potential_tips, cnv4, cnv8)
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

    cnv4, cnv8 = get_neighbor_comparisons(pad_skeleton)
    potential_tips = get_terminations_and_their_connected_nodes(pad_skeleton, cnv4, cnv8)
    pad_vertices, potential_tips = get_inner_vertices(pad_skeleton, potential_tips, cnv4, cnv8)
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

    cnv4, cnv8 = get_neighbor_comparisons(pad_skeleton)
    potential_tips = get_terminations_and_their_connected_nodes(pad_skeleton, cnv4, cnv8)
    pad_vertices, potential_tips = get_inner_vertices(pad_skeleton, potential_tips, cnv4, cnv8)
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

    cnv4, cnv8 = get_neighbor_comparisons(pad_skeleton)
    potential_tips = get_terminations_and_their_connected_nodes(pad_skeleton, cnv4, cnv8)
    pad_vertices, potential_tips = get_inner_vertices(pad_skeleton, potential_tips, cnv4, cnv8)
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

    cnv4, cnv8 = get_neighbor_comparisons(pad_skeleton)
    potential_tips = get_terminations_and_their_connected_nodes(pad_skeleton, cnv4, cnv8)
    pad_vertices, potential_tips = get_inner_vertices(pad_skeleton, potential_tips, cnv4, cnv8)
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
    pad_distances = pad_skeleton.copy()
    cnv4, cnv8 = get_neighbor_comparisons(pad_skeleton)
    potential_tips = get_terminations_and_their_connected_nodes(pad_skeleton, cnv4, cnv8)
    pad_vertices, potential_tips = get_inner_vertices(pad_skeleton, potential_tips, cnv4, cnv8)
    vertices = pad_vertices[1:-1, 1:-1]
    target = np.array([
        [0, 0, 1, 1, 1, 0, 0, 1, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 1, 0, 0, 1, 0, 0],
        [0, 0, 0, 1, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 1, 0, 0]], dtype=np.uint8)
    print(f"{test_name}{np.array_equal(vertices, target)}")


    test_nb += 1
    test_name = f"Test {test_nb}, thick node, correct if wrong, this should not exist: "
    skeleton = np.array([
        [0, 0, 1, 0, 0],
        [0, 0, 1, 1, 0],
        [1, 1, 1, 1, 1]], dtype=np.uint8)
    pad_skeleton = np.pad(skeleton, [(1,), (1,)], mode='constant')

    cnv4, cnv8 = get_neighbor_comparisons(pad_skeleton)
    potential_tips = get_terminations_and_their_connected_nodes(pad_skeleton, cnv4, cnv8)
    pad_vertices, potential_tips = get_inner_vertices(pad_skeleton, potential_tips, cnv4, cnv8)
    vertices = pad_vertices[1:-1, 1:-1]
    target = np.array([
        [0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0],
        [1, 0, 1, 0, 1]], dtype=np.uint8)
    print(f"{test_name}{not np.array_equal(vertices, target)}")


    test_nb += 1
    test_name = f"Test {test_nb}, another cross: "
    skeleton = np.array([
        [0, 1, 0, 0, 1],
        [0, 0, 1, 1, 0],
        [0, 0, 1, 0, 0],
        [1, 1, 1, 0, 0],
        [0, 0, 0, 1, 0]], dtype=np.uint8)
    pad_skeleton = np.pad(skeleton, [(1,), (1,)], mode='constant')

    cnv4, cnv8 = get_neighbor_comparisons(pad_skeleton)
    potential_tips = get_terminations_and_their_connected_nodes(pad_skeleton, cnv4, cnv8)
    pad_vertices, potential_tips = get_inner_vertices(pad_skeleton, potential_tips, cnv4, cnv8)
    vertices = pad_vertices[1:-1, 1:-1]
    target = np.array([
        [0, 1, 0, 0, 1],
        [0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [1, 0, 0, 0, 0],
        [0, 0, 0, 1, 0]], dtype=np.uint8)
    print(f"{test_name}{np.array_equal(vertices, target)}")


    test_nb += 1
    test_name = f"Test {test_nb}, another cross: "
    skeleton = np.array([
        [0, 0, 0, 0, 1],
        [1, 0, 0, 1, 0],
        [0, 1, 1, 1, 0],
        [0, 1, 0, 0, 1],
        [0, 1, 0, 0, 0]], dtype=np.uint8)
    pad_skeleton = np.pad(skeleton, [(1,), (1,)], mode='constant')
    cnv4, cnv8 = get_neighbor_comparisons(pad_skeleton)
    potential_tips = get_terminations_and_their_connected_nodes(pad_skeleton, cnv4, cnv8)
    pad_vertices, potential_tips = get_inner_vertices(pad_skeleton, potential_tips, cnv4, cnv8)
    vertices = pad_vertices[1:-1, 1:-1]
    target = np.array([
        [0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1],
        [0, 1, 0, 0, 0]], dtype=np.uint8)
    print(f"{test_name}{np.array_equal(vertices, target)}")

    test_nb += 1
    test_name = f"Test {test_nb}, another cross: "
    skeleton = np.array([
        [0, 1, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 1, 0],
        [1, 0, 1, 0, 0, 0, 1],
        [0, 0, 1, 0, 0, 0, 0]], dtype=np.uint8)
    pad_skeleton = np.pad(skeleton, [(1,), (1,)], mode='constant')
    cnv4, cnv8 = get_neighbor_comparisons(pad_skeleton)
    potential_tips = get_terminations_and_their_connected_nodes(pad_skeleton, cnv4, cnv8)
    pad_vertices, potential_tips = get_inner_vertices(pad_skeleton, potential_tips, cnv4, cnv8)
    vertices = pad_vertices[1:-1, 1:-1]
    target = np.array([
        [0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 1],
        [0, 0, 1, 0, 0, 0, 0]], dtype=np.uint8)
    print(f"{test_name}{np.array_equal(vertices, target)}")

    test_nb += 1
    test_name = f"Test {test_nb}, another cross: "
    arr_skel = np.array([
        [0, 0, 0, 1],
        [0, 1, 1, 0],
        [0, 0, 1, 0],
        [0, 0, 1, 0]], dtype=np.uint8)
    pad_skeleton = np.pad(arr_skel, [(1,), (1,)], mode='constant')
    cnv4, cnv8 = get_neighbor_comparisons(pad_skeleton)
    potential_tips = get_terminations_and_their_connected_nodes(pad_skeleton, cnv4, cnv8)
    pad_vertices, potential_tips = get_inner_vertices(pad_skeleton, potential_tips, cnv4, cnv8)
    vertices = pad_vertices[1:-1, 1:-1]
    target = np.array([
        [0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 1],
        [0, 0, 1, 0, 0, 0, 0]], dtype=np.uint8)
    print(f"{test_name}{np.array_equal(vertices, target)}")


def test_remove_small_loops():
    test_nb = 1
    test_name = f"Test {test_nb}, small loop: "
    skeleton =  np.array([
        [0, 1, 0, 1, 0],
        [0, 0, 1, 0, 0],
        [1, 1, 0, 1, 1],
        [0, 0, 1, 0, 0],], dtype=np.uint8)
    pad_skeleton = np.pad(skeleton, [(1,), (1,)], mode='constant')
    pad_skeleton = remove_small_loops(pad_skeleton)
    new_skeleton = remove_padding([pad_skeleton])[0]
    target = np.array([
       [0, 1, 0, 1, 0],
       [0, 0, 1, 0, 0],
       [1, 1, 0, 1, 1],
       [0, 0, 0, 0, 0]], dtype=uint8)
    print(f"{test_name}{np.array_equal(new_skeleton, target)}")

    test_nb += 1
    test_name = f"Test {test_nb}, bow loop: "
    skeleton =  np.array([
        [1, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0],
        [0, 1, 0, 1, 1, 1, 1],
        [0, 0, 1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0]], dtype=np.uint8)
    pad_skeleton = np.pad(skeleton, [(1,), (1,)], mode='constant')
    pad_skeleton = remove_small_loops(pad_skeleton)
    new_skeleton = remove_padding([pad_skeleton])[0]
    target = np.array([
        [1, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 1, 1],
        [0, 0, 1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0]], dtype=uint8)
    print(f"{test_name}{np.array_equal(new_skeleton, target)}")

    test_nb += 1
    test_name = f"Test {test_nb}, weird line distances: "
    filled_loops = np.array([
        [1, 1, 1, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 0, 0],
        [0, 1, 1, 1, 1, 1, 0],
        [0, 0, 1, 1, 1, 1, 0],
        [0, 0, 0, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 1, 0]], dtype=np.uint8)
    skeleton = np.array([
        [1, 0, 1, 0, 0, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 0, 0, 1, 0, 1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 0, 0, 0, 0, 1, 0]], dtype=np.uint8)
    distances = np.array([
        [2, 0, 2, 0, 0, 0, 0],
        [0, 2, 0, 2, 0, 0, 0],
        [0, 0, 2, 0, 2, 0, 0],
        [0, 0, 0, 2, 0, 2, 0],
        [0, 0, 0, 0, 2, 0, 2],
        [0, 0, 0, 0, 0, 2, 0]], dtype=np.float64)
    pad_skeleton = np.pad(skeleton, [(1,), (1,)], mode='constant')
    pad_distances = np.pad(distances, [(1,), (1,)], mode='constant')
    pad_skeleton, pad_distances = remove_small_loops(pad_skeleton, pad_distances)
    pas_distances_target = np.array([
        [0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 2., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 2., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 2., 2., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 2., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0.]], dtype=float)
    print(f"{test_name}{np.array_equal(pad_distances, pas_distances_target)}")

    test_nb += 1
    test_name = f"Test {test_nb}, weird line tips: "
    cnv4, cnv8 = get_neighbor_comparisons(pad_skeleton)
    potential_tips = get_terminations_and_their_connected_nodes(pad_skeleton, cnv4, cnv8)
    pad_vertices, potential_tips = get_inner_vertices(pad_skeleton, potential_tips, cnv4, cnv8)
    vertices = pad_vertices[1:-1, 1:-1]
    target = np.array([
        [1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0]], dtype=np.uint8)
    print(f"{test_name}{np.array_equal(vertices, target)}")

    test_nb += 1
    test_name = f"Test {test_nb}, another bow loop: "
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
    potential_tips = get_terminations_and_their_connected_nodes(pad_skeleton, cnv4, cnv8)
    pad_vertices, potential_tips = get_inner_vertices(pad_skeleton, potential_tips, cnv4, cnv8)
    vertices = pad_vertices[1:-1, 1:-1]
    target = np.array([
        [0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0]], dtype=np.uint8)
    print(f"{test_name}{np.array_equal(vertices, target)}")


    test_nb += 1
    test_name = f"Test {test_nb}, small loop that comes back with medial axis: "
    skeleton = np.array([
        [0, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 1],
        [1, 1, 0, 1, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 1, 0, 0]], dtype=np.uint8)
    pad_skeleton = np.pad(skeleton, [(1,), (1,)], mode='constant')
    pad_skeleton = remove_small_loops(pad_skeleton)
    new_skeleton = remove_padding([pad_skeleton])[0]
    target = np.array([
       [0, 0, 1, 0, 0, 0],
       [0, 0, 1, 0, 1, 1],
       [1, 1, 1, 1, 0, 0],
       [0, 0, 1, 0, 0, 0],
       [0, 0, 0, 1, 0, 0],
       [0, 0, 0, 1, 0, 0]], dtype=uint8)
    print(f"{test_name}{np.array_equal(new_skeleton, target)}")


    test_nb += 1
    test_name = f"Test {test_nb}, holed cross with large side: "
    skeleton =  np.array([
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 1, 1],
        [0, 1, 0, 1, 1, 0, 0],
        [1, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 1]], dtype=np.uint8)
    pad_skeleton = np.pad(skeleton, [(1,), (1,)], mode='constant')
    pad_skeleton = remove_small_loops(pad_skeleton)
    new_skeleton = remove_padding([pad_skeleton])[0]
    target = np.array([
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 1, 1, 0, 1, 1],
        [0, 1, 0, 0, 1, 0, 0],
        [1, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 1]], dtype=uint8)
    print(f"{test_name}{np.array_equal(new_skeleton, target)}")



def test_get_vertices_and_tips_from_skeleton():
    """
    NEW
    """
    test_nb = 1
    test_name = f"Test {test_nb}, check that no tips gets connected: "
    skeleton = np.array([
        [1, 0, 1, 1, 1, 0, 0, 1, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 1, 0, 0],
        [0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
        [0, 0, 0, 1, 0, 1, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 1, 0, 0],
        [0, 0, 0, 1, 0, 1, 0, 1, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 1, 0, 0]], dtype=np.uint8)
    pad_skeleton = np.pad(skeleton, [(1,), (1,)], mode='constant')
    pad_vertices, pad_tips = get_vertices_and_tips_from_skeleton(pad_skeleton)
    nb, im = cv2.connectedComponents(pad_tips)
    print(f"{test_name}{(im > 0).sum() == (nb - 1) }")

    test_nb = 1
    test_name = f"Test {test_nb}, check that no tips gets connected: "
    skeleton = np.array([
        [0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, 0, 1, 0, 1, 0],
        [0, 1, 1, 1, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 1, 1]], dtype=uint8)
    pad_skeleton = np.pad(skeleton, [(1,), (1,)], mode='constant')
    pad_vertices, pad_tips = get_vertices_and_tips_from_skeleton(pad_skeleton)
    vertices = remove_padding([pad_vertices])[0]

    nb, im = cv2.connectedComponents(pad_tips)
    print(f"{test_name}{(im > 0).sum() == (nb - 1) }")


    test_nb += 1
    test_name = f"Test {test_nb}, check that no tips gets connected: "
    skeleton = np.array([
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [1, 0, 1, 1, 1, 0, 0],
        [0, 1, 0, 1, 0, 1, 1],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0]], dtype=uint8)
    pad_skeleton = np.pad(skeleton, [(1,), (1,)], mode='constant')
    pad_vertices, pad_tips = get_vertices_and_tips_from_skeleton(pad_skeleton)
    vertices = remove_padding([pad_vertices])[0]

    test_nb += 1
    test_name = f"Test {test_nb}, swimming thing: "
    skeleton = np.array([
        [0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 1, 0, 1, 0, 1, 0],
        [0, 0, 1, 1, 1, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0, 0]], dtype=uint8)
    pad_skeleton = np.pad(skeleton, [(1,), (1,)], mode='constant')
    pad_skeleton = keep_one_connected_component(pad_skeleton)
    pad_vertices, pad_tips = get_vertices_and_tips_from_skeleton(pad_skeleton)
    vertices = remove_padding([pad_vertices])[0]
    target = np.array([
        [0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0]], dtype=uint8)
    print(f"{test_name}{np.array_equal(vertices, target)}")

    test_nb += 1
    test_name = f"Test {test_nb}, false tip: "
    skeleton = np.array([
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 1, 1, 1],
        [1, 1, 1, 1, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0]], dtype=uint8)
    pad_skeleton = np.pad(skeleton, [(1,), (1,)], mode='constant')
    pad_skeleton = keep_one_connected_component(pad_skeleton)
    pad_vertices, pad_tips = get_vertices_and_tips_from_skeleton(pad_skeleton)
    vertices = remove_padding([pad_vertices])[0]
    target = np.array([
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1],
        [1, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0]], dtype=uint8)
    print(f"{test_name}{np.array_equal(vertices, target)}")

    test_nb += 1
    test_name = f"Test {test_nb}, false tip: "
    skeleton = np.array([
        [1, 0, 0, 0, 0, 1, 0],
        [0, 1, 0, 0, 0, 1, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 0, 0, 1, 1, 0, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [1, 1, 0, 0, 0, 0, 1]], dtype=np.uint8)
    pad_skeleton = np.pad(skeleton, [(1,), (1,)], mode='constant')
    pad_skeleton = keep_one_connected_component(pad_skeleton)
    pad_vertices, pad_tips = get_vertices_and_tips_from_skeleton(pad_skeleton)
    vertices = remove_padding([pad_vertices])[0]
    print(f"{test_name}{np.array_equal(vertices, target)}")

    test_nb += 1
    test_name = f"Test {test_nb}, false tip: "
    skeleton = np.array([
        [1, 0, 0, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0, 1, 1],
        [0, 1, 1, 1, 1, 0, 0],
        [1, 0, 1, 0, 0, 1, 0],
        [0, 0, 1, 0, 0, 0, 1]], dtype=np.uint8)
    pad_skeleton = np.pad(skeleton, [(1,), (1,)], mode='constant')
    pad_skeleton = keep_one_connected_component(pad_skeleton)
    pad_vertices, pad_tips = get_vertices_and_tips_from_skeleton(pad_skeleton)

    test_nb += 1
    test_name = f"Test {test_nb}, false tip: "
    skeleton = np.array([
        [0, 0, 1, 0],
        [0, 0, 1, 0],
        [1, 1, 1, 0],
        [0, 0, 1, 0],
        [0, 1, 1, 0],
        [1, 0, 0, 1]], dtype=np.uint8)
    pad_skeleton = np.pad(skeleton, [(1,), (1,)], mode='constant')
    pad_skeleton = keep_one_connected_component(pad_skeleton)
    pad_vertices, pad_tips = get_vertices_and_tips_from_skeleton(pad_skeleton)

    test_nb += 1
    test_name = f"Test {test_nb}, false tip: "
    skeleton = np.array([
        [0, 1, 0, 0, 1, 0, 0],
        [0, 0, 1, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 1, 1],
        [0, 1, 1, 1, 1, 0, 0],
        [1, 0, 0, 0, 0, 1, 0]], dtype=np.uint8)
    pad_skeleton = np.pad(skeleton, [(1,), (1,)], mode='constant')
    pad_skeleton = keep_one_connected_component(pad_skeleton)
    pad_vertices, pad_tips = get_vertices_and_tips_from_skeleton(pad_skeleton)

    test_nb += 1
    test_name = f"Test {test_nb}, false tip: "
    skeleton = np.array([
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0],
        [1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1]], dtype=uint8)
    pad_skeleton = np.pad(skeleton, [(1,), (1,)], mode='constant')
    pad_skeleton = keep_one_connected_component(pad_skeleton)
    pad_vertices, pad_tips = get_vertices_and_tips_from_skeleton(pad_skeleton)

    test_nb += 1
    test_name = f"Test {test_nb}, false tip: "
    skeleton = np.array([
        [0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0],
        [0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
        [0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0],
        [1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        [0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1],
        [0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0],
        [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=uint8)
    pad_skeleton = np.pad(skeleton, [(1,), (1,)], mode='constant')
    pad_skeleton = keep_one_connected_component(pad_skeleton)
    pad_vertices, pad_tips = get_vertices_and_tips_from_skeleton(pad_skeleton)

    test_nb += 1
    test_name = f"Test {test_nb}, cross with two edges on one branch: "
    skeleton = np.array([
        [0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0],
        [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
        [1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0],
        [0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1],
        [0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0],
        [1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]], dtype=uint8)
    pad_skeleton = np.pad(skeleton, [(1,), (1,)], mode='constant')
    pad_skeleton = keep_one_connected_component(pad_skeleton)
    pad_vertices, pad_tips = get_vertices_and_tips_from_skeleton(pad_skeleton)


    test_nb += 1
    test_name = f"Test {test_nb}, cross with two edges on one branch: "
    skeleton = np.array([
        [1, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 1],
        [0, 0, 1, 0, 1, 0],
        [1, 1, 1, 1, 0, 0],
        [0, 0, 0, 0, 1, 1]], dtype=np.uint8)
    pad_skeleton = np.pad(skeleton, [(1,), (1,)], mode='constant')
    pad_skeleton = keep_one_connected_component(pad_skeleton)
    pad_vertices, pad_tips = get_vertices_and_tips_from_skeleton(pad_skeleton)



    test_nb += 1
    test_name = f"Test {test_nb}, cross with two edges on one branch: "
    skeleton = np.array([
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1],
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0],
        [1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]], dtype=np.uint8)
    pad_skeleton = np.pad(skeleton, [(1,), (1,)], mode='constant')
    pad_skeleton = keep_one_connected_component(pad_skeleton)
    pad_vertices, pad_tips = get_vertices_and_tips_from_skeleton(pad_skeleton)



def test_get_branches_and_tips_coord():
    """
    NEW
    """
    test_nb = 1
    test_name = f"Test {test_nb}, check tip and branching vertex coord: "
    skeleton = np.array([
        [1, 0, 1, 1, 1, 0, 0, 1, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 1, 0, 0],
        [0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
        [0, 0, 0, 1, 0, 1, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 1, 0, 0],
        [0, 0, 0, 1, 0, 1, 0, 1, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 1, 0, 0]], dtype=np.uint8)
    pad_skeleton = np.pad(skeleton, [(1,), (1,)], mode='constant')
    pad_vertices, pad_tips = get_vertices_and_tips_from_skeleton(pad_skeleton)
    non_tip_vertices, tips_coord = get_branches_and_tips_coord(pad_vertices, pad_tips)
    vertices_and_tips = np.zeros_like(pad_skeleton)
    vertices_and_tips[tips_coord[:, 0], tips_coord[:, 1]] = 1
    vertices_and_tips[non_tip_vertices[:, 0], non_tip_vertices[:, 1]] = 2
    vertices_and_tips_target = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                               [0, 0, 0, 1, 2, 1, 0, 0, 1, 0, 0, 0],
                                               [0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0],
                                               [0, 0, 1, 0, 0, 2, 0, 0, 2, 0, 0, 0],
                                               [0, 0, 0, 0, 2, 0, 2, 0, 0, 0, 0, 0],
                                               [0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0],
                                               [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                                               [0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0],
                                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=uint8)
    print(f"{test_name}{np.array_equal(vertices_and_tips, vertices_and_tips_target)}")


def test_find_closest_vertices():
    """
    NEW HERE IS THE NEW WAY
    """
    test_nb = 1
    skeleton = np.array([
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 1, 0, 0]], dtype=np.uint8)

    pad_skeleton = np.pad(skeleton, [(1,), (1,)], mode='constant')
    pad_distances = pad_skeleton.copy()

    pad_vertices, pad_tips = get_vertices_and_tips_from_skeleton(pad_skeleton)
    non_tip_vertices, tips_coord = get_branches_and_tips_coord(pad_vertices, pad_tips)
    all_vertices_coord = np.vstack((tips_coord, non_tip_vertices))
    starting_vertices_coord = tips_coord
    # Manually:
    # all_vertices_coord = np.array([[0, 7], [2, 0], [2, 8], [4, 2], [6, 3], [6, 7], [4, 3], [2, 7], [4, 7]])
    # all_vertices = np.zeros_like(skeleton)
    # all_vertices[all_vertices_coord[:, 0], all_vertices_coord[:, 1]] = 1
    # starting_vertices_coord = np.array([[0, 7], [2, 0], [2, 8], [4, 2], [6, 3], [6, 7]])
    # all_vertices[starting_vertices_coord[:, 0], starting_vertices_coord[:, 1]] = 2

    # vertex_coord_with_edge_id, edge_lengths, edge_pix_coord = find_closest_vertices(pad_skeleton, all_vertices_coord, starting_vertices_coord)
    vertex_coord_with_edge_id, edge_lengths, edge_pix_coord = find_closest_vertices(pad_skeleton, non_tip_vertices, starting_vertices_coord)

    pad_skeleton, pad_distances, tips_coord, non_tip_vertices, edge_pix_coord, vertices_branching_tips, ordered_v_coord, numbered_vertices, edges_labels = remove_tipped_edge_smaller_than_branch_width(
        pad_skeleton, pad_distances, tips_coord, non_tip_vertices, edge_pix_coord, vertex_coord_with_edge_id, edge_lengths)

    computed_vertices = remove_padding([numbered_vertices])[0]
    target = np.array([
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [2, 0, 0, 0, 0, 0, 0, 3, 4, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 5, 6, 0, 0, 0, 7, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 8, 0, 0, 0, 9, 0, 0]], dtype=np.uint32)
    test_name = f"Test {test_nb}, numbered_vertices: "
    print(f"{test_name}{np.array_equal(computed_vertices, target)}")

    test_nb += 1
    test_name = f"Test {test_nb}, check that writting edge id do not overwrite vertex id: "
    edge_vertex_id = numbered_vertices.copy()
    edge_vertex_id[edge_pix_coord[:, 0], edge_pix_coord[:, 1]] = edge_pix_coord[:, 2]
    edge_vertex_id_overwritten = edge_vertex_id.copy()
    edge_vertex_id_overwritten[np.nonzero(numbered_vertices)] = numbered_vertices[np.nonzero(numbered_vertices)]
    print(f"{test_name}{np.array_equal(edge_vertex_id, edge_vertex_id_overwritten)}")

    for i in edges_labels[:, 0]:
        test_nb += 1
        test_name = f"Test {test_nb}, adding vertices to edge {i} shows equal edge lengths as before: "
        complete_edge_id = np.zeros_like(numbered_vertices)
        first_vertex = edges_labels[edges_labels[:, 0] == i, 1]
        second_vertex = edges_labels[edges_labels[:, 0] == i, 2]
        complete_edge_id[numbered_vertices == first_vertex] = i
        complete_edge_id[numbered_vertices == second_vertex] = i
        complete_edge_id[edge_pix_coord[edge_pix_coord[:, 2] == i, 0], edge_pix_coord[edge_pix_coord[:, 2] == i, 1]] = i
        print(f"{test_name}{((complete_edge_id == i).sum() - 1) == edge_lengths[i - 1]}")

    edges_labels, edge_pix_coord, edge_lengths  = identify_other_edges(pad_skeleton, edges_labels, edge_pix_coord, edge_lengths, numbered_vertices, tips_coord, non_tip_vertices, vertices_branching_tips)

    for i in edges_labels[:, 0]:
        test_nb += 1
        test_name = f"Test {test_nb}, adding vertices to edge {i} shows equal edge lengths as before: "
        complete_edge_id = np.zeros_like(numbered_vertices)
        first_vertex = edges_labels[edges_labels[:, 0] == i, 1]
        second_vertex = edges_labels[edges_labels[:, 0] == i, 2]
        complete_edge_id[numbered_vertices == first_vertex] = i
        complete_edge_id[numbered_vertices == second_vertex] = i
        complete_edge_id[edge_pix_coord[edge_pix_coord[:, 2] == i, 0], edge_pix_coord[edge_pix_coord[:, 2] == i, 1]] = i
        print(f"{test_name}{((complete_edge_id == i).sum() - 1) == edge_lengths[i - 1]}")
    test_nb += 1
    test_name = f"Test {test_nb}, check that no vertex is in edge_pix_coord"
    test1 = np.zeros_like(pad_skeleton)
    test1[edge_pix_coord[:, 0], edge_pix_coord[:, 1]] = 1
    test2 = np.zeros_like(pad_skeleton)
    test2[vertices_branching_tips[:, 0], vertices_branching_tips[:, 1]] = 1
    print(f"{test_name}{(test1 * test2).sum() == 0}")


    ############## STOPED HERE ##################

def test_identify_all_vertices_and_edges():
    """
    NEW
    """
    test_nb = 1
    test_name = f"Test {test_nb}, detect edges connecting tips: "
    skeleton = np.array([
        [0, 0, 1, 1, 1, 0, 0, 1, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 1, 0, 0],
        [0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
        [0, 0, 0, 1, 0, 1, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 1, 0, 0],
        [0, 0, 0, 1, 0, 1, 0, 1, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 1, 0, 0]], dtype=np.uint8)
    pad_skeleton = np.pad(skeleton, [(1,), (1,)], mode='constant')
    pad_vertices, pad_tips = get_vertices_and_tips_from_skeleton(pad_skeleton)
    non_tip_vertices, tips_coord = get_branches_and_tips_coord(pad_vertices, pad_tips)
    vertex_coord_with_edge_id, edge_lengths, edge_pix_coord = find_closest_vertices(pad_skeleton,
                                                                                        non_tip_vertices,
                                                                                        tips_coord)
    only_edges_connecting_tips = np.zeros_like(pad_skeleton)
    only_edges_connecting_tips[edge_pix_coord[:, 0], edge_pix_coord[:, 1]] = 1
    only_edges_connecting_tips = remove_padding([only_edges_connecting_tips])[0]
    # only_edges_connecting_tips_target = np.array([
    #     [0, 0, 1, 1, 1, 0, 0, 1, 0, 0],
    #     [0, 0, 0, 1, 0, 0, 0, 1, 0, 0],
    #     [0, 1, 1, 0, 0, 0, 0, 1, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    #     [0, 0, 0, 0, 1, 0, 0, 1, 0, 0],
    #     [0, 0, 0, 1, 0, 1, 0, 1, 0, 0],
    #     [0, 0, 1, 0, 0, 0, 0, 1, 0, 0]], dtype=np.uint8)
    # print(f"{test_name}{np.array_equal(only_edges_connecting_tips, only_edges_connecting_tips_target)}")

    test_nb += 1
    test_name = f"Test {test_nb}, detect vertices connecting edges connecting tips: "

    vertices_connecting_edges_connecting_tips = np.zeros_like(pad_skeleton)
    vertices_connecting_edges_connecting_tips[vertex_coord_with_edge_id[:, 0], vertex_coord_with_edge_id[:, 1]] = 1
    vertices_connecting_edges_connecting_tips = remove_padding([vertices_connecting_edges_connecting_tips])[0]
    vertices_connecting_edges_connecting_tips_target = np.array([
       [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=uint8)
    print(f"{test_name}{np.array_equal(vertices_connecting_edges_connecting_tips, vertices_connecting_edges_connecting_tips_target)}")

    test_nb += 1
    test_name = f"Test {test_nb}, the whole network is described in edge_pix_coord: "
    pad_distances = pad_skeleton.copy()
    pad_skeleton, pad_distances, edges_labels, numbered_vertices, edge_pix_coord, tips_coord, non_tip_vertices, vertices_branching_tips = remove_tipped_edge_smaller_than_branch_width(
        pad_skeleton, vertex_coord_with_edge_id, pad_distances, edge_lengths, tips_coord, non_tip_vertices,
        edge_pix_coord)


    edges_labels, edge_pix_coord = identify_other_edges(pad_skeleton, edges_labels, numbered_vertices,
                                                            edge_pix_coord, tips_coord, non_tip_vertices,
                                                            vertices_branching_tips)
    all_edges = np.zeros_like(pad_skeleton)
    all_edges[edge_pix_coord[:, 0], edge_pix_coord[:, 1]] = 1
    all_edges = remove_padding([all_edges])[0]
    print(f"{test_name}{np.array_equal(all_edges, skeleton)}")


    test_nb += 1
    test_name = f"Test {test_nb}, find closest vertices: "
    nb_skeleton = np.array([
        [0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 10, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 2, 10, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 10, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 3, 15, 0],
        [0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 3, 3, 3, 0, 2, 0],
        [10, 0, 0, 0, 0, 0, 0, 0, 10, 0, 3, 15, 0, 0, 0, 0, 2, 0],
        [0, 2, 2, 2, 0, 0, 0, 3, 0, 3, 0, 2, 0, 0, 0, 0, 10, 0],
        [0, 0, 0, 0, 2, 2, 15, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 2, 2, 0, 10, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [3, 3, 3, 3, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.uint8)
    skeleton = (nb_skeleton > 0).astype(np.uint8)
    pad_skeleton = add_padding([skeleton])[0]
    pad_distances = pad_skeleton.copy()
    pad_vertices, pad_tips = get_vertices_and_tips_from_skeleton(pad_skeleton)
    non_tip_vertices, tips_coord = get_branches_and_tips_coord(pad_vertices, pad_tips)
    vertex_coord_with_edge_id, edge_lengths, edge_pix_coord = find_closest_vertices(pad_skeleton,
                                                                                        non_tip_vertices,
                                                                                        tips_coord)

    pad_skeleton, pad_distances, edges_labels, numbered_vertices, edge_pix_coord, tips_coord, non_tip_vertices, vertices_branching_tips = remove_tipped_edge_smaller_than_branch_width(
        pad_skeleton, vertex_coord_with_edge_id, pad_distances, edge_lengths, tips_coord, non_tip_vertices,
        edge_pix_coord)

    edges_labels, edge_pix_coord = identify_other_edges(pad_skeleton, edges_labels, numbered_vertices,
                                                            edge_pix_coord, tips_coord, non_tip_vertices,
                                                            vertices_branching_tips)



########## OLD ###########
def test_get_graph_from_vertices_and_edges():
    test_nb = 1
    test_name = f"Test {test_nb}, small loop: "
    skeleton = np.array([
        [0, 0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0, 1],
        [1, 1, 0, 1, 1, 0],
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 1, 0]], dtype=np.uint8)
    pad_skeleton = np.pad(skeleton, [(1,), (1,)], mode='constant')
    pad_vertices, pad_tips = get_vertices_and_tips_from_skeleton(pad_skeleton)
    numbered_vertices, numbered_edges, vertices_table, edges_labels = get_graph_from_vertices_and_edges(pad_vertices, pad_tips, pad_distances)
    target = skeleton.copy()
    print(f"{test_name}{np.array_equal(((numbered_vertices + numbered_edges) > 0).astype(np.uint8), target)}")


    test_nb += 1
    test_name = f"Test {test_nb}, small loop: "
    skeleton = np.array([
        [0, 0, 0, 1, 1, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 1, 1, 1],
        [1, 1, 0, 1, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0]], dtype=np.uint8)
    pad_vertices, pad_tips = get_vertices_and_tips_from_skeleton(skeleton)
    numbered_vertices, numbered_edges, vertices_table, edges_labels = get_graph_from_vertices_and_edges(pad_vertices, pad_tips)
    target = skeleton.copy()
    print(f"{test_name}{np.array_equal(((numbered_vertices + numbered_edges) > 0).astype(np.uint8), target)}")


def test_get_tipped_edges():
    test_nb = 1
    test_name = f"Test {test_nb}, bigger network: "
    skeleton = np.array([
        [0, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 1, 0],
        [0, 1, 1, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 1, 1, 1],
        [0, 0, 0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 1, 1, 0]], dtype=np.uint8)
    pad_skeleton = np.pad(skeleton, [(1,), (1,)], mode='constant')
    pad_vertices, pad_tips = get_vertices_and_tips_from_skeleton(pad_skeleton)
    nb_e, tempo_numbered_edges, numbered_branches, numbered_tips = get_numbered_edges_and_vertices(pad_skeleton, pad_vertices, pad_tips)




    pad_skeleton = np.array([
        [0, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 2, 0],
        [0, 2, 1, 2, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 2, 2, 2],
        [0, 0, 0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 1, 2, 0]], dtype=np.uint8)
    skeleton = np.array([
        [0, 0, 1, 1, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 0, 1, 0],
        [0, 1, 1, 1, 1, 1, 1, 1, 0],
        [0, 0, 0, 1, 0, 1, 0, 1, 0],
        [0, 0, 0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 1, 0, 1, 0],
        [0, 0, 1, 0, 0, 0, 0, 1, 0]], dtype=np.uint8)


    pad_distances = pad_skeleton.copy()
    pad_skeleton, distances = remove_small_loops(pad_skeleton, pad_distances)
    cnv4, cnv8 = get_neighbor_comparisons(pad_skeleton)
    pad_tips = get_terminations_and_their_connected_nodes(pad_skeleton, cnv4, cnv8)
    pad_vertices = get_inner_vertices(pad_tips, cnv4, cnv8)
####### OLD #####

if __name__ == '__main__':
    test_get_terminations_and_their_connected_nodes()
    test_get_inner_vertices()