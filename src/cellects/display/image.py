#!/usr/bin/env python3
"""
This script contains functions to display images.
"""
import cv2
import numpy as np
from numpy.typing import NDArray
from matplotlib import pyplot as plt
from cellects.image.morphological_operations import get_line_points, cross_33
from cellects.display.param import dark_grey_rgb, crimson_rgb, lightgreen_rgb, yellow_rgb


def show(img, interactive: bool=True, cmap=None, axes: bool=True, show: bool=True):
    """
    Display a 2D image using Matplotlib.

    Parameters
    ----------
    img
        Image data array with shape (height, width) or (height, width,
        channels).  Must be convertible to a NumPy array.
    interactive
        If ``True``, enable Matplotlib's interactive mode (``ion``);
        otherwise disable it (``ioff``).  Default is ``True``.
    cmap
        Colormap to apply when displaying the image.  If ``None`` the
        default image colors are used.  Default is ``None``.
    axes
        When ``False``, hide axis ticks and labels.  Default is ``True``.
    show
        If ``True``, call ``fig.tight_layout()`` and display the figure
        immediately.  If ``False``, the figure can be further modified
        before being shown.  Default is ``True``.

    Returns
    -------
    fig
        The created Matplotlib ``Figure`` object.
    ax
        The ``Axes`` instance on which the image is drawn.

    Notes
    -----
    This function alters Matplotlib's global interactive state, which may
    affect subsequent plotting commands.  Use ``interactive=False`` when
    creating figures programmatically to avoid unintended side effects.
    """
    if interactive:
        plt.ion()
    else:
        plt.ioff()
    sizes = img.shape[0] / 100,  img.shape[1] / 100
    fig = plt.figure(figsize=(sizes[1], sizes[0]))
    ax = fig.gca()
    if cmap is None:
        ax.imshow(img, interpolation="none", extent=(0, sizes[1], 0, sizes[0]))
    else:
        ax.imshow(img, cmap=cmap, interpolation="none", extent=(0, sizes[1], 0, sizes[0]))
    if not axes:
        ax.axis('off')
    if show:
        fig.tight_layout()
        fig.show()

    return fig, ax


def zoom_on_nonzero(binary_image:NDArray, padding: int = 2, return_coord: bool=True):
    """
    Crops a binary image around non-zero elements with optional padding and returns either coordinates or cropped region.

    Parameters
    ----------
    binary_image : NDArray
        2D NumPy array containing binary values (0/1)
    padding : int, default=2
        Amount of zero-padding to add around the minimum bounding box
    return_coord : bool, default=True
        If True, return slice coordinates instead of cropped image

    Returns
    -------
        If `return_coord` is True: [y_min, y_max, x_min, x_max] as 4-element Tuple.
        If False: 2D binary array representing the cropped region defined by non-zero elements plus padding.

    Examples
    --------
    >>> img = np.zeros((10,10))
    >>> img[3:7,4:6] = 1
    >>> result = zoom_on_nonzero(img)
    >>> print(result)
    [1 8 2 7]
    >>> cropped = zoom_on_nonzero(img, return_coord=False)
    >>> print(cropped.shape)
    (6, 5)

    Notes
    -----
    - Returns empty slice coordinates if input contains no non-zero elements.
    - Coordinate indices are 0-based and compatible with NumPy array slicing syntax.
    """
    y, x = np.nonzero(binary_image)
    cy_min = np.max((0, y.min() - padding))
    cy_max = np.min((binary_image.shape[0], y.max() + padding + 1))
    cx_min = np.max((0, x.min() - padding))
    cx_max = np.min((binary_image.shape[1], x.max() + padding + 1))
    if return_coord:
        return cy_min, cy_max, cx_min, cx_max
    else:
        return binary_image[cy_min:cy_max, cx_min:cx_max]

def display_boxes(binary_image: NDArray, box_diameter: int, show: bool = True):
    """
    Display grid lines on a binary image at specified box diameter intervals.

    This function displays the given binary image with vertical and horizontal
    grid lines drawn at regular intervals defined by `box_diameter`. The function
    returns the total number of grid lines drawn.

    Parameters
    ----------
    binary_image : ndarray
        Binary image on which to draw the grid lines.
    box_diameter : int
        Diameter of each box in pixels.

    Returns
    -------
    line_nb : int
        Number of grid lines drawn, both vertical and horizontal.

    Examples
    --------
    >>> import numpy as np
    >>> binary_image = np.random.randint(0, 2, (100, 100), dtype=np.uint8)
    >>> display_boxes(binary_image, box_diameter=25)
    """
    plt.imshow(binary_image, cmap='gray', extent=(0, binary_image.shape[1], 0, binary_image.shape[0]))
    height, width = binary_image.shape
    line_nb = 0
    for x in range(0, width + 1, box_diameter):
        line_nb += 1
        plt.axvline(x=x, color='white', linewidth=1)
    for y in range(0, height + 1, box_diameter):
        line_nb += 1
        plt.axhline(y=y, color='white', linewidth=1)

    if show:
        plt.show()

    return line_nb


def display_network_methods(network_detection: object, save_path: str=None):
    """

    Display segmentation results from a network detection object.

    Extended Description
    --------------------

    Plots the binary segmentation results for various methods stored in ``network_detection.all_results``.
    Highlights the best result based on quality metrics and allows for saving the figure to a file.

    Parameters
    ----------
    network_detection : object
        An object containing segmentation results and quality metrics.
    save_path : str, optional
        Path to save the figure. If ``None``, the plot is displayed.

    """
    row_nb = 6
    fig, axes = plt.subplots(int(np.ceil(len(network_detection.all_results) / row_nb)), row_nb, figsize=(100, 100))
    fig.suptitle(f'Segmentation Comparison: Frangi + Sato Variations', fontsize=16)

    # Plot all results
    for idx, result in enumerate(network_detection.all_results):
        row = idx // row_nb
        col = idx % row_nb

        ax = axes[row, col]

        # Display binary segmentation result
        ax.imshow(result['binary'], cmap='gray')

        # Create title with filter info and quality score
        title = f"{result['method']}: {str(np.round(network_detection.quality_metrics[idx], 0))}"

        # Highlight the best result
        if idx == network_detection.best_idx:
            ax.set_title(title, fontsize=8, color='red', fontweight='bold')
            ax.add_patch(plt.Rectangle((0, 0), result['binary'].shape[1] - 1,
                                       result['binary'].shape[0] - 1,
                                       fill=False, edgecolor='red', linewidth=3))
        else:
            ax.set_title(title, fontsize=8)

        ax.axis('off')
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0., transparent=True, dpi=500)
        plt.close()
    else:
        plt.show()

def rgb_gradient(value:int, min_val: int=0, max_val: int=None):
    # Normalize the value to a range between 0 and 1
    if max_val is None:
        max_val = 3
        if value > 3:
            value = 3
    normalized = (value - min_val) / (max_val - min_val)

    # Define the colors at the endpoints of the gradient
    start_color = (0, 0, 255)   # blue
    end_color = (255, 40, 40)     #

    # # Compute the RGB values based on the normalized value
    red = start_color[0] * (1 - normalized) + end_color[0] * normalized
    green = start_color[1] * (1 - normalized) + end_color[1] * normalized
    blue = start_color[2] * (1 - normalized) + end_color[2] * normalized

    return int(red), int(green), int(blue)

def draw_graph(img: NDArray[np.uint8], vertices, edges, cell_contours=None):
    graph = np.full((img.shape[0], img.shape[1], 3), 255, dtype=np.uint8)
    if cell_contours is not None:
        # i) Draw the cell contour on the graph:
        graph[cell_contours > 0] = 200
    # ii) Draw the shortest path of every edge on the graph
    edge_names = np.unique(edges['edge_id'])
    min_width, max_width = edges['average_width'].min(), edges['average_width'].max()
    for edge_id in edge_names:
        edge = edges.loc[edges['edge_id'] == edge_id, :]
        v1 = vertices.loc[vertices['vertex_id'] == int(edge['vertex1'].values[0]), :]
        v2 = vertices.loc[vertices['vertex_id'] == int(edge['vertex2'].values[0]), :]
        v1_coord = v1['y'].values[0], v1['x'].values[0]
        v2_coord = v2['y'].values[0], v2['x'].values[0]
        shortest_path = get_line_points(v1_coord, v2_coord)

        if np.isnan(edge['average_width'].values[0]):
            graph[shortest_path[:, 0], shortest_path[:, 1], :] = crimson_rgb
        else:
            graph[shortest_path[:, 0], shortest_path[:, 1], :] = rgb_gradient(edge['average_width'].values[0],
                                                                              min_width)

    # Draw a green cross on branching vertices
    vertex_img = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    branching_vertices = vertices.loc[vertices['is_tip'] == 0, :]
    vertex_img[branching_vertices['y'], branching_vertices['x']] = 1
    dil_v_coord = np.nonzero(cv2.dilate(vertex_img, cross_33))
    graph[dil_v_coord[0], dil_v_coord[1], :] = lightgreen_rgb  # 255, 165, 0

    # Draw a red cross on food vertices
    origin_img = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    origin_vertices = vertices.loc[vertices['origin'] == 1, :]
    origin_img[origin_vertices['y'], origin_vertices['x']] = 1
    dil_o_coord = np.nonzero(cv2.dilate(origin_img, cross_33))
    graph[dil_o_coord[0], dil_o_coord[1], :] = yellow_rgb

    # Draw a blue cross on tips
    tips_img = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    tips = vertices.loc[vertices['is_tip'] == 1, :]
    tips_img[tips['y'], tips['x']] = 1
    dil_tips_coord = np.nonzero(cv2.dilate(tips_img, cross_33))
    graph[dil_tips_coord[0], dil_tips_coord[1], :] = dark_grey_rgb

    # Draw a black dot on all vertices
    graph[vertices['y'], vertices['x'], :] = 20, 20, 20

    return graph