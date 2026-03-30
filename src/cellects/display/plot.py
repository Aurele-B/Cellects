#!/usr/bin/env python3
"""
This script contains functions to make graphical representation of the dynamical descriptors extracted from Cellects
"""
import matplotlib.ticker as mticker
from numpy.typing import NDArray
from cellects.display.param import axes_label_dict, cblind, curve_width, curve_alpha, dark_grey_hexa, font_plot_titles, font_plot_ticks, teal_hexa, font_size
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
from cellects.image.morphological_operations import cross_33
from cellects.utils.formulas import eudist



def scientific(x, pos):
    return f'{x:.1e}'

def curve(y, x=None, y_label: str="", x_label: str="", color=None, save_path: str=None):
    """
    Plot a one‑dimensional curve with customizable appearance.

    Parameters
    ----------
    y : array-like
        Sequence of y‑coordinates to plot.
    x : array-like, optional
        Sequence of x‑coordinates. If ``None``, an integer range
        ``0 .. len(y)-1`` is generated. Must have the same length as
        `y` when provided.
    y_label : str, optional
        Label for the y‑axis. Default is ``""``.
    x_label : str, optional
        Label for the x‑axis. Default is ``""``.
    color : color spec, optional
        Matplotlib colour specification used for the line. If ``None``,
        a predefined teal colour is applied.
    save_path : str, optional
        File path to save the figure. If ``None``, the plot is shown
        interactively; otherwise the figure is saved and closed.

    Returns
    -------
    None
        The function either displays the figure or writes it to
        `save_path`.

    Raises
    ------
    ValueError
        If `x` is provided and its length does not match ``len(y)``.

    Notes
    -----
    * If all values in `y` are greater than ``10000`` or smaller than
      ``0.001``, the y‑axis uses a scientific‑notation formatter.
    * The figure size is fixed at 10 × 10 inches; saved figures use a DPI
      of 500 with a tight layout and minimal padding.
    * ``transparent=True`` is used when saving, making the background
      invisible.

    Examples
    --------
    >>> y = [i**2 for i in range(10)]
    >>> curve(y, y_label='y', x_label='x')
    # A window displaying the plot of y = x² appears.

    >>> x = list(range(10))
    >>> curve(y, x=x, color='red', save_path='quadratic.png')
    # The plot is saved to ``quadratic.png`` without being shown.
    """
    fig, ax = plt.subplots(figsize=(10, 10))

    # Plot the data using one of the specified colors
    if x is None:
        x = np.arange(0, y.shape[0])
    if color is None:
        color = teal_hexa
    ax.plot(x, y, color=color, linewidth=6)
    if (y > 10000).all() or (y < 0.001).all():
        ax.yaxis.set_major_formatter(FuncFormatter(scientific))
    # Adding labels and title
    ax.set_xlabel(x_label, fontsize=font_plot_titles)
    ax.set_ylabel(y_label, fontsize=font_plot_titles)

    # Improve the aesthetics
    ax.tick_params(axis='both', which='major', labelsize=font_plot_ticks)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_ylim(ymin=y.min(), ymax=y.max())
    ax.set_xlim(xmin=x.min(), xmax=x.max())

    # Adjust the layout
    plt.tight_layout()
    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1, transparent=True, dpi=500)
        plt.close()


def plot_growth_features(y: NDArray, time_step: float, growth_features: dict, save_path: str=None):
    """
    Plot growth surface data with linear and exponential fits, and rupture markers.

    Parameters
    ----------
    y :
        Array of surface area measurements.
    time_step :
        Time interval (in minutes) between successive measurements.
    growth_features :
        Dictionary containing regression coefficients, fit intervals,
        rupture information, and optional R‑squared values. Expected keys
        include ``'lin_growth_rate_mm2s'``, ``'lin_intercept'``,
        ``'lin_start'``, ``'lin_end'``, ``'exp_growth_rate_mm2s'``,
        ``'exp_intercept'``, ``'exp_start'``, ``'exp_end'``,
        ``'growth_rupture_time_min'``, and ``'growth_rupture_surface_mm2'``.
        Optional keys ``'lin_r_squared'`` / ``'lin_r2'`` and
        ``'exp_r_squared'`` / ``'exp_r2'`` are used for legend display.
    save_path : optional
        File path where the figure will be saved. If ``None``, the plot is
        shown interactively via ``plt.show()``; otherwise ``plt.savefig`` is
        called with high‑resolution settings.

    Returns
    -------
    None

    Notes
    -----
    * The function uses Matplotlib to construct a figure with a reduced
      right margin (``fig.subplots_adjust(right=0.85)``) so that the custom
      legend can be placed inside the reserved space without being clipped.
    * The legend is anchored at a fixed figure coordinate ``(.8, .5)``; this
      works well for typical figure sizes but may require adjustment for
      unusually wide or tall plots.
    * All numeric values displayed in the legend are rounded to two decimal
      places for readability.

    """
    x = np.arange(0, y.shape[0] * time_step, time_step)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, y, color=dark_grey_hexa)

    # ----- linear fit ------------------------------------------------
    lin_reg = growth_features['lin_growth_rate_mm2s'] * x + growth_features['lin_intercept']
    to_draw = np.logical_and(x > growth_features['lin_start'], x < growth_features['lin_end'])
    ax.plot(x[to_draw], lin_reg[to_draw],
            color=cblind['darkblue'], linewidth=curve_width, alpha=curve_alpha)
    ax.axvline(growth_features['lin_start'], color=cblind['darkblue'],
               linestyle='--', alpha=curve_alpha)
    ax.axvline(growth_features['lin_end'],   color=cblind['darkblue'],
               linestyle='--', alpha=curve_alpha)

    # ----- exponential fit -------------------------------------------
    exp_reg = np.exp(growth_features['exp_growth_rate_mm2s'] * x + growth_features['exp_intercept'])
    to_draw = np.logical_and(x > growth_features['exp_start'], x < growth_features['exp_end'])
    ax.plot(x[to_draw], exp_reg[to_draw],
            color=cblind['darkgreen'], linewidth=curve_width, alpha=curve_alpha)
    ax.axvline(growth_features['exp_start'], color=cblind['darkgreen'],
               linestyle='--', alpha=curve_alpha)
    ax.axvline(growth_features['exp_end'],   color=cblind['darkgreen'],
               linestyle='--', alpha=curve_alpha)

    # ----- rupture markers --------------------------------------------
    ax.axvline(growth_features['growth_rupture_time_min'],
               color=cblind['bordeaux'], linestyle=':')
    ax.axhline(growth_features['growth_rupture_surface_mm2'],
               color=cblind['bordeaux'], linestyle=':')

    # Prepare La‑TeX legend strings (fixed key names)
    lin_r2 = round(growth_features.get('lin_r_squared',
                                       growth_features.get('lin_r2', np.nan)), 2)
    exp_r2 = round(growth_features.get('exp_r_squared',
                                       growth_features.get('exp_r2', np.nan)), 2)

    # Round coefficients to two decimals (readability)
    lin_a = round(growth_features['lin_growth_rate_mm2s'], 2)
    lin_b = round(growth_features['lin_intercept'], 2)
    exp_a = round(growth_features['exp_growth_rate_mm2s'], 2)
    exp_b = round(growth_features['exp_intercept'], 2)

    # La‑TeX formatted equations
    if lin_b > 0:
        lin_eq_latex = rf"$y = {lin_a}x + {lin_b}$"
    else:
        lin_eq_latex = rf"$y = {lin_a}x {lin_b}$"
    if exp_b > 0:
        exp_eq_latex = rf"$y = e^{{{exp_a}\,x + {exp_b}}}$"
    else:
        exp_eq_latex = rf"$y = e^{{{exp_a}\,x {exp_b}}}$"

    fig.subplots_adjust(right=0.85)

    # Custom legend
    legend_handles = [
        # Linear regression – visible line + title
        Line2D([0], [0], color=cblind['darkblue'], lw=2,
               label=rf"Linear fit:"),
        Line2D([0], [0], color='none', lw=0,
               label=lin_eq_latex),
        Line2D([0], [0], color='none', lw=0,
               label=rf"$R²={lin_r2}$"),

        # Exponential regression – visible line + title
        Line2D([0], [0], color=cblind['darkgreen'], lw=2,
               label=rf"Exponential fit:"),
        Line2D([0], [0], color='none', lw=0,
               label=exp_eq_latex),
        Line2D([0], [0], color='none', lw=0,
               label=rf"$R²={exp_r2}$"),

        # Rupture marker – single visible entry
        Line2D([0], [0], color=cblind['bordeaux'], lw=2,
               ls=':', label="Growth rupture")
    ]

    # Anchor
    legend_anchor = (.8, .5)

    # Anchor the legend a little inside the reserved margin
    legend = ax.legend(handles=legend_handles,
                       loc='center left',
                       bbox_to_anchor=legend_anchor,
                       fontsize=font_size,
                       frameon=False)   # no contour/box

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_ylim(ymin=0)
    ax.set_xlim(xmin=0)

    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path,
                    bbox_inches='tight',
                    pad_inches=0.1,
                    transparent=True,
                    dpi=500,
                    bbox_extra_artists=(legend,))
        plt.close()


def draw_arrow(canvas, start_point, end_point, color=(0, 0, 0), thickness=2, head_size=15):
    """
    Draw an anti‑aliased arrow on an image.

    Parameters
    ----------
    canvas : ndarray
        BGR image on which the arrow is rendered; the array is modified
        in‑place.
    start_point : tuple of int
        ``(x, y)`` coordinates of the arrow tail.
    end_point : tuple of int
        ``(x, y)`` coordinates of the arrow tip.
    color : tuple of int, optional
        BGR colour of the arrow; defaults to black ``(0, 0, 0)``.
    thickness : int, optional
        Line thickness of the arrow shaft; defaults to ``2``.
    head_size : int, optional
        Length of each side of the triangular arrowhead; defaults to ``15``.

    Returns
    -------
    None
        The function modifies ``canvas`` directly and does not return a value.

    Notes
    -----
    * The function uses OpenCV's ``cv2.line`` with ``LINE_AA`` for
      anti‑aliased drawing.
    * The shaft is shortened by ``head_size`` pixels to avoid overlapping the
      filled arrowhead drawn with ``cv2.fillConvexPoly``.
    * ``head_size`` is interpreted in pixel units; very large values may cause
      the arrowhead to exceed the image boundaries.

    Examples
    --------
    >>> canvas = np.zeros((120, 160, 3), dtype=np.uint8)
    >>> draw_arrow(canvas, (20, 30), (100, 90),
    ...            color=(0, 255, 0), thickness=3, head_size=12)
    >>> # ``canvas`` now contains a green arrow from (20"""
    # Calculate the direction vector and angle of the arrow
    dx, dy = end_point[0] - start_point[0], end_point[1] - start_point[1]
    angle = np.arctan2(dy, dx)

    # Offset the end point slightly to avoid overlapping with the arrowhead
    line_end = (
        int(end_point[0] - head_size * np.cos(angle)),
        int(end_point[1] - head_size * np.sin(angle))
    )

    # Draw the main line slightly short of the intended end point
    cv2.line(canvas, start_point, line_end, color, thickness, lineType=cv2.LINE_AA)

    # Calculate the two points of the arrowhead triangle based on head_size
    arrow_point1 = (
        int(end_point[0] - head_size * np.cos(angle - np.pi / 6)),
        int(end_point[1] - head_size * np.sin(angle - np.pi / 6))
    )
    arrow_point2 = (
        int(end_point[0] - head_size * np.cos(angle + np.pi / 6)),
        int(end_point[1] - head_size * np.sin(angle + np.pi / 6))
    )

    # Draw the arrowhead as a filled triangle at the exact end point
    arrowhead_points = np.array([end_point, arrow_point1, arrow_point2], np.int32)
    cv2.fillConvexPoly(canvas, arrowhead_points, color)

def plot_blob_directions(pixel_data, colony_centroids, boundaries: list=None, starting_time: int=None, ending_time: int=None, max_colony_number: int=None, selection_criteria: str='size'):
    """
    Summary
    -------
    Plot colony contours and movement arrows on a white canvas.

    Parameters
    ----------
    pixel_data
        ndarray or array-like
        Array of shape (N, 4) containing ``[time, colony_id, y, x]`` for every
        pixel belonging to a colony.
    colony_centroids
        ndarray or array-like
        Array of shape (M, 4) containing ``[time, colony_id, centroid_y,
        centroid_x]`` for each colony at each time point.
    boundaries
        list, optional
        ``[y_min, y_max, x_min, x_max]`` specifying the region to plot.
        If ``None`` the bounds are inferred from ``pixel_data``.
    starting_time
        int, optional
        Only data with ``time`` greater than this value are kept.
    ending_time
        int, optional
        Only data with ``time`` smaller than this value are kept.
    max_colony_number
        int, optional
        Maximum number of colonies to visualise. If ``None`` all colonies are
        considered.
    selection_criteria
        str, optional
        Method used to select colonies when ``max_colony_number`` is given:
        ``'distance'`` selects colonies with the largest travel distance,
        ``'lifetime'`` selects colonies that exist for the longest period,
        any other value selects colonies with the largest size at the final
        frame (default is ``'size'``).

    Returns
    -------
    canvas : ndarray
        RGB image (uint8) of shape ``(y_max - y_min + 1, x_max - x_min + 1, 3)`` where
        each selected colony is filled with a unique colour, its initial
        contour is drawn in black, and an arrow indicates movement from the
        initial to the final centroid.

    Notes
    -----
    * The function creates a white background and draws on it using
      ``cv2`` and a custom ``draw_arrow`` helper; make sure those
      dependencies are available.
    * Arrow head size and line thickness are automatically reduced for
      large numbers of colonies to keep the figure readable.
    """
    if not isinstance(pixel_data, np.ndarray):
        pixel_data = np.array(pixel_data)
    if not isinstance(colony_centroids, np.ndarray):
        colony_centroids = np.array(colony_centroids)

    if starting_time is not None:
        pixel_data = pixel_data[pixel_data[:, 0] > starting_time]
        colony_centroids = colony_centroids[colony_centroids[:, 0] > starting_time]
    if ending_time is not None:
        pixel_data = pixel_data[pixel_data[:, 0] < ending_time]
        colony_centroids = colony_centroids[colony_centroids[:, 0] < ending_time]

    if max_colony_number is None:
        max_colony_number = len(np.unique(pixel_data[:, 1]))
    if boundaries is None:
        y_min, y_max, x_min, x_max = 0, pixel_data[:, 2].max(), 0, pixel_data[:, 3].max()
    else:
        y_min, y_max, x_min, x_max = boundaries

    if selection_criteria == "distance":
        unique_colony_id, colony_counts = np.unique(colony_centroids[:, 1], axis=0, return_counts=True)
        colony_distances = np.zeros((len(unique_colony_id), 2), dtype=float)
        for _i, colony_id in enumerate(unique_colony_id):
            # colony_id = unique_colony_id[np.argmax(colony_counts)]
            centroid_coord = colony_centroids[colony_centroids[:, 1] == colony_id, 2:4]
            start_coord, end_coord = centroid_coord[0, :], centroid_coord[-1, :]
            colony_distances[_i, 0] = colony_id
            colony_distances[_i, 1] = eudist(start_coord, end_coord)
        selected_colony_ids = colony_distances[np.argsort(colony_distances[:, 1]), 0][-max_colony_number:]
    elif selection_criteria == "lifetime":
        unique_colony_id, colony_counts = np.unique(colony_centroids[:, 1], axis=0, return_counts=True)
        selected_colony_ids = unique_colony_id[np.argsort(colony_counts)][-max_colony_number:]
    else:
        # Step 1: Get unique (time, colony_id) pairs with their pixel counts
        unique_time_colony, colony_counts = np.unique(pixel_data[:, :2], axis=0, return_counts=True)
        colony_sizes = np.hstack((unique_time_colony, colony_counts.reshape(-1, 1)))  # [time, colony_id, size]
        # Initialize a dictionary to store the size at the last frame for each colony
        max_sizes_by_colony = {}
        # Loop through each unique colony ID
        for colony_id in np.unique(colony_sizes[:, 1]):
            # Select frames for the current colony
            colony_frames = colony_sizes[colony_sizes[:, 1] == colony_id]

            # Get the last frame for the colony and its size at that frame
            last_frame = colony_frames[colony_frames[:, 0] == colony_frames[:, 0].max()]
            max_sizes_by_colony[colony_id] = last_frame[0]  # Store time, colony_id, and size

        # Step 3: Select the 50 largest colonies based on their size at the last frame
        selected_colonies = sorted(max_sizes_by_colony.items(), key=lambda x: x[1][2], reverse=True)[:max_colony_number]
        selected_colony_ids = [item[1][1] for item in selected_colonies]



    # Create RGB canvas within specified bounds
    canvas_shape = (y_max - y_min + 1, x_max - x_min + 1, 3)
    canvas = np.ones(canvas_shape, dtype=np.uint8) * 255
    import colorsys
    colony_colors = [
        tuple(
            (np.array(colorsys.hsv_to_rgb((0.6 + i / len(selected_colony_ids)) % 1, 0.3, 0.85))[::-1] * 255).astype(int))
        for i in range(len(selected_colony_ids))
    ]
    colony_color_map = {colony: colony_colors[i] for i, colony in enumerate(selected_colony_ids)}


    # Plot contours and centroids
    head_size=15
    thickness=2
    if max_colony_number > 30:
        head_size=10
    if max_colony_number > 100:
        head_size=5
        thickness=1

    # Iterate through the 50 largest colonies, plotting their contours and centroids
    for i, colony_id in enumerate(selected_colony_ids):
        color = colony_color_map[colony_id]

        # Get pixel data and centroid data for the current colony
        colony_pixel_data = pixel_data[pixel_data[:, 1] == colony_id]
        colony_colony_centroids = colony_centroids[colony_centroids[:, 1] == colony_id]

        # Initial and final states
        initial_time = colony_pixel_data[:, 0].min()
        final_time = colony_pixel_data[:, 0].max()

        initial_pixels = colony_pixel_data[colony_pixel_data[:, 0] == initial_time][:, 2:4].astype(int)
        final_pixels = colony_pixel_data[colony_pixel_data[:, 0] == final_time][:, 2:4].astype(int)

        # Create binary images for the initial and final occurrences
        final_binary = np.zeros((y_max - y_min + 1, x_max - x_min + 1), dtype=np.uint8)
        initial_binary = np.zeros((y_max - y_min + 1, x_max - x_min + 1), dtype=np.uint8)
        final_binary[final_pixels[:, 0] - y_min, final_pixels[:, 1] - x_min] = 1
        initial_binary[initial_pixels[:, 0] - y_min, initial_pixels[:, 1] - x_min] = 1

        # Extract contours for initial state only
        initial_contour = cv2.morphologyEx(initial_binary, cv2.MORPH_GRADIENT, cross_33)

        # Overlay contours and filled shapes on canvas
        canvas[final_binary > 0] = color  # Full color for final state
        canvas[initial_contour > 0] = (0,0,0)# (color * 0.6).astype(int)  # Lighter color for initial state contour

        # Get centroid coordinates
        initial_centroid = colony_colony_centroids[colony_colony_centroids[:, 0] == initial_time][0, 2:4]# - [y_min, x_min]
        final_centroid = colony_colony_centroids[colony_colony_centroids[:, 0] == final_time][0, 2:4]# - [y_min, x_min]
        start_point = (int(initial_centroid[1]), int(initial_centroid[0]))
        end_point = (int(final_centroid[1]), int(final_centroid[0]))
        draw_arrow(canvas, start_point, end_point, color=(0, 0, 0), thickness=thickness, head_size=head_size)

    return canvas

def get_colony_average_direction(colony_centroids: pd.DataFrame) -> dict:
    """
    Compute the average direction taken by colonies from their first to last position.

    The direction is computed from initial to final centroid location in polar coordinates,
    with y-axis reversed due to image data convention (0 at top, increasing downward).

    Parameters:
        colony_centroids (pd.DataFrame): DataFrame containing colony centroids over time.
            Expected columns: 'time', 'colony', 'y', 'x'.

    Returns:
        dict: {colony_id: average direction in radians}

    Example:
        >>> df = pd.read_csv('centroids.csv')
        >>> directions = get_colony_average_direction(df)
        >>> print(directions[1])
        0.785398...
    """
    if not {'time', 'colony', 'y', 'x'}.issubset(colony_centroids.columns):
        raise ValueError("DataFrame must contain columns: 'time', 'colony', 'y', and 'x'.")

    def compute_direction(group: pd.DataFrame) -> float:
        group = group.sort_values(by='time')
        initial, final = group.iloc[0], group.iloc[-1]

        delta_y = initial.y - final.y  # y-axis is reversed in image data
        delta_x = final.x - initial.x

        if delta_x == 0 and delta_y == 0:
            return np.nan

        direction_rad = np.arctan2(delta_y, delta_x)
        return float(direction_rad)

    directions_series = colony_centroids.groupby('colony').apply(compute_direction, include_groups=False).dropna()
    return directions_series.to_dict()


def plot_spider_binned(colony_centroids, num_intervals: int = 12) -> None:
    """
    Plot a polar “spider” diagram of colony orientation frequencies.

    The function computes the average direction of each colony, bins the
    directions into ``num_intervals`` angular sectors, normalises the counts,
    and visualises the distribution on a polar plot.  A dashed line marks the
    overall mean direction when it can be defined.

    Parameters
    ----------
    colony_centroids : dict or iterable
        Collection that ``get_colony_average_direction`` can process to obtain
        per‑colony orientation vectors (in radians).  Typical input is a mapping
        ``{colony_id: (x, y)}`` where ``(x, y)`` are centroid coordinates.
    num_intervals : int, optional
        Number of angular bins to use for the histogram.  Default is ``12`` which
        yields 30° sectors.

    Returns
    -------
    None
        The function creates a Matplotlib figure; it does not return a value.

    Notes
    -----
    * The polar axis is set to a north‑up, clockwise convention
      (``ax.set_theta_zero_location("N")`` and ``ax.set_theta_direction(-1)``).
    * Counts are normalised by the maximum bin count, so the radial axis always
      spans ``[0, 1]``.
    * If no directions are available the function prints a message and exits
      early without raising an exception.
    * The implementation assumes that ``numpy`` (`np`) and ``matplotlib.pyplot``
      (`plt`) are already imported in the surrounding module.

    """
    colony_directions = get_colony_average_direction(colony_centroids)
    if not colony_directions:
        print("No directions provided. Plot skipped.")
        return
    line_width = 3

    values = np.array(list(colony_directions.values()))

    # Normalize angles to [0, 2π] and convert to polar system (North-up, clockwise)
    angles = (values + 2 * np.pi) % (2 * np.pi)
    theta_polar_clockwise = (np.pi / 2 - angles) % (2 * np.pi)

    # Binning
    bins = np.linspace(0, 2 * np.pi, num_intervals + 1)
    counts, _ = np.histogram(theta_polar_clockwise, bins=bins)

    # Midpoints and normalization for plot
    bin_edges = [(b[0] + b[1]) / 2 for b in zip(bins[:-1], bins[1:])]
    standardized_counts = counts / np.max(counts) if np.max(counts) > 0 else np.zeros_like(counts)

    # Close the polygon to complete the circle
    bin_edges_closed = np.concatenate([bin_edges, [bin_edges[0]]])
    counts_closed = np.concatenate([standardized_counts, [standardized_counts[0]]])

    # Plotting with improved visual layout
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(10, 10))

    color_peach = '#FFDAB9'
    line_color = '#E67E22'

    # Custom radial grid (circles)
    num_circles = 5
    max_radius = 1.0

    # Improved circle placement to avoid visual repetition with outer contour
    for i in range(1, num_circles):
        ax.plot(np.linspace(0, 2 * np.pi, 100),
                [max_radius * (i / num_circles)] * 100,
                color='gray', alpha=0.3, linewidth=0.8)


    # Compute average direction from all colony directions (in radians)
    if len(values) > 0:
        sum_x = np.sum(np.cos(values))
        sum_y = np.sum(np.sin(values))
        resultant_length = np.sqrt(sum_x ** 2 + sum_y ** 2)

        if resultant_length == 0:
            mean_angle_rad = np.nan
        else:
            mean_angle_rad = np.arctan2(sum_y, sum_x) % (2 * np.pi)
            # Convert to polar-clockwise convention for plotting
            theta_avg_polar_clockwise = (np.pi / 2 - mean_angle_rad) % (2 * np.pi)

            ax.plot([theta_avg_polar_clockwise, theta_avg_polar_clockwise], [0.0, 1.0],
                    color=line_color, linewidth=3, linestyle='--')

    # Main plot with adjusted position to avoid clipping
    ax.plot(bin_edges_closed, counts_closed, color=line_color, linewidth=line_width)
    ax.fill(bin_edges_closed, counts_closed, color=color_peach, alpha=0.5)

    # Set polar axis to North-up and clockwise
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)  # Clockwise

    # Customize radial grid and ticks
    r_ticks = np.linspace(0, max_radius, num_circles)

    # Align tick labels with the right (90°) side of the plot
    for i, rtick in enumerate(r_ticks):
        if rtick == 1.:
            rtick = 1
        ax.text(np.pi / 2,  # 90 degrees position
                rtick,
                f'{rtick}',
                horizontalalignment='left',
                verticalalignment='center', # if i != len(r_ticks) - 1 else 'bottom',
                fontsize=font_size)

    # Remove default radial and theta grids/ticks for cleaner look
    ax.set_yticks([])

    # Set the angle (theta) tick labels font size
    ax.set_xticks(np.linspace(0, 2 * np.pi, num_intervals))
    lab = ax.get_xticklabels()
    lab[0] = ''
    ax.set_xticklabels(lab, fontsize=font_size)
    # ax.set_xticklabels([f'{int(angle*180/np.pi)}°' for angle in np.linspace(0, 2 * np.pi, num_intervals)], fontsize=FONT)
    ax.grid(False, axis='y')
    plt.tight_layout()

def superimposed_barplot(big_bars, small_bars, colors, y_label, x_label, big_bar_labels, small_bar_labels, big_bar_width, space_between_small_bars, plot_saving_location="", fig_size=None, adjust_big_bar_number_pos=-.1, adjust_small_bar_number_pos=-.1, adjust_legend_pos=(1.3, 0.6)):
    """
    :param big_bars: a list of 3 lists:
        The first contain the height of each big bar.
        The second contain its 0.025 CI, and its 0.975 CI
        The third contain its 0.975 CI
    :param small_bars: a list of length of the number of big_bars. Each element contain k lists:
        k is the number of small bar within each big bar. Each k element contain 3 values:
        the height of the small bar, its 0.025 CI, and its 0.975 CI
    :param colors:
    :param labels:
    :return:
    """
    # adjust_big_bar_number_pos = - 0.1
    # adjust_small_bar_number_pos = - 0.1
    # adjust_legend_pos = (1.3, 0.6)
    # if plot_aspect_ratio is None:
    #     plot_aspect_ratio = np.max((1, big_bars.shape[0] / 2))
    plt.rc('font', size=16)
    plt.rc('xtick', labelsize=17)
    plt.rc('axes', labelsize=20)
    plt.rc('legend', fontsize=20)
    indices = np.arange(big_bars.shape[0])
    fig, ax = plt.subplots()
    ax.axhline(y=1, color='black', linestyle=':', label="_nolegend_")
    # error_bars = 1 - np.swapaxes(big_bars[:, 1:], 0, 1)
    # ax.bar(indices, big_bars[:, 0], width=big_bar_width,
    #         color=colors[0], alpha=0.4, yerr=error_bars)
    for xi, yi in zip(indices, big_bars[:, 0]):
        yi_to_display = f"{np.round(yi * 100, 2)} %"# str(np.round(yi, 3))
        ax.text(xi + adjust_big_bar_number_pos, 1.02, yi_to_display, color="black", fontweight='bold')
    sb_nb = small_bars.shape[2]
    #Formula: big_bar_width = sb_width * sb_nb + space_between_small_bars * (sb_nb + 1)
    sb_width = (big_bar_width - space_between_small_bars * (sb_nb + 2)) / sb_nb
    patterns = ["", "/"]
    for sb_i in np.arange(sb_nb):
        #sb_i = 0
        error_bars = 1 - np.swapaxes(small_bars[:, 1:, sb_i], 0, 1)
        x_pos = [i - (sb_width/2 + space_between_small_bars/2) + (sb_i * (sb_width + space_between_small_bars)) for i in indices]
        y_pos = small_bars[:, 0, sb_i]
        # ax.bar(x_pos, y_pos, width=sb_width, color="lightgray", alpha=0.9, yerr=error_bars,
        #        label=small_bar_labels[1 + sb_i])#, color=colors[1 + sb_i]
        # ax.bar(x_pos, y_pos, width=sb_width, color="silver", hatch=patterns[sb_i], alpha=0.9, yerr=error_bars, label=small_bar_labels[1 + sb_i])#, color=colors[1 + sb_i]
        ax.bar(x_pos, y_pos, width=sb_width, color=colors[1 + sb_i], alpha=0.9, yerr=error_bars, label=small_bar_labels[1 + sb_i])#,
        # Display small bar text
        for xi, yi in zip(x_pos, y_pos):
            yi_to_display = int(np.round(yi, 2) * 100)
        #    print(yi)
        #    if yi < .95:
        #        yi_to_display = ""
        #    elif yi < .99:
        #        yi_to_display = "  * "
        #    elif yi < .999:
        #        yi_to_display = " * * "
        #    else:
        #        yi_to_display = " *** "
            adjust_small_bar_number_pos = 0.01 - (len(str(yi_to_display)) * 0.04)
            #if len(str(yi_to_display))==2:
            #    adjust_small_bar_number_pos = - 0.07
            #else:
            #    adjust_small_bar_number_pos = - 0.11
            ax.text(xi + adjust_small_bar_number_pos, 0.1, yi_to_display, color='black', fontweight='bold')
        #ax.bar([i + 0.125 * width for i in indices], small_bars[:, 0, small_bar_within],
        #        width=0.5 * width, color='r', alpha=0.5, label='Min Power in mW')
    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)

    ax.set_ylabel(y_label, fontdict=axes_label_dict)
    ax.set_xlabel(x_label, fontdict=axes_label_dict, labelpad=10)

    # Fix ticks locators to avoid a warning when ax.set_xticklabels(labels)
    ticks_loc = ax.get_xticks().tolist()
    ax.xaxis.set_major_locator(mticker.FixedLocator(ticks_loc))

    labels = [item.get_text() for item in ax.get_xticklabels()]
    labels[1:(len(big_bar_labels) + 1)] = big_bar_labels
    ax.set_xticklabels(labels)
    ax.set_yticklabels([0, 20, 40, 60, 80, 100])
    leg = ax.legend([small_bar_labels[1], small_bar_labels[2]], loc="center right", title="Assessed:", bbox_to_anchor=adjust_legend_pos)
    leg._legend_box.align = "left"
    fig.tight_layout()
    if fig_size is not None:
        fig.set_size_inches(fig_size[0], fig_size[1])

    # ax.set_aspect(plot_aspect_ratio)
    fig.show()

    if len(str(plot_saving_location)) > 0:
        plt.savefig(plot_saving_location / "paper_figs_jpg" / "validation_plot.jpg", dpi=500, pil_kwargs={'quality': 100})
        plt.savefig(plot_saving_location / "paper_figs_tif" / "validation_plot.tif", dpi=500)
        plt.close('all')