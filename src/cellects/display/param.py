#!/usr/bin/env python3
"""
This script contains color, font, and size parameters for displaying images, videos, and plots.
"""
import numpy as np
from matplotlib import pyplot as plt
import colorsys
from matplotlib.colors import to_hex

# Colors:
cblind = {"darkblue": "#332288", "blue": "#5E77BB", "darkgreen": "#117733", "lightgreen": "#44AA99", "lightblue": "#88CCEE",
                        "yellow": "#DDCC77", "salmon": "#CC6677", "fuchsia": "#AA4499","bordeaux": "#882255"}
cblind_rgb = {"darkblue": (51, 34, 136), "blue": (94, 119, 187), "darkgreen": (17, 119, 51), "lightgreen": (68, 170, 153), "lightblue": (136, 204, 238),
                        "yellow": (	221, 204, 119), "salmon": (204, 102, 119), "fuchsia": (170, 68, 153),"bordeaux": (136, 34, 85)}
light_grey_hexa = "#787878"
dark_grey_rgb = (50, 50, 50)
dark_grey_hexa = "#323232"
red_bgr = (0, 0, 255)
red_rgb = (255, 0, 0)
red_hexa = "#ff0000"
firebrick_bgr = (34, 34, 178)
firebrick_rgb = (178, 34, 34)
firebrick_hexa = "#b22222"
crimson_bgr = (0, 0, 204)
crimson_rgb = (204, 0, 0)
crimson_hexa = '#cc0000'
orange_bgr = (0, 165, 255)
orange_rgb = (255, 165, 0)
orange_hexa = "#ffa500"
yellow_bgr = (79, 196, 254)
yellow_rgb = (254, 196, 79)
yellow_hexa = "#fec44f"
mauve_bgr = (195, 142, 153)
mauve_rgb = (153, 142, 195)
mauve_hexa = "#998ec3"
purple_bgr = (165, 80, 138)
purple_rgb = (138, 80, 165)
purple_hexa = "	#8a50a5"
blue_bgr = (202, 162, 67)
blue_rgb = (67, 162, 202)
blue_hexa = "#43a2ca"
darkblue_bgr = (178, 34, 34)
darkblue_rgb = (34, 34, 178)
darkblue_hexa = "#2222b2"
lightgreen_rgb = (0, 255, 0)
lightgreen_bgr = (0, 255, 0)
lightgreen_hexa = "#00ff00"
peach_bgr = (98, 141, 252)
peach_rgb = (252, 141, 98)
peach_hexa = '#fc8d62'
teal_bgr = (153, 153, 1)
teal_rgb = (1, 153, 153)
teal_hexa = '#73C6AC'

# Font parameters:
plt.rc('font', size=15)
font_size = 15
font_plot_titles = 35
font_plot_ticks = 30
axes_ticks_label_font = 24
cbar_ticks_label_font = 40
axes_label_font = 30
axes_label_family = 'Arial'
axes_label_dict = {'family': axes_label_family, 'size': axes_label_font}

# Other graphic parameters:
curve_width = 3
curve_alpha = 0.7

def get_mpl_colormap(cmap_name: str):
    """
    Returns a linear color range array for the given matplotlib colormap.

    Parameters
    ----------
    cmap_name : str
        The name of the colormap to get.

    Returns
    -------
    numpy.ndarray
        A 256x1x3 array of bytes representing the linear color range.

    Examples
    --------
    >>> result = get_mpl_colormap('viridis')
    >>> print(result.shape)
    (256, 1, 3)

    """
    cmap = plt.get_cmap(cmap_name)

    # Initialize the matplotlib color map
    sm = plt.cm.ScalarMappable(cmap=cmap)

    # Obtain linear color range
    color_range = sm.to_rgba(np.linspace(0, 1, 256), bytes=True)[:, 2::-1]

    return color_range.reshape(256, 1, 3)

def generate_color_gradient(colors: list, n_steps: int) -> list:
    """
    Generate a linear color gradient interpolating between a sequence of RGB colors.

    Parameters
    ----------
    colors : list
        List of RGB tuples defining the colors to interpolate through.
    n_steps : int
        Total number of color tuples to generate to fill the gradient between the provided colors.

    Returns
    -------
    list
        A list of RGB tuples forming a gradient between the provided colors.

    Examples
    --------
    >>> colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    >>> gradient = generate_color_gradient(colors, 5)
    >>> print(gradient)
    [(255.0, 0.0, 0.0), (127.5, 127.5, 0.0), (0.0, 255.0, 0.0),
     (0.0, 127.5, 127.5), (0.0, 0.0, 255.0)]
    """
    gradient = []
    n_segments = len(colors) - 1
    steps_per_segment = n_steps // n_segments

    for i in range(n_segments):
        start_color = colors[i]
        end_color = colors[i + 1]
        for step in range(steps_per_segment):
            t = step / steps_per_segment
            interpolated_color = tuple(
                start_color[j] + t * (end_color[j] - start_color[j])
                for j in range(3)
            )
            gradient.append(interpolated_color)

    gradient.append(colors[-1])
    return gradient[:n_steps]


def random_pastel_colors(color_nb, hexadecimal: bool=False):
    """
    Generate random pastel colors.

    This function generates a specified number of random pastel colors by manipulating the HSV color
    space to ensure a soft, pastel-like appearance. The generated colors can be returned either in
    hexadecimal format or as RGB tuples.

    Parameters
    ----------
    color_nb : int
        Number of pastel colors to generate.
    hexadecimal : bool, default=False
        If `True`, the colors are returned in hexadecimal format.
        If `False`, the colors are returned as RGB tuples with each value ranging from 0 to 255.

    Returns
    -------
    List[Union[tuple, str]]
        A list of generated pastel colors. Each color is represented as an RGB tuple or hexadecimal
        string, depending on the value of `hexadecimal`.

    Raises
    ------
    IndexError
        If the `colony_colors` list is accessed using an incorrect index.

    Examples
    --------
    >>> colors = random_pastel_colors(3, hexadecimal=True)
    >>> print(colors)
    ['#d9d8ff', '#ffdabd', '#ffe5d3']

    >>> colors = random_pastel_colors(3, hexadecimal=False)
    >>> print(colors)
    [(221, 187, 255), (255, 218, 187), (255, 227, 211)]
    """
    colony_colors = []
    for i in range(color_nb):
        color_code = np.array(colorsys.hsv_to_rgb((0.6 + i / color_nb) % 1, 0.3, 0.85))[::-1]
        if hexadecimal:
            colony_colors.append(to_hex(color_code))
        else:
            colony_colors.append(tuple((color_code * 255).astype(int)))
    return colony_colors
