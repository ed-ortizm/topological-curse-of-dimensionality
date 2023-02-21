"""
Functionality to plot figures of the behavior of the distance metrics in
the Euclidean space, the n-Spheres and the n-hyperbolic spaces.
"""

from typing import Tuple

import matplotlib.pyplot as plt


def contrast_plot(
    metric: str,
    title: str,
    euclidean: bool = False,
    p: float = None,
    figsize: tuple = None,
) -> Tuple[plt.Figure, plt.Axes]:

    """
    Plot the contrast of the distance metrics in the Euclidean space,
    the n-Spheres and the n-hyperbolic spaces.
    The contrast is defined as the average of the distances between
    the points in the manifold and the distance to the closest point
    to the origin of the manifold.

    INPUT

    metric: name of the metric in the corresponding manifold
    title: title of the plot
    euclidean: True if the metric is in the Euclidean space
    p: parameter of the metric in the Euclidean space
    figsize: size of the figure

    OUTPUT

    (fig, ax): figure and axis of the plot

    """

    fig, ax = plt.subplots(
        nrows=1, ncols=1, tight_layout=True, figsize=figsize
    )

    ax.minorticks_on()

    ax.set_xlabel("Number of Diemnsions", fontsize="xx-large")
    ax.set_ylabel("Contrast", fontsize="xx-large")

    if euclidean is True:

        ax.set_title(
            f"{title}\n {metric} $ \\to p=$ {p:.1f}", fontsize="xx-large"
        )

    else:

        ax.set_title(f"{title}\n {metric}", fontsize="xx-large")

    return fig, ax
