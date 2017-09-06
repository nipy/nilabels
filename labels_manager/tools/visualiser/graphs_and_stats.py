import numpy as np
import matplotlib
from matplotlib.patches import Circle, Wedge, Polygon
from matplotlib.collections import PatchCollection
import matplotlib.pyplot as plt
import pandas as pd


def plot_tri_metrics_x_y_radius(list_df, list_metrics=('dice_score', 'dispersion'), num_fig=1, title='tri-metric'):
    """

    :param list_df: list data-frames containing the corresponding metrics.
    :param list_metrics: list of metrics in the data-frame as names of the columns.
    :param num_fig: figure number
    :param title: figure title
    :return:
    """
    fig = plt.figure(num_fig, figsize=(6, 7.5), dpi=100)
    ax = fig.add_subplot(111)
    ax.set_position([0.1, 0.29, 0.8, 0.7])

    fig.canvas.set_window_title(title)

    for df in list_df:
        for me in list_metrics:
            pass


    #     df.
    #
    # patches = []
    # for x1, y1, r in zip(x, y, radii):
    #     circle = Circle((x1, y1), r)
    #     patches.append(circle)

    Wedge((.3, .7), .1, 0, 360)