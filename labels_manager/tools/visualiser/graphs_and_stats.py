import numpy as np
import matplotlib
from matplotlib.patches import Circle, Wedge, Polygon
from matplotlib.collections import PatchCollection
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd


def bulls_eye(ax, data, cmap=None, norm=None, raidal_subdivisions=(2, 8, 8, 11),
              centered=(True, False, False, True), add_nomenclatures=True, cell_resolution=128):
    """
    Clockwise, from smaller radius to bigger radius.
    :param ax:
    :param data:
    :param cmap:
    :param norm:
    :param raidal_subdivisions:
    :param centered:
    :param add_nomenclatures:
    :param cell_resolution:
    :return:
    """
    line_width = 1.5
    data = np.array(data).ravel()

    if cmap is None:
        cmap = plt.cm.viridis

    if norm is None:
        norm = mpl.colors.Normalize(vmin=data.min(), vmax=data.max())

    theta = np.linspace(0, 2*np.pi, 768)
    r = np.linspace(0, 1, len(raidal_subdivisions)+1)

    nomenclatures = []
    if isinstance(add_nomenclatures, bool):
        if add_nomenclatures:
            nomenclatures = range(sum(raidal_subdivisions))
    elif isinstance(add_nomenclatures, list) or isinstance(add_nomenclatures, tuple):
        assert len(add_nomenclatures) == sum(raidal_subdivisions)
        nomenclatures = add_nomenclatures[:]
        add_nomenclatures = True

    # Create the circular bounds
    line_width_circular = line_width
    for i in range(r.shape[0]):
        if i == range(r.shape[0])[-1]:
            line_width_circular = int(line_width / 2.)
        ax.plot(theta, np.repeat(r[i], theta.shape), '-k', lw=line_width_circular)

    # iterate over cells divided by radial subdivision
    for rs_id, rs in enumerate(raidal_subdivisions):
        for i in range(rs):
            cell_id = sum(raidal_subdivisions[:rs_id]) + i
            theta_i = - i * 2 * np.pi / rs + np.pi / 2
            if not centered[rs_id]:
                theta_i += (2 * np.pi / rs) / 2
            theta_i_plus_one = theta_i - 2 * np.pi / rs  # clockwise
            # Create colour fillings for each cell:
            theta_interval = np.linspace(theta_i, theta_i_plus_one, cell_resolution)
            r_interval = np.array([r[rs_id], r[rs_id+1]])
            angle  = np.repeat(theta_interval[:, np.newaxis], 2, axis=1)
            radius = np.repeat(r_interval[:, np.newaxis], cell_resolution, axis=1).T
            z = np.ones((cell_resolution, 2)) * data[cell_id]
            ax.pcolormesh(angle, radius, z, cmap=cmap, norm=norm)

            # Create radial bounds
            if rs  > 1:
                ax.plot([theta_i, theta_i], [r[rs_id], r[rs_id+1]], '-k', lw=line_width)
            # Add centered nomenclatures if needed
            if add_nomenclatures:
                if rs == 1 and rs_id ==0:
                    cell_center = (0, 0)
                else:
                    cell_center = ((theta_i + theta_i_plus_one) / 2., r[rs_id] + .5 * r[1] )
                ax.annotate(str(nomenclatures[cell_id]), xy=cell_center,
                            xytext=(cell_center[0], cell_center[1]),
                            horizontalalignment='center', verticalalignment='center')

    ax.grid(False)
    ax.set_ylim([0, 1])
    ax.set_yticklabels([])
    ax.set_xticklabels([])


def multi_bull_eyes(multi_data, cbar=None, cmaps=None, normalisations=None,
                    titles=None, units=None, raidal_subdivisions=(2, 8, 8, 11),
                    centered=(True, False, False, True), add_nomenclatures=True):
    n_fig = len(multi_data)
    if cbar is None:
        cbar = [True] * n_fig
    if cmaps is None:
        cmaps = [mpl.cm.viridis] * n_fig
    if normalisations is None:
        normalisations = [mpl.colors.Normalize(vmin=np.min(multi_data[i]), vmax=np.max(multi_data[i]))
                          for i in range(n_fig)]
    if titles is None:
        titles = ['Title {}'.format(i) for i in range(n_fig)]
    if units is None:
        units = ['Units {}'.format(i) for i in range(n_fig)]

    h_space = 0.15 / n_fig
    h_dim_fig = .8
    w_dim_fig = .8 / n_fig

    # Make a figure and axes with dimensions as desired.
    fig = plt.figure(figsize=(3 * n_fig, 6))
    fig.canvas.set_window_title('Bulls Eyes - segmentation assessment')

    for n in range(n_fig):
        origin_fig = (h_space * (n + 1) + w_dim_fig * n, 0.15)
        ax = fig.add_axes([origin_fig[0], origin_fig[1], w_dim_fig, h_dim_fig], polar=True)
        bulls_eye(ax, multi_data[n], cmap=cmaps[n], norm=normalisations[n], raidal_subdivisions=raidal_subdivisions,
                  centered=centered, add_nomenclatures=True)
        ax.set_title(titles[n])

        if cbar[n]:
            origin_cbar = (h_space * (n + 1) + w_dim_fig * n, .15)
            axl = fig.add_axes([origin_cbar[0], origin_cbar[1], w_dim_fig, .05])
            cb1 = mpl.colorbar.ColorbarBase(axl, cmap=cmaps[n], norm=normalisations[n], orientation='horizontal')
            cb1.set_label(units[n])


def confusion_matrix(confusion_data_frame, annotation_data_frame=None, fig_size=(4,4), title='Title', cmap=plt.cm.jet,
                     pfi_where_to_save=None, show_fig=True):

    fig = plt.figure(figsize=fig_size)
    plt.clf()
    ax = fig.add_subplot(111)
    ax.set_aspect(1)
    res = ax.imshow(confusion_data_frame.as_matrix(), cmap=cmap,
                    interpolation='nearest', origin='lower')

    rows, cols = confusion_data_frame.shape
    if annotation_data_frame is not None:
        for x in xrange(rows):
            for y in xrange(cols):
                ax.annotate(str(annotation_data_frame.as_matrix()[x, y]), xy=(y, x),
                            horizontalalignment='center',
                            verticalalignment='center')

    fig.colorbar(res)

    rows_index_list = list(confusion_data_frame.index)
    cols_index_list = list(confusion_data_frame.columns)

    ax.set_xticks(range(cols))
    ax.set_xticklabels(cols_index_list, rotation=45, ha='center')
    ax.set_yticks(range(rows))
    ax.set_yticklabels(rows_index_list)
    fig.text(.5, .05, title, ha='center')

    ax.invert_yaxis()
    ax.xaxis.tick_top()

    # plt.tight_layout()
    if show_fig:
        plt.show()
    if pfi_where_to_save is not None:
        plt.savefig(pfi_where_to_save, format='pdf', dpi=200)


if __name__ == '__main__':

    # TODO move this part in examples.

    # Very dummy data:
    data = np.array(range(29)) + 1

    # TEST bull-eye three-fold
    if False:

        fig, ax = plt.subplots(figsize=(12, 8), nrows=1, ncols=3,
                               subplot_kw=dict(projection='polar'))
        fig.canvas.set_window_title('Left Ventricle Bulls Eyes (AHA)')

        # First one:
        cmap = mpl.cm.viridis
        norm = mpl.colors.Normalize(vmin=1, vmax=29)

        bulls_eye(ax[0], data, cmap=cmap, norm=norm)
        ax[0].set_title('Bulls Eye (AHA)')

        axl = fig.add_axes([0.14, 0.15, 0.2, 0.05])
        cb1 = mpl.colorbar.ColorbarBase(axl, cmap=cmap, norm=norm, orientation='horizontal')
        cb1.set_label('Some Units')

        # Second one
        cmap2 = mpl.cm.cool
        norm2 = mpl.colors.Normalize(vmin=1, vmax=29)

        bulls_eye(ax[1], data, cmap=cmap2, norm=norm2)
        ax[1].set_title('Bulls Eye (AHA)')

        axl2 = fig.add_axes([0.41, 0.15, 0.2, 0.05])
        cb2 = mpl.colorbar.ColorbarBase(axl2, cmap=cmap2, norm=norm2, orientation='horizontal')
        cb2.set_label('Some other units')

        # Third one
        cmap3 = mpl.cm.winter
        norm3 = mpl.colors.Normalize(vmin=1, vmax=29)

        bulls_eye(ax[2], data, cmap=cmap3, norm=norm3)
        ax[2].set_title('Bulls Eye third')

        axl3 = fig.add_axes([0.69, 0.15, 0.2, 0.05])
        cb3 = mpl.colorbar.ColorbarBase(axl3, cmap=cmap3, norm=norm3, orientation='horizontal')
        cb3.set_label('Some more units')

        plt.show()

    if False:
        fig = plt.figure(figsize=(5, 7))
        fig.canvas.set_window_title('Bulls Eyes - segmentation assessment')

        # First and only:
        cmap = mpl.cm.viridis
        norm = mpl.colors.Normalize(vmin=1, vmax=29)

        ax = fig.add_axes([0.1, 0.2, 0.8, 0.7], polar=True)
        bulls_eye(ax, data, cmap=cmap, norm=norm)
        ax.set_title('Bulls Eye (AHA)')

        axl = fig.add_axes([0.1, 0.15, 0.8, 0.05])
        cb1 = mpl.colorbar.ColorbarBase(axl, cmap=cmap, norm=norm, orientation='horizontal')
        cb1.set_label('Some Units')

        plt.show()

    if False:

        # multi_data = [data for _ in range(3)]
        # print multi_data
        # multi_bull_eyes(multi_data)
        #
        # plt.show(block=False)

        multi_data = [data[:16] for _ in range(3)]
        print multi_data
        multi_bull_eyes(multi_data, raidal_subdivisions=(3,3,4,6),
                  centered=(True, True, True, True), add_nomenclatures=True)

        plt.show(block=True)

    if True:
        d = {'one': pd.Series([1., 2., 3.], index=['a', 'b', 'c']),
             'two': pd.Series([1.5, 2.5, 3.5, 4.5], index=['a', 'b', 'c', 'd'])}
        df = pd.DataFrame(d)

        confusion_matrix(df, annotation_data_frame=df, title='a title', fig_size=(5.5, 8))
