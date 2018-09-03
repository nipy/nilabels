import os
from os.path import join as jph
import numpy as np
import nibabel as nib
from matplotlib import rc

import matplotlib.pyplot as plt

from nilabels.tools.aux_methods.utils import print_and_run


def see_array(in_array, pfo_tmp='./z_tmp', in_array_segm=None, pfi_label_descriptor=None, block=False):
    """
    Itk-snap based quick array visualiser.
    :param in_array: numpy array or list of numpy array same dimension (GIGO).
    :param pfo_tmp: path to file temporary folder.
    :param in_array_segm: if there is a single array representing a segmentation (in this case all images must
    have the same shape).
    :param pfi_label_descriptor: path to file to a label descriptor in ITK-snap standard format.
    :param block: if want to stop after each show.
    :return:
    """
    if isinstance(in_array, list):
        assert len(in_array) > 0
        sh = in_array[0].shape
        for arr in in_array[1:]:
            assert sh == arr.shape
        print_and_run('mkdir {}'.format(pfo_tmp))
        cmd = 'itksnap -g '
        for arr_id, arr in enumerate(in_array):
            im = nib.Nifti1Image(arr, affine=np.eye(4))
            pfi_im = jph(pfo_tmp,  'im_{}.nii.gz'.format(arr_id))
            nib.save(im, pfi_im)
            if arr_id == 1:
                cmd += ' -o {} '.format(pfi_im)
            else:
                cmd += ' {} '.format(pfi_im)
    elif isinstance(in_array, np.ndarray):
        print_and_run('mkdir {}'.format(pfo_tmp))
        im = nib.Nifti1Image(in_array, affine=np.eye(4))
        pfi_im = jph(pfo_tmp, 'im_0.nii.gz')
        nib.save(im, pfi_im)
        cmd = 'itksnap -g {}'.format(pfi_im)
    else:
        raise IOError
    if in_array_segm is not None:
        im_segm = nib.Nifti1Image(in_array_segm, affine=np.eye(4))
        pfi_im_segm = jph(pfo_tmp, 'im_segm_0.nii.gz')
        nib.save(im_segm, pfi_im_segm)
        cmd += ' -s {} '.format(pfi_im_segm)
        if pfi_label_descriptor:
            if os.path.exists(pfi_label_descriptor):
                cmd += ' -l {} '.format(pfi_im_segm)
    print_and_run(cmd)
    if block:
        _ = raw_input("Press any key to continue.")


def see_image_slice_with_a_grid(pfi_image, fig_num=1, axis_quote=('y', 230), vmin=None, vmax=None, cmap='gray',
                                pfi_where_to_save=None):
    rc('text', usetex=True)
    fig = plt.figure(fig_num, figsize=(6, 6))
    fig.canvas.set_window_title('canvas {}'.format(fig_num))
    ax = fig.add_subplot(111)

    im = nib.load(pfi_image)
    if axis_quote[0] == 'x':

        data = im.get_data()[axis_quote[1], :, :].T
        shape = data.shape

        voxel_origin = np.array([axis_quote[1], 0, 0, 1])
        voxel_x = np.array([axis_quote[1], shape[1], 0, 1])
        voxel_y = np.array([axis_quote[1], 0, shape[0], 1])

        affine = im.affine

        pt_origin = affine.dot(voxel_origin)
        pt_x = affine.dot(voxel_x)
        pt_y = affine.dot(voxel_y)

        horizontal_min = pt_origin[1]
        horizontal_max = pt_x[1]
        vertical_min = pt_origin[2]
        vertical_max = pt_y[2]

        extent = [horizontal_min, horizontal_max, vertical_min, vertical_max]
        print(extent)

    elif axis_quote[0] == 'y':
        data = im.get_data()[:, axis_quote[1], :].T
        shape = data.shape

        voxel_origin = np.array([0, axis_quote[1], 0, 1])
        voxel_x = np.array([shape[1], axis_quote[1], 0, 1])
        voxel_y = np.array([0, axis_quote[1], shape[0], 1])

        affine = im.affine

        pt_origin = affine.dot(voxel_origin)
        pt_x = affine.dot(voxel_x)
        pt_y = affine.dot(voxel_y)

        horizontal_min = pt_origin[0]
        horizontal_max = pt_x[0]
        vertical_min = pt_origin[2]
        vertical_max = pt_y[2]

        extent = [horizontal_min, horizontal_max, vertical_min, vertical_max]

    elif axis_quote[0] == 'z':

        data = im.get_data()[:, :, axis_quote[1]].T
        shape = data.shape

        voxel_origin = np.array([0, 0, axis_quote[1], 1])
        voxel_x = np.array([shape[1], 0, axis_quote[1], 1])
        voxel_y = np.array([0, shape[0], axis_quote[1], 1])

        affine = im.affine

        pt_origin = affine.dot(voxel_origin)
        pt_x = affine.dot(voxel_x)
        pt_y = affine.dot(voxel_y)

        horizontal_min = pt_origin[0]
        horizontal_max = pt_x[0]
        vertical_min = pt_origin[1]
        vertical_max = pt_y[1]

        extent = [horizontal_min, horizontal_max, vertical_min, vertical_max]

    else:
        raise IOError

    print(voxel_origin)
    print(voxel_x)
    print(voxel_y)
    print(pt_origin)
    print(pt_x)
    print(pt_y)
    print(extent)
    res = ax.imshow(data,
                    extent=extent,
                    origin='lower',
                    interpolation='nearest',
                    cmap=cmap, vmin=vmin, vmax=vmax)

    ax.grid(color='grey', linestyle='-', linewidth=0.5)
    ax.set_aspect('equal')

    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(8)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(8)

    if pfi_where_to_save is not None:
        plt.savefig(pfi_where_to_save, format='pdf', dpi=200)






