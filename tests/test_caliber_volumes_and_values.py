import os
from os.path import join as jph

import numpy as np
import nibabel as nib
from nose.tools import assert_raises, assert_almost_equal, assert_equal

from labels_manager.tools.caliber.volumes_and_values import get_total_num_voxels, get_num_voxels_from_labels_list
from labels_manager.tools.phantoms_generator.shapes_for_phantoms import cube_shape


def test_volumes_and_values_total_num_voxels():

    omega = [80, 80, 80]
    cube_a = [[10, 60, 55], 11, 1]
    cube_b = [[50, 55, 42], 17, 2]
    cube_c = [[25, 20, 20], 19, 3]
    cube_d = [[55, 16, 9], 9, 4]

    sky = cube_shape(omega, center=cube_a[0], side_length=cube_a[1], foreground_intensity=cube_a[2]) + \
           cube_shape(omega, center=cube_b[0], side_length=cube_b[1], foreground_intensity=cube_b[2]) + \
           cube_shape(omega, center=cube_c[0], side_length=cube_c[1], foreground_intensity=cube_c[2]) + \
           cube_shape(omega, center=cube_d[0], side_length=cube_d[1], foreground_intensity=cube_d[2])
    im_segm = nib.Nifti1Image(sky, affine=np.eye(4))

    num_voxels = get_total_num_voxels(im_segm)
    assert_equal(num_voxels, 11 ** 3 + 17 **3 + 19 ** 3 + 9 **3)

    num_voxels = get_total_num_voxels(im_segm, list_labels_to_exclude=[2, 4])
    assert_equal(num_voxels, 11 ** 3 +  19 ** 3)


def test_volumes_and_values_total_num_voxels_empthy():

    omega = [80, 80, 80]
    im_segm = nib.Nifti1Image(np.zeros(omega), affine=np.eye(4))

    num_voxels = get_total_num_voxels(im_segm)
    print num_voxels
    assert_equal(num_voxels, 0)


def test_volumes_and_values_total_num_voxels_full():

    omega = [80, 80, 80]
    im_segm = nib.Nifti1Image(np.ones(omega), affine=np.eye(4))

    num_voxels = get_total_num_voxels(im_segm)
    print num_voxels
    assert_equal(num_voxels, 80 ** 3)


def test_get_num_voxels_from_labels_list():
    pass