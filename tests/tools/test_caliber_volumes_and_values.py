import numpy as np
import nibabel as nib
import pytest
from numpy.testing import assert_array_equal, assert_equal, assert_raises

from nilabels.tools.caliber.volumes_and_values import get_total_num_nonzero_voxels, get_num_voxels_from_labels_list, \
    get_values_below_labels_list


def cube_shape(omega, center, side_length, background_intensity=0, foreground_intensity=100, dtype=np.uint8):
    sky = background_intensity * np.ones(omega, dtype=dtype)
    half_side_length = int(np.ceil(int(side_length / 2)))

    for lx in range(-half_side_length, half_side_length + 1):
        for ly in range(-half_side_length, half_side_length + 1):
            for lz in range(-half_side_length, half_side_length + 1):
                sky[center[0] + lx, center[1] + ly, center[2] + lz] = foreground_intensity
    return sky


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

    num_voxels = get_total_num_nonzero_voxels(im_segm)
    assert num_voxels == 11 ** 3 + 17 ** 3 + 19 ** 3 + 9 **3

    num_voxels = get_total_num_nonzero_voxels(im_segm, list_labels_to_exclude=[2, 4])
    assert_equal(num_voxels, 11 ** 3 + 19 ** 3)


def test_volumes_and_values_total_num_voxels_empty():

    omega = [80, 80, 80]
    im_segm = nib.Nifti1Image(np.zeros(omega), affine=np.eye(4))

    num_voxels = get_total_num_nonzero_voxels(im_segm)
    print(num_voxels)
    assert_equal(num_voxels, 0)


def test_volumes_and_values_total_num_voxels_full():

    omega = [80, 80, 80]
    im_segm = nib.Nifti1Image(np.ones(omega), affine=np.eye(4))

    num_voxels = get_total_num_nonzero_voxels(im_segm)
    assert_equal(num_voxels, 80 ** 3)


def test_get_num_voxels_from_labels_list():

    omega = [80, 80, 80]
    cube_a = [[10, 60, 55], 11, 1]
    cube_b = [[50, 55, 42], 15, 2]
    cube_c = [[25, 20, 20], 13, 3]
    cube_d = [[55, 16, 9], 7, 4]

    sky = cube_shape(omega, center=cube_a[0], side_length=cube_a[1], foreground_intensity=cube_a[2]) + \
          cube_shape(omega, center=cube_b[0], side_length=cube_b[1], foreground_intensity=cube_b[2]) + \
          cube_shape(omega, center=cube_c[0], side_length=cube_c[1], foreground_intensity=cube_c[2]) + \
          cube_shape(omega, center=cube_d[0], side_length=cube_d[1], foreground_intensity=cube_d[2])
    im_segm = nib.Nifti1Image(sky, affine=np.eye(4))

    num_voxels = get_num_voxels_from_labels_list(im_segm, labels_list=[1, 2, 3, 4])
    print(num_voxels, [11 **3, 15 **3, 13 **3, 7 ** 3])
    assert_array_equal(num_voxels, [11 **3, 15 **3, 13 **3, 7 ** 3])

    num_voxels = get_num_voxels_from_labels_list(im_segm, labels_list=[1, [2, 3], 4])
    print(num_voxels, [11 ** 3, 15 ** 3 + 13 ** 3, 7 ** 3])
    assert_array_equal(num_voxels, [11 ** 3, 15 ** 3 + 13 ** 3, 7 ** 3])

    num_voxels = get_num_voxels_from_labels_list(im_segm, labels_list=[[1, 2, 3], 4])
    print(num_voxels, [11 ** 3, 15 ** 3 + 13 ** 3, 7 ** 3])
    assert_array_equal(num_voxels, [11 ** 3 + 15 ** 3 + 13 ** 3, 7 ** 3])

    num_voxels = get_num_voxels_from_labels_list(im_segm, labels_list=[[1, 2, 3, 4]])
    print(num_voxels, [11 ** 3, 15 ** 3 + 13 ** 3, 7 ** 3])
    assert_array_equal(num_voxels, [11 ** 3 + 15 ** 3 + 13 ** 3 + 7 ** 3])


def test_get_num_voxels_from_labels_list_unexisting_labels():

    omega = [80, 80, 80]
    cube_a = [[10, 60, 55], 11, 1]
    cube_b = [[50, 55, 42], 15, 2]
    cube_c = [[25, 20, 20], 13, 3]
    cube_d = [[55, 16, 9], 7, 4]

    sky = cube_shape(omega, center=cube_a[0], side_length=cube_a[1], foreground_intensity=cube_a[2]) + \
          cube_shape(omega, center=cube_b[0], side_length=cube_b[1], foreground_intensity=cube_b[2]) + \
          cube_shape(omega, center=cube_c[0], side_length=cube_c[1], foreground_intensity=cube_c[2]) + \
          cube_shape(omega, center=cube_d[0], side_length=cube_d[1], foreground_intensity=cube_d[2])
    im_segm = nib.Nifti1Image(sky, affine=np.eye(4))

    num_voxels = get_num_voxels_from_labels_list(im_segm, labels_list=[1, 2, 3, 5])
    print(num_voxels, [11 ** 3, 15 ** 3, 13 ** 3, 0])
    assert_array_equal(num_voxels, [11 ** 3, 15 ** 3, 13 ** 3, 0])

    num_voxels = get_num_voxels_from_labels_list(im_segm, labels_list=[1, 2, [3, 5]])
    print(num_voxels, [11 ** 3, 15 ** 3, 13 ** 3 + 0])
    assert_array_equal(num_voxels, [11 ** 3, 15 ** 3, 13 ** 3 + 0])

    num_voxels = get_num_voxels_from_labels_list(im_segm, labels_list=[[1, 2], [7, 8]])
    print(num_voxels, [11 ** 3 + 15 ** 3,  0])
    assert_array_equal(num_voxels, [11 ** 3 + 15 ** 3, 0])

    num_voxels = get_num_voxels_from_labels_list(im_segm, labels_list=[[1, 2], [7, -8]])
    print(num_voxels, [11 ** 3 + 15 ** 3, 0])
    assert_array_equal(num_voxels, [11 ** 3 + 15 ** 3, 0])


def test_get_num_voxels_from_labels_list_wrong_input():
    omega = [80, 80, 80]
    cube_a_seg = [[10, 60, 55], 11, 1]
    sky_s = cube_shape(omega, center=cube_a_seg[0], side_length=cube_a_seg[1], foreground_intensity=cube_a_seg[2])
    im_segm = nib.Nifti1Image(sky_s, affine=np.eye(4))
    with assert_raises(IOError):
        get_num_voxels_from_labels_list(im_segm, [1, [2, 3], '3'])


def test_get_values_below_labels_list():
    omega = [80, 80, 80]

    cube_a_seg = [[10, 60, 55], 11, 1]
    cube_b_seg = [[50, 55, 42], 15, 2]
    cube_c_seg = [[25, 20, 20], 13, 3]
    cube_d_seg = [[55, 16, 9], 7, 4]

    cube_a_anat = [[10, 60, 55], 11, 1.5]
    cube_b_anat = [[50, 55, 42], 15, 2.5]
    cube_c_anat = [[25, 20, 20], 13, 3.5]
    cube_d_anat = [[55, 16, 9], 7, 4.5]

    sky_s = cube_shape(omega, center=cube_a_seg[0], side_length=cube_a_seg[1], foreground_intensity=cube_a_seg[2])
    sky_s += cube_shape(omega, center=cube_b_seg[0], side_length=cube_b_seg[1], foreground_intensity=cube_b_seg[2])
    sky_s += cube_shape(omega, center=cube_c_seg[0], side_length=cube_c_seg[1], foreground_intensity=cube_c_seg[2])
    sky_s += cube_shape(omega, center=cube_d_seg[0], side_length=cube_d_seg[1], foreground_intensity=cube_d_seg[2])

    sky_a = cube_shape(omega, center=cube_a_anat[0], side_length=cube_a_anat[1], foreground_intensity=cube_a_anat[2], dtype=np.float32)
    sky_a += cube_shape(omega, center=cube_b_anat[0], side_length=cube_b_anat[1], foreground_intensity=cube_b_anat[2], dtype=np.float32)
    sky_a += cube_shape(omega, center=cube_c_anat[0], side_length=cube_c_anat[1], foreground_intensity=cube_c_anat[2], dtype=np.float32)
    sky_a += cube_shape(omega, center=cube_d_anat[0], side_length=cube_d_anat[1], foreground_intensity=cube_d_anat[2], dtype=np.float32)

    im_segm = nib.Nifti1Image(sky_s, affine=np.eye(4))
    im_anat = nib.Nifti1Image(sky_a, affine=np.eye(4))

    assert im_segm.shape == im_anat.shape

    labels_list = [[1, 2], [3, 4], 4]

    vals_below = get_values_below_labels_list(im_segm, im_anat, labels_list)

    assert_array_equal(vals_below[0], np.array([1.5, ] * (11**3) + [2.5] * (15**3)) )
    assert_array_equal(vals_below[1], np.array([3.5, ] * (13**3) + [4.5] * (7**3)) )
    assert_array_equal(vals_below[2], np.array([4.5] * (7 ** 3)))


def test_get_values_below_labels_list_wrong_input():
    omega = [80, 80, 80]
    cube_a_seg = [[10, 60, 55], 11, 1]
    cube_a_anat = [[10, 60, 55], 11, 1.5]
    sky_a = cube_shape(omega, center=cube_a_anat[0], side_length=cube_a_anat[1], foreground_intensity=cube_a_anat[2], dtype=np.float32)
    sky_s = cube_shape(omega, center=cube_a_seg[0], side_length=cube_a_seg[1], foreground_intensity=cube_a_seg[2])

    im_segm = nib.Nifti1Image(sky_s, affine=np.eye(4))
    im_anat = nib.Nifti1Image(sky_a, affine=np.eye(4))

    with assert_raises(IOError):
        get_values_below_labels_list(im_segm, im_anat, [1, [2, 3], '3', '4'])


def test_get_volumes_per_label():
    # TODO
    pass


if __name__ == '__main__':
    test_volumes_and_values_total_num_voxels()
    test_volumes_and_values_total_num_voxels_empty()
    test_volumes_and_values_total_num_voxels_full()

    test_get_num_voxels_from_labels_list()
    test_get_num_voxels_from_labels_list_wrong_input()

    test_get_num_voxels_from_labels_list_unexisting_labels()
    test_get_values_below_labels_list()
    test_get_values_below_labels_list_wrong_input()
