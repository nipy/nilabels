import os
from os.path import join as jph

import nibabel as nib
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal, assert_raises

from nilabels.tools.aux_methods.utils_nib import replace_translational_part, remove_nan_from_im, \
    set_new_data, compare_two_nib, one_voxel_volume


# TEST aux_methods.utils_nib.py


def test_set_new_data_simple_modifications():
    aff = np.eye(4)
    aff[2, 1] = 42.0

    im_0 = nib.Nifti1Image(np.zeros([3, 3, 3]), affine=aff)
    im_0_header = im_0.header
    # default intent_code
    assert cmp(im_0_header['intent_code'], 0) == 0
    # change intento code
    im_0_header['intent_code'] = 5

    # generate new nib from the old with new data
    im_1 = set_new_data(im_0, np.ones([3, 3, 3]))
    im_1_header = im_1.header
    # see if the infos are the same as in the modified header
    assert_array_equal(im_1.get_data()[:], np.ones([3, 3, 3]))
    assert cmp(im_1_header['intent_code'], 5) == 0
    assert_array_equal(im_1.affine, aff)


def test_set_new_data_new_data_type():

    im_0 = nib.Nifti1Image(np.zeros([3, 3, 3], dtype=np.uint8), affine=np.eye(4))
    assert im_0.get_data_dtype() == 'uint8'

    new_data = np.zeros([3, 3, 3], dtype=np.float64)
    new_im = set_new_data(im_0, new_data)
    assert new_im.get_data_dtype() == 'uint8'

    new_data = np.zeros([3, 3, 3], dtype=np.float64)
    new_im_update_data = set_new_data(im_0, new_data, new_dtype=np.float64)
    assert new_im_update_data.get_data_dtype() == '<f8'


def test_set_new_data_for_nifti2():

    im_0 = nib.Nifti2Image(np.zeros([3, 3, 3]), affine=np.eye(4))

    new_data = np.ones([5, 5, 5])
    new_im_nifti2 = set_new_data(im_0, new_data)

    hd = new_im_nifti2.header

    assert hd['sizeof_hdr'] == 540


def test_set_new_data_for_buggy_image_header():

    im_0 = nib.Nifti2Image(np.zeros([3, 3, 3]), affine=np.eye(4))
    im_0.header['sizeof_hdr'] = 550

    with assert_raises(IOError):
        set_new_data(im_0, np.array([6, 6, 6]))


# TEST compare two nifti images


def test_compare_two_nib_equals():
    im_0 = nib.Nifti1Image(np.zeros([3, 3, 3]), affine=np.eye(4))
    im_1 = nib.Nifti1Image(np.zeros([3, 3, 3]), affine=np.eye(4))
    assert cmp(compare_two_nib(im_0, im_1), True) == 0


def test_compare_two_nib_different_nifti_version():
    im_0 = nib.Nifti1Image(np.zeros([3, 3, 3]), affine=np.eye(4))
    im_1 = nib.Nifti2Image(np.zeros([3, 3, 3]), affine=np.eye(4))
    assert cmp(compare_two_nib(im_0, im_1), False) == 0


def test_compare_two_nib_different_nifti_version2():
    im_0 = nib.Nifti2Image(np.zeros([3, 3, 3]), affine=np.eye(4))
    im_1 = nib.Nifti1Image(np.zeros([3, 3, 3]), affine=np.eye(4))
    assert cmp(compare_two_nib(im_0, im_1), False) == 0


def test_compare_two_nib_different_data_dtype():
    im_0 = nib.Nifti1Image(np.zeros([3, 3, 3], dtype=np.uint8), affine=np.eye(4))
    im_1 = nib.Nifti2Image(np.zeros([3, 3, 3], dtype=np.float64), affine=np.eye(4))
    assert cmp(compare_two_nib(im_0, im_1), False) == 0


def test_compare_two_nib_different_data():
    im_0 = nib.Nifti1Image(np.zeros([3, 3, 3]), affine=np.eye(4))
    im_1 = nib.Nifti2Image(np.ones([3, 3, 3]), affine=np.eye(4))
    assert cmp(compare_two_nib(im_0, im_1), False) == 0


def test_compare_two_nib_different_affine():
    aff_1 = np.eye(4)
    aff_1[3, 3] = 5
    im_0 = nib.Nifti1Image(np.zeros([3, 3, 3]), affine=np.eye(4))
    im_1 = nib.Nifti1Image(np.zeros([3, 3, 3]), affine=aff_1)
    assert cmp(compare_two_nib(im_0, im_1), False) == 0


# TEST one voxel volumes

def test_one_voxel_volume():
    diag = [0.3, 0.4, 0.5]
    aff = np.diag(diag + [1])
    im = nib.Nifti1Image(np.ones([5, 5, 5]), affine=aff)

    voxel_vol_expected = np.prod(diag)
    voxel_vol = one_voxel_volume(im)

    assert voxel_vol == voxel_vol_expected


def test_one_voxel_volume_decimals():
    diag = [0.3, 0.4, 0.5]
    aff = np.diag(diag + [1])
    im = nib.Nifti1Image(np.ones([5, 5, 5]), affine=aff)

    voxel_vol_expected = np.round(np.prod(diag), decimals=2)
    voxel_vol = one_voxel_volume(im, decimals=2)

    assert voxel_vol == voxel_vol_expected


# TEST replace translational part


def test_replace_translational_part():
    pass


# TEST remove nan

def test_remove_nan():
    data_ts = np.array([[[0, 1, 2, 3],
                         [4, 5, np.nan, 7],
                         [8, 9, 10, np.nan]],
                        [[12, 13, 14, 15],
                         [16, np.nan, 18, 19],
                         [20, 21, 22, 23]]])

    im = nib.Nifti1Image(data_ts, np.eye(4))
    im_no_nan = remove_nan_from_im(im)

    data_no_nan_expected = np.array([[[0, 1, 2, 3],
                                     [4, 5, 0, 7],
                                     [8, 9, 10, 0]],
                                    [[12, 13, 14, 15],
                                     [16, 0, 18, 19],
                                     [20, 21, 22, 23]]])

    assert_array_equal(im_no_nan.get_data(), data_no_nan_expected)


if __name__ == '__main__':
    # test_set_new_data_simple_modifications()
    # test_set_new_data_new_data_type()
    # test_set_new_data_for_nifti2()
    # test_set_new_data_for_buggy_image_header()
    #
    # test_compare_two_nib_equals()
    # test_compare_two_nib_different_nifti_version()
    # test_compare_two_nib_different_nifti_version2()
    # test_compare_two_nib_different_data_dtype()
    # test_compare_two_nib_different_data()
    # test_compare_two_nib_different_affine()

    test_one_voxel_volume()
    test_one_voxel_volume_decimals()




    test_remove_nan()