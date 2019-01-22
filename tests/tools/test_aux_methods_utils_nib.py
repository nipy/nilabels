import os
from os.path import join as jph

import nibabel as nib
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal, assert_raises

from nilabels.tools.aux_methods.utils_nib import replace_translational_part, remove_nan_from_im, \
    set_new_data, compare_two_nib, one_voxel_volume, modify_image_data_type, modify_affine_transformation, \
    images_are_overlapping


# TEST aux_methods.utils_nib.py


def test_set_new_data_simple_modifications():
    aff = np.eye(4)
    aff[2, 1] = 42.0

    im_0 = nib.Nifti1Image(np.zeros([3, 3, 3]), affine=aff)
    im_0_header = im_0.header
    # default intent_code
    assert im_0_header['intent_code'] == 0
    # change intento code
    im_0_header['intent_code'] = 5

    # generate new nib from the old with new data
    im_1 = set_new_data(im_0, np.ones([3, 3, 3]))
    im_1_header = im_1.header
    # see if the infos are the same as in the modified header
    assert_array_equal(im_1.get_data()[:], np.ones([3, 3, 3]))
    assert im_1_header['intent_code'] == 5
    assert_array_equal(im_1.affine, aff)


def test_set_new_data_new_data_type():

    im_0 = nib.Nifti1Image(np.zeros([3, 3, 3], dtype=np.uint8), affine=np.eye(4))
    assert im_0.get_data_dtype() == 'uint8'

    # check again the original had not changed
    new_data = np.zeros([3, 3, 3], dtype=np.float64)
    new_im = set_new_data(im_0, new_data)
    assert new_im.get_data_dtype() == '<f8'

    new_data = np.zeros([3, 3, 3], dtype=np.uint8)
    new_im_update_data = set_new_data(im_0, new_data, new_dtype=np.float64)
    assert new_im_update_data.get_data_dtype() == '<f8'

    new_data = np.zeros([3, 3, 3], dtype=np.float64)
    new_im_update_data = set_new_data(im_0, new_data, new_dtype=np.float64)
    assert new_im_update_data.get_data_dtype() == '<f8'

    # check again the original had not changed
    assert im_0.get_data_dtype() == 'uint8'


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
    assert compare_two_nib(im_0, im_1) == True


def test_compare_two_nib_different_nifti_version():
    im_0 = nib.Nifti1Image(np.zeros([3, 3, 3]), affine=np.eye(4))
    im_1 = nib.Nifti2Image(np.zeros([3, 3, 3]), affine=np.eye(4))
    assert compare_two_nib(im_0, im_1) == False


def test_compare_two_nib_different_nifti_version2():
    im_0 = nib.Nifti2Image(np.zeros([3, 3, 3]), affine=np.eye(4))
    im_1 = nib.Nifti1Image(np.zeros([3, 3, 3]), affine=np.eye(4))
    assert compare_two_nib(im_0, im_1) == False


def test_compare_two_nib_different_data_dtype():
    im_0 = nib.Nifti1Image(np.zeros([3, 3, 3], dtype=np.uint8), affine=np.eye(4))
    im_1 = nib.Nifti2Image(np.zeros([3, 3, 3], dtype=np.float64), affine=np.eye(4))
    assert compare_two_nib(im_0, im_1) == False


def test_compare_two_nib_different_data():
    im_0 = nib.Nifti1Image(np.zeros([3, 3, 3]), affine=np.eye(4))
    im_1 = nib.Nifti2Image(np.ones([3, 3, 3]), affine=np.eye(4))
    assert compare_two_nib(im_0, im_1) == False


def test_compare_two_nib_different_affine():
    aff_1 = np.eye(4)
    aff_1[3, 3] = 5
    im_0 = nib.Nifti1Image(np.zeros([3, 3, 3]), affine=np.eye(4))
    im_1 = nib.Nifti1Image(np.zeros([3, 3, 3]), affine=aff_1)
    assert compare_two_nib(im_0, im_1) == False


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


# TEST HEADER MODIFICATION modify image type


def test_modify_image_type_simple():
    im = nib.Nifti1Image(np.ones([5, 5, 5], dtype=np.float64), affine=np.eye(4))
    assert im.get_data_dtype() == 'float64'
    im_new = modify_image_data_type(im, np.uint8, verbose=False)
    assert im_new.get_data_dtype() == 'uint8'
    assert im.get_data_dtype() == 'float64'


# Possible bug in NiBabel
def test_modify_image_type_update_description_header():
    im = nib.Nifti1Image(np.ones([5, 5, 5], dtype=np.float64), affine=np.eye(4))
    im_new = modify_image_data_type(im, np.uint8, update_descrip_field_header='spam')
    hd = im_new.header
    if isinstance(hd['descrip'], str):
        assert hd['descrip'] == 'spam'
    elif isinstance(hd['descrip'], np.ndarray):
        assert hd['descrip'].item() == b'spam'
    else:
        assert False
    assert im_new.get_data_dtype() == 'uint8'
    assert im.get_data_dtype() == 'float64'


def test_modify_image_type_update_description_header_no_string_input():
    im = nib.Nifti1Image(np.ones([5, 5, 5], dtype=np.float64), affine=np.eye(4))
    with assert_raises(IOError):
        modify_image_data_type(im, np.uint8, update_descrip_field_header=123)


def test_modify_image_type_remove_nan():
    data = np.ones([5, 5, 5], dtype=np.float64)
    data[1, 1, 1] = np.nan
    data[1, 2, 1] = np.nan
    im = nib.Nifti1Image(data, affine=np.eye(4))
    im_new = modify_image_data_type(im, np.uint8, remove_nan=True)
    assert im_new.get_data_dtype() == 'uint8'
    assert im.get_data_dtype() == 'float64'
    assert np.nan not in im_new.get_data()


def test_modify_image_type_wrong_input():
    im = nib.Nifti1Image(np.ones([5, 5, 5], dtype=np.float64), affine=np.eye(4))
    flag = False

    # noinspection PyBroadException
    try:
        modify_image_data_type(im, 'fake_type')
    except Exception:
        flag = True
    assert flag


# TEST HEADER MODIFICATION : modify affine transformation


def test_modify_affine_transformation_replace():
    aff = np.eye(4)
    aff[:3, :3] = np.random.randn(3, 3)
    aff = np.round(aff, decimals=5)

    im = nib.Nifti1Image(np.zeros([5, 5, 5]), affine=aff)

    aff_new = np.eye(4)
    aff_new[:3, :3] = np.round(np.random.randn(3, 3), decimals=5)
    aff_new = np.round(aff_new, decimals=5)

    new_im = modify_affine_transformation(im, new_aff=aff_new, multiplication_side='replace',  q_form=True, s_form=True)

    assert_array_almost_equal(new_im.affine, aff_new, decimal=5)


def test_modify_affine_transformation_io_nifti1_nifti2():
    aff = np.eye(4)
    aff[:3, :3] = np.random.randn(3, 3)
    aff = np.round(aff, decimals=5)

    im1 = nib.Nifti1Image(np.zeros([5, 5, 5]), affine=aff)
    im2 = nib.Nifti2Image(np.zeros([5, 5, 5]), affine=aff)

    im1_new = modify_affine_transformation(im1, new_aff=aff)
    im2_new = modify_affine_transformation(im2, new_aff=aff)

    assert im1_new.header['sizeof_hdr'] == 348
    assert im2_new.header['sizeof_hdr'] == 540


def test_modify_affine_transformation_left():
    aff = np.eye(4)
    aff[:3, :3] = np.random.randn(3, 3)
    aff = np.round(aff, decimals=5)

    im = nib.Nifti1Image(np.zeros([5, 5, 5]), affine=aff)

    aff_new = np.eye(4)
    aff_new[:3, :3] = np.round(np.random.randn(3, 3), decimals=5)
    aff_new = np.round(aff_new, decimals=5)

    new_im = modify_affine_transformation(im, new_aff=aff_new, multiplication_side='left',  q_form=True, s_form=True)

    assert_array_almost_equal(new_im.affine, aff_new.dot(aff), decimal=5)


def test_modify_affine_transformation_right():
    aff = np.eye(4)
    aff[:3, :3] = np.random.randn(3, 3)
    aff = np.round(aff, decimals=5)

    im = nib.Nifti1Image(np.zeros([5, 5, 5]), affine=aff)

    aff_new = np.eye(4)
    aff_new[:3, :3] = np.round(np.random.randn(3, 3), decimals=5)
    aff_new = np.round(aff_new, decimals=5)

    new_im = modify_affine_transformation(im, new_aff=aff_new, multiplication_side='right',  q_form=True, s_form=True)

    assert_array_almost_equal(new_im.affine, aff.dot(aff_new), decimal=5)


def test_modify_affine_transformation_replace_sform():
    aff = np.eye(4)
    aff[:3, :3] = np.random.randn(3, 3)
    aff = np.round(aff, decimals=5)

    im = nib.Nifti1Image(np.zeros([5, 5, 5]), affine=aff)

    aff_new = np.eye(4)
    aff_new[:3, :3] = np.round(np.random.randn(3, 3), decimals=5)
    aff_new = np.round(aff_new, decimals=5)

    new_im_sform_true = modify_affine_transformation(im, new_aff=aff_new, multiplication_side='replace',
                                                     q_form=False, s_form=True)
    new_im_sform_false = modify_affine_transformation(im, new_aff=aff_new, multiplication_side='replace',
                                                      q_form=False, s_form=False)

    assert_array_almost_equal(new_im_sform_true.get_sform(), aff_new, decimal=5)
    assert_array_almost_equal(new_im_sform_false.get_sform(), aff, decimal=5)


def test_modify_affine_transformation_wrong_input_parameter():
    aff = np.eye(4)
    aff[:3, :3] = np.random.randn(3, 3)
    aff = np.round(aff, decimals=5)

    im = nib.Nifti1Image(np.zeros([5, 5, 5]), affine=aff)

    with assert_raises(IOError):
        modify_affine_transformation(im, new_aff=aff, multiplication_side='spam')


def test_modify_affine_transformation_dumb_input_header_nifti_version():
    im = nib.Nifti1Image(np.zeros([5, 5, 5]), affine=0.4 * np.eye(4))
    im.header['sizeof_hdr'] = 42

    with assert_raises(IOError):
        modify_affine_transformation(im, new_aff=np.eye(4))


# TEST HEADER MODIFICATION replace translational part


def test_replace_translational_part_simple():
    aff = np.eye(4)
    aff[:3, :3] = np.random.randn(3, 3)
    aff[:3, 3] = [1, 2, 3]
    aff = np.round(aff, decimals=5)

    im = nib.Nifti1Image(np.zeros([5, 5, 5]), affine=aff)

    aff_new = np.copy(aff)
    aff_new[:3, 3] = [3, 1, 2]

    im_new = replace_translational_part(im, new_translation=[3, 1, 2])

    assert_array_equal(im.affine[:3, 3], [1, 2, 3])
    assert_array_equal(im_new.affine[:3, 3], [3, 1, 2])


# TEST remove nan from im


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


# TEST images are overlapping


def test_images_are_overlapping_simple():
    im1 = nib.Nifti1Image(np.zeros([5, 5, 5]), affine=np.eye(4))
    im2 = nib.Nifti1Image(np.zeros([5, 5, 5]), affine=np.eye(4))
    im3 = nib.Nifti1Image(np.zeros([5, 5, 5]), affine=0.5 * np.eye(4))
    im4 = nib.Nifti1Image(np.zeros([5, 5, 4]), affine=np.eye(4))

    assert images_are_overlapping(im1, im2)
    assert not images_are_overlapping(im1, im3)
    assert not images_are_overlapping(im1, im4)


if __name__ == '__main__':
    test_set_new_data_simple_modifications()
    test_set_new_data_new_data_type()
    test_set_new_data_for_nifti2()
    test_set_new_data_for_buggy_image_header()

    test_compare_two_nib_equals()
    test_compare_two_nib_different_nifti_version()
    test_compare_two_nib_different_nifti_version2()
    test_compare_two_nib_different_data_dtype()
    test_compare_two_nib_different_data()
    test_compare_two_nib_different_affine()

    test_one_voxel_volume()
    test_one_voxel_volume_decimals()

    test_modify_image_type_simple()
    test_modify_image_type_update_description_header()
    test_modify_image_type_update_description_header_no_string_input()
    test_modify_image_type_remove_nan()
    test_modify_image_type_wrong_input()

    test_modify_affine_transformation_replace()
    test_modify_affine_transformation_io_nifti1_nifti2()
    test_modify_affine_transformation_left()
    test_modify_affine_transformation_right()
    test_modify_affine_transformation_replace_sform()
    test_modify_affine_transformation_wrong_input_parameter()
    test_modify_affine_transformation_dumb_input_header_nifti_version()

    test_replace_translational_part_simple()
    test_remove_nan()
    test_images_are_overlapping_simple()
