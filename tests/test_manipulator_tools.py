import numpy as np
import nibabel as nib

from nose.tools import assert_raises
from numpy.testing import assert_array_equal


''' From manipulations.merger.py'''
from labels_manager.tools.manipulations.merger import stack_images, merge_labels_from_4d


def test_merge_labels_from_4d_fake_input():

    data = np.zeros([3,3,3])
    with assert_raises(AssertionError):
        merge_labels_from_4d(data)


def test_merge_labels_from_4d_shape_output():

    data000 = np.zeros([3, 3, 3])
    data111 = np.zeros([3, 3, 3])
    data222 = np.zeros([3, 3, 3])
    data000[0,0,0] = 1
    data111[1,1,1] = 2
    data222[2,2,2] = 4
    data = np.stack([data000, data111, data222], axis=3)

    out = merge_labels_from_4d(data)
    assert_array_equal([out[0,0,0], out[1,1,1], out[2,2,2]], [1, 2, 4])

    out = merge_labels_from_4d(data, keep_original_values=False)
    assert_array_equal([out[0,0,0], out[1,1,1], out[2,2,2]], [1, 2, 3])


def test_stack_images_cascade():

    d = 2
    im1 = nib.Nifti1Image(np.zeros([d,d]), affine=np.eye(4))
    assert_array_equal(im1.shape, (d, d))

    list_images1 = [im1] * d
    im2 = stack_images(list_images1)
    assert_array_equal(im2.shape, (d,d,d))

    list_images2 = [im2] * d
    im3 = stack_images(list_images2)
    assert_array_equal(im3.shape, (d,d,d,d))

    list_images3 = [im3] * d
    im4 = stack_images(list_images3)
    assert_array_equal(im4.shape, (d, d, d, d, d))


''' From  manipulations.relabeller.py'''
from labels_manager.tools.manipulations.relabeller import relabeller, permute_labels, erase_labels, \
    assign_all_other_labels_the_same_value, keep_only_one_label


def test_relabeller_basic():
    data = np.array(range(10)).reshape(2, 5)
    relabelled_data = relabeller(data, range(10), range(10)[::-1])
    assert_array_equal(relabelled_data, np.array(range(10)[::-1]).reshape(2,5))


def test_relabeller_one_element():
    data = np.array(range(10)).reshape(2, 5)
    relabelled_data = relabeller(data, 0, 1)
    expected_output = data[:]
    expected_output[0, 0] = 1
    assert_array_equal(relabelled_data, expected_output)


def test_permute_label_invalid_permutation():
    with assert_raises(AssertionError):
        invalid_permutation = [[3, 3, 3], [1, 1, 1]]
        permute_labels(np.zeros([3,3]), invalid_permutation)


def test_erase_label_simple():
    data = np.array(range(10)).reshape(2, 5)
    data_erased_1 = erase_labels(data, 1)
    expected_output = data[:]
    expected_output[0,1] = 0
    assert_array_equal(data_erased_1, expected_output)


def test_assign_all_other_labels_the_same_values_simple():
    data = np.array(range(10)).reshape(2, 5)
    data_erased_1 = erase_labels(data, 1)
    data_labels_to_keep = assign_all_other_labels_the_same_value(data, range(2,10), same_value_label=0)
    assert_array_equal(data_erased_1, data_labels_to_keep)


def test_keep_only_one_label_label_not_present():
    data = np.array(range(10)).reshape(2, 5)
    new_data = keep_only_one_label(data, 1)
    expected_data = np.zeros([2,5])
    expected_data[0,1] = 1
    assert_array_equal(new_data, expected_data)


''' From manipulations.splitter.py '''
from labels_manager.tools.manipulations.splitter import split_labels_to_4d


def test_split_labels_to_4d():
    data = np.array(range(8)).reshape(2, 2, 2)
    splitted_4d = split_labels_to_4d(data, list_labels=range(8))
    for t in range(8):
        expected_slice = np.zeros(8)
        expected_slice[t] = t
        assert_array_equal(splitted_4d[...,t], expected_slice.reshape(2,2,2))


''' From manipulations.symmetriser.py '''
from labels_manager.tools.manipulations.spatial_adjuster import basic_rot_ax, axial_rotations, flip_data, symmetrise_data


def test_basic_rotation_ax_simple_and_visual():

    cube_id = np.array([[[ 0, 1, 2, 3],
                         [ 4, 5, 6, 7],
                         [ 8, 9, 10, 11]],

                        [[12, 13, 14, 15],
                         [16, 17, 18, 19],
                         [20, 21, 22, 23]]])

    # counterclockwise looking front face axis
    cube_ax0 = np.array([[[ 3,  7, 11],
                          [ 2,  6, 10],
                          [ 1,  5,  9],
                          [ 0,  4,  8]],

                         [[15, 19, 23],
                          [14, 18, 22],
                          [13, 17, 21],
                          [12, 16, 20]]])

    # clockwise looking upper face axis
    cube_ax1 = np.array([[[ 3, 15],
                          [ 7, 19],
                          [11, 23]],

                         [[ 2, 14],
                          [ 6, 18],
                          [10, 22]],

                         [[ 1, 13],
                          [ 5, 17],
                          [ 9, 21]],

                         [[ 0, 12],
                          [ 4, 16],
                          [ 8, 20]]])

    # clockwise looking right face axis
    cube_ax2 = np.array([[[ 8,  9, 10, 11],
                          [20, 21, 22, 23]],

                         [[ 4,  5,  6,  7],
                          [16, 17, 18, 19]],

                         [[ 0,  1,  2,  3],
                          [12, 13, 14, 15]]])

    assert_array_equal(basic_rot_ax(cube_id, ax=0), cube_ax0)
    assert_array_equal(basic_rot_ax(cube_id, ax=1), cube_ax1)
    assert_array_equal(basic_rot_ax(cube_id, ax=2), cube_ax2)


def test_axial_rotations_identity_test():

    cube = np.array(range(2 * 3 * 4)).reshape(2, 3, 4)
    for x in range(3):
        assert_array_equal(axial_rotations(cube,rot=4, ax=x), cube)


def test_flip_data():

    cube_id = np.array([[[0, 1, 2, 3],
                         [4, 5, 6, 7],
                         [8, 9, 10, 11]],

                        [[12, 13, 14, 15],
                         [16, 17, 18, 19],
                         [20, 21, 22, 23]]])

    cube_flip_x = np.array([[[8, 9, 10, 11],
                             [4, 5, 6, 7],
                             [0, 1, 2, 3]],

                            [[20, 21, 22, 23],
                             [16, 17, 18, 19],
                             [12, 13, 14, 15]]])

    cube_flip_y = np.array([[[3, 2, 1, 0],
                             [7, 6, 5, 4],
                             [11, 10, 9, 8]],

                            [[15, 14, 13, 12],
                             [19, 18, 17, 16],
                             [23, 22, 21, 20]]])

    cube_flip_z = np.array([[[12, 13, 14, 15],
                              [16, 17, 18, 19],
                              [20, 21, 22, 23]],

                             [[0, 1, 2, 3],
                              [4, 5, 6, 7],
                              [8, 9, 10, 11]]])

    assert_array_equal(flip_data(cube_id, axis='x'), cube_flip_x)
    assert_array_equal(flip_data(cube_id, axis='y'), cube_flip_y)
    assert_array_equal(flip_data(cube_id, axis='z'), cube_flip_z)


def test_symmetrise_data_():

    cube_id = np.array([[[0, 1, 2, 3],
                         [4, 5, 6, 7],
                         [8, 9, 10, 11],
                         [12, 13, 14, 15]],

                        [[16, 17, 18, 19],
                         [20, 21, 22, 23],
                         [24, 25, 26, 27],
                         [28, 29, 30, 31]]])

    cube_sym_x2_ab_T = np.array([[[ 0,  1,  2,  3],
                                  [ 4,  5,  6,  7],
                                  [ 4,  5,  6,  7],
                                  [ 0,  1,  2,  3]],

                                 [[16, 17, 18, 19],
                                  [20, 21, 22, 23],
                                  [20, 21, 22, 23],
                                  [16, 17, 18, 19]]])

    cube_sym_x3_ab_F = np.array([[[ 0,  1,  2,  3],
                                  [ 4,  5,  6,  7],
                                  [ 8,  9, 10, 11],
                                  [ 8,  9, 10, 11],
                                  [ 4,  5,  6,  7],
                                  [ 0,  1,  2,  3]],

                                 [[16, 17, 18, 19],
                                  [20, 21, 22, 23],
                                  [24, 25, 26, 27],
                                  [24, 25, 26, 27],
                                  [20, 21, 22, 23],
                                  [16, 17, 18, 19]]])

    cube_sym_x3_ab_T = np.array([[[0, 1, 2, 3],
                                  [4, 5, 6, 7],
                                  [8, 9, 10, 11],
                                  [8, 9, 10, 11]],

                                 [[16, 17, 18, 19],
                                  [20, 21, 22, 23],
                                  [24, 25, 26, 27],
                                  [24, 25, 26, 27]]])

    assert_array_equal(symmetrise_data(cube_id, axis='x', plane_intercept=2, side_to_copy='below', keep_in_data_dimensions=True), cube_sym_x2_ab_T)
    assert_array_equal(symmetrise_data(cube_id, axis='x', plane_intercept=3, side_to_copy='below', keep_in_data_dimensions=False), cube_sym_x3_ab_F)
    assert_array_equal(symmetrise_data(cube_id, axis='x', plane_intercept=3, side_to_copy='below', keep_in_data_dimensions=True), cube_sym_x3_ab_T)


''' From manipulations.cutter.py '''
from labels_manager.tools.manipulations.cutter import cut_4d_volume_with_a_1_slice_mask


def test_cut_4d_volume_with_a_1_slice_mask():

    data_0 = np.stack([np.array(range(5*5*5)).reshape(5, 5, 5)] * 4, axis=3)
    mask = np.zeros([5,5,5])
    for k in range(5):
        mask[k, k, k] = 1
    expected_answer_for_each_slice = np.zeros([5, 5, 5])
    for k in range(5):
        expected_answer_for_each_slice[k, k, k] = 30 * k + k
    ans = cut_4d_volume_with_a_1_slice_mask(data_0, mask)

    for k in range(4):
        assert_array_equal(ans[..., k], expected_answer_for_each_slice)


