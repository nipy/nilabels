import nibabel as nib
import numpy as np
import pytest
from numpy.testing import assert_array_equal


''' From  manipulations.relabeller.py'''
from nilabels.tools.image_colors_manipulations.relabeller import relabeller, permute_labels, erase_labels, \
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
    invalid_permutation = [[3, 3, 3], [1, 1, 1]]
    with pytest.raises(IOError):
        permute_labels(np.zeros([3, 3]), invalid_permutation)


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
from nilabels.tools.image_shape_manipulations.splitter import split_labels_to_4d


def test_split_labels_to_4d():
    data = np.array(range(8)).reshape(2, 2, 2)
    splitted_4d = split_labels_to_4d(data, list_labels=range(8))
    for t in range(8):
        expected_slice = np.zeros(8)
        expected_slice[t] = t
        assert_array_equal(splitted_4d[...,t], expected_slice.reshape(2,2,2))


if __name__ == '__main__':
    test_permute_label_invalid_permutation()

# def test_cut_4d_volume_with_a_1_slice_mask():
#
#     data_0 = np.stack([np.array(range(5*5*5)).reshape(5, 5, 5)] * 4, axis=3)
#     mask = np.zeros([5,5,5])
#     for k in range(5):
#         mask[k, k, k] = 1
#     expected_answer_for_each_slice = np.zeros([5, 5, 5])
#     for k in range(5):
#         expected_answer_for_each_slice[k, k, k] = 30 * k + k
#     ans = cut_4d_volume_with_a_1_slice_mask(data_0, mask)
#
#     for k in range(4):
#         assert_array_equal(ans[..., k], expected_answer_for_each_slice)


