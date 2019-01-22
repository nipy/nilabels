import numpy as np
import pytest

from nilabels.tools.image_colors_manipulations.relabeller import relabeller, permute_labels, erase_labels, \
    assign_all_other_labels_the_same_value, keep_only_one_label, relabel_half_side_one_label


def test_relabeller_basic():
    data = np.array(range(10)).reshape(2, 5)
    relabelled_data = relabeller(data, range(10), range(10)[::-1])
    np.testing.assert_array_equal(relabelled_data, np.array(range(10)[::-1]).reshape(2,5))


def test_relabeller_one_element():
    data = np.array(range(10)).reshape(2, 5)
    relabelled_data = relabeller(data, 0, 1, verbose=1)
    expected_output = data[:]
    expected_output[0, 0] = 1
    np.testing.assert_array_equal(relabelled_data, expected_output)


def test_relabeller_one_element_not_in_array():
    data = np.array(range(10)).reshape(2, 5)
    relabelled_data = relabeller(data, 15, 1, verbose=1)
    np.testing.assert_array_equal(relabelled_data, data)


def test_relabeller_wrong_input():
    data = np.array(range(10)).reshape(2, 5)
    with np.testing.assert_raises(IOError):
        relabeller(data, [1, 2], [3, 4, 4])


def test_permute_labels_invalid_permutation():
    invalid_permutation = [[3, 3, 3], [1, 1, 1]]
    with pytest.raises(IOError):
        permute_labels(np.zeros([3, 3]), invalid_permutation)


def test_permute_labels_valid_permutation():
    data = np.array([[1, 2, 3],
                     [1, 2, 3],
                     [1, 2, 3]])
    valid_permutation = [[1, 2, 3], [1, 3, 2]]
    perm_data = permute_labels(data, valid_permutation)
    expected_data = np.array([[1, 3, 2],
                              [1, 3, 2],
                              [1, 3, 2]])
    np.testing.assert_equal(perm_data, expected_data)


def test_erase_label_simple():
    data = np.array(range(10)).reshape(2, 5)
    data_erased_1 = erase_labels(data, 1)
    expected_output = data[:]
    expected_output[0, 1] = 0
    np.testing.assert_array_equal(data_erased_1, expected_output)


def test_assign_all_other_labels_the_same_values_simple():
    data = np.array(range(10)).reshape(2, 5)
    data_erased_1 = erase_labels(data, 1)
    data_labels_to_keep = assign_all_other_labels_the_same_value(data, range(2, 10), same_value_label=0)
    np.testing.assert_array_equal(data_erased_1, data_labels_to_keep)


def test_assign_all_other_labels_the_same_values_single_value():
    data = np.array(range(10)).reshape(2, 5)
    data_erased_1 = np.zeros_like(data)
    data_erased_1[0, 1] = 1
    data_labels_to_keep = assign_all_other_labels_the_same_value(data, 1, same_value_label=0)
    np.testing.assert_array_equal(data_erased_1, data_labels_to_keep)


def test_keep_only_one_label_label_simple():
    data = np.array(range(10)).reshape(2, 5)
    new_data = keep_only_one_label(data, 1)
    expected_data = np.zeros([2, 5])
    expected_data[0, 1] = 1
    np.testing.assert_array_equal(new_data, expected_data)


def test_keep_only_one_label_label_not_present():
    data = np.array(range(10)).reshape(2, 5)
    new_data = keep_only_one_label(data, 120)
    np.testing.assert_array_equal(new_data, data)


def test_relabel_half_side_one_label_wrong_input_shape():
    data = np.array(range(10)).reshape(2, 5)
    with np.testing.assert_raises(IOError):
        relabel_half_side_one_label(data, label_old=[1, 2], label_new=[2, 1], side_to_modify='above',
                                    axis='x', plane_intercept=2)


def test_relabel_half_side_one_label_wrong_input_side():
    data = np.array(range(27)).reshape(3, 3, 3)
    with np.testing.assert_raises(IOError):
        relabel_half_side_one_label(data, label_old=[1, 2], label_new=[2, 1], side_to_modify='spam',
                                    axis='x', plane_intercept=2)


def test_relabel_half_side_one_label_wrong_input_axis():
    data = np.array(range(27)).reshape(3, 3, 3)
    with np.testing.assert_raises(IOError):
        relabel_half_side_one_label(data, label_old=[1, 2], label_new=[2, 1], side_to_modify='above',
                                    axis='spam', plane_intercept=2)


def test_relabel_half_side_one_label_wrong_input_simple():
    data = np.array(range(3 ** 3)).reshape(3, 3, 3)
    # Z above
    new_data = relabel_half_side_one_label(data, label_old=1, label_new=100, side_to_modify='above',
                                           axis='z', plane_intercept=1)
    expected_data = data[:]
    expected_data[0, 0, 1] = 100

    np.testing.assert_array_equal(new_data, expected_data)

    # Z below
    new_data = relabel_half_side_one_label(data, label_old=3, label_new=300, side_to_modify='below',
                                           axis='z', plane_intercept=2)
    expected_data = data[:]
    expected_data[0, 1, 0] = 300

    np.testing.assert_array_equal(new_data, expected_data)

    # Y above
    new_data = relabel_half_side_one_label(data, label_old=8, label_new=800, side_to_modify='above',
                                           axis='y', plane_intercept=1)
    expected_data = data[:]
    expected_data[0, 2, 2] = 800

    np.testing.assert_array_equal(new_data, expected_data)

    # Y below
    new_data = relabel_half_side_one_label(data, label_old=6, label_new=600, side_to_modify='below',
                                           axis='y', plane_intercept=2)
    expected_data = data[:]
    expected_data[0, 2, 0] = 600
    np.testing.assert_array_equal(new_data, expected_data)

    # X above
    new_data = relabel_half_side_one_label(data, label_old=18, label_new=180, side_to_modify='above',
                                           axis='x', plane_intercept=1)
    expected_data = data[:]
    expected_data[2, 0, 0] = 180
    np.testing.assert_array_equal(new_data, expected_data)

    # X below
    new_data = relabel_half_side_one_label(data, label_old=4, label_new=400, side_to_modify='below',
                                           axis='x', plane_intercept=2)
    expected_data = data[:]
    expected_data[0, 1, 1] = 400
    np.testing.assert_array_equal(new_data, expected_data)


if __name__ == '__main__':
    test_relabeller_basic()
    test_relabeller_one_element()
    test_relabeller_one_element_not_in_array()
    test_relabeller_wrong_input()

    test_permute_labels_invalid_permutation()
    test_permute_labels_valid_permutation()

    test_erase_label_simple()

    test_assign_all_other_labels_the_same_values_simple()
    test_assign_all_other_labels_the_same_values_single_value()

    test_keep_only_one_label_label_simple()
    test_keep_only_one_label_label_not_present()

    test_relabel_half_side_one_label_wrong_input_shape()
    test_relabel_half_side_one_label_wrong_input_side()
    test_relabel_half_side_one_label_wrong_input_axis()

    test_relabel_half_side_one_label_wrong_input_simple()
