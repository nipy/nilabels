import numpy as np
import os
import nibabel as nib

from definitions import root_dir
from numpy.testing import assert_array_almost_equal

from nose.tools import assert_equals, assert_raises, assert_almost_equals
from numpy.testing import assert_array_equal, assert_almost_equal


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


''' From  manipulations.relabeller.py'''
from labels_manager.tools.manipulations.relabeller import relabeller, permute_labels, erase_labels, \
    assign_all_other_labels_the_same_value, keep_only_one_label


