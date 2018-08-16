from os.path import join as jph

import nibabel as nib
import numpy as np
from nose.tools import assert_equals, assert_raises
from numpy.testing import assert_array_equal

from LABelsToolkit.tools.defs import root_dir
from LABelsToolkit.tools.aux_methods.sanity_checks import check_pfi_io
from LABelsToolkit.tools.aux_methods.utils_nib import set_new_data, compare_two_nib
from LABelsToolkit.tools.aux_methods.utils import eliminates_consecutive_duplicates, lift_list, labels_query
from LABelsToolkit.tools.aux_methods.utils import permutation_from_cauchy_to_disjoints_cycles, \
    permutation_from_disjoint_cycles_to_cauchy


''' Test aux_methods.morphological.py'''
from LABelsToolkit.tools.aux_methods.morpological_operations import get_morphological_patch, get_patch_values, \
    get_circle_shell_for_given_radius


def test_get_morpological_patch():
    expected = np.ones([3, 3]).astype(np.bool)
    expected[0, 0] = False
    expected[0, 2] = False
    expected[2, 0] = False
    expected[2, 2] = False
    assert_array_equal(get_morphological_patch(2, 'circle'), expected)
    assert_array_equal(get_morphological_patch(2, 'square'), np.ones([3, 3]).astype(np.bool))


def test_get_patch_values_simple():
    # toy mask on a simple image:
    image = np.random.randint(0, 10, (7, 7))
    patch = np.zeros_like(image).astype(np.bool)
    patch[2, 2] = True
    patch[2, 3] = True
    patch[3, 2] = True
    patch[3, 3] = True

    vals = get_patch_values([2, 2, 2], image, morfo_mask=patch)
    assert_array_equal([image[2, 2], image[2, 3], image[3, 2], image[3, 3]], vals)


def test_get_shell_for_given_radius():
    expected_ans = [(-2, 0, 0), (-1, -1, -1), (-1, -1, 0), (-1, -1, 1), (-1, 0, -1), (-1, 0, 1), (-1, 1, -1),
                    (-1, 1, 0), (-1, 1, 1), (0, -2, 0), (0, -1, -1), (0, -1, 1), (0, 0, -2), (0, 0, 2), (0, 1, -1),
                    (0, 1, 1), (0, 2, 0), (1, -1, -1), (1, -1, 0), (1, -1, 1), (1, 0, -1), (1, 0, 1), (1, 1, -1),
                    (1, 1, 0), (1, 1, 1), (2, 0, 0)]
    computed_ans = get_circle_shell_for_given_radius(2)

    assert len(expected_ans) == len(computed_ans)
    assert set(tuple(expected_ans)) == set(tuple(computed_ans))


def test_check_pfi_io():
    assert check_pfi_io(root_dir, None)
    assert check_pfi_io(root_dir, root_dir)

    non_existing_file = jph(root_dir, 'non_existing_file.txt')
    file_in_non_existing_folder = jph(root_dir, 'non_existing_folder/non_existing_file.txt')

    with assert_raises(IOError):
        check_pfi_io(non_existing_file, None)
    with assert_raises(IOError):
        check_pfi_io(root_dir, file_in_non_existing_folder)


''' Test aux_methods.utils_nib.py '''


def test_set_new_data_simple_modifications():
    aff = np.eye(4)
    aff[2, 1] = 42.0

    im_0 = nib.Nifti1Image(np.zeros([3, 3, 3]), affine=aff)
    im_0_header = im_0.header
    # default intent_code
    assert_equals(im_0_header['intent_code'], 0)
    # change intento code
    im_0_header['intent_code'] = 5

    # generate new nib from the old with new data
    im_1 = set_new_data(im_0, np.ones([3, 3, 3]))
    im_1_header = im_1.header
    # see if the infos are the same as in the modified header
    assert_array_equal(im_1.get_data()[:], np.ones([3, 3, 3]))
    assert_equals(im_1_header['intent_code'], 5)
    assert_array_equal(im_1.get_affine(), aff)


def test_compare_two_nib_equals():
    im_0 = nib.Nifti1Image(np.zeros([3, 3, 3]), affine=np.eye(4))
    im_1 = nib.Nifti1Image(np.zeros([3, 3, 3]), affine=np.eye(4))
    assert_equals(compare_two_nib(im_0, im_1), True)


def test_compare_two_nib_different_nifti_version():
    im_0 = nib.Nifti1Image(np.zeros([3, 3, 3]), affine=np.eye(4))
    im_1 = nib.Nifti2Image(np.zeros([3, 3, 3]), affine=np.eye(4))
    assert_equals(compare_two_nib(im_0, im_1), False)


def test_compare_two_nib_different_affine():
    aff_1 = np.eye(4)
    aff_1[3, 3] = 5
    im_0 = nib.Nifti1Image(np.zeros([3, 3, 3]), affine=np.eye(4))
    im_1 = nib.Nifti1Image(np.zeros([3, 3, 3]), affine=aff_1)
    assert_equals(compare_two_nib(im_0, im_1), False)


''' Test tools.aux_methods.utils.py'''


def test_eliminates_consecutive_duplicates():
    l_in, l_out = [0, 0, 0, 1, 1, 2, 3, 4, 5, 5, 5, 6, 7, 8, 9], range(10)
    assert_array_equal(eliminates_consecutive_duplicates(l_in), l_out)


def test_lift_list_1():
    l_in, l_out = [[0, 1], 2, 3, [4, [5, 6]], 7, [8, [9]]], range(10)
    assert_array_equal(lift_list(l_in), l_out)


def test_lift_list_2():
    l_in, l_out = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], range(10)
    assert_array_equal(lift_list(l_in), l_out)


def test_lift_list_3():
    l_in, l_out = [], []
    assert_array_equal(lift_list(l_in), l_out)


def test_labels_query_int_input():
    lab, lab_names = labels_query(1)
    assert_array_equal(lab, [1])
    assert_array_equal(lab_names, ['1'])


def test_labels_query_list_input1():
    lab, lab_names = labels_query([1, 2, 3])
    assert_array_equal(lab, [1, 2, 3])
    assert_array_equal(lab_names, ['1', '2', '3'])


def test_labels_query_list_input2():
    lab, lab_names = labels_query([1, 2, 3, [4, 5, 6]])
    assert_array_equal(lift_list(lab), lift_list([1, 2, 3, [4, 5, 6]]))
    assert_array_equal(lab_names, ['1', '2', '3', '[4, 5, 6]'])


def test_labels_query_all_or_tot_input():
    v = np.arange(10).reshape(5, 2)
    lab, lab_names = labels_query('all', v, remove_zero=False)
    assert_array_equal(lab, np.arange(10))
    lab, lab_names = labels_query('tot', v, remove_zero=False)
    assert_array_equal(lab, np.arange(10))
    lab, lab_names = labels_query('tot', v, remove_zero=True)
    assert_array_equal(lab, np.arange(10)[1:])


# Test permutations:


def test_from_permutation_to_disjoints_cycles():
    cauchy_perm = [[1, 2, 3, 4, 5], [3, 4, 5, 2, 1]]
    cycles_perm = permutation_from_cauchy_to_disjoints_cycles(cauchy_perm)
    expected_ans = [[1, 3, 5], [2, 4]]
    for c1, c2 in zip(expected_ans, cycles_perm):
        assert_array_equal(c1, c2)


def test_from_disjoint_cycles_to_permutation():
    cycles_perm = [[1, 3, 5], [2, 4]]
    cauchy_perm = permutation_from_disjoint_cycles_to_cauchy(cycles_perm)
    expected_ans = [[1, 2, 3, 4, 5], [3, 4, 5, 2, 1]]
    for c1, c2 in zip(cauchy_perm, expected_ans):
        assert_array_equal(c1, c2)


def test_from_permutation_to_disjoints_cycles_single_cycle():
    cauchy_perm = [[1, 2, 3, 4, 5, 6, 7],
                   [3, 4, 5, 1, 2, 7, 6]]
    cycles_perm = permutation_from_cauchy_to_disjoints_cycles(cauchy_perm)
    expected_ans = [[1, 3, 5, 2, 4], [6,7]]

    print expected_ans
    print cycles_perm

    for c1, c2 in zip(expected_ans, cycles_perm):
        assert_array_equal(c1, c2)


def test_from_disjoint_cycles_to_permutation_single_cycle():
    cycles_perm = [[1, 3, 5, 2, 4]]
    cauchy_perm = permutation_from_disjoint_cycles_to_cauchy(cycles_perm)
    expected_ans = [[1, 2, 3, 4, 5], [3, 4, 5, 1, 2]]

    print expected_ans
    print cauchy_perm

    for c1, c2 in zip(cauchy_perm, expected_ans):
        assert_array_equal(c1, c2)


test_from_permutation_to_disjoints_cycles_single_cycle()
test_from_disjoint_cycles_to_permutation_single_cycle()

test_from_disjoint_cycles_to_permutation()
test_from_permutation_to_disjoints_cycles()

#
# permutation_from_cauchy_to_disjoints_cycles,