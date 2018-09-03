from os.path import join as jph
from numpy.testing import assert_array_equal, assert_raises

from nilabels.definitions import root_dir
from nilabels.tools.aux_methods.sanity_checks import check_pfi_io, check_path_validity, is_valid_permutation

from nilabels.tools.aux_methods.utils_path import connect_path_tail_head, get_pfi_in_pfi_out

from .decorators_tools import create_and_erase_temporary_folder_with_a_dummy_nifti_image, pfo_tmp_test


# TEST: methods sanity_checks


def test_check_pfi_io():
    assert check_pfi_io(root_dir, None)
    assert check_pfi_io(root_dir, root_dir)

    non_existing_file = jph(root_dir, 'non_existing_file.txt')
    file_in_non_existing_folder = jph(root_dir, 'non_existing_folder/non_existing_file.txt')

    with assert_raises(IOError):
        check_pfi_io(non_existing_file, None)
    with assert_raises(IOError):
        check_pfi_io(root_dir, file_in_non_existing_folder)


def test_check_path_validity_not_existing_path():
    with assert_raises(IOError):
        check_path_validity('/Spammer/path_to_spam')


@create_and_erase_temporary_folder_with_a_dummy_nifti_image
def test_check_path_validity_for_a_nifti_image():
    assert check_path_validity(jph(pfo_tmp_test, 'dummy_image.nii.gz'))


def test_check_path_validity_root():
    assert check_path_validity(root_dir)


def test_is_valid_permutation():
    assert not is_valid_permutation([1, 2, 3])
    assert not is_valid_permutation([[1, 2, 3, 4], [3, 1, 2]])
    assert not is_valid_permutation([[1, 2, 3], [4, 5, 6]])
    assert not is_valid_permutation([[1, 1, 3], [1, 3, 1]])
    assert not is_valid_permutation([[1.2, 2, 3], [2, 1.2, 3]])
    assert is_valid_permutation([[1.2, 2, 3], [2, 1.2, 3]], for_labels=False)
    assert is_valid_permutation([[1, 2, 3], [3, 1, 2]])


# TEST aux_methods.sanity_checks.py - NOTE - this is the core of the manager design '''


def test_connect_tail_head_path():
    # Case 1:
    assert cmp(connect_path_tail_head('as/df/gh', 'lm.txt'), 'as/df/gh/lm.txt') == 0
    # Case 2:
    assert cmp(connect_path_tail_head('as/df/gh', 'as/df/gh/lm/nb.txt'), 'as/df/gh/lm/nb.txt') == 0
    # Case 3:
    assert cmp(connect_path_tail_head('as/df/gh', 'lm/nb.txt'), 'as/df/gh/lm/nb.txt') == 0


def test_get_pfi_in_pfi_out():

    tail_a = jph(root_dir, 'tests')
    tail_b = root_dir
    head_a = 'test_auxiliary_methods.py'
    head_b = 'head_b.txt'

    assert_array_equal(get_pfi_in_pfi_out(head_a, None, tail_a, None), (jph(tail_a, head_a), jph(tail_a, head_a)))
    assert_array_equal(get_pfi_in_pfi_out(head_a, head_b, tail_a, None), (jph(tail_a, head_a), jph(tail_a, head_b)))
    assert_array_equal(get_pfi_in_pfi_out(head_a, head_b, tail_a, tail_b), (jph(tail_a, head_a), jph(tail_b, head_b)))


if __name__ == '__main__':
    test_check_pfi_io()
    test_check_path_validity_not_existing_path()
    test_check_path_validity_for_a_nifti_image()
    test_check_path_validity_root()
    test_is_valid_permutation()

    test_connect_tail_head_path()
    test_get_pfi_in_pfi_out()
