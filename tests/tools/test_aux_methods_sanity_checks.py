from os.path import join as jph
from numpy.testing import assert_raises

from nilabels.definitions import root_dir
from nilabels.tools.aux_methods.sanity_checks import check_pfi_io, check_path_validity, is_valid_permutation

from tests.tools.decorators_tools import create_and_erase_temporary_folder_with_a_dummy_nifti_image, pfo_tmp_test


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


if __name__ == '__main__':
    test_check_pfi_io()
    test_check_path_validity_not_existing_path()
    test_check_path_validity_for_a_nifti_image()
    test_check_path_validity_root()
    test_is_valid_permutation()
