import os
from os.path import join as jph

import nibabel as nib
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal

from LABelsToolkit.tools.defs import root_dir
from LABelsToolkit.tools.phantoms_generator.generate_simple_phantoms import generate_figures
from LABelsToolkit.tools.aux_methods.utils_nib import replace_translational_part, remove_nan, \
    set_new_header_description


def test_adjust_nifti_replace_translational_part_F_F():

    pfi_input = jph(root_dir, 'data_examples', 'acee.nii.gz')
    if not os.path.exists(pfi_input):
        generate_figures()

    translation = [1, 2, 3]
    pfi_output = jph(root_dir, 'data_examples', 'acee_new_header.nii.gz')

    im_a_cee = nib.load(pfi_input)
    initial_affine = im_a_cee.affine
    initial_qform = im_a_cee.get_qform()
    initial_sform = im_a_cee.get_sform()

    new_im = replace_translational_part(im_a_cee, translation, q_form=False, s_form=False)
    nib.save(new_im, pfi_output)

    im_a_cee_new = nib.load(pfi_output)
    final_affine = im_a_cee_new.affine
    final_qform = im_a_cee_new.get_qform()
    final_sform = im_a_cee_new.get_sform()

    new_transformation_expected = np.copy(initial_affine)
    new_transformation_expected[:3, 3] = translation

    # see documentaiton http://nipy.org/nibabel/nifti_images.html#choosing-image-affine
    if im_a_cee.header['sform_code'] > 0 :  # affine -> sform affine
        assert_array_almost_equal(initial_affine, final_affine)
        assert_array_almost_equal(initial_qform, final_qform)
        assert_array_almost_equal(initial_sform, final_sform)
    elif im_a_cee.header['qform_code'] > 0: # affine -> qform affine
        assert_array_almost_equal(initial_affine, final_affine)
        assert_array_almost_equal(initial_qform, final_qform)
        assert_array_almost_equal(initial_sform, final_sform)
    else: # affine -> header-off affine
        assert_array_almost_equal(new_transformation_expected, final_affine)
        assert_array_almost_equal(initial_qform, final_qform)
        assert_array_almost_equal(initial_sform, final_sform)


def test_adjust_nifti_translation_path_F_T():
    pfi_input = jph(root_dir, 'data_examples', 'acee.nii.gz')
    if not os.path.exists(pfi_input):
        generate_figures()
    translation = [1, 2, 3]
    pfi_output = jph(root_dir, 'data_examples', 'acee_new_header.nii.gz')

    im_a_cee = nib.load(pfi_input)
    initial_affine = im_a_cee.affine  # sform
    initial_qform = im_a_cee.get_qform()
    initial_sform = im_a_cee.get_sform()

    new_im = replace_translational_part(im_a_cee, translation, q_form=False, s_form=True)
    nib.save(new_im, pfi_output)

    im_a_cee_new = nib.load(pfi_output)
    final_affine = im_a_cee_new.affine
    final_qform = im_a_cee_new.get_qform()
    final_sform = im_a_cee_new.get_sform()

    new_transformation_expected = np.copy(initial_affine)
    new_transformation_expected[:3, 3] = translation

    # see documentaiton http://nipy.org/nibabel/nifti_images.html#choosing-image-affine
    if im_a_cee.header['sform_code'] > 0:  # affine -> sform affine
        assert_array_almost_equal(new_transformation_expected, final_affine)
        assert_array_almost_equal(initial_qform, final_qform)
        assert_array_almost_equal(new_transformation_expected, final_sform)
    elif im_a_cee.header['qform_code'] > 0:  # affine -> qform affine
        assert_array_almost_equal(initial_affine, final_affine)
        assert_array_almost_equal(initial_qform, final_qform)
        assert_array_almost_equal(new_transformation_expected, final_sform)
    else:  # affine -> header-off affine
        assert_array_almost_equal(new_transformation_expected, final_affine)
        assert_array_almost_equal(initial_qform, final_qform)
        assert_array_almost_equal(new_transformation_expected, final_sform)

    # check all of the original image is still the same
    im_a_cee_reloaded = nib.load(pfi_input)
    reloaded_affine = im_a_cee_reloaded.affine  # sform
    reloaded_qform = im_a_cee_reloaded.get_qform()
    reloaded_sform = im_a_cee_reloaded.get_sform()

    assert_array_almost_equal(initial_affine, reloaded_affine)
    assert_array_almost_equal(initial_qform, reloaded_qform)
    assert_array_almost_equal(initial_sform, reloaded_sform)


def test_adjust_nifti_translation_path_T_F():
    pfi_input = jph(root_dir, 'data_examples', 'acee.nii.gz')
    if not os.path.exists(pfi_input):
        generate_figures()
    translation = [1, 2, 3]
    pfi_output = jph(root_dir, 'data_examples', 'acee_new_header.nii.gz')

    im_a_cee = nib.load(pfi_input)
    initial_affine = im_a_cee.affine  # sform
    initial_qform = im_a_cee.get_qform()
    initial_sform = im_a_cee.get_sform()

    new_im = replace_translational_part(im_a_cee, translation, q_form=True, s_form=False)
    nib.save(new_im, pfi_output)

    im_a_cee_new = nib.load(pfi_output)
    final_affine = im_a_cee_new.affine
    final_qform = im_a_cee_new.get_qform()
    final_sform = im_a_cee_new.get_sform()

    new_transformation_expected = np.copy(initial_affine)
    new_transformation_expected[:3, 3] = translation

    # sform must be the same
    # see documentaiton http://nipy.org/nibabel/nifti_images.html#choosing-image-affine
    if im_a_cee.header['sform_code'] > 0:  # affine -> sform affine
        assert_array_almost_equal(initial_affine, final_affine)
        assert_array_almost_equal(new_transformation_expected, final_qform)
        assert_array_almost_equal(initial_sform, final_sform)
    elif im_a_cee.header['qform_code'] > 0:  # affine -> qform affine
        assert_array_almost_equal(new_transformation_expected, final_affine)
        assert_array_almost_equal(new_transformation_expected, final_qform)
        assert_array_almost_equal(initial_sform, final_sform)
    else:  # affine -> header-off affine
        assert_array_almost_equal(new_transformation_expected, final_affine)
        assert_array_almost_equal(new_transformation_expected, final_qform)
        assert_array_almost_equal(initial_sform, final_sform)


    # check all of the original image is still the same
    im_a_cee_reloaded = nib.load(pfi_input)
    reloaded_affine = im_a_cee_reloaded.affine  # sform
    reloaded_qform = im_a_cee_reloaded.get_qform()
    reloaded_sform = im_a_cee_reloaded.get_sform()

    assert_array_almost_equal(initial_affine, reloaded_affine)
    assert_array_almost_equal(initial_qform, reloaded_qform)
    assert_array_almost_equal(initial_sform, reloaded_sform)


def test_adjust_nifti_translation_path_T_T():
    pfi_input = jph(root_dir, 'data_examples', 'acee.nii.gz')
    if not os.path.exists(pfi_input):
        generate_figures()
    translation = [1, 2, 3]
    pfi_output = jph(root_dir, 'data_examples', 'acee_new_header.nii.gz')

    im_a_cee = nib.load(pfi_input)
    initial_affine = im_a_cee.affine  # sform
    initial_qform = im_a_cee.get_qform()
    initial_sform = im_a_cee.get_sform()

    new_im = replace_translational_part(im_a_cee, translation, q_form=True, s_form=True)
    nib.save(new_im, pfi_output)

    im_a_cee_new = nib.load(pfi_output)
    final_affine = im_a_cee_new.affine
    final_qform = im_a_cee_new.get_qform()
    final_sform = im_a_cee_new.get_sform()

    # all must be new
    new_transformation = np.copy(initial_affine)
    new_transformation[:3, 3] = translation
    assert_array_almost_equal(new_transformation, final_qform)
    assert_array_almost_equal(new_transformation, final_affine)
    assert_array_almost_equal(new_transformation, final_sform)

    # check all of the original image is still the same
    im_a_cee_reloaded = nib.load(pfi_input)
    reloaded_affine = im_a_cee_reloaded.affine  # sform
    reloaded_qform = im_a_cee_reloaded.get_qform()
    reloaded_sform = im_a_cee_reloaded.get_sform()

    assert_array_almost_equal(initial_affine, reloaded_affine)
    assert_array_almost_equal(initial_qform, reloaded_qform)
    assert_array_almost_equal(initial_sform, reloaded_sform)


def test_remove_nan():
    data_ts = np.array([[[0, 1, 2, 3],
                         [4, 5, np.nan, 7],
                         [8, 9, 10, np.nan]],

                        [[12, 13, 14, 15],
                         [16, np.nan, 18, 19],
                         [20, 21, 22, 23]]])
    im = nib.Nifti1Image(data_ts, np.eye(4))
    im_no_nan = remove_nan(im)

    data_no = np.array([[[0, 1, 2, 3],
                         [4, 5, 0, 7],
                         [8, 9, 10, 0]],

                        [[12, 13, 14, 15],
                         [16, 0, 18, 19],
                         [20, 21, 22, 23]]])

    assert_array_equal(im_no_nan.get_data(), data_no)


def test_set_new_header_description():
    arr_data = np.zeros([10,10,10])
    im = nib.Nifti1Image(arr_data, np.eye(4))

    hd = im.header
    hd['descrip'] = 'Old Spam'

    im_new_header = set_new_header_description(im, new_header_description='New Spam')

    new_hd = im_new_header.header
    assert new_hd['descrip'] == 'New Spam'
