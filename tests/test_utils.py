import os
from os.path import join as jph

import nibabel as nib
import numpy as np
from nose.tools import assert_equals, assert_raises
from numpy.testing import assert_array_equal, assert_array_almost_equal

from defs import root_dir
from labels_manager.tools.phantoms_generator.generate_data_examples import generate_figures
from labels_manager.tools.aux_methods.utils import adjust_nifti_translation_path

def test_adjust_nifti_translation_path_F_F():
    pfi_input = jph(root_dir, 'data_examples', 'acee.nii.gz')
    if not os.path.exists(pfi_input):
        generate_figures()

    translation = [1, 2, 3]
    pfi_output = jph(root_dir, 'data_examples', 'acee_new_header.nii.gz')

    im_a_cee = nib.load(pfi_input)
    initial_affine = im_a_cee.affine  # sform
    initial_qform = im_a_cee.get_qform()
    initial_sform = im_a_cee.get_sform()

    adjust_nifti_translation_path(pfi_input, translation, pfi_output, q_form=False, s_form=False)

    im_a_cee_new = nib.load(pfi_output)
    final_affine = im_a_cee_new.affine
    final_qform = im_a_cee_new.get_qform()
    final_sform = im_a_cee_new.get_sform()

    # all must be the same
    assert_array_almost_equal(initial_affine, final_affine)
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

    adjust_nifti_translation_path(pfi_input, translation, pfi_output, q_form=False, s_form=True)

    im_a_cee_new = nib.load(pfi_output)
    final_affine = im_a_cee_new.affine
    final_qform = im_a_cee_new.get_qform()
    final_sform = im_a_cee_new.get_sform()

    # qform must be the same
    assert_array_almost_equal(initial_qform, final_qform)

    # sform must be new
    new_transformation = np.copy(initial_affine)
    new_transformation[:3, 3] = translation
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

    adjust_nifti_translation_path(pfi_input, translation, pfi_output, q_form=True, s_form=False)

    im_a_cee_new = nib.load(pfi_output)
    final_affine = im_a_cee_new.affine
    final_qform = im_a_cee_new.get_qform()
    final_sform = im_a_cee_new.get_sform()

    # sform must be the same
    assert_array_almost_equal(initial_affine, final_affine)
    assert_array_almost_equal(initial_sform, final_sform)

    # aform must be new
    new_transformation = np.copy(initial_affine)
    new_transformation[:3, 3] = translation
    assert_array_almost_equal(new_transformation, final_qform)

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

    adjust_nifti_translation_path(pfi_input, translation, pfi_output, q_form=True, s_form=True)

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
