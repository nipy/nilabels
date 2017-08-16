import os
from os.path import join as jph
import nibabel as nib
import numpy as np

from labels_manager.tools.aux_methods.utils import print_and_run


def adjust_affine_header(pfi_input, pfi_output, theta, trasl=np.array([0, 0, 0])):

    if theta != 0:
        # transformations parameters
        rot_x = np.array([[1,            0,           0,      trasl[0]],
                         [0,  np.cos(theta),  -np.sin(theta), trasl[1]],
                         [0,  np.sin(theta), np.cos(theta),   trasl[2]],
                         [0,             0,          0,       1]])

        # Load input image:
        im_input = nib.load(pfi_input)

        # generate new affine transformation (from bicommissural to histological)
        new_transf = rot_x.dot(im_input.get_affine())

        # create output image on the input
        if im_input.header['sizeof_hdr'] == 348:
            new_image = nib.Nifti1Image(im_input.get_data(), new_transf, header=im_input.get_header())
        # if nifty2
        elif im_input.header['sizeof_hdr'] == 540:
            new_image = nib.Nifti2Image(im_input.get_data(), new_transf, header=im_input.get_header())
        else:
            raise IOError

        # print intermediate results
        print('Affine input image: \n')
        print(im_input.get_affine())
        print('Affine after transformation: \n')
        print(new_image.get_affine())

        # sanity check
        np.testing.assert_almost_equal(np.linalg.det(new_transf), np.linalg.det(im_input.get_affine()))

        # save output image
        nib.save(new_image, pfi_output)
    else:
        if not pfi_input == pfi_output:
            print_and_run('cp {0} {1}'.format(pfi_input, pfi_output))


def adjust_nifti_translation_path(pfi_nifti_input, new_traslation, pfi_nifti_output, q_form=True, s_form=True,
                                  verbose=1):
    """
    Change q_form or s_form or both translational part.
    :param pfi_nifti_input: path to file of the input image
    :param new_traslation: 3dim array, affine coordinates, will be the future translational part of the affine.
    :param pfi_nifti_output: path to file of the image with the modifed translation. Try not to be destructive, unless you do not really want.
    :param q_form: [True] affect q_form
    :param s_form: [True] affect s_form
    :param verbose:
    :return: None. It creates a new image in pfi_nifti_output with defined translational part.
    """
    im_input = nib.load(pfi_nifti_input)

    # generate new affine transformation (from bicommissural to histological)
    aff = im_input.affine
    # create output image on the input
    if im_input.header['sizeof_hdr'] == 348:
        new_image = nib.Nifti1Image(im_input.get_data(), aff, header=im_input.header)
    # if nifty2
    elif im_input.header['sizeof_hdr'] == 540:
        new_image = nib.Nifti2Image(im_input.get_data(), aff, header=im_input.header)
    else:
        raise IOError

    new_transf = np.copy(aff)
    if len(new_traslation) == 4 and new_traslation[-1] == 1:
        new_transf[:, 3] = new_traslation
    elif len(new_traslation) == 3:
        new_transf[:3, 3] = new_traslation
    else:
        raise IOError

    if q_form:
        new_image.set_qform(new_transf)

    if s_form:
        new_image.set_sform(new_transf)

    new_image.update_header()

    if verbose > 0:
        # print intermediate results
        print('Affine input image:')
        print(im_input.get_affine())
        print('Affine after update:')
        print(new_image.get_affine())
        print('Q-form after update:')
        print(new_image.get_qform())
        print('S-form after update:')
        print(new_image.get_sform())

    # save output image
    nib.save(new_image, pfi_nifti_output)


def apply_orientation_matrix_to_image(pfi_nifti_image, affine_transformation_left,
                                      pfo_output=None, pfi_b_vects=None, suffix='new', verbose=1):
    """

    :param pfi_nifti_image: path to file to a nifti image.
    :param affine_transformation_left: a reorientation matrix.
    :param pfo_output: path to folder where the output will be stored
    :param pfi_b_vects: path to file to b-vectors.
    :param suffix: added at the end of the new files.
    :param verbose:
    :return: None. It saves the input image and the optional input b-vectors transformed according to a matrix.
    """
    # TODO nib images as input.
    assert isinstance(affine_transformation_left, np.ndarray)

    if pfo_output is None:
        pfo_output = os.path.dirname(pfi_nifti_image)

    pfi_new_image = jph(pfo_output, os.path.basename(pfi_nifti_image).split('.')[0] + '{}.nii.gz'.format(suffix))

    im = nib.load(pfi_nifti_image)

    new_affine = im.affine.dot(affine_transformation_left)

    if im.header['sizeof_hdr'] == 348:
        new_im = nib.Nifti1Image(im.get_data(), new_affine, header=im.get_header())
    elif im.header['sizeof_hdr'] == 540:
        new_im = nib.Nifti2Image(im.get_data(), new_affine, header=im.get_header())
    else:
        raise IOError

    # sanity check
    msg = 'Is the input matrix a re-orientation matrix?'
    np.testing.assert_almost_equal(np.linalg.det(new_affine), np.linalg.det(im.get_affine()), err_msg=msg)

    # save output image
    nib.save(new_im, pfi_new_image)
    if pfi_b_vects is not None:
        bvects_name = os.path.basename(pfi_b_vects).split('.')[0]
        bvects_ext = os.path.basename(pfi_b_vects).split('.')[-1]
        pfi_new_bvects = jph(pfo_output, '{0}_{1}.{2}'.format(bvects_name, suffix, bvects_ext))

        if bvects_ext == 'txt':
            b_vects = np.loadtxt(pfi_b_vects)
            new_bvects = np.einsum('ij, kj -> ki', affine_transformation_left[:3, :3], b_vects)
            np.savetxt(pfi_new_bvects, new_bvects, fmt='%10.14f')
        else:
            b_vects = np.load(pfi_b_vects)
            new_bvects = np.einsum('ij, kj -> ki', affine_transformation_left[:3, :3], b_vects)
            np.save(pfi_new_bvects, new_bvects)

    if verbose > 0:
        # print intermediate results
        print('Affine input image: \n')
        print(im.get_affine())
        print('Affine after transformation: \n')
        print(new_im.get_affine())
