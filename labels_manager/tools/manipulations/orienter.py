import os
from os.path import join as jph
import nibabel as nib
import numpy as np



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
