import nibabel as nib
import numpy as np
import os

from labels_manager.tools.aux_methods.utils_nib import set_new_data


def modify_image_type(im_input, new_dtype, update_description=None, verbose=1):
    if update_description is not None:
        if not isinstance(update_description, str):
            raise IOError('update_description must be a string')
        hd = im_input.header
        hd['descrip'] = update_description
        im_input.update_header()
    new_im = set_new_data(im_input, im_input.get_data().astype(new_dtype), new_dtype=new_dtype, remove_nan=True)
    if verbose > 0:
        print('Data type before {}'.format(im_input.get_data_dtype()))
        print('Data type after {}'.format(new_im.get_data_dtype()))
    return new_im


def modify_affine_transformation(im_input, new_aff, q_form=True, s_form=True, verbose=1, multiplication_side='left'):
    """
    Change q_form or s_form or both translational part and rotational part.
    :param im_input: nibabel input image
    :param q_form: [True] affect q_form
    :param s_form: [True] affect s_form
    :param multiplication_side: can be lef, right, or replace.
    :param verbose:

    :return: None. It creates a new image in pfi_nifti_output with defined translational part.
    """
    if np.linalg.det(new_aff) < 0 :
        print('WARNING: affine matrix proposed has negative determinant.')
    if multiplication_side is 'left':
        new_transf = new_aff.dot(im_input.get_affine())
    elif multiplication_side is 'right':
        new_transf = im_input.get_affine().dot(new_aff)
    elif multiplication_side is 'replace':
        new_transf = new_aff
    else:
        raise IOError('multiplication_side parameter can be lef, right, or replace.')

    # create output image on the input
    if im_input.header['sizeof_hdr'] == 348:
        new_image = nib.Nifti1Image(im_input.get_data(), new_transf, header=im_input.get_header())
    # if nifty2
    elif im_input.header['sizeof_hdr'] == 540:
        new_image = nib.Nifti2Image(im_input.get_data(), new_transf, header=im_input.get_header())
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

    return new_image
