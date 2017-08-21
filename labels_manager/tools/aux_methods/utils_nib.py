import numpy as np
import nibabel as nib


def set_new_data(image, new_data, new_dtype=None, remove_nan=True):
    """
    From a nibabel image and a numpy array it creates a new image with
    the same header of the image and the new_data as its data.
    :param image: nibabel image
    :param new_data:
    :param new_dtype: numpy array
    :param remove_nan:
    :return: nibabel image
    """
    hd = image.header
    if remove_nan:
        new_data = np.nan_to_num(new_data)

    # update data type:
    if new_dtype is not None:
        new_data = new_data.astype(new_dtype)
        image.set_data_dtype(new_dtype)

    # if nifty1
    if hd['sizeof_hdr'] == 348:
        new_image = nib.Nifti1Image(new_data, image.affine, header=hd)
    # if nifty2
    elif hd['sizeof_hdr'] == 540:
        new_image = nib.Nifti2Image(new_data, image.affine, header=hd)
    else:
        raise IOError('Input image header problem')

    return new_image


def compare_two_nib(im1, im2):
    """
    :param im1: one nibabel image
    :param im2: another nibabel image
    :return: true false and plot to console if the images are the same or not (up to a tollerance in the data)
    """

    im1_name = 'First argument'
    im2_name = 'Second argument'
    msg = ''

    hd1 = im1.header
    hd2 = im2.header

    images_are_equals = True

    # compare nifty version:
    if not hd1['sizeof_hdr'] == hd2['sizeof_hdr']:

        if hd1['sizeof_hdr'] == 348:
            msg += '{0} is nifti1\n{1} is nifti2.'.format(im1_name, im2_name)
        else:
            msg += '{0} is nifti2\n{1} is nifti1.'.format(im1_name, im2_name)

        images_are_equals = False

    # Compare headers:
    else:
        for k in hd1.keys():
            if k not in ['scl_slope', 'scl_inter']:
                val1, val2 = hd1[k], hd2[k]
                are_different = val1 != val2
                if isinstance(val1, np.ndarray):
                    are_different = are_different.any()

                if are_different:
                    images_are_equals = False
                    msg += 'Header Key {0} for {1} is {2} - for {3} is {4}  \n'.format(k, im1_name , hd1[k], im2_name, hd2[k])

            elif not np.isnan(hd1[k]) and np.isnan(hd2[k]):
                images_are_equals = False
                msg += 'Header Key {0} for {1} is {2} - for {3} is {4}  \n'.format(k, im1_name, hd1[k], im2_name, hd2[k])

    # Compare values and type:
    if not im1.get_data_dtype() == im2.get_data_dtype():
        msg += 'Dtype are different consistent {0} {1} - {2} {3} \n'.format(im1_name, im1.get_data_dtype(), im2_name, im2.get_data_dtype())
        images_are_equals = False
    if not np.array_equal(im1.get_data(), im2.get_data()):
        msg += 'Data are different. \n'
        images_are_equals = False
    if not np.array_equal(im1.get_affine(), im2.get_affine()):
        msg += 'Affine transformations are different. \n'
        images_are_equals = False

    print(msg)
    return images_are_equals


def compare_two_nifti(path_img_1, path_img_2):
    """
    ... assuming nibabel take into account all the information in the nifty header properly!
    :param path_img_1:
    :param path_img_2:
    :return:
    """
    im1 = nib.load(path_img_1)
    im2 = nib.load(path_img_2)

    return compare_two_nib(im1, im2)


def one_voxel_volume(im):
    return np.round(np.abs(np.prod(np.diag(im.get_affine()))), decimals=6)


# ---------- Header modifications ---------------


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
    :param new_aff: new affine transformation to be multiplied of replaced
    :param q_form: [True] affect q_form
    :param s_form: [True] affect s_form
    :param multiplication_side: can be lef, right, or replace.
    :param verbose:

    :return: None. It creates a new image in pfi_nifti_output with defined translational part.

    NOTE: see the documentation http://nipy.org/nibabel/nifti_images.html#choosing-image-affine to
    understand relationships between s_form affine, q_form affine and fall-back header affine.
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
    else:
        new_image.set_qform(im_input.get_affine())

    if s_form:
        new_image.set_sform(new_transf)
    else:
        new_image.set_sform(im_input.get_affine())

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


def replace_translational_part(im_input, new_translation, q_form=True, s_form=True):
    """
    Create new image with translational part replaced. Not destructive.
    :param im_input: nibabel input image
    :param new_translation: new translational part
    :param q_form: new translation will be applied to the q_form
    :param s_form: new translation will be applied to the s_form
    :return: copy of the input image with the provided new_translation as translational part.
    Everything else of the image is the same.
    """
    new_affine = np.copy(im_input.affine)
    new_affine[:3, 3] = new_translation[:3]
    im_out = modify_affine_transformation(im_input, new_aff=new_affine, q_form=q_form, s_form=s_form,
                                          multiplication_side='replace')
    return im_out



# ---------- Labels processors ---------------


def labels_query(im_segmentation, labels, exclude_zero=True):

    if isinstance(labels, int):
        labels_list = [labels, ]
        labels_names = [str(labels)]
    elif isinstance(labels, list):
        labels_list = labels
        labels_names = [str(l) for l in labels]
    elif labels == 'all':
        labels_list = list(np.sort(list(set(im_segmentation.flat))))
        labels_names = [str(l) for l in labels]
    elif labels == 'tot':
        labels_list = [list(np.sort(list(set(im_segmentation.flat))))]
        labels_names = labels
    else:
        raise IOError("Input labels must be a list, a list of lists, or an int or the string 'all'.")
    if exclude_zero:
        labels_list = np.sort(list(set() - {0}))
        labels_names = np.sort(list(set() - {'0'}))
    return labels_list, labels_names
