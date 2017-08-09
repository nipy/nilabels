import numpy as np
import nibabel as nib
import os
import subprocess


def set_new_data(image, new_data, new_dtype=None, remove_nan=True):
    """
    From a nibabel image and a numpy array it creates a new image with
    the same header of the image and the new_data as its data.
    :param image: nibabel image
    :apram new_data:
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
    :param toll: tolerance to the dissimilarity in the data - if headers are different images are different.
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


# def eliminates_consecutive_duplicates(input_list):
#     """
#     :param input_list: a list
#     :return: the same list with no consecutive duplicates.
#     """
#     output_list = [input_list[0],]
#     for i in xrange(1, len(input_list)):
#         if not input_list[i] == input_list[i-1]:
#             output_list.append(input_list[i])
#
#     return output_list


def binarise_a_matrix(in_matrix, dtype=np.bool):
    """
    All the values above zeros will be ones.
    :param in_matrix: any matrix
    :param dtype: the output matrix is forced to this data type (bool by default).
    :return: The same matrix, where all the non-zero elements are equals to 1.
    """
    out_matrix = np.zeros_like(in_matrix)
    non_zero_places = in_matrix != 0
    np.place(out_matrix, non_zero_places, 1)
    return out_matrix.astype(dtype)


# def get_values_below_label(image, segmentation, label):
#     """
#     Given an image (matrix) and a segmentation (another matrix), provides a list
#     :param image: np.array of an image
#     :param segmentation: np.array of the segmentation of the same image
#     :param label: a label in the segmentation
#     :return: np.array with all the values below the label.
#     """
#     np.testing.assert_array_equal(image.shape, segmentation.shape)
#     below_label_places = segmentation == label
#     coord = np.nonzero(below_label_places.flatten())[0]
#     return np.take(image.flatten(), coord)


def is_valid_permutation(in_perm):
    """
    A permutation is a list of 2 lists of same size:
    a = [[1,2,3], [2,3,1]]
    means permute 1 with 2, 2 with 3, 3 with 1.
    """

    if not len(in_perm) == 2:
        return False
    if not len(in_perm[0]) == len(in_perm[1]):
        return False
    if not all(isinstance(n, int) for n in in_perm[0]):
        return False
    if not all(isinstance(n, int) for n in in_perm[1]):
        return False
    if not set(in_perm[0]) == set(in_perm[1]):
        return False
    return True


def scan_and_remove_path(msg):
    """
    Take a string with a series of paths separated by a space and keeps only the base-names of each path.
    """
    a = [os.path.basename(p) for p in msg.split(' ')]
    return ' '.join(a)


def print_and_run(cmd, msg=None, safety_on=False, short_path_output=True):
    """
    run the command to console and print the message.
    if msg is None print the command itself.
    :param cmd: command for the terminal
    :param msg: message to show before running the command
    on the top of the command itself.
    :param short_path_output: the message provided at the prompt has only the filenames without the paths.
    :param safety_on: safety, in case you want to see the messages at a first run.
    :return:
    """

    # if len(cmd) > 249:
    #     print(cmd)
    #     raise IOError('input command is too long, this may create problems. Please use shortest names!')
    if short_path_output:
        path_free_cmd = scan_and_remove_path(cmd)
    else:
        path_free_cmd = cmd

    if msg is not None:
        print('\n' + msg + '\n')
    else:
        print('\n-> ' + path_free_cmd + '\n')

    if not safety_on:
        # os.system(cmd)
        subprocess.call(cmd, shell=True)


def adjust_affine_header(pfi_input, pfi_output, theta, trasl=np.array([0, 0, 0])):

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


def adjust_nifti_translation_path(pfi_nifti_input, new_traslation, pfi_nifti_output, q_form=True, s_form=True,
                                  verbose=1):
    """
    Change q_form or s_form or both translational part.
    :param pfi_nifti_input: path to file of the input image
    :param new_traslation: 3dim array, affine coordinates, will be the future translational part of the affine.
    :param pfi_nifti_output: path to file of the image with the modifed translation. Try not to be destructive, unless you do not really want.
    :param q_form: [True] affect q_form
    :param s_form: [True] affect s_form
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


def adjust_nifti_image_type_path(pfi_nifti_input, new_dtype, pfi_nifti_output, update_description=None, verbose=1):
    im_input = nib.load(pfi_nifti_input)
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
    nib.save(new_im, pfi_nifti_output)


def reproduce_slice_fourth_dimension(nib_image, num_slices=10, repetition_axis=3):

    im_sh = nib_image.shape
    if not (len(im_sh) == 2 or len(im_sh) == 3):
        raise IOError('Methods can be used only for 2 or 3 dim images. No conflicts with existing multi, slices')

    new_data = np.stack([nib_image.get_data(), ] * num_slices, axis=repetition_axis)
    output_im = set_new_data(nib_image, new_data)

    return output_im


def reproduce_slice_fourth_dimension_path(pfi_input_image, pfi_output_image, num_slices=10, repetition_axis=3):
    old_im = nib.load(pfi_input_image)
    new_im = reproduce_slice_fourth_dimension(old_im, num_slices=num_slices, repetition_axis=repetition_axis)
    nib.save(new_im, pfi_output_image)
    print 'New image created and saved in {0}'.format(pfi_output_image)



def apply_a_mask_path(pfi_input, pfi_mask, pfi_output):
    """
    Adaptative - if the mask is 3D and the image is 4D, will create a temporary mask,
    generate the stack of masks, and apply the stacks to the image.
    :param pfi_input: path to file 3d x T image
    :param pfi_mask: 3d mask same dimension as the 3d of the pfi_input
    :param pfi_output: apply the mask to each time point T in the fourth dimension if any.
    :return: None, save the output in pfi_output.
    """
    im_input = nib.load(pfi_input)
    im_mask = nib.load(pfi_mask)

    assert len(im_mask.shape) == 3

    if not im_mask.shape == im_input.shape[:3]:
        msg = 'Mask {0} and image {1} does not have compatible dimension: {2} and {3}'.format(
            pfi_input, pfi_mask, im_input, im_mask.shape)
        raise IOError(msg)

    if len(im_input.shape) == 3:
        new_data = im_input.get_data() * im_mask.get_data().astype(np.bool)
    else:
        new_data = np.zeros_like(im_input.get_data())
        for t in range(im_input.shape[3]):
            new_data[..., t] = im_input.get_data()[..., t] * im_mask.get_data().astype(np.bool)

    new_im = set_new_data(image=im_input, new_data=new_data)

    nib.save(new_im, filename=pfi_output)


# ---------- Distributions ---------------

def triangular_density_function(x, a, mu, b):

    if a <= x < mu:
        return 2 * (x - a) / float((b - a) * (mu - a))
    elif x == mu:
        return 2 / float(b - a)
    elif mu < x <= b:
        return 2 * (b - x) / float((b - a) * (b - mu))
    else:
        return 0
