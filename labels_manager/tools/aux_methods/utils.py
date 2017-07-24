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
    if remove_nan:
        new_data = np.nan_to_num(new_data)

    # update data type:
    if new_dtype is not None:
        new_data.astype(new_dtype)

    # if nifty1
    if image.header['sizeof_hdr'] == 348:
        new_image = nib.Nifti1Image(new_data, image.affine, header=image.header)
    # if nifty2
    elif image.header['sizeof_hdr'] == 540:
        new_image = nib.Nifti2Image(new_data, image.affine, header=image.header)
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


def adjust_affine_header(pfi_input, pfi_output, theta, trasl):

    # transformations parameters
    rot_x = np.array([[1,            0,           0,     trasl[0]],
                     [0,  np.cos(theta),  -np.sin(theta),     trasl[1]],
                     [0,  np.sin(theta), np.cos(theta),   trasl[2]],
                     [0,             0,          0,      1]])

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
