import numpy as np
import nibabel as nib
import collections

from labels_manager.tools.descriptions.colours_rgb_lab import get_random_rgb
from labels_manager.tools.descriptions.manipulate_descriptors import descriptor_standard_header



def set_new_data(image, new_data, new_dtype=None, remove_nan=False):
    """
    From a nibabel image and a numpy array it creates a new image with
    the same header of the image and the new_data as its data.
    :param image: nibabel image
    :param new_data: numpy array
    :param new_dtype: preferred output data type. Default is the type of the new_data
    :param remove_nan: remove nan in case there is some in the new_data.
    :return: nibabel image
    ----
    Example:
    To grab the third time-point of a nibabel image:
    > input_4d_im = nib.load('from_somewhere.nii.gz')
    > data_3d_third_slice = set_new_data(input_4d_im, input_4d_im.get_data()[..., 2])
    """
    if remove_nan:
        new_data = np.nan_to_num(new_data)

    # if nifty1
    if image.header['sizeof_hdr'] == 348:
        new_image = nib.Nifti1Image(new_data, image.affine, header=image.header)
    # if nifty2
    elif image.header['sizeof_hdr'] == 540:
        new_image = nib.Nifti2Image(new_data, image.affine, header=image.header)
    else:
        raise IOError('input_image_problem')

    # update data type:
    if new_dtype is None:
        new_image.set_data_dtype(new_data.dtype)
    else:
        new_image.set_data_dtype(new_dtype)

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


def eliminates_consecutive_duplicates(input_list):
    """
    :param input_list: a list
    :return: the same list with no consecutive duplicates.
    """
    output_list = [input_list[0],]
    for i in xrange(1, len(input_list)):
        if not input_list[i] == input_list[i-1]:
            output_list.append(input_list[i])

    return output_list


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


def get_values_below_label(image, segmentation, label):
    """
    Given an image (matrix) and a segmentation (another matrix), provides a list
    :param image: np.array of an image
    :param segmentation: np.array of the segmentation of the same image
    :param label: a label in the segmentation
    :return: np.array with all the values below the label.
    """
    np.testing.assert_array_equal(image.shape, segmentation.shape)
    below_label_places = segmentation == label
    coord = np.nonzero(below_label_places.flatten())[0]
    return np.take(image.flatten(), coord)


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


# ---------- O C generators experiments ---------------

def generate_o(omega=(250, 250), radius=50,
               background_intensity=0, foreground_intensity=20, dtype=np.uint8):

    m = background_intensity * np.ones(omega, dtype=dtype)

    if len(omega) == 2:
        c = [omega[j] / 2 for j in range(len(omega))]
        for x in xrange(omega[0]):
            for y in xrange(omega[1]):
                if (x - c[0])**2 + (y - c[1])**2 < radius**2:
                    m[x, y] = foreground_intensity
    elif len(omega) == 3:
        c = [omega[j] / 2 for j in range(len(omega))]
        for x in xrange(omega[0]):
            for y in xrange(omega[1]):
                for z in xrange(omega[2]):
                    if (x - c[0])**2 + (y - c[1])**2 + (z - c[2])**2 < radius**2:
                        m[x, y, z] = foreground_intensity
    return m


def generate_c(omega=(250, 250), internal_radius=40, external_radius=60, opening_height=50,
               background_intensity=0, foreground_intensity=20, dtype=np.uint8, margin=None):

    def get_a_2d_c(omega, internal_radius, external_radius, opening_height, background_intensity,
                   foreground_intensity, dtype):

        m = background_intensity * np.ones(omega[:2], dtype=dtype)

        c = [omega[j] / 2 for j in range(len(omega))]
        # create the crown
        for x in xrange(omega[0]):
            for y in xrange(omega[1]):
                if internal_radius**2 < (x - c[0])**2 + (y - c[1])**2 < external_radius**2:
                    m[x, y] = foreground_intensity

        # open the c
        low_lim = int(omega[0] / 2) - int(opening_height / 2)
        high_lim = int(omega[0] / 2) + int(opening_height / 2)

        for x in xrange(omega[0]):
            for y in xrange(int(omega[1] / 2), omega[1]):
                if low_lim < x < high_lim and m[x, y] == foreground_intensity:
                    m[x, y] = background_intensity

        return m

    c_2d = get_a_2d_c(omega=omega[:2], internal_radius=internal_radius, external_radius=external_radius,
                      opening_height=opening_height, background_intensity=background_intensity,
                      foreground_intensity=foreground_intensity, dtype=dtype)

    if len(omega) == 2:
        return c_2d

    elif len(omega) == 3:
        if margin is None:
            return np.repeat(c_2d, omega[2]).reshape(omega)
        else:
            res = np.zeros(omega, dtype=dtype)
            for z in xrange(margin, omega[2] - 2 * margin):
                res[..., z] = c_2d
            return res


# ---------- Ellipsoids experiments ---------------

def generate_ellipsoid(omega, focus_1, focus_2, distance, background_intensity=0, foreground_intensity=100,
                       dtype=np.uint8):
    sky = background_intensity * np.ones(omega, dtype=dtype)
    for xi in range(omega[0]):
        for yi in range(omega[1]):
            for zi in range(omega[2]):
                if np.sqrt( (focus_1[0] - xi) ** 2 + (focus_1[1] - yi) ** 2 + (focus_1[2] - zi) ** 2 ) + \
                    np.sqrt( (focus_2[0] - xi) ** 2 + (focus_2[1] - yi) ** 2 + (focus_2[2] - zi) ** 2 ) \
                        <= distance:
                    sky[xi, yi, zi] = foreground_intensity
    return sky

# ---------- Cubes experiments ---------------


def generate_cube(omega, center, side_length, background_intensity=0, foreground_intensity=100, dtype=np.uint8):
    sky = background_intensity * np.ones(omega, dtype=dtype)
    half_side_length = int(np.ceil(side_length / 2))

    for lx in range(-half_side_length, half_side_length + 1):
        for ly in range(-half_side_length, half_side_length + 1):
            for lz in range(-half_side_length, half_side_length + 1):
                sky[center[0] + lx, center[1] + ly, center[2] + lz] = foreground_intensity
    return sky


# ---------- Label descriptors experiments ---------------


def generate_dummy_label_descriptor(pfi_output=None, list_labels=range(5), list_roi_names=None):
    """
    For testing purposes, it creates a dummy label descriptor.
    :param pfi_output: where to save the eventual label descriptor
    :param list_labels: list of labels range, default 0:5
    :param list_roi_names: names of the regions of interests. If None, default names are assigned.
    :return: label descriptor as a dictionary.
    """
    d = collections.OrderedDict()
    d.update({'type': 'Label descriptor parsed'})
    num_labels = len(list_labels)
    colors = [get_random_rgb() for _ in range(num_labels)]
    visibility = [(1, 1, 1)] * num_labels
    if list_roi_names is None:
        list_roi_names = ["label {}".format(j) for j in list_labels]
    else:
        assert len(list_labels) == len(list_roi_names)
    for j in range(num_labels):
        up_d = {str(j): [colors[j], visibility[j], list_roi_names[j]]}
        d.update(up_d)
    f = open(pfi_output, 'w+')
    f.write(descriptor_standard_header)
    for j in d.keys():
        if j.isdigit():
            line = '{0: >5}{1: >6}{2: >4}{3: >4}{4: >9}{5: >3}{6: >3}    "{7: >5}"\n'.format(j,
                d[j][0][0], d[j][0][1], d[j][0][2], d[j][1][0], d[j][1][1], d[j][1][2], d[j][2])
            f.write(line)
    f.close()
    return d