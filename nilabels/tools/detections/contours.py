import numpy as np
from scipy import ndimage as nd

from nilabels.tools.aux_methods.utils_nib import set_new_data


def contour_from_array_at_label(im_arr, lab, thr=0.3, omit_axis=None, verbose=0):
    """
    Get the contour of a single label
    :param im_arr: input array with segmentation
    :param lab: considered label
    :param thr: threshold (default 0.3) increase to increase the contour thickness.
    :param omit_axis: a directional axis preference for the contour creation, to avoid "walls" when scrolling
    the 3d image in a particular direction. None if no preference axis is expected.
    :param verbose:
    :return: boolean mask with the array labels.
    """
    if verbose > 0:
        print('Getting contour for label {}'.format(lab))
    array_label_l = im_arr == lab
    assert isinstance(array_label_l, np.ndarray)
    gra = np.gradient(array_label_l.astype(np.bool).astype(np.float64))
    if omit_axis is None:
        thresholded_gra = np.sqrt(gra[0] ** 2 + gra[1] ** 2 + gra[2] ** 2) > thr
    elif omit_axis == 'x':
        thresholded_gra = np.sqrt(gra[1] ** 2 + gra[2] ** 2) > thr
    elif omit_axis == 'y':
        thresholded_gra = np.sqrt(gra[0] ** 2 + gra[2] ** 2) > thr
    elif omit_axis == 'z':
        thresholded_gra = np.sqrt(gra[0] ** 2 + gra[1] ** 2) > thr
    else:
        raise IOError
    return thresholded_gra


def contour_from_segmentation(im_segm, omit_axis=None, verbose=0):
    """
    From an input nibabel image segmentation, returns the contour of each segmented region with the original
    label.
    :param im_segm:
    :param omit_axis: a directional axis preference for the contour creation, to avoid "walls" when scrolling
    the 3d image in a particular direction. None if no preference axis is expected.
    :param verbose: 0 no, 1 yes.
    :return: return the contour of the provided segmentation
    """
    list_labels = sorted(list(set(im_segm.get_data().flat)))[1:]
    output_arr = np.zeros_like(im_segm.get_data(), dtype=im_segm.get_data_dtype())

    for la in list_labels:
        output_arr += contour_from_array_at_label(im_segm.get_data(), la, omit_axis=omit_axis, verbose=verbose)

    return set_new_data(im_segm, output_arr.astype(np.bool) * im_segm.get_data(), new_dtype=im_segm.get_data_dtype())


def get_xyz_borders_of_a_label(segm_arr, label):
    """
    :param segm_arr: array representing a segmentation
    :param label: a single integer label
    :return: box coordinates containing the given label in the segmentation, None if the label is not present.
    """
    assert segm_arr.ndim == 3

    if label not in segm_arr:
        return None

    X, Y, Z = np.where(segm_arr == label)
    return [np.min(X), np.max(X), np.min(Y), np.max(Y), np.min(Z), np.max(Z)]


def get_internal_contour_with_erosion_at_label(segm_arr, lab, thickness=1):
    """
    Get the internal contour for a given thickness.
    :param segm_arr: input segmentation where to extract the contour
    :param lab: label to extract the contour
    :param thickness: final thickness of the segmentation
    :return: image with only the contour of the given input image.
    """
    im_lab = segm_arr == lab
    return (im_lab ^ nd.morphology.binary_erosion(im_lab, iterations=thickness).astype(np.bool)).astype(segm_arr.dtype) * lab

