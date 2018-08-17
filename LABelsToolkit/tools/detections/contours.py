import numpy as np

from LABelsToolkit.tools.aux_methods.utils_nib import set_new_data


def contour_from_array_at_label_l(im_arr, l, thr=0.3, omit_axis=None, verbose=0):
    """
    Get the contour of a single label
    :param im_arr: input array with segmentation
    :param l: considered label
    :param thr: threshold (default 0.3) increase to increase the contour thickness.
    :param omit_axis: a directional axis preference for the contour creation, to avoid "walls" when scrolling
    the 3d image in a particular direction. None if no preference axis is expected.
    :param verbose:
    :return:
    """
    if verbose > 0:
        print('eroding label {}'.format(l))
    array_label_l = im_arr == l
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
        output_arr += contour_from_array_at_label_l(im_segm.get_data(), la, omit_axis=omit_axis, verbose=verbose)

    return set_new_data(im_segm, output_arr.astype(np.bool) * im_segm.get_data(), new_dtype=im_segm.get_data_dtype())
