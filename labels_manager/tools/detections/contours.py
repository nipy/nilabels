import numpy as np
from ..aux_methods.utils_nib import set_new_data


def contour_from_segmentation(im_segm, omit_axis=None, verbose=0):
    """
    From an input nibabel image segmentation, returns the contour of each segmented region with the original
    label.
    :param im_segm:
    :return:
    """
    def get_contour_single_label_from_array(im_arr, l, thr=0.3):
        if verbose > 0:
            print(l)
        array_label_l = im_arr == l
        assert isinstance(array_label_l, np.ndarray)
        gra = np.gradient(array_label_l.astype(np.bool).astype(np.float64))
        if omit_axis is None:
            norm_gra = np.sqrt(gra[0] ** 2 + gra[1] ** 2 + gra[2] ** 2) > thr
        elif omit_axis == 'x':
            norm_gra = np.sqrt(gra[1] ** 2 + gra[2] ** 2) > thr
        elif omit_axis == 'y':
            norm_gra = np.sqrt(gra[0] ** 2 + gra[2] ** 2) > thr
        elif omit_axis == 'z':
            norm_gra = np.sqrt(gra[0] ** 2 + gra[1] ** 2) > thr
        else:
            raise IOError
        return norm_gra

    list_labels = sorted(list(set(im_segm.get_data().flat)))[1:]
    output_arr = np.zeros_like(im_segm.get_data(), dtype=im_segm.get_data_dtype())

    for la in list_labels:
        output_arr += get_contour_single_label_from_array(im_segm.get_data(), la)

    return set_new_data(im_segm, output_arr.astype(np.bool) * im_segm.get_data(), new_dtype=im_segm.get_data_dtype())