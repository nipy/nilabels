import numpy as np

from labels_manager.tools.aux_methods.utils import set_new_data


def merge_labels_from_4d(in_data, keep_original_values=True):
    """
    Can be the inverse function of split label with default parameters.
    The labels are assuming to have no overlaps.
    :param in_data: 4d volume
    :param keep_original_values: merge the labels with their values, otherwise it uses the values of the slice numbering (including zero label!).
    :return:
    """
    msg = 'Input array must be 4-dimensional.'
    assert in_data.ndim == 4, msg

    in_data_shape = in_data.shape
    out_data = np.zeros(in_data_shape[:3], dtype=in_data.dtype)

    for t in xrange(in_data.shape[3]):
        slice_t = in_data[...,t]
        if keep_original_values:
            out_data = out_data + slice_t
        else:
            out_data = out_data + ((t + 1) * slice_t.astype(np.bool)).astype(in_data.dtype)
    return out_data


def stack_images(list_images):
    """
    From a list of image of the same shape, the stack of these images in the new dimension.
    :param list_images:
    :return: stack image of the input list
    """
    msg = 'input images shapes are not all of the same dimension'
    assert False not in [list_images[0].shape == im.shape for im in list_images[1:]], msg
    new_data = np.stack([nib_image.get_data() for nib_image in list_images] , axis=len(list_images[0].shape))
    stack_im = set_new_data(list_images[0], new_data)
    return stack_im
