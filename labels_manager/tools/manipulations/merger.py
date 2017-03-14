import os
import copy
import numpy as np
import nibabel as nib

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
    assert len(in_data.shape) == 4, msg

    in_data_shape = in_data.shape
    out_data = np.zeros(in_data_shape[:3], dtype=in_data.dtype)

    for t in xrange(in_data.shape[3]):
        slice_t = in_data[...,t]
        if keep_original_values:
            out_data = out_data + slice_t
        else:
            out_data = (t * slice_t.astype(np.bool)).astype(in_data.dtype)
    return out_data


def merge_labels_from_4d_path(input_im_path, output_im_path):

    # TODO erase after testing facade

    if not os.path.isfile(input_im_path):
        raise IOError('input image file does not exist.')
    if not os.path.isfile(output_im_path):
        raise IOError('input image file does not exist.')

    im_labels = nib.load(input_im_path)
    data_labels = im_labels.get_data()
    data_relabelled = merge_labels_from_4d(data_labels)

    im_relabelled = set_new_data(im_labels, data_relabelled)
    nib.save(im_relabelled, output_im_path)
