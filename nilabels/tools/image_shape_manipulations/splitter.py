import numpy as np


def split_labels_to_4d(in_data, list_labels=(), keep_original_values=True):
    """
    Split labels of a 3d segmentation in a 4d segmentation,
    one label for each slice in ascending order.
    Labels can be relabelled in consecutive order or can keep the
    original labels value.
    :param in_data: segmentation (only positive labels allowed).
    :param list_labels: list of labels to split.
    :param keep_original_values: boolean otherwise keep the same
    value for all the
    :return:
    """
    msg = 'Input array must be 3-dimensional.'
    assert in_data.ndim == 3, msg

    out_data = np.zeros(list(in_data.shape) + [len(list_labels)], dtype=in_data.dtype)

    for l_index, l in enumerate(list_labels):
        places_l = in_data == l
        if keep_original_values:
            out_data[..., l_index] = l * places_l  # .astype(in_data.dtype)
        else:
            out_data[..., l_index] = places_l  # .astype(in_data.dtype)

    return out_data
