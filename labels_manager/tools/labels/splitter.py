import os
import numpy as np
import nibabel as nib

from labels_manager.tools.aux_methods.utils import set_new_data


def split_labels_to_4d(in_data, relabel_in_consecutive_numbers=True):
    """
    Split labels of a 3d segmentation in a 4d segmentation,
    one label for each slice, reordered in ascending order.
    Masks are relabelled in ascending order, if remove gaps is True.
    or kept the same if remove_gaps is False. In the second case,
    the number of the 4d does not correspond to the label index.
    remove_gaps=False makes split_labels biiective with the function
    merge_labels.
    :param in_data: labels (only positive labels allowed).
    :param relabel_in_consecutive_numbers:
    :return:
    """
    in_data_shape = in_data.shape

    msg = 'Input array must be 3-dimensional.'
    assert len(in_data.shape) == 3, msg

    list_labels = list(set(in_data.flat))
    list_labels.sort()
    max_label = max(list_labels)

    out_data = np.zeros(list(in_data_shape) + [max_label], dtype=in_data.dtype)

    for i in xrange(in_data_shape[0]):
        for j in xrange(in_data_shape[1]):
            for k in xrange(in_data_shape[2]):
                l = in_data[i, j, k]
                if not l == 0:
                    out_data[i, j, k, int(l) - 1] = 1

    if relabel_in_consecutive_numbers:
        # remove the empty slices:
        complementary_labels = list(set(list_labels) - set(range(1, max_label + 1)))
        complementary_labels.sort()
        out_data = np.delete(out_data, complementary_labels, axis=3)

    return out_data


def split_labels_path(input_im_path, output_im_path, remove_gaps=True):

    # TODO remove after testing manager

    # check parameters
    if not os.path.isfile(input_im_path):
        raise IOError('input image file does not exist.')
    if not os.path.isfile(output_im_path):
        raise IOError('input image file does not exist.')

    im_labels = nib.load(input_im_path)
    data_labels = im_labels.get_data()
    data_relabelled = split_labels(data_labels, remove_gaps=remove_gaps)

    im_relabelled = set_new_data(im_labels, data_relabelled)
    nib.save(im_relabelled, output_im_path)
