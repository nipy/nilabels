import numpy as np

from labels_manager.tools.manipulations.relabeller import keep_only_one_label
from labels_manager.tools.aux_methods.utils import binarise_a_matrix


def box_sides(in_segmentation, label_to_box=1, affine=np.eye(4), dtype_output=np.float64):
    """
    We assume the component with label equals to label_to_box is connected
    :return:
    """
    one_label_data = keep_only_one_label(in_segmentation, label_to_keep=label_to_box)
    ans = []
    for d in range(len(one_label_data.shape)):
        ans.append(np.sum(binarise_a_matrix(np.sum(one_label_data, axis=d), dtype=np.int)))
    return ans
