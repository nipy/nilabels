import numpy as np
from scipy import ndimage

from LABelsToolkit.tools.image_colors_manipulations.relabeller import relabeller


def island_for_label(array_segm, label, emphasis_max=False, special_label=-1):
    """
    As ndimage.label, with output ordered by the size of the connected component.
    :param array_segm:
    :param label:
    :param emphasis_max: if True, the max component will be one and all the others will be -1.
    :return:
    """
    if label not in array_segm:
        print('Label {} not in the provided array.'.format(label))
        return array_segm

    binary_segm_comp, num_comp = ndimage.label(array_segm == label)

    if num_comp == 1:
        return binary_segm_comp

    voxels_per_components = np.array([np.count_nonzero(binary_segm_comp == l + 1) for l in range(num_comp)])
    scores = voxels_per_components.argsort()[::-1] + 1
    new_labels = np.arange(1, num_comp + 1)

    # if label == 201:  # temporary workaround
    #     new_labels[1] = 1

    binary_segm_components_sorted = relabeller(binary_segm_comp, scores, new_labels, verbose=1)

    if emphasis_max:
        binary_segm_components_sorted[binary_segm_components_sorted > 1] = special_label

    return binary_segm_components_sorted
