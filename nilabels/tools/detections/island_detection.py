import numpy as np
from scipy import ndimage

from nilabels.tools.image_colors_manipulations.relabeller import relabeller


def island_for_label(array_segm, label, m=0, special_label=-1):
    """
    As ndimage.label, with output ordered by the size of the connected component.
    :param array_segm:
    :param label:
    :param m: integer. If m = 0 the n connected components will be numbered from 1 (biggest) to n (smallest).
                             If m > 0, from the m-th largest components, all the components are set to the special
                             special_label.
    :param special_label: value used if m > 0.
    :return: segmentation with the components sorted.
    -----
    e.g.
    array_segm has 4 components of for the input label.
    If m = 0 it returns the components labelled from 1 to 4, where 1 is the biggest.
    if m = 2 the first two largest components are numbered 1 and 2, and the remaining 2 are labelled with special_label.
    """

    if label not in array_segm:
        print('Label {} not in the provided array.'.format(label))
        return array_segm

    binary_segm_comp, num_comp = ndimage.label(array_segm == label)

    if num_comp == 1:
        return binary_segm_comp

    voxels_per_components = np.array([np.count_nonzero(binary_segm_comp == l + 1) for l in range(num_comp)])
    scores = voxels_per_components.argsort()[::-1] + 1
    new_labels = np.arange(num_comp) + 1

    binary_segm_components_sorted = relabeller(binary_segm_comp, scores, new_labels, verbose=1)

    if m > 0:
        binary_segm_components_sorted[binary_segm_components_sorted > m] = special_label

    return binary_segm_components_sorted
