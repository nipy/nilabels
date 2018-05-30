import numpy as np
from scipy import ndimage

from LABelsToolkit.tools.detections.island_detection import island_for_label

# Dirty
a = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 1, 0, 0, 0, 0, 2, 2, 1, 2, 2, 0],
              [0, 1, 0, 2, 1, 1, 2, 2, 1, 2, 2, 0],
              [0, 0, 0, 1, 1, 1, 0, 2, 2, 2, 2, 0],
              [0, 1, 0, 1, 2, 1, 0, 0, 2, 2, 2, 0],
              [0, 0, 0, 1, 1, 1, 1, 0, 2, 2, 2, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 1, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

# Cleaned expected
b = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 0],
              [0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 2, 0],
              [0, 0, 0, 1, 1, 1, 0, 2, 2, 2, 2, 0],
              [0, 0, 0, 1, 1, 1, 0, 0, 2, 2, 2, 0],
              [0, 0, 0, 1, 1, 1, 1, 0, 2, 2, 2, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])


c = np.array([[0,   0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
              [0,   0,  0,  0,  0,  0, -1, -1,  2,  2,  2,  0],
              [0,   0,  0,  1,  1,  1, -1, -1,  2,  2,  2,  0],
              [0,   0,  0,  1, -1,  1, -1,  2,  2,  2,  2,  0],
              [0,   0,  0,  1,  1,  1,  0,  0,  2,  2,  2,  0],
              [0,   0,  0,  1,  1,  1,  1,  0,  2, -1,  2,  0],
              [-1, -1,  0,  0,  0,  0,  0,  0,  0, -1,  0,  0],
              [-1, -1, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0],
              [ 0, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0]])


def multi_lab_segmentation_dilate_1_above_selected_label(arr_segm, selected_label=-1, labels_to_dilate=None):
    """
    The orders of labels to dilate counts.
    :param arr_segm:
    :param selected_label:
    :param labels_to_dilate: if None all labels are dilated, in ascending order (algorithm is NOT order invariant).
    :return:
    """
    answer = np.copy(arr_segm)
    selected_labels_mask = arr_segm[arr_segm == selected_label]
    if labels_to_dilate is None:
        labels_to_dilate = sorted(list(set(arr_segm.flat)))

    for l in labels_to_dilate:
        bin_label_l = np.zeros_like(arr_segm, dtype=np.bool)
        bin_label_l[arr_segm == l] = 1
        dilated_bin_label_l = ndimage.morphology.binary_dilation(bin_label_l)
        dilation_l_over_selected_label = dilated_bin_label_l * selected_labels_mask
        answer[dilation_l_over_selected_label > 0] = l

    return answer


def holes_filler(arr_segm_with_holes, holes_label=-1):
    """
    Given a segmentation with holes (holes are specified by a special labels called holes_label)
    the holes are filled with the closest labels around
    :param arr_segm_with_holes:
    :param holes_label:
    :return:
    """
    arr_segm_no_holes = np.copy(arr_segm_with_holes)
    while holes_label in arr_segm_no_holes:
        arr_segm_no_holes = multi_lab_segmentation_dilate_1_above_selected_label(arr_segm_no_holes,
                                                                                 selected_label=holes_label)
    return arr_segm_no_holes


def clean_semgentation(arr_segm):
    """
    Given an array representing a binary segmentation, the connected components of the segmentations
    :param arr_segm:
    :return:
    """
    labels_segm = sorted(list(set(arr_segm.flat)))
    segm_no_holes_lab = np.copy(arr_segm)
    for lab in labels_segm:
        islands = island_for_label(segm_no_holes_lab, lab, emphasis_max=True)
        segm_with_holes = (1 - islands.astype(np.bool)) * segm_no_holes_lab + islands
        segm_no_holes_lab = holes_filler(segm_with_holes)
    return segm_no_holes_lab







