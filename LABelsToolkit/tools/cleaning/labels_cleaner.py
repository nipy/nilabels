import numpy as np
from scipy import ndimage

from LABelsToolkit.tools.detections.island_detection import island_for_label


def multi_lab_segmentation_dilate_1_above_selected_label(arr_segm, selected_label=-1, labels_to_dilate=()):
    """
    The orders of labels to dilate counts.
    :param arr_segm:
    :param selected_label:
    :param labels_to_dilate: if None all labels are dilated, in ascending order (algorithm is NOT order invariant).
    :return:
    """
    answer = np.copy(arr_segm)
    if labels_to_dilate is ():
        labels_to_dilate = sorted(list(set(arr_segm.flat) - {selected_label}))

    for l in labels_to_dilate:
        selected_labels_mask = np.zeros_like(answer, dtype=np.bool)
        selected_labels_mask[answer == selected_label] = 1
        bin_label_l = np.zeros_like(answer, dtype=np.bool)
        bin_label_l[answer == l] = 1
        dilated_bin_label_l = ndimage.morphology.binary_dilation(bin_label_l)
        dilation_l_over_selected_label = dilated_bin_label_l * selected_labels_mask
        answer[dilation_l_over_selected_label > 0] = l

    return answer


def holes_filler(arr_segm_with_holes, holes_label=-1, labels_sequence=()):
    """
    Given a segmentation with holes (holes are specified by a special labels called holes_label)
    the holes are filled with the closest labels around.
    It applies multi_lab_segmentation_dilate_1_above_selected_label until all the holes
    are filled.
    :param arr_segm_with_holes:
    :param holes_label:
    :param labels_sequence: As multi_lab_segmentation_dilate_1_above_selected_label is not invariant
    for the selected sequence, this iterative version is not invariant too.
    :return:
    """
    arr_segm_no_holes = np.copy(arr_segm_with_holes)
    while holes_label in arr_segm_no_holes:
        arr_segm_no_holes = multi_lab_segmentation_dilate_1_above_selected_label(arr_segm_no_holes,
                                selected_label=holes_label, labels_to_dilate=labels_sequence)
    return arr_segm_no_holes


def clean_semgentation(arr_segm, labels_to_clean=(), verbose=1):
    """
    Given an array representing a binary segmentation, the connected components of the segmentations.
    If an hole could be filled by 2 different labels, wins the label with lower value.
    This should be improved with a probabilistic framework [future work].
    Only the largest connected component of each label will remain in the final
    segmentation. The smaller components will be filled by the surrounding labels.
    :param arr_segm: an array of a segmentation.
    :param labels_to_clean: select the labels you want to consider for the cleaning.
    :param verbose:
    :return:
    """
    if labels_to_clean == ():
        labels_to_clean = sorted(list(set(arr_segm.flat)))
    segm_with_holes   = np.copy(arr_segm)
    for lab in labels_to_clean:
        if verbose:
            print('Cleaning label {}'.format(lab))
        islands = island_for_label(arr_segm, lab, emphasis_max=True)
        segm_with_holes[islands == -1] = -1

    return holes_filler(segm_with_holes)
