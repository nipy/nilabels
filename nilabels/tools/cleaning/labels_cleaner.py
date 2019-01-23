import numpy as np
from scipy import ndimage

from nilabels.tools.detections.island_detection import island_for_label


def multi_lab_segmentation_dilate_1_above_selected_label(arr_segm, selected_label=-1, labels_to_dilate=(), verbose=2):
    """
    The orders of labels to dilate counts.
    :param arr_segm:
    :param selected_label:
    :param labels_to_dilate: if None all labels are dilated, in ascending order (algorithm is NOT order invariant).
    :param verbose:
    :return:
    """
    answer = np.copy(arr_segm)
    if labels_to_dilate is ():
        labels_to_dilate = sorted(list(set(arr_segm.flat) - {selected_label}))

    num_labels_dilated = 0
    for l in labels_to_dilate:
        if verbose > 1:
            print('Dilating label {} over hole-label {}'.format(l, selected_label))
        selected_labels_mask = np.zeros_like(answer, dtype=np.bool)
        selected_labels_mask[answer == selected_label] = 1
        bin_label_l = np.zeros_like(answer, dtype=np.bool)
        bin_label_l[answer == l] = 1
        dilated_bin_label_l = ndimage.morphology.binary_dilation(bin_label_l)
        dilation_l_over_selected_label = dilated_bin_label_l * selected_labels_mask
        answer[dilation_l_over_selected_label > 0] = l
        num_labels_dilated += 1
    if verbose > 0:
        print('Number of labels_dilated: {}\n'.format(num_labels_dilated))

    return answer


def holes_filler(arr_segm_with_holes, holes_label=-1, labels_sequence=(), verbose=1):
    """
    Given a segmentation with holes (holes are specified by a special labels called holes_label)
    the holes are filled with the closest labels around.
    It applies multi_lab_segmentation_dilate_1_above_selected_label until all the holes
    are filled.
    :param arr_segm_with_holes:
    :param holes_label:
    :param labels_sequence: As multi_lab_segmentation_dilate_1_above_selected_label is not invariant
    for the selected sequence.
    :param verbose:
    :return:
    """
    num_rounds = 0
    arr_segm_no_holes = np.copy(arr_segm_with_holes)

    if verbose:
        print('Filling holes in the segmentation:')

    while holes_label in arr_segm_no_holes:
        arr_segm_no_holes = multi_lab_segmentation_dilate_1_above_selected_label(arr_segm_no_holes,
                                selected_label=holes_label, labels_to_dilate=labels_sequence)
        num_rounds += 1

    if verbose:
        print('Number of dilations required to remove the holes: {}'.format(num_rounds))

    return arr_segm_no_holes


def clean_semgentation(arr_segm, labels_to_clean=(), label_for_holes=-1, verbose=1):
    """
    Given an array representing a binary segmentation, the connected components of the segmentations.
    If an hole could be filled by 2 different labels, wins the label with lower value.
    This should be improved with a probabilistic framework [future work].
    Only the largest connected component of each label will remain in the final
    segmentation. The smaller components will be filled by the surrounding labels.
    :param arr_segm: an array of a segmentation.
    :param labels_to_clean: list of binaries lists. [[z_1, zc_1], ... , [z_J, zc_J]] where z_j is the label you want to
     clean and zc_1 is the number of components you want to keep. If empty tuple, by default cleans all the labels
     keeping only one component.
    :prarm return_filled_holes: if True returns two arrayw, one with the holes filled, and the other with the
    binarised holes that had been filled.
    :param label_for_holes: internal variable for the dummy labels that will be used for the 'holes'. This must
     not be a label already present in the segmentation.
    :param verbose:
    :return:
    """
    if labels_to_clean == ():
        labels_to_clean = sorted(list(set(arr_segm.flat)))
        labels_to_clean = [[z, 1] for z in labels_to_clean]

    segm_with_holes = np.copy(arr_segm)
    for lab, num_components in labels_to_clean:
        if verbose:
            print('Cleaning label {}, keeping {} components'.format(lab, num_components))
        islands = island_for_label(arr_segm, lab, m=num_components, special_label=label_for_holes)
        segm_with_holes[islands == label_for_holes] = label_for_holes

    return holes_filler(segm_with_holes, holes_label=label_for_holes)
