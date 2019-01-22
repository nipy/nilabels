import copy
import numpy as np

from nilabels.tools.aux_methods.sanity_checks import is_valid_permutation


def relabeller(in_data, list_old_labels, list_new_labels, verbose=True):
    """
    :param in_data: array corresponding to an image segmentation.
    :param list_old_labels: list or tuple of labels
    :param list_new_labels: list or tuple of labels of the same len as list_old_labels
    :param verbose:
    :return: array where all the labels in list_new_labels are substituted with list_new_label in the same order.
    """

    if isinstance(list_new_labels, int):
        list_new_labels = [list_new_labels, ]
    if isinstance(list_old_labels, int):
        list_old_labels = [list_old_labels, ]

    # sanity check: old and new must have the same number of elements
    if not len(list_old_labels) == len(list_new_labels):
        raise IOError('Labels lists old and new do not have the same length.')

    new_data = copy.deepcopy(in_data)

    for k in range(len(list_new_labels)):
        places = in_data == list_old_labels[k]
        if np.any(places):
            np.place(new_data, places, list_new_labels[k])
            if verbose:
                print('Label {0} substituted with label {1}'.format(list_old_labels[k], list_new_labels[k]))
        else:
            if verbose:
                print('Label {0} not present in the array'.format(list_old_labels[k]))

    return new_data


def permute_labels(in_data, permutation):
    """
    Permute the values of the labels in an int image.
    :param in_data: array corresponding to an image segmentation.
    :param permutation:
    :return:
    """
    if not is_valid_permutation(permutation):
        raise IOError('Input permutation not valid.')
    return relabeller(in_data, permutation[0], permutation[1])


def erase_labels(in_data, labels_to_erase):
    """
    :param in_data: array corresponding to an image segmentation.
    :param labels_to_erase: list or tuple of labels
    :return: all the labels in the list labels_to_erase will be assigned to zero.
    """
    if isinstance(labels_to_erase, int):
        labels_to_erase = [labels_to_erase, ]
    return relabeller(in_data, list_old_labels=labels_to_erase,
                      list_new_labels=[0, ] * len(labels_to_erase))


def assign_all_other_labels_the_same_value(in_data, labels_to_keep, same_value_label=255):
    """
    All the labels that are not in the list labels_to_keep will be given the value same_value_label
    :param in_data: array corresponding to an image segmentation.
    :param labels_to_keep: list or tuple of value in in_data
    :param same_value_label: a single label value.
    :return: segmentation of the same size where all the labels not in the list label_to_keep will be assigned to the
             value same_value_label.
    """
    list_labels = sorted(list(set(in_data.flat)))
    if isinstance(labels_to_keep, int):
        labels_to_keep = [labels_to_keep, ]

    labels_that_will_have_the_same_value = list(set(list_labels) - set(labels_to_keep) - {0})

    return relabeller(in_data, list_old_labels=labels_that_will_have_the_same_value,
                      list_new_labels=[same_value_label, ] * len(labels_that_will_have_the_same_value))


def keep_only_one_label(in_data, label_to_keep):
    """
    From a segmentation keeps only the values in the list labels_to_keep.
    :param in_data: a segmentation (only positive labels allowed).
    :param label_to_keep: the single label that will be kept.
    :return:
    """

    list_labels = sorted(list(set(in_data.flat)))

    if label_to_keep not in list_labels:
        print('the label {} you want to keep is not present in the segmentation'.format(label_to_keep))
        return in_data

    labels_not_to_keep = list(set(list_labels) - {label_to_keep})
    return relabeller(in_data, list_old_labels=labels_not_to_keep, list_new_labels=[0, ]*len(labels_not_to_keep),
                      verbose=False)


def relabel_half_side_one_label(in_data, label_old, label_new, side_to_modify, axis, plane_intercept):
    """
    :param in_data: input data array (must be 3d)
    :param label_old: single label to be replaced
    :param label_new: single label to replace with
    :param side_to_modify: can be the string 'above' or 'below'
    :param axis: can be 'x', 'y', 'z'
    :param plane_intercept: voxel along the selected direction plane where to consider the symmetry.
    :return:
    """
    if not in_data.ndim == 3:
        msg = 'Input array must be 3-dimensional.'
        raise IOError(msg)

    if side_to_modify not in ['below', 'above']:
        msg = 'side_to_copy must be one of the two {}.'.format(['below', 'above'])
        raise IOError(msg)

    if axis not in ['x', 'y', 'z']:
        msg = 'axis variable must be one of the following: {}.'.format(['x', 'y', 'z'])
        raise IOError(msg)

    positions = in_data == label_old
    halfed_positions = np.zeros_like(positions)
    if axis == 'x':
        if side_to_modify == 'above':
            halfed_positions[plane_intercept:, :, :] = positions[plane_intercept:, :, :]
        if side_to_modify == 'below':
            halfed_positions[:plane_intercept, :, :] = positions[:plane_intercept, :, :]
    if axis == 'y':
        if side_to_modify == 'above':
            halfed_positions[:, plane_intercept:, :] = positions[:, plane_intercept:, :]
        if side_to_modify == 'below':
            halfed_positions[:, plane_intercept, :] = positions[:, plane_intercept, :]
    if axis == 'z':
        if side_to_modify == 'above':
            halfed_positions[:, :, plane_intercept:] = positions[ :, :, plane_intercept:]
        if side_to_modify == 'below':
            halfed_positions[:, :, :plane_intercept] = positions[:, :, :plane_intercept]

    new_data = in_data * np.invert(halfed_positions) + label_new * halfed_positions.astype(np.int)
    return new_data
