import copy
import numpy as np

from labels_manager.tools.aux_methods.permutations import is_valid_permutation


def relabeller(in_data, list_old_labels, list_new_labels):
    """
    :param in_data: array corresponding to an image segmentation.
    :param list_old_labels: list or tuple of labels
    :param list_new_labels: list or tuple of labels of the same len as list_old_labels
    :return: array where all the labels in list_new_labels are substituted with list_new_label in the same order.
    """

    if isinstance(list_new_labels, int):
        list_new_labels = [list_new_labels, ]
    if isinstance(list_old_labels, int):
        list_old_labels = [list_old_labels, ]

    # sanity check: old and new have the same number of elements
    if not len(list_old_labels) == len(list_new_labels):
        raise IOError('Labels list does not have the same length.')

    new_data = copy.deepcopy(in_data)

    for k in range(len(list_new_labels)):
        places = in_data == list_old_labels[k]
        if np.any(places):
            np.place(new_data, places, list_new_labels[k])
            print('Label {0} substituted with label {1}'.format(list_old_labels[k], list_new_labels[k]))
        else:
            print('Label {0} not present in the array'.format(list_old_labels[k]))

    return new_data


def permute_labels(in_data, permutation):
    """
    Permute the values of the labels in an int image.
    :param in_data: array corresponding to an image segmentation.
    :param permutation:
    :return:
    """
    assert is_valid_permutation(permutation), 'Input permutation not valid.'
    return relabeller(in_data, permutation[0], permutation[1])


def erase_labels(in_data, labels_to_erase):
    """
    :param in_data: array corresponding to an image segmentation.
    :param labels_to_erase: list or tuple of labels
    :return: all the labels in the list labels_to_erase will be assigned to zero.
    """
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
        print 'labels_to_keep {} in not delineated in the image'
        return

    labels_not_to_keep = list(set(list_labels) - {label_to_keep})
    return relabeller(in_data, list_old_labels=labels_not_to_keep, list_new_labels=[0,]*len(labels_not_to_keep))
