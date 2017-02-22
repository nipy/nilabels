import os
import copy
import numpy as np
import nibabel as nib

from labels_manager.tools.aux_methods.permutations import is_valid_permutation
from labels_manager.tools.aux_methods.utils import set_new_data


def relabeller(in_data, list_old_labels, list_new_labels):

    new_data = copy.deepcopy(in_data)

    # sanity check: old and new have the same number of elements
    if not len(list_old_labels) == len(list_new_labels):
        raise IOError('Labels list does not have the same length.')

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
    :param in_data:
    :param permutation:
    :return:
    """
    assert is_valid_permutation(permutation), 'Input permutation not valid.'
    return relabeller(in_data, permutation[0], permutation[1])


def erase_labels(in_data, labels_to_erase):
    return relabeller(in_data, list_old_labels=labels_to_erase,
                      list_new_labels=[0, ] * len(labels_to_erase))


def assign_all_other_labels_the_same_value(in_data, labels_to_keep, same_value_label=255):
    """
    All the labels that are not in the list labels_to_keep will be given the value same_value_label
    :param in_data:
    :param labels_to_keep:
    :param same_value_label:
    :return:
    """

    list_labels = np.sort(list(set(in_data.flat)))
    list_labels.sort()

    labels_that_will_have_the_same_value = list(set(list_labels) - set(labels_to_keep) - {0})

    places = np.zeros_like(in_data).astype(bool)
    new_data = copy.deepcopy(in_data)

    for k in labels_that_will_have_the_same_value:
        places += new_data == k

    np.place(new_data, places, same_value_label)

    return new_data
