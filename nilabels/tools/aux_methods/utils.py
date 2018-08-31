import numpy as np
import os
import subprocess
from sympy.combinatorics import Permutation

from nilabels.tools.aux_methods.sanity_checks import is_valid_permutation


# ---- List utils ----


def lift_list(input_list):
    """
    List of nested lists becomes a list with the element exposed in the main list.
    :param input_list: a list of lists.
    :return: eliminates the first nesting levels of lists.
    E.G.
    >> lift_list([1, 2, [1,2,3], [1,2], [4,5, 6], [3,4]])
    [1, 2, 1, 2, 3, 1, 2, 4, 5, 6, 3, 4]
    """
    if input_list == []:
        return []
    else:
        return lift_list(input_list[0]) + (lift_list(input_list[1:]) if len(input_list) > 1 else []) \
            if type(input_list) is list else [input_list]


def eliminates_consecutive_duplicates(input_list):
    """
    :param input_list: a list
    :return: the same list with no consecutive duplicates.
    """
    output_list = [input_list[0], ]
    for i in range(1, len(input_list)):
        if not input_list[i] == input_list[i-1]:
            output_list.append(input_list[i])
    return output_list


# ---- Command executions utils ----


def print_and_run(cmd, msg=None, safety_on=False, short_path_output=True):
    """
    run the command to console and print the message.
    if msg is None print the command itself.
    :param cmd: command for the terminal
    :param msg: message to show before running the command
    on the top of the command itself.
    :param short_path_output: the message provided at the prompt has only the filenames without the paths.
    :param safety_on: safety, in case you want to see the messages at a first run.
    :return:
    """
    def scan_and_remove_path(msg):
        """
        Take a string with a series of paths separated by a space and keeps only the base-names of each path.
        """
        a = [os.path.basename(p) for p in msg.split(' ')]
        return ' '.join(a)

    if short_path_output:
        output_msg = scan_and_remove_path(cmd)
    else:
        output_msg = cmd

    if msg is not None:
        output_msg += '{}\n{}'.format(msg, output_msg)

    print('\n-> {}\n'.format(output_msg))

    if not safety_on:
        subprocess.call(cmd, shell=True)

    return output_msg


# ---------- Labels processors ---------------


def labels_query(labels, segmentation_array=None, remove_zero=True):
    """
    labels_list can be a list or a list of lists in case some labels have to be considered together. labels_names
    :param labels: can be int, list, string as 'all' or 'tot', or a string containing a path to a .txt or a numpy array
    :param segmentation_array: optional segmentation image data (array)
    :param remove_zero: do not return zero
    :return: labels_list, labels_names
    """
    labels_names = []
    if labels is None:
        labels = 'all'

    if isinstance(labels, int):
        if segmentation_array is not None:
            assert labels in segmentation_array
        labels_list = [labels, ]
    elif isinstance(labels, list):
        labels_list = labels
    elif isinstance(labels, str):
        if labels == 'all' and segmentation_array is not None:
            labels_list = list(np.sort(list(set(segmentation_array.astype(np.int).flat))))
        elif labels == 'tot' and segmentation_array is not None:
            labels_list = [list(np.sort(list(set(segmentation_array.astype(np.int).flat))))]
            if labels_list[0][0] == 0:
                if remove_zero:
                    labels_list = labels_list[0][1:]  # remove initial zero
                else:
                    labels_list = labels_list[0]
        elif os.path.exists(labels):
            if labels.endswith('.txt'):
                labels_list = list(np.loadtxt(labels))
            else:
                labels_list = list(np.load(labels))
        else:
            raise IOError("Input labels must be a list, a list of lists, or an int or the string 'all' (with "
                          "segmentation array not set to none)) or the path to a file with the labels.")
    elif isinstance(labels, dict):
        # expected input is the output of manipulate_descriptor.get_multi_label_dict (keys are labels names id are
        # list of labels)
        labels_list = []
        labels_names = labels.keys()
        for k in labels_names:
            if len(labels[k]) > 1:
                labels_list.append(labels[k])
            else:
                labels_list.append(labels[k][0])
    else:
        raise IOError("Input labels must be a list, a list of lists, or an int or the string 'all' or the path to a"
                      "file with the labels.")
    if not isinstance(labels, dict):
        labels_names = [str(l) for l in labels_list]

    return labels_list, labels_names


# ------------ Permutations --------------
# Thanks to Martin R and Accumulation
# https://codereview.stackexchange.com/questions/201725/disjoint-cycles-of-a-permutation


def permutation_from_cauchy_to_disjoints_cycles(cauchy_perm):
    """
    from [[1, 2, 3, 4, 5], [3, 4, 5, 2, 1]]
    to   [[1, 3, 5], [2, 4]]
    :param cauchy_perm: permutation in generalised Cauchy convention (any object, not necessarily numbers from 1 to n
    or from 0 ot n-1) where the objects that are not permuted do not appear.
    :return: input permutation written in disjoint cycles.
    """
    if not is_valid_permutation(cauchy_perm):
        raise IOError('Input permutation is not valid')
    list_cycles = []
    cycle = [cauchy_perm[0][0]]
    while len(cauchy_perm[0]) > 0:
        first_row_element, second_row_element = cycle[-1], cauchy_perm[1][cauchy_perm[0].index(cycle[-1])]
        cycle.append(second_row_element)
        cauchy_perm[0].remove(first_row_element)
        cauchy_perm[1].remove(second_row_element)
        if cycle[0] == cycle[-1]:
            if len(cycle) > 2:
                list_cycles += [cycle[:-1]]
            if len(cauchy_perm[0]) > 0:
                cycle = [cauchy_perm[0][0]]
    return list_cycles


def permutation_from_disjoint_cycles_to_cauchy(cyclic_perm):
    """
    from [[1, 3, 5], [2, 4]]
    to   [[1, 2, 3, 4, 5], [3, 4, 5, 2, 1]]
    """
    pairs = sorted([(a, b) for cycle in cyclic_perm for a, b in zip(cycle, cycle[1:] + cycle[:1])])
    return [list(i) for i in zip(*pairs)]

