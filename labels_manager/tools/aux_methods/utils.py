import numpy as np
import os
import subprocess


# ---- List utils ----


def lift_list(input_list):
    return [val for sublist in input_list for val in sublist]


def eliminates_consecutive_duplicates(input_list):
    """
    :param input_list: a list
    :return: the same list with no consecutive duplicates.
    """
    output_list = [input_list[0],]
    for i in xrange(1, len(input_list)):
        if not input_list[i] == input_list[i-1]:
            output_list.append(input_list[i])

    return output_list


# ---- Matrices and Arrays utils ----


def binarise_a_matrix(in_matrix, labels=None, dtype=np.bool):
    """
    All the values above zeros will be ones.
    :param in_matrix: any matrix
    :param labels: input labels.
    :param dtype: the output matrix is forced to this data type (bool by default).
    :return: The same matrix, where all the non-zero elements are equals to 1.
    """
    out_matrix = np.zeros_like(in_matrix)
    if labels is None:
        non_zero_places = in_matrix != 0
    else:
        non_zero_places = np.zeros_like(in_matrix, dtype=np.bool)
        for l in labels:
            non_zero_places += in_matrix == l

    np.place(out_matrix, non_zero_places, 1)
    return out_matrix.astype(dtype)


# def get_values_below_label(image, segmentation, label):
#     """
#     Given an image (matrix) and a segmentation (another matrix), provides a list
#     :param image: np.array of an image
#     :param segmentation: np.array of the segmentation of the same image
#     :param label: a label in the segmentation
#     :return: np.array with all the values below the label.
#     """
#     np.testing.assert_array_equal(image.shape, segmentation.shape)
#     below_label_places = segmentation == label
#     coord = np.nonzero(below_label_places.flatten())[0]
#     return np.take(image.flatten(), coord)

# ---- Paths Utils


def scan_and_remove_path(msg):
    """
    Take a string with a series of paths separated by a space and keeps only the base-names of each path.
    """
    a = [os.path.basename(p) for p in msg.split(' ')]
    return ' '.join(a)


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

    # if len(cmd) > 249:
    #     print(cmd)
    #     raise IOError('input command is too long, this may create problems. Please use shortest names!')
    if short_path_output:
        path_free_cmd = scan_and_remove_path(cmd)
    else:
        path_free_cmd = cmd

    if msg is not None:
        print('\n' + msg + '\n')
    else:
        print('\n-> ' + path_free_cmd + '\n')

    if not safety_on:
        # os.system(cmd)
        subprocess.call(cmd, shell=True)

# ---------- Labels processors ---------------


def labels_query(labels, segmentation_array=None):
    """
    Will return a list with the labels and the labels names (same order list of strings with labels names)
    for a labels list provided in some way, and the optional segmentation image data (array)
    :param labels: can be int, list, string as 'all' or 'tot', or a string containing a path to a .txt or a numpy array
    :param segmentation_array: optional segmentation image data (array)
    :return: labels_list, labels_names
    """
    if labels is None:
        labels = 'all'

    if isinstance(labels, int):
        assert labels in segmentation_array
        labels_list = [labels, ]
    elif isinstance(labels, list):
        labels_list = labels
    elif isinstance(labels, str):
        if labels == 'all' and segmentation_array is not None:
            labels_list = list(np.sort(list(set(segmentation_array.astype(np.int).flat))))
        elif labels == 'tot' and segmentation_array is not None:
            labels_list = [list(np.sort(list(set(segmentation_array.astype(np.int).flat))))]
        elif os.path.exists(labels):
            if labels.endswith('.txt'):
                labels_list = list(np.loadtxt(labels))
            else:
                labels_list = list(np.load(labels))
        else:
            raise IOError("Input labels must be a list, a list of lists, or an int or the string 'all' or the path to a"
                          "file with the labels.")
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


# ---------- Distributions ---------------


def triangular_density_function(x, a, mu, b):

    if a <= x < mu:
        return 2 * (x - a) / float((b - a) * (mu - a))
    elif x == mu:
        return 2 / float(b - a)
    elif mu < x <= b:
        return 2 * (b - x) / float((b - a) * (b - mu))
    else:
        return 0
