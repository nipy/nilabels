"""
Module to manipulate descriptors as formatted by ITK-snap,
as the one in the descriptor below:

################################################
# ITK-SnAP Label Description File
# File format:
# IDX   -R-  -G-  -B-  -A--  VIS MSH  LABEL
# Fields:
#    IDX:   Zero-based index
#    -R-:   Red color component (0..255)
#    -G-:   Green color component (0..255)
#    -B-:   Blue color component (0..255)
#    -A-:   Label transparency (0.00 .. 1.00)
#    VIS:   Label visibility (0 or 1)
#    IDX:   Label mesh visibility (0 or 1)
#  LABEL:   Label description - name (abbreviation)
################################################
    0     0    0    0        0  0  0    "background"
    1   255    0    0        1  1  1    "label one (l1)"
    2   204    0    0        1  1  1    "label two (l2)"
    3    51   51  255        1  1  1    "label three"
    4   102  102  255        1  1  1    "label four"
    5     0  204   51        1  1  1    "label five (l5)"
    6    51  255  102        1  1  1    "label six"
    7   255  255    0        1  1  1    "label seven"
    8   255  50    50        1  1  1    "label eight"
    ...

"""
import collections
import os
import copy
import numpy as np

from LABelsToolkit.tools.aux_methods.sanity_checks import is_valid_permutation
from LABelsToolkit.tools.aux_methods.utils import from_permutation_to_disjoints_cycles


descriptor_standard_header = \
"""################################################
# ITK-SnAP Label Description File
# File format:
# IDX   -R-  -G-  -B-  -A--  VIS MSH  LABEL
# Fields:
#    IDX:   Zero-based index
#    -R-:   Red color component (0..255)
#    -G-:   Green color component (0..255)
#    -B-:   Blue color component (0..255)
#    -A-:   Label transparency (0.00 .. 1.00)
#    VIS:   Label visibility (0 or 1)
#    IDX:   Label mesh visibility (0 or 1)
#  LABEL:   Label description
################################################
"""

descriptor_data_examples = \
"""    0     0    0    0        0  0  0    "background"
    1   255    0    0        1  1  1    "label one (l1)"
    2   204    0    0        1  1  1    "label two (l2)"
    3    51   51  255        1  1  1    "label three"
    4   102  102  255        1  1  1    "label four"
    5     0  204   51        1  1  1    "label five (l5)"
    6    51  255  102        1  1  1    "label six"
    7   255  255    0        1  1  1    "label seven"
    8   255  50    50        1  1  1    "label eight"
"""


def generate_dummy_label_descriptor(pfi_output, list_labels=range(5),
                                    list_roi_names=None, list_colors_triplets=None):
    """
    For testing purposes, it creates a dummy label descriptor with the itk-snap convention
    :param pfi_output: where to save the eventual label descriptor
    :param list_labels: list of labels range, default 0:5
    :param list_roi_names: names of the regions of interests. If None, default names are assigned.
    :param list_colors_triplets: list of lists of rgb same lenght of list_roi_names if any.
    :return: label descriptor as a dictionary.
    """
    d = collections.OrderedDict()
    num_labels = len(list_labels)
    visibility = [(1.0, 1, 1)] * num_labels
    if list_roi_names is None:
        list_roi_names = ["label {}".format(j) for j in list_labels]
    else:
        if not len(list_labels) == len(list_roi_names):
            raise IOError('Wrong input data')
    if list_colors_triplets is None:
        list_colors_triplets = [list(np.random.choice(range(256), 3)) for _ in range(num_labels)]
    else:
        if not len(list_labels) == len(list(list_colors_triplets)):
            raise IOError('Wrong input data')
    for j in range(num_labels):
        up_d = {str(j): [list_colors_triplets[j], visibility[j], list_roi_names[j]]}
        d.update(up_d)
    f = open(pfi_output, 'w+')
    f.write(descriptor_standard_header)
    for j in d.keys():
        if j.isdigit():
            line = '{0: >5}{1: >6}{2: >6}{3: >6}{4: >9}{5: >6}{6: >6}    "{7}"\n'.format(
                j, d[j][0][0], d[j][0][1], d[j][0][2], d[j][1][0], d[j][1][1], d[j][1][2], d[j][2])
            f.write(line)
    f.close()
    return d


class LabelsDescriptorManager(object):

    def __init__(self, pfi_label_descriptor, convention='itk-snap'):
        self.pfi_label_descriptor = pfi_label_descriptor
        self.convention = convention
        self._check_path()
        if self.convention == 'itk-snap':
            self.dict_label_descriptor = self.get_dict_itk_snap()
        elif self.convention == 'fsl':
            self.dict_label_descriptor = self.get_dict_fsl()
        else:
            raise IOError("Signature for the variable **convention** can be only 'itk-snap' or 'fsl'.")

    # Sanity checks:

    def _check_path(self):
        """
        The path to labels descriptor it must exist.
        :return: input path sanity check.
        """
        if not os.path.exists(self.pfi_label_descriptor):
            msg = 'Label descriptor file {} does not exist'.format(self.pfi_label_descriptor)
            raise IOError(msg)

    # Getters-setters:

    def get_dict_itk_snap(self):
        """
        Parse the ITK-Snap label descriptor into an ordered dict data structure.
        Each element of the ordered dict is of the kind
         {218 : [[128, 0, 128], [1.0, 1.0, 1.0], 'Corpus callosum']}
          key : [[RGB],         [B A vis],       "label_name"]

        :return: dict with information relative to the parsed label descriptor.
        id : ''
        """
        label_descriptor_dict = collections.OrderedDict()
        for l in open(self.pfi_label_descriptor, 'r'):
            if not l.strip().startswith('#') and not l == '':
                parsed_line = [j.strip() for j in l.split('  ') if not j == '']
                args        = [tuple(parsed_line[1:4]), tuple(parsed_line[4:7]), parsed_line[7].replace('"', '')]
                args[0]     = [int(k) for k in args[0]]
                args[1]     = [float(k) for k in args[1]]
                label_descriptor_dict.update({int(parsed_line[0]): args})

        return label_descriptor_dict

    def get_dict_fsl(self):
        """
        Parse the fsl label descriptor into an ordered dict data structure.
        """
        label_descriptor_dict = collections.OrderedDict()
        for l in open(self.pfi_label_descriptor, 'r'):
            if not l.strip().startswith('#'):
                parsed_line = [j.strip() for j in l.split(' ') if not j == '']
                args        = [list(parsed_line[2:5]), float(parsed_line[5]), parsed_line[1]]
                args[0]     = [int(k) for k in args[0]]
                args[1]     = [args[1], 1, 1]
                label_descriptor_dict.update({int(parsed_line[0]): args})
        return label_descriptor_dict

    def get_multi_label_dict(self, keep_duplicate=False, combine_right_left=True):
        """
        Different data structure to allow for multiple labels.
        :param keep_duplicate:
        :param combine_right_left:
        :return:
        """
        mld = collections.OrderedDict()
        mld_tmp = collections.OrderedDict()
        # first round, fill mld_tmp, with the same values in the label descriptor switching the label name with
        # the label id - note: possible abbreviations gets lost.
        for k in self.dict_label_descriptor.keys():
            if combine_right_left:
                mld_tmp.update({self.dict_label_descriptor[k][2].replace('"', ''): [int(k)]})
                # second round, add the left right in dictionary entry.
                for k1 in mld_tmp.keys():
                    if keep_duplicate:
                        mld.update({k1: mld_tmp[k1]})
                    if 'Right' in k1:
                        left_key = k1.replace('Right', 'Left')
                        key = k1.replace('Right', '').strip()
                        if key.startswith('-'):
                            key = key[1:]
                        mld.update({key : mld_tmp[left_key] + mld_tmp[k1]})
                    elif 'Left' in k1:
                        pass
                    else:
                        if not keep_duplicate:
                            mld.update({k1: mld_tmp[k1]})
            else:
                mld.update({self.dict_label_descriptor[k][2].replace('"', ''): [int(k)]})

        return mld

    def save_label_descriptor(self, pfi_where_to_save):
        f = open(pfi_where_to_save, 'w+')
        if self.convention == 'itk-snap':
            f.write(descriptor_standard_header)

        for j in self.dict_label_descriptor.keys():
            if self.convention == 'itk-snap':
                line = '{0: >5}{1: >6}{2: >6}{3: >6}{4: >9}{5: >6}{6: >6}    "{7}"\n'.format(
                        j,
                        self.dict_label_descriptor[j][0][0],
                        self.dict_label_descriptor[j][0][1],
                        self.dict_label_descriptor[j][0][2],
                        self.dict_label_descriptor[j][1][0],
                        int(self.dict_label_descriptor[j][1][1]),
                        int(self.dict_label_descriptor[j][1][2]),
                        self.dict_label_descriptor[j][2])

            elif self.convention == 'fsl':
                line = '{0} {1} {2} {3} {4} {5}\n'.format(
                        j,
                        self.dict_label_descriptor[j][2].replace(' ', '-'),
                        self.dict_label_descriptor[j][0][0],
                        self.dict_label_descriptor[j][0][1],
                        self.dict_label_descriptor[j][0][2],
                        self.dict_label_descriptor[j][1][0])
            else:
                return
            f.write(line)
        f.close()

    def save_as_multi_label_descriptor(self, pfi_destination):
        """
        Multi label descriptor looks like:
        ---
        Clear Label          &     0
        Prefrontal Left      &     5
        Prefrontal Right     &     6
        Prefrontal           &     5  &     6
        ...
        ----
        :param pfi_destination: where to save the multi label descriptor in .txt or compatible format.
        :return: a saved multi label descriptor
        """
        mld = self.get_multi_label_dict()
        f = open(pfi_destination, 'w+')
        for k in mld.keys():
            line  = '{0: <40}'.format(k)
            for j in mld[k]:
                line += '&{0: ^10}'.format(j)
            f.write(line)
            f.write('\n')
        f.close()

    # Methods for labels manipulations -  all these methods are not destructive.

    def relabel(self, old_labels, new_labels, sort=True):
        """
        :param old_labels: list of existing labels
        :param new_labels: list of new labels to substitute the previous one
        :param sort: provide sorted OrderedDict output.
        E.G. old_labels = [3, 6, 9]  new_labels = [4, 4, 100]
        will transform label 3 in label 4, label 6 in label 4 and label 9 in label 100, even if label 100 was not
        present before in the labels descriptor.
        Note: where multiple labels are relabelled to the same one, the arguments kept in the labels descriptor are the
        one of the last label. In the example, label 4 will have the arguments of label 6 and arguments of label 3 will
        be lost in the output ldm.
        """
        ldm_new = copy.deepcopy(self)
        for k in old_labels:
            del ldm_new.dict_label_descriptor[k]
        for k_new in new_labels:
            k_old = old_labels[new_labels.index(k_new)]
            if k_old in self.dict_label_descriptor.keys():
                ldm_new.dict_label_descriptor.update({k_new : self.dict_label_descriptor[k_old]})
            else:
                ldm_new.dict_label_descriptor.update({k_new: [[255, 0, 255], [1.0, 1.0, 1.0], 'NewLabel']})
        if sort:
            d_sorted = collections.OrderedDict()
            for k in sorted(ldm_new.dict_label_descriptor.keys()):
                d_sorted.update({k : ldm_new.dict_label_descriptor[k]})
                ldm_new.dict_label_descriptor = d_sorted
        return ldm_new

    def permute_labels(self, permutation):
        """
        Given a permutation, as specified in is_valid_permutation, provides a new labels descriptor manager
        with the permuted labels.
        :param permutation: a permutation defined by a list of integers (checked by is_valid_permutation under
        sanity_check).
        :return:
        """
        if not is_valid_permutation(permutation):
            raise IOError('Not valid input permutation, please see the documentation.')
        ldm_new = copy.deepcopy(self)

        cycles = from_permutation_to_disjoints_cycles(permutation)
        for cycle in cycles:
            len_perm = len(cycle)
            for i in range(len_perm):
                ldm_new.dict_label_descriptor[cycle[i]] = self.dict_label_descriptor[cycle[(i + 1) % len_perm]]
        return ldm_new

    def erase_labels(self, labels_to_erase):
        """
        :param labels_to_change: is a list of labels.
        :param single_new_label:
        E.G. labels_to_change = [1,2,3,4] single_new_label = 7
        will transform the labels 1,2,3,4 in the label 7.
        Note: the arguments kept in the labels descriptor corresponding to the new label are the one corresponding to
        the last lables of labels_to_change. Other labels argumetns are lost in the output ldm.
        """
        ldm_new = copy.deepcopy(self)
        for l in labels_to_erase:
            del ldm_new.dict_label_descriptor[l]
        return ldm_new

    def assign_all_other_labels_the_same_value(self, labels_to_keep, other_value):
        """
        :param labels_to_keep:
        :param other_values:
        :return:
        """
        labels_that_will_have_the_same_value = list(set(self.dict_label_descriptor.keys()) - set(labels_to_keep) - {0})
        return self.relabel(labels_that_will_have_the_same_value,
                            [other_value] * len(labels_that_will_have_the_same_value))

    def keep_one_label(self, label_to_keep=1):
        """
        :param label_to_keep:
        :return:
        """
        labels_not_to_keep = list(set(self.dict_label_descriptor.keys()) - {label_to_keep})
        return self.relabel(labels_not_to_keep, [0] * len(labels_not_to_keep))
