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

An instance of a label descriptor manager is contains the path to a file with
a label descriptor in itk-snap convention or fsl convention and a parameter specifying the convetion.
"""
import collections
import os
import copy
import numpy as np

from nilabels.tools.aux_methods.sanity_checks import is_valid_permutation
from nilabels.tools.aux_methods.utils import permutation_from_cauchy_to_disjoints_cycles


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
    visibility = [(1.0, 1, 1)] * len(list_labels)
    if list_roi_names is None:
        list_roi_names = ["label {}".format(j) for j in list_labels]
    else:
        if not len(list_labels) == len(list_roi_names):
            raise IOError('Wrong input data')
    if list_colors_triplets is None:
        list_colors_triplets = [list(np.random.choice(range(256), 3)) for _ in list_labels]
    else:
        if not len(list_labels) == len(list(list_colors_triplets)):
            raise IOError('Wrong input data')
    for j_in, j in enumerate(list_labels):
        up_d = {str(j): [list_colors_triplets[j_in], visibility[j_in], list_roi_names[j_in]]}
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

    def __init__(self, pfi_label_descriptor, labels_descriptor_convention='itk-snap'):
        self.pfi_label_descriptor = pfi_label_descriptor
        self.convention = labels_descriptor_convention
        self._check_path()
        if self.convention == 'itk-snap':
            self.dict_label_descriptor = self.get_dict_itk_snap()
        elif self.convention == 'fsl':
            self.dict_label_descriptor = self.get_dict_fsl()
        else:
            raise IOError("Signature for the variable **convention** can be only 'itk-snap' or 'fsl'.")

    # ----------- Sanity checks -----------

    def _check_path(self):
        """
        The path to labels descriptor it must exist.
        :return: input path sanity check.
        """
        if not os.path.exists(self.pfi_label_descriptor):
            msg = 'Label descriptor file {} does not exist'.format(self.pfi_label_descriptor)
            raise IOError(msg)

    # ---------- Getters-setters -----------

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
                args[0]     = [int(k) for k in list(args[0])]
                args[1]     = [args[1], 1, 1]
                label_descriptor_dict.update({int(parsed_line[0]): args})
        return label_descriptor_dict

    def get_multi_label_dict(self, combine_right_left=True):
        """
        Different data structure to allow for multiple labels with the same name Left/Right (capital letters)
        :param combine_right_left:
        :return:
        """
        mld = collections.OrderedDict()
        mld_tmp = collections.OrderedDict()
        # Fill mld_tmp, with the same values in the label descriptor switching the label name with
        # the label id - note: possible abbreviations gets lost.

        for k_label_id in self.dict_label_descriptor.keys():
            mld_tmp.update({self.dict_label_descriptor[k_label_id][2].replace('"', '').strip(): [int(k_label_id)]})

        mld_tmp_keys = list(mld_tmp.keys())
        while len(mld_tmp_keys) > 0:
            if 'Right' in mld_tmp_keys[0]:
                right_key = mld_tmp_keys[0]
                left_key = right_key.replace('Right', 'Left')
                if left_key in mld_tmp_keys:
                    mld.update({right_key: mld_tmp[right_key]})
                    mld.update({left_key: mld_tmp[left_key]})
                    mld_tmp_keys.remove(left_key)
                    if combine_right_left:
                        mld.update({right_key.replace('Right', '').strip(): mld_tmp[right_key] + mld_tmp[left_key]})
                else:
                    print('Warning: Left key for {} not present'.format(right_key))
                    mld.update({right_key: mld_tmp[right_key]})

            elif 'Left' in mld_tmp_keys[0]:
                left_key = mld_tmp_keys[0]
                right_key = left_key.replace('Left', 'Right')
                if right_key in mld_tmp_keys:
                    mld.update({left_key: mld_tmp[left_key]})
                    mld.update({right_key: mld_tmp[right_key]})
                    mld_tmp_keys.remove(right_key)
                    if combine_right_left:
                        mld.update({left_key.replace('Left', '').strip(): mld_tmp[left_key] + mld_tmp[right_key]})
                else:
                    print('Warning: Right key for {} not present'.format(left_key))
                    mld.update({left_key: mld_tmp[left_key]})
            else:
                mld.update({mld_tmp_keys[0]: mld_tmp[mld_tmp_keys[0]]})

            mld_tmp_keys = mld_tmp_keys[1:]

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
                raise IOError("Assigned variable for **convention** can be only 'itk-snap' or 'fsl'.")
            f.write(line)
        f.close()

    def save_as_multi_label_descriptor(self, pfi_destination):
        """
        Multi label descriptor is a custom file. It shows labels name and labels number keeping the Left and Right
        joined together in an additional line. It looks like:
        ---
        Clear Label          &     0
        Prefrontal Left      &     5
        Prefrontal Right     &     6
        Prefrontal           &     5  &     6
        Corona Left          &     7
        Corona Right         &     8
        Prefrontal           &     7  &     8
        ----
        :param pfi_destination: where to save the multi label descriptor in .txt or compatible format.
        :return: a saved multi label descriptor
        """
        mld = self.get_multi_label_dict(combine_right_left=True)
        f = open(pfi_destination, 'w+')
        for k in mld.keys():
            line  = '{0: <40}'.format(k)
            for j in mld[k]:
                line += '&{0: ^10}'.format(j)
            f.write(line)
            f.write('\n')
        f.close()

    # ----------- Methods for labels manipulations -  all these methods are not destructive -----------

    def relabel(self, list_old_labels, list_new_labels, sort=True):
        """
        :param list_old_labels: list of existing labels
        :param list_new_labels: list of new labels to substitute the previous one
        :param sort: provide sorted OrderedDict output.
        E.G. old_labels = [3, 6, 9]  new_labels = [4, 4, 100]
        will transform label 3 in label 4, label 6 in label 4 and label 9 in label 100, even if label 100 was not
        present before in the labels descriptor.
        Note: where multiple labels are relabelled to the same one, the arguments kept in the labels descriptor are the
        one of the last label. In the example, label 4 will have the arguments of label 6 and arguments of label 3 will
        be lost in the output ldm.
        """
        ldm_new = copy.deepcopy(self)
        for k in list_old_labels:
            if k not in ldm_new.dict_label_descriptor.keys():
                raise IOError('Label {} in the input list not present'.format(k))
            del ldm_new.dict_label_descriptor[k]
        for k_new in list_new_labels:
            k_old = list_old_labels[list_new_labels.index(k_new)]
            if k_old in self.dict_label_descriptor.keys():
                ldm_new.dict_label_descriptor.update({k_new : self.dict_label_descriptor[k_old]})
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

        cycles = permutation_from_cauchy_to_disjoints_cycles(permutation)
        for cycle in cycles:
            len_perm = len(cycle)
            for i in range(len_perm):
                ldm_new.dict_label_descriptor[cycle[i]] = self.dict_label_descriptor[cycle[(i + 1) % len_perm]]
        return ldm_new

    def erase_labels(self, labels_to_erase, verbose=True):
        """
        :param labels_to_erase: is a list of labels that will be erased.
        :param verbose: raise warning message if label to erase is not present
        """
        ldm_new = copy.deepcopy(self)
        for lab in labels_to_erase:
            if verbose and lab not in ldm_new.dict_label_descriptor.keys():
                print('Labels descriptor manager: Label {} can not be erased as not present.'.format(lab))
            else:
                del ldm_new.dict_label_descriptor[lab]
        return ldm_new

    def assign_all_other_labels_the_same_value(self, labels_to_keep, same_value_label):
        """
        :param labels_to_keep:
        :param same_value_label:
        :return:
        """
        labels_that_will_have_the_same_value = list(set(self.dict_label_descriptor.keys()) - set(labels_to_keep) - {0})
        return self.relabel(labels_that_will_have_the_same_value,
                            [same_value_label] * len(labels_that_will_have_the_same_value))

    def keep_one_label(self, label_to_keep=1):
        """
        :param label_to_keep: all other values will be set to zero.
        :return:
        """
        labels_not_to_keep = list(set(self.dict_label_descriptor.keys()) - {label_to_keep})
        return self.relabel(labels_not_to_keep, [0] * len(labels_not_to_keep))
