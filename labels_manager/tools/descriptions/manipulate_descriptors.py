import os
import collections
import numpy as np

from labels_manager.tools.descriptions.colours_rgb_lab import get_random_rgb
from labels_manager.tools.aux_methods.utils_nib import set_new_data

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
    
A label descriptor can be as well saved as 
"""
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

class LabelsDescriptorManager(object):

    def __init__(self, pfi_label_descriptor):
        self.pfi_label_descriptor = pfi_label_descriptor
        self._dict_label_descriptor = self.get_dict(as_string=True)

    def _check_path(self):
        if not os.path.exists(self.pfi_label_descriptor):
            msg = 'Label descriptor file {} does not exist'.format(self.pfi_label_descriptor)
            raise IOError(msg)

    def get_dict(self, as_string=False):
        """
        Parse the ITK-Snap label descriptor into a dict.
        :return: dict with information relative to the parsed label descriptor
        """
        self._check_path()
        label_descriptor_dict = collections.OrderedDict()
        for l in open(self.pfi_label_descriptor, 'r'):
            if not l.startswith('#'):
                parsed_line = [j.strip() for j in l.split('  ') if not j == '']
                args = [tuple(parsed_line[1:4]), tuple(parsed_line[4:7]), parsed_line[7].replace('"', '')]
                if '(' in args[2]:  # there is the abbreviation - it gets separated in a new args element:
                    name = args[2].split('(')[0].strip()
                    abbrev = args[2].split('(')[1].replace(')', '').strip()
                    args[2] = name
                    args += [abbrev]
                if as_string:
                    dd = {parsed_line[0]: args}
                else:
                    args[0] = [int(k) for k in args[0]]
                    args[1] = [int(k) for k in args[1]]
                    dd = {int(parsed_line[0]): args}
                label_descriptor_dict.update(dd)

        return label_descriptor_dict

    def get_multi_label_dict(self, keep_duplicate=False, combine_right_left=True):
        mld = collections.OrderedDict()
        mld_tmp = collections.OrderedDict()
        # first round, fill mld_tmp, with the same values in the label descriptor switching the label name with
        # the label id - note: possible abbreviations gets lost.
        for k in self._dict_label_descriptor.keys():
            if combine_right_left:
                mld_tmp.update({self._dict_label_descriptor[k][2].replace('"', ''): [int(k)]})
                # second round, add the left right in dictionary entry.
                for k in mld_tmp.keys():
                    if keep_duplicate:
                        mld.update({k: mld_tmp[k]})
                    if 'Right' in k:
                        left_key = k.replace('Right', 'Left')
                        mld.update({k.replace('Right', '').strip() : mld_tmp[left_key] + mld_tmp[k]})
                    elif 'Left' in k:
                        pass
                    else:
                        if not keep_duplicate:
                            mld.update({k: mld_tmp[k]})
            else:
                mld.update({self._dict_label_descriptor[k][2].replace('"', ''): [int(k)]})

        return mld

    def save_label_descriptor(self, pfi_where_to_save):
        f = open(pfi_where_to_save, 'w+')
        f.write(descriptor_standard_header)
        ld_dict = self.get_dict(as_string=False)
        for j in ld_dict.keys():
            if j.isdigit():
                if len(ld_dict[j]) == 4:  # there is an abbreviation
                    name = '{0} ({1})'.format(ld_dict[j][2], ld_dict[j][3])
                else:
                    name = ld_dict[j][2]
                line = '{0: >5}{1: >6}{2: >4}{3: >4}{4: >9}{5: >3}{6: >3}    "{7: >5}"\n'.format(j,
                        ld_dict[j][0][0],
                        ld_dict[j][0][1],
                        ld_dict[j][0][2],
                        ld_dict[j][1][0],
                        ld_dict[j][1][1],
                        ld_dict[j][1][2],
                        name)
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

    def save_labels_and_abbreviations(self, pfi_where_to_save):
        f = open(pfi_where_to_save, 'w+')
        f.write(descriptor_standard_header)
        ld_dict = self.get_dict(as_string=False)
        for j in ld_dict.keys():
            if j.isdigit():
                if len(ld_dict[j]) == 4:  # there is an abbreviation
                    abbr =  ld_dict[j][3]
                else:
                    abbr = ''
                line = '{0: >5}{1: >6}\n'.format(j, abbr)
                f.write(line)
        f.close()

    def permute_labels(self, permutation):
        # permute the label
        # TODO
        # save on the same place: yes it is destructive!
        self.save_label_descriptor(self.pfi_label_descriptor)

    def get_corresponding_rgb_image(self, im_segm):
        """
        From the labels descriptor and a nibabel segmentation image, it returns the
        :param im_segm: nibabel segmentation whose labels corresponds to the input labels descriptor.
        :return: a 4d image, where at each voxel there is the [rgb ]
        """
        labels_in_image = list(np.sort(list(set(im_segm.get_data().flatten()))))
        labels_dict = self.get_dict(as_string=False)

        assert len(im_segm.shape) == 3

        rgb_image_arr = np.ones(list(im_segm.shape) + [3])

        for l in labels_dict.keys():
            if l not in labels_in_image:
                msg = 'get_corresponding_rgb_image: Label {} present in the label descriptor and not in ' \
                      'selected image'.format(l)
                print(msg)
            pl = im_segm.get_data() == l
            rgb_image_arr[pl, :] = labels_dict[l][0]

        return set_new_data(im_segm, rgb_image_arr, new_dtype=np.int32)


def generate_dummy_label_descriptor(pfi_output=None, list_labels=range(5), list_roi_names=None):
    """
    For testing purposes, it creates a dummy label descriptor.
    :param pfi_output: where to save the eventual label descriptor
    :param list_labels: list of labels range, default 0:5
    :param list_roi_names: names of the regions of interests. If None, default names are assigned.
    :return: label descriptor as a dictionary.
    """
    d = collections.OrderedDict()
    d.update({'type': 'Label descriptor parsed'})
    num_labels = len(list_labels)
    colors = [get_random_rgb() for _ in range(num_labels)]
    visibility = [(1, 1, 1)] * num_labels
    if list_roi_names is None:
        list_roi_names = ["label {}".format(j) for j in list_labels]
    else:
        assert len(list_labels) == len(list_roi_names)
    for j in range(num_labels):
        up_d = {str(j): [colors[j], visibility[j], list_roi_names[j]]}
        d.update(up_d)
    f = open(pfi_output, 'w+')
    f.write(descriptor_standard_header)
    for j in d.keys():
        if j.isdigit():
            line = '{0: >5}{1: >6}{2: >4}{3: >4}{4: >9}{5: >3}{6: >3}    "{7: >5}"\n'.format(j,
                d[j][0][0], d[j][0][1], d[j][0][2], d[j][1][0], d[j][1][1], d[j][1][2], d[j][2])
            f.write(line)
    f.close()
    return d

# temporary test
# if __name__ == '__main__':
#
#     # from labels_manager.tools.descriptions.manipulate_descriptors import LabelsDescriptorManager
#
#     pfi_descriptor = '/Users/sebastiano/Dropbox/RabbitEOP-MRI/study/A_internal_template/LabelsDescriptors/labels_descriptor_v9.txt'
#
#     ldm = LabelsDescriptorManager(pfi_descriptor)
#     ldm.get_dict()
#     import nibabel as nib
#
#     im_se = nib.load('/Users/sebastiano/Desktop/sphdec/1201/segm/automatic/1201_S0_segm_IN_TEMPLATE.nii.gz')
#
#     im = ldm.get_corresponding_rgb_image(im_se)
#     nib.save(im, '/Users/sebastiano/Desktop/zzz_rgb.nii.gz')
