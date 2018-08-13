import collections
import os
import numpy as np

from LABelsToolkit.tools.aux_methods.utils_nib import set_new_data

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


class LabelsDescriptorManager(object):

    def __init__(self, pfi_label_descriptor, convention='itk-snap'):
        self.pfi_label_descriptor = pfi_label_descriptor
        self._convention = convention
        self._check_path()
        if self._convention == 'itk-snap':
            self._dict_label_descriptor = self.get_dict_itk_snap()
        elif self._convention == 'fsl':
            self._dict_label_descriptor = self.get_dict_fsl()
        else:
            raise IOError("Signature for the variable **convention** can be only 'itk-snap' or 'fsl'.")

    def _check_path(self):
        """
        The path to labels descriptor it must exist.
        :return: input path sanity check.
        """
        if not os.path.exists(self.pfi_label_descriptor):
            msg = 'Label descriptor file {} does not exist'.format(self.pfi_label_descriptor)
            raise IOError(msg)

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
            if not l.strip().startswith('#'):
                parsed_line = [j.strip() for j in l.split('  ') if not j == '']
                args        = [tuple(parsed_line[1:4]), tuple(parsed_line[4:7]), parsed_line[7].replace('"', '')]

                args[0] = [int(k) for k in args[0]]
                args[1] = [float(k) for k in args[1]]
                dd = {int(parsed_line[0]): args}
                label_descriptor_dict.update(dd)

        return label_descriptor_dict

    def get_dict_fsl(self):
        """
        Parse the fsl label descriptor into an ordered dict data structure.
        """
        label_descriptor_dict = collections.OrderedDict()
        for l in open(self.pfi_label_descriptor, 'r'):
            if not l.strip().startswith('#'):
                parsed_line = [j.strip() for j in l.split(' ') if not j == '']
                args        = [list(parsed_line[2:5]), list(parsed_line[5]), parsed_line[1]]

                args[0] = [int(k) for k in args[0]]
                args[1] = [int(k) for k in args[1]]
                dd = {int(parsed_line[0]): args}
                label_descriptor_dict.update(dd)
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
        for k in self._dict_label_descriptor.keys():
            if combine_right_left:
                mld_tmp.update({self._dict_label_descriptor[k][2].replace('"', ''): [int(k)]})
                # second round, add the left right in dictionary entry.
                for k in mld_tmp.keys():
                    if keep_duplicate:
                        mld.update({k: mld_tmp[k]})
                    if 'Right' in k:
                        left_key = k.replace('Right', 'Left')
                        key = k.replace('Right', '').strip()
                        if key.startswith('-'):
                            key = key[1:]
                        mld.update({key : mld_tmp[left_key] + mld_tmp[k]})
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
        if self._convention == 'itk-snap':
            f.write(descriptor_standard_header)

        for j in self._dict_label_descriptor.keys():
            if self._convention == 'itk-snap':
                line = '{0: >5}{1: >6}{2: >4}{3: >4}{4: >9}{5: >3}{6: >3}    "{7: >5}"\n'.format(
                        j,
                        self._dict_label_descriptor[j][0][0],
                        self._dict_label_descriptor[j][0][1],
                        self._dict_label_descriptor[j][0][2],
                        self._dict_label_descriptor[j][1][0],
                        int(self._dict_label_descriptor[j][1][1]),
                        int(self._dict_label_descriptor[j][1][2]),
                        self._dict_label_descriptor[j][2])

            elif self._convention == 'fsl':
                line = '{0} {1} {2} {3} {4} {5}\n'.format(
                        j,
                        self._dict_label_descriptor[j][2].replace(' ', '-'),
                        self._dict_label_descriptor[j][0][0],
                        self._dict_label_descriptor[j][0][1],
                        self._dict_label_descriptor[j][0][2],
                        self._dict_label_descriptor[j][1][0])
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

    def get_corresponding_rgb_image(self, im_segm, invert_black_white=False):
        """
        From the labels descriptor and a nibabel segmentation image.
        :param im_segm: nibabel segmentation whose labels corresponds to the input labels descriptor.
        :return: a 4d image, where at each voxel there is the [r, g, b] vector in the fourth dimension.
        """
        labels_in_image = list(np.sort(list(set(im_segm.get_data().flatten()))))

        assert len(im_segm.shape) == 3

        rgb_image_arr = np.ones(list(im_segm.shape) + [3])

        for l in self._dict_label_descriptor.keys():
            if l not in labels_in_image:
                msg = 'get_corresponding_rgb_image: Label {} present in the label descriptor and not in ' \
                      'selected image'.format(l)
                print(msg)
            pl = im_segm.get_data() == l
            rgb_image_arr[pl, :] = self._dict_label_descriptor[l][0]

        if invert_black_white:
            pl = im_segm.get_data() == 0
            rgb_image_arr[pl, :] = np.array([255, 255, 255])
        return set_new_data(im_segm, rgb_image_arr, new_dtype=np.int32)
