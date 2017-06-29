import os
import collections


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
#  LABEL:   Label description
################################################
    0     0    0    0        0  0  0    "background"
    1   255    0    0        1  1  1    "label one"
    2   204    0    0        1  1  1    "label two"
    3    51   51  255        1  1  1    "label three"
    4   102  102  255        1  1  1    "label four"
    5     0  204   51        1  1  1    "label five"
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
        self._dict_label_descriptor = self._get_dict()

    def _check_path(self):
        if not os.path.exists(self.pfi_label_descriptor):
            msg = 'Label descriptor file {} does not exist'.format(self.pfi_label_descriptor)
            raise IOError(msg)

    def _get_dict(self):
        """
        Parse the ITK-Snap label descriptor into a dict.
        :return: dict with information relative to the parsed label descriptor
        """
        self._check_path()
        label_descriptor_dict = collections.OrderedDict()
        label_descriptor_dict.update({'type': 'Label descriptor parsed'})  # special tag for my dictionary

        for l in open(self.pfi_label_descriptor, 'r'):
            if not l.startswith('#'):
                parsed_line = [j.strip() for j in l.split('  ') if not j == '']
                dd = {
                    parsed_line[0]: [tuple(parsed_line[1:4]), tuple(parsed_line[4:7]), parsed_line[7].replace('"', '')]}
                label_descriptor_dict.update(dd)

        return label_descriptor_dict

    def get_as_dict(self):
        return self._dict_label_descriptor

    def get_multi_label_dict(self):
        mld = collections.OrderedDict()
        mld.update({'type': 'Multi label descriptor parsed'})
        # first round, just switch
        for k in self._dict_label_descriptor.keys():
            mld.update({self._dict_label_descriptor[k][-1].replace('"', '') : [self._dict_label_descriptor[k]]})
        # second round, add the left right in dictionary entry.
        for k in mld.keys():
            if 'Right' in k:
                left_key = k.replace('Right', 'Left')
                mld.update({k.replace('Right', '').strip() : mld[left_key] + mld[k]})
        return mld

    def save_label_descriptor(self, pfi_where_to_save):
        f = open(pfi_where_to_save, 'w+')
        f.write(descriptor_standard_header)
        for j in self._dict_label_descriptor.keys():
            if j.isdigit():
                line = '{0: >5}{1: >6}{2: >4}{3: >4}{4: >9}{5: >3}{6: >3}    "{7: >5}"\n'.format(j,
                        self._dict_label_descriptor[j][0][0],
                        self._dict_label_descriptor[j][0][1],
                        self._dict_label_descriptor[j][0][2],
                        self._dict_label_descriptor[j][1][0],
                        self._dict_label_descriptor[j][1][1],
                        self._dict_label_descriptor[j][1][2],
                        self._dict_label_descriptor[j][2])
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
            if not k == 'type':
                line  = '{0: <40}'.format(k)
                for j in mld[k]:
                    line += '&{0: ^10}'.format(j)
                f.write(line)
        f.close()

    def permute_labels(self, permutation):
        # permute the label
        # TODO
        # save on the same place: yes it is destructive!
        self.save_label_descriptor(self.pfi_label_descriptor)
