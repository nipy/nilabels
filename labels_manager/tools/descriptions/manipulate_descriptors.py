import os
import collections

from labels_manager.tools.descriptions.colours_rgb_lab import get_random_rgb
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


def parse_label_descriptor_to_dict(pfi_label_descriptor):
    """
    Parse the ITK-Snap label descriptor into a dict.
    :param pfi_label_descriptor: path to file to label descriptor.
    :return: list of lists, where each sublist contains the information of each line.
    """
    if not os.path.exists(pfi_label_descriptor):
        msg = 'Label descriptor file {} does not exist'.format(pfi_label_descriptor)
        raise IOError(msg)

    label_descriptor_dict = collections.OrderedDict()
    label_descriptor_dict.update({'type': 'Label descriptor parsed'})

    for l in open(pfi_label_descriptor, 'r'):
        if not l.startswith('#'):
            parsed_line = [j.strip() for j in l.split('  ') if not j == '']
            d = {parsed_line[0] : [tuple(parsed_line[1:4]), tuple(parsed_line[4:7]), parsed_line[7].replace('"', '') ]}
            label_descriptor_dict.update(d)

    return label_descriptor_dict


def from_dict_to_label_descriptor(dict_input, pfi_output):
    assert dict_input['type'] == 'Label descriptor parsed'
    f = open(pfi_output, 'w+')
    f.write(descriptor_standard_header)
    for j in dict_input.keys():
        if j.isdigit():
            line = '{0: >5}{1: >6}{2: >4}{3: >4}{4: >9}{5: >3}{6: >3}    "{7: >5}"\n'.format(j,
                                             dict_input[j][0][0], dict_input[j][0][1], dict_input[j][0][2],
                                             dict_input[j][1][0], dict_input[j][1][1], dict_input[j][1][2],
                                             dict_input[j][2])
            f.write(line)
    f.close()


def generate_sample_label_descriptor(list_labels=range(5), pfi_output=None):
    d = collections.OrderedDict()
    d.update({'type' :'Label descriptor parsed'})
    num_labels = len(list_labels)
    colors = [get_random_rgb() for j in range(num_labels)]
    visibility = [(1, 1, 1)] * num_labels
    label = ["label {}".format(j) for j in list_labels]
    for j in range(num_labels):
        up_d = {str(j) : [colors[j], visibility[j], label[j]]}
        d.update(up_d)
    if pfi_output is not None:
        from_dict_to_label_descriptor(d, pfi_output)
        print 'Dummy dictionary saved in {}'.format(pfi_output)
    return d


def permute_idx(pfi_descriptor, in_permutation, pfi_new_descriptor):
    # TODO
    pass


if __name__ == '__main__':
    # test on the fly:
    pfo_ld = '/Users/sebastiano/Dropbox/RabbitEOP-MRI/study/A_internal_template/LabelsDescriptors'
    pfi_dict_test = os.path.join(pfo_ld, 'labels_descriptor_v7.txt')
    d = parse_label_descriptor_to_dict(pfi_dict_test)
    print d
    pfi_output_descr = '/Users/sebastiano/Desktop/z_lab_desc_test.txt'

    from_dict_to_label_descriptor(d, pfi_output_descr)
    pfi_output_descr1 = '/Users/sebastiano/Desktop/z_lab_desc_test2.txt'
    generate_sample_label_descriptor(num_labels=5, pfi_output=pfi_output_descr1)
