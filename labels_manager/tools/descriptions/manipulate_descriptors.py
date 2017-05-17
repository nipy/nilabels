import os
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


def generate_example(num_labels=5, ):
    # TODO and to move in exmaples
    pass


def permute_idx(in_descriptor, in_permutation):
    # TODO
    pass


def standardise_colour_convention_left_right(in_descriptor, out_descriptor):
    # TODO
    pass


def parse_label_descriptor_in_a_list(pfi_label_descriptor):
    """
    parse the ITK-Snap into a list.
    :param pfi_label_descriptor: path to file to label descriptor.
    :return: list of lists, where each sublist contains the information of each line.
    """
    if not os.path.exists(pfi_label_descriptor):
        msg = 'Label descriptor file {} does not exist'.format(pfi_label_descriptor)
        raise IOError(msg)

    f = open(pfi_label_descriptor, 'r')
    lines = f.readlines()

    label_descriptor_list = []

    for l in lines:
        if not l.startswith('#'):

            parsed_line = [j.strip() for j in l.split('  ') if not j == '']
            for position_element, element in enumerate(parsed_line):
                if element.isdigit():
                    parsed_line[position_element] = int(element)
                if element.startswith('"') or element.endswith('"'):
                    parsed_line[position_element] = element.replace('"', '')

            parsed_line.insert(1, parsed_line[-1])

            label_descriptor_list.append(parsed_line[:-1])

    return label_descriptor_list
