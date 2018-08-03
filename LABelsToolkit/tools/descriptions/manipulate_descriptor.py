import collections

from LABelsToolkit.tools.aux_methods.colours_rgb_lab import get_random_rgb
from LABelsToolkit.tools.descriptions.label_descriptor_manager import LabelsDescriptorManager, \
    descriptor_standard_header


def permute_labels_from_descriptor(in_ldm, permutation):
    """
    Given a permutation, as specified in
    Save on the same place: yes it is destructive, but results can be undone with the inverse permutation
    :param in_ldm: instance of a class labels descriptor manager.
    :param permutation: a permutation defined by a list of integers (checked by is_valid_permutation under
    sanity_check).
    :return:
    """
    # permute the label
    # TODO

    return in_ldm, permutation


def generate_dummy_label_descriptor(pfi_output=None, list_labels=range(5), list_roi_names=None):
    """
    For testing purposes, it creates a dummy label descriptor with the itk-snap convention
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
            line = '{0: >5}{1: >6}{2: >4}{3: >4}{4: >9}{5: >3}{6: >3}    "{7: >5}"\n'.format(
                j, d[j][0][0], d[j][0][1], d[j][0][2], d[j][1][0], d[j][1][1], d[j][1][2], d[j][2])
            f.write(line)
    f.close()
    return d


# temporary test
if __name__ == '__main__':

    # TODO move to examples

    pfi_descriptor = '/Users/sebastiano/Desktop/colour_label.txt'

    ldm = LabelsDescriptorManager(pfi_descriptor, convention='fsl')

    print ldm.get_dict_fsl()

    ldm.save_label_descriptor('/Users/sebastiano/Desktop/colour_label_fsl.txt')


    pfi_descriptor = '/Users/sebastiano/Desktop/labels_descriptor.txt'

    ldm = LabelsDescriptorManager(pfi_descriptor, convention='itk-snap')

    print ldm._dict_label_descriptor

    ldm._convention = 'fsl'

    ldm.save_label_descriptor('/Users/sebastiano/Desktop/colour_label_as_fsl.txt')
