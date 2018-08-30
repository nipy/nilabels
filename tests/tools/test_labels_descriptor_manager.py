import collections
import os
from os.path import join as jph

from nose.tools import assert_raises

from nilabels.tools.aux_methods.label_descriptor_manager import LabelsDescriptorManager, \
    generate_dummy_label_descriptor


# PATH MANAGER

test_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
pfo_tmp_test = jph(test_dir, 'z_tmp_test')


# AUXILIARIES


def is_a_string_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


# DECORATORS


def write_and_erase_temporary_folder(test_func):
    def wrap(*args, **kwargs):
        # 1) Before: create folder
        os.system('mkdir {}'.format(pfo_tmp_test))
        # 2) Run test
        test_func(*args, **kwargs)
        # 3) After: delete folder and its content
        os.system('rm -r {}'.format(pfo_tmp_test))

    return wrap


def write_and_erase_temporary_folder_with_dummy_labels_descriptor(test_func):
    def wrap(*args, **kwargs):

        # 1) Before: create folder
        os.system('mkdir {}'.format(pfo_tmp_test))
        # 1bis) Then, generate image in the generated folder
        descriptor_dummy = \
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
    0     0    0    0        0  0  0    "background"
    1   255    0    0        1  1  1    "label one (l1)"
    2   204    0    0        1  1  1    "label two (l2)"
    3    51   51  255        1  1  1    "label three"
    4   102  102  255        1  1  1    "label four"
    5     0  204   51        1  1  1    "label five (l5)"
    6    51  255  102        1  1  1    "label six"
    7   255  255    0        1  1  1    "label seven"
    8   255  50    50        1  1  1    "label eight" """
        with open(jph(pfo_tmp_test, 'labels_descriptor.txt'), 'w+') as f:
            f.write(descriptor_dummy)
        # 2) Run test
        test_func(*args, **kwargs)
        # 3) After: delete folder and its content
        os.system('rm -r {}'.format(pfo_tmp_test))

    return wrap


# TESTING:

# --- > Testing generate dummy descriptor

@write_and_erase_temporary_folder
def test_generate_dummy_labels_descriptor_wrong_input1():
    with assert_raises(IOError):
        generate_dummy_label_descriptor(jph(pfo_tmp_test, 'label_descriptor.txt'), list_labels=range(5),
                                        list_roi_names=['1', '2'])


@write_and_erase_temporary_folder
def test_generate_dummy_labels_descriptor_wrong_input2():
    with assert_raises(IOError):
        generate_dummy_label_descriptor(jph(pfo_tmp_test, 'label_descriptor.txt'), list_labels=range(5),
                                        list_roi_names=['1', '2', '3', '4', '5'],
                                        list_colors_triplets=[[0, 0, 0], [1, 1, 1]])


@write_and_erase_temporary_folder
def test_generate_labels_descriptor_list_roi_names_None():
    d = generate_dummy_label_descriptor(jph(pfo_tmp_test, 'dummy_label_descriptor.txt'), list_labels=range(5),
                                        list_roi_names=None, list_colors_triplets=[[1, 1, 1], ] * 5)

    for k in d.keys():
        assert d[k][-1] == 'label {}'.format(k)


@write_and_erase_temporary_folder
def test_generate_labels_descriptor_list_colors_triplets_None():
    d = generate_dummy_label_descriptor(jph(pfo_tmp_test, 'dummy_label_descriptor.txt'), list_labels=range(5),
                                        list_roi_names=None, list_colors_triplets=[[1, 1, 1], ] * 5)
    for k in d.keys():
        assert len(d[k][1]) == 3


@write_and_erase_temporary_folder
def test_generate_labels_descriptor_general():
    list_labels         = [1, 2, 3, 4, 5]
    list_color_triplets = [[1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4], [5, 5, 5]]
    list_roi_names      = ['one', 'two', 'three', 'four', 'five']

    d = generate_dummy_label_descriptor(jph(pfo_tmp_test, 'dummy_label_descriptor.txt'), list_labels=list_labels,
                                        list_roi_names=list_roi_names, list_colors_triplets=list_color_triplets)
    for k_num, k in enumerate(d.keys()):
        assert int(k) == list_labels[k_num]
        assert d[k][0]  == list_color_triplets[k_num]
        assert d[k][-1] == list_roi_names[k_num]

# --- > Testing basics methods labels descriptor class manager


@write_and_erase_temporary_folder
def test_basics_methods_labels_descriptor_manager_wrong_input_path():

    pfi_unexisting_label_descriptor_manager = 'zzz_path_to_spam'
    with assert_raises(IOError):
        LabelsDescriptorManager(pfi_unexisting_label_descriptor_manager)


@write_and_erase_temporary_folder
def test_basics_methods_labels_descriptor_manager_wrong_input_convention():

    not_allowed_convention_name = 'just_spam'
    with assert_raises(IOError):
        LabelsDescriptorManager(jph(pfo_tmp_test, 'labels_descriptor.txt'), not_allowed_convention_name)


@write_and_erase_temporary_folder_with_dummy_labels_descriptor
def test_basic_dict_input():
    dict_ld = collections.OrderedDict()
    # note that in the dictionary there are no double quotes " ", but by default the strings are '"label name"'
    dict_ld.update({0: [[0, 0, 0],   [0, 0, 0], 'background']})
    dict_ld.update({1: [[255, 0, 0], [1, 1, 1], 'label one (l1)']})
    dict_ld.update({2: [[204, 0, 0], [1, 1, 1], 'label two (l2)']})
    dict_ld.update({3: [[51, 51, 255], [1, 1, 1], 'label three']})
    dict_ld.update({4: [[102, 102, 255], [1, 1, 1], 'label four']})
    dict_ld.update({5: [[0, 204, 51], [1, 1, 1], 'label five (l5)']})
    dict_ld.update({6: [[51, 255, 102], [1, 1, 1], 'label six']})
    dict_ld.update({7: [[255, 255, 0], [1, 1, 1], 'label seven']})
    dict_ld.update({8: [[255, 50, 50], [1, 1, 1], 'label eight']})

    ldm = LabelsDescriptorManager(jph(pfo_tmp_test, 'labels_descriptor.txt'))

    cmp(ldm.dict_label_descriptor.keys(), dict_ld.keys())
    for k in ldm.dict_label_descriptor.keys():
        cmp(ldm.dict_label_descriptor[k], dict_ld[k])
        assert cmp(ldm.dict_label_descriptor[k], dict_ld[k]) == 0


@write_and_erase_temporary_folder_with_dummy_labels_descriptor
def test_load_save_and_compare():
    ldm = LabelsDescriptorManager(jph(pfo_tmp_test, 'labels_descriptor.txt'))
    ldm.save_label_descriptor(jph(pfo_tmp_test, 'labels_descriptor2.txt'))

    f1 = open(jph(pfo_tmp_test, 'labels_descriptor.txt'), 'r')
    f2 = open(jph(pfo_tmp_test, 'labels_descriptor2.txt'), 'r')

    for l1, l2 in zip(f1.readlines(), f2.readlines()):
        split_l1 = [float(a) if is_a_string_number(a) else a for a in [a.strip() for a in l1.split(' ') if a is not '']]
        split_l2 = [float(b) if is_a_string_number(b) else b for b in [b.strip() for b in l2.split(' ') if b is not '']]
        assert cmp(split_l1, split_l2) == 0


@write_and_erase_temporary_folder_with_dummy_labels_descriptor
def test_save_in_fsl_convention_reload_as_dict_and_compare():
    ldm_itk = LabelsDescriptorManager(jph(pfo_tmp_test, 'labels_descriptor.txt'))
    # change convention
    ldm_itk.convention = 'fsl'
    ldm_itk.save_label_descriptor(jph(pfo_tmp_test, 'labels_descriptor_fsl.txt'))

    ldm_fsl = LabelsDescriptorManager(jph(pfo_tmp_test, 'labels_descriptor_fsl.txt'),
                                      labels_descriptor_convention='fsl')

    # NOTE: test works only with default 1.0 values - fsl convention is less informative than itk-snap..
    cmp(ldm_itk.dict_label_descriptor.keys(), ldm_fsl.dict_label_descriptor.keys())
    for k in ldm_itk.dict_label_descriptor.keys():
        cmp(ldm_itk.dict_label_descriptor[k], ldm_fsl.dict_label_descriptor[k])


# TESTING: labels permutations - permute_labels_in_descriptor


@write_and_erase_temporary_folder_with_dummy_labels_descriptor
def test_relabel_labels_descriptor():

    dict_expected = collections.OrderedDict()
    dict_expected.update({0: [[0, 0, 0], [0, 0, 0], 'background']})
    dict_expected.update({10: [[255, 0, 0], [1, 1, 1], 'label one (l1)']})
    dict_expected.update({11: [[204, 0, 0], [1, 1, 1], 'label two (l2)']})
    dict_expected.update({12: [[51, 51, 255], [1, 1, 1], 'label three']})
    dict_expected.update({4: [[102, 102, 255], [1, 1, 1], 'label four']})
    dict_expected.update({5: [[0, 204, 51], [1, 1, 1], 'label five (l5)']})
    dict_expected.update({6: [[51, 255, 102], [1, 1, 1], 'label six']})
    dict_expected.update({7: [[255, 255, 0], [1, 1, 1], 'label seven']})
    dict_expected.update({8: [[255, 50, 50], [1, 1, 1], 'label eight']})

    ldm_original = LabelsDescriptorManager(jph(pfo_tmp_test, 'labels_descriptor.txt'))

    old_labels = [1, 2, 3]
    new_labels = [10, 11, 12]

    ldm_relabelled = ldm_original.relabel(old_labels, new_labels, sort=True)

    cmp(dict_expected.keys(), ldm_relabelled.dict_label_descriptor.keys())
    for k in dict_expected.keys():
        cmp(dict_expected[k], ldm_relabelled.dict_label_descriptor[k])


@write_and_erase_temporary_folder_with_dummy_labels_descriptor
def test_relabel_labels_descriptor_with_merging():

    dict_expected = collections.OrderedDict()
    dict_expected.update({0: [[0, 0, 0], [0, 0, 0], 'background']})
    # dict_expected.update({1: [[255, 0, 0], [1, 1, 1], 'label one (l1)']})  # copied over label two
    dict_expected.update({1: [[204, 0, 0], [1, 1, 1], 'label two (l2)']})
    dict_expected.update({5: [[51, 51, 255], [1, 1, 1], 'label three']})
    dict_expected.update({4: [[102, 102, 255], [1, 1, 1], 'label four']})
    dict_expected.update({5: [[0, 204, 51], [1, 1, 1], 'label five (l5)']})
    dict_expected.update({6: [[51, 255, 102], [1, 1, 1], 'label six']})
    dict_expected.update({7: [[255, 255, 0], [1, 1, 1], 'label seven']})
    dict_expected.update({8: [[255, 50, 50], [1, 1, 1], 'label eight']})

    ldm_original = LabelsDescriptorManager(jph(pfo_tmp_test, 'labels_descriptor.txt'))

    old_labels = [1, 2, 3]
    new_labels = [1, 1, 5]

    ldm_relabelled = ldm_original.relabel(old_labels, new_labels, sort=True)

    cmp(dict_expected.keys(), ldm_relabelled.dict_label_descriptor.keys())
    for k in dict_expected.keys():
        cmp(dict_expected[k], ldm_relabelled.dict_label_descriptor[k])


@write_and_erase_temporary_folder_with_dummy_labels_descriptor
def test_permute_labels_from_descriptor_wrong_input_permutation():
    ldm = LabelsDescriptorManager(jph(pfo_tmp_test, 'labels_descriptor.txt'))
    perm = [[1, 2, 3], [1, 1]]

    with assert_raises(IOError):
        ldm.permute_labels(perm)


@write_and_erase_temporary_folder_with_dummy_labels_descriptor
def test_permute_labels_from_descriptor_check():
    dict_expected = collections.OrderedDict()
    dict_expected.update({0: [[0, 0, 0], [0, 0, 0], 'background']})
    dict_expected.update({3: [[255, 0, 0], [1, 1, 1], 'label one (l1)']})  # copied over label two
    dict_expected.update({4: [[204, 0, 0], [1, 1, 1], 'label two (l2)']})
    dict_expected.update({2: [[51, 51, 255], [1, 1, 1], 'label three']})
    dict_expected.update({1: [[102, 102, 255], [1, 1, 1], 'label four']})
    dict_expected.update({5: [[0, 204, 51], [1, 1, 1], 'label five (l5)']})
    dict_expected.update({6: [[51, 255, 102], [1, 1, 1], 'label six']})
    dict_expected.update({7: [[255, 255, 0], [1, 1, 1], 'label seven']})
    dict_expected.update({8: [[255, 50, 50], [1, 1, 1], 'label eight']})

    ldm_original = LabelsDescriptorManager(jph(pfo_tmp_test, 'labels_descriptor.txt'))
    perm = [[1, 2, 3, 4], [3, 4, 2, 1]]
    ldm_relabelled = ldm_original.permute_labels(perm)

    cmp(dict_expected.keys(), ldm_relabelled.dict_label_descriptor.keys())
    for k in dict_expected.keys():
        cmp(dict_expected[k], ldm_relabelled.dict_label_descriptor[k])


@write_and_erase_temporary_folder_with_dummy_labels_descriptor
def test_erase_labels():
    dict_expected = collections.OrderedDict()
    dict_expected.update({0: [[0, 0, 0], [0, 0, 0], 'background']})
    dict_expected.update({1: [[255, 0, 0], [1, 1, 1], 'label one (l1)']})  # copied over label two
    dict_expected.update({4: [[102, 102, 255], [1, 1, 1], 'label four']})
    dict_expected.update({5: [[0, 204, 51], [1, 1, 1], 'label five (l5)']})
    dict_expected.update({6: [[51, 255, 102], [1, 1, 1], 'label six']})
    dict_expected.update({8: [[255, 50, 50], [1, 1, 1], 'label eight']})

    ldm_original = LabelsDescriptorManager(jph(pfo_tmp_test, 'labels_descriptor.txt'))
    labels_to_erase = [2, 3, 7]
    ldm_relabelled = ldm_original.erase_labels(labels_to_erase)

    cmp(dict_expected.keys(), ldm_relabelled.dict_label_descriptor.keys())
    for k in dict_expected.keys():
        cmp(dict_expected[k], ldm_relabelled.dict_label_descriptor[k])


@write_and_erase_temporary_folder_with_dummy_labels_descriptor
def test_erase_labels_unexisting_labels():
    dict_expected = collections.OrderedDict()
    dict_expected.update({0: [[0, 0, 0], [0, 0, 0], 'background']})
    dict_expected.update({1: [[255, 0, 0], [1, 1, 1], 'label one (l1)']})  # copied over label two
    # dict_expected.update({2: [[204, 0, 0], [1, 1, 1], 'label two (l2)']})
    dict_expected.update({3: [[51, 51, 255], [1, 1, 1], 'label three']})
    # dict_expected.update({4: [[102, 102, 255], [1, 1, 1], 'label four']})
    dict_expected.update({5: [[0, 204, 51], [1, 1, 1], 'label five (l5)']})
    dict_expected.update({6: [[51, 255, 102], [1, 1, 1], 'label six']})
    dict_expected.update({7: [[255, 255, 0], [1, 1, 1], 'label seven']})
    dict_expected.update({8: [[255, 50, 50], [1, 1, 1], 'label eight']})

    ldm_original = LabelsDescriptorManager(jph(pfo_tmp_test, 'labels_descriptor.txt'))
    labels_to_erase = [2, 4, 16, 32]
    ldm_relabelled = ldm_original.erase_labels(labels_to_erase)

    cmp(dict_expected.keys(), ldm_relabelled.dict_label_descriptor.keys())
    for k in dict_expected.keys():
        cmp(dict_expected[k], ldm_relabelled.dict_label_descriptor[k])


@write_and_erase_temporary_folder_with_dummy_labels_descriptor
def test_assign_all_other_labels_the_same_value():
    dict_expected = collections.OrderedDict()
    dict_expected.update({0: [[0, 0, 0], [0, 0, 0], 'background']})
    dict_expected.update({1: [[255, 0, 0], [1, 1, 1], 'label one (l1)']})  # copied over label two
    dict_expected.update({4: [[102, 102, 255], [1, 1, 1], 'label four']})
    dict_expected.update({7: [[255, 255, 0], [1, 1, 1], 'label seven']})

    ldm_original = LabelsDescriptorManager(jph(pfo_tmp_test, 'labels_descriptor.txt'))
    labels_to_keep = [0, 1, 4, 7]
    other_value = 0
    ldm_relabelled = ldm_original.assign_all_other_labels_the_same_value(labels_to_keep, other_value)

    cmp(dict_expected.keys(), ldm_relabelled.dict_label_descriptor.keys())
    for k in dict_expected.keys():
        cmp(dict_expected[k], ldm_relabelled.dict_label_descriptor[k])


@write_and_erase_temporary_folder_with_dummy_labels_descriptor
def test_keep_one_label():
    dict_expected = collections.OrderedDict()
    dict_expected.update({3: [[51, 51, 255], [1, 1, 1], 'label three']})

    ldm_original = LabelsDescriptorManager(jph(pfo_tmp_test, 'labels_descriptor.txt'))
    label_to_keep = 3
    ldm_relabelled = ldm_original.keep_one_label(label_to_keep)

    cmp(dict_expected.keys(), ldm_relabelled.dict_label_descriptor.keys())
    for k in dict_expected.keys():
        cmp(dict_expected[k], ldm_relabelled.dict_label_descriptor[k])


if __name__ == '__main__':
    test_generate_dummy_labels_descriptor_wrong_input1()
    test_generate_dummy_labels_descriptor_wrong_input2()
    test_generate_labels_descriptor_list_roi_names_None()
    test_generate_labels_descriptor_list_colors_triplets_None()
    test_generate_labels_descriptor_general()

    test_basics_methods_labels_descriptor_manager_wrong_input_path()
    test_basics_methods_labels_descriptor_manager_wrong_input_convention()
    test_basic_dict_input()
    test_load_save_and_compare()
    test_save_in_fsl_convention_reload_as_dict_and_compare()

    test_relabel_labels_descriptor()
    test_relabel_labels_descriptor_with_merging()
    test_permute_labels_from_descriptor_wrong_input_permutation()
    test_permute_labels_from_descriptor_check()
    test_erase_labels()

    test_erase_labels_unexisting_labels()
    test_assign_all_other_labels_the_same_value()
    test_keep_one_label()
