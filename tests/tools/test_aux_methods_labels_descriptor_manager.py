import collections
from os.path import join as jph

import pytest

from nilabels.tools.aux_methods.label_descriptor_manager import LabelsDescriptorManager, \
    generate_dummy_label_descriptor


from tests.tools.decorators_tools import write_and_erase_temporary_folder, pfo_tmp_test, \
    write_and_erase_temporary_folder_with_dummy_labels_descriptor, is_a_string_number, \
    write_and_erase_temporary_folder_with_left_right_dummy_labels_descriptor


# TESTING:

# --- > Testing generate dummy descriptor

@write_and_erase_temporary_folder
def test_generate_dummy_labels_descriptor_wrong_input1():
    with pytest.raises(IOError):
        generate_dummy_label_descriptor(jph(pfo_tmp_test, 'labels_descriptor.txt'), list_labels=range(5),
                                        list_roi_names=['1', '2'])


@write_and_erase_temporary_folder
def test_generate_dummy_labels_descriptor_wrong_input2():
    with pytest.raises(IOError):
        generate_dummy_label_descriptor(jph(pfo_tmp_test, 'labels_descriptor.txt'), list_labels=range(5),
                                        list_roi_names=['1', '2', '3', '4', '5'],
                                        list_colors_triplets=[[0, 0, 0], [1, 1, 1]])


@write_and_erase_temporary_folder
def test_generate_labels_descriptor_list_roi_names_None():
    d = generate_dummy_label_descriptor(jph(pfo_tmp_test, 'dummy_labels_descriptor.txt'), list_labels=range(5),
                                        list_roi_names=None, list_colors_triplets=[[1, 1, 1], ] * 5)

    for k in d.keys():
        assert d[k][-1] == 'label {}'.format(k)


@write_and_erase_temporary_folder
def test_generate_labels_descriptor_list_colors_triplets_None():
    d = generate_dummy_label_descriptor(jph(pfo_tmp_test, 'dummy_labels_descriptor.txt'), list_labels=range(5),
                                        list_roi_names=None, list_colors_triplets=[[1, 1, 1], ] * 5)
    for k in d.keys():
        assert len(d[k][1]) == 3


@write_and_erase_temporary_folder
def test_generate_none_list_colour_triples():
    generate_dummy_label_descriptor(jph(pfo_tmp_test, 'labels_descriptor.txt'), list_labels=range(5),
                                    list_roi_names=['1', '2', '3', '4', '5'], list_colors_triplets=None)
    loaded_dummy_ldm = LabelsDescriptorManager(jph(pfo_tmp_test, 'labels_descriptor.txt'))
    for k in loaded_dummy_ldm.dict_label_descriptor.keys():
        assert len(loaded_dummy_ldm.dict_label_descriptor[k][0]) == 3
        for k_rgb in loaded_dummy_ldm.dict_label_descriptor[k][0]:
            assert 0 <= k_rgb < 256


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
    with pytest.raises(IOError):
        LabelsDescriptorManager(pfi_unexisting_label_descriptor_manager)


@write_and_erase_temporary_folder
def test_basics_methods_labels_descriptor_manager_wrong_input_convention():

    not_allowed_convention_name = 'just_spam'
    with pytest.raises(IOError):
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

    for k in ldm.dict_label_descriptor.keys():
        assert ldm.dict_label_descriptor[k] == dict_ld[k]


@write_and_erase_temporary_folder_with_dummy_labels_descriptor
def test_load_save_and_compare():
    ldm = LabelsDescriptorManager(jph(pfo_tmp_test, 'labels_descriptor.txt'))
    ldm.save_label_descriptor(jph(pfo_tmp_test, 'labels_descriptor2.txt'))

    f1 = open(jph(pfo_tmp_test, 'labels_descriptor.txt'), 'r')
    f2 = open(jph(pfo_tmp_test, 'labels_descriptor2.txt'), 'r')

    for l1, l2 in zip(f1.readlines(), f2.readlines()):
        split_l1 = [float(a) if is_a_string_number(a) else a for a in [a.strip() for a in l1.split(' ') if a is not '']]
        split_l2 = [float(b) if is_a_string_number(b) else b for b in [b.strip() for b in l2.split(' ') if b is not '']]
        assert split_l1 == split_l2


@write_and_erase_temporary_folder_with_dummy_labels_descriptor
def test_save_in_fsl_convention_reload_as_dict_and_compare():
    ldm_itk = LabelsDescriptorManager(jph(pfo_tmp_test, 'labels_descriptor.txt'))
    # change convention
    ldm_itk.convention = 'fsl'
    ldm_itk.save_label_descriptor(jph(pfo_tmp_test, 'labels_descriptor_fsl.txt'))

    ldm_fsl = LabelsDescriptorManager(jph(pfo_tmp_test, 'labels_descriptor_fsl.txt'),
                                      labels_descriptor_convention='fsl')

    # NOTE: test works only with default 1.0 values - fsl convention is less informative than itk-snap..
    for k in ldm_itk.dict_label_descriptor.keys():
        ldm_itk.dict_label_descriptor[k] ==  ldm_fsl.dict_label_descriptor[k]


@write_and_erase_temporary_folder_with_dummy_labels_descriptor
def test_signature_for_variable_convention_wrong_input():
    with pytest.raises(IOError):
        LabelsDescriptorManager(jph(pfo_tmp_test, 'labels_descriptor.txt'),
                                    labels_descriptor_convention='spam')


@write_and_erase_temporary_folder_with_dummy_labels_descriptor
def test_signature_for_variable_convention_wrong_input_after_initialisation():
    my_ldm = LabelsDescriptorManager(jph(pfo_tmp_test, 'labels_descriptor.txt'),
                                     labels_descriptor_convention='itk-snap')

    with pytest.raises(IOError):
        my_ldm.convention = 'spam'
        my_ldm.save_label_descriptor(jph(pfo_tmp_test, 'labels_descriptor_again.txt'))

# --> Testing labels permutations - permute_labels_in_descriptor


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

    for k in dict_expected.keys():
        dict_expected[k] == ldm_relabelled.dict_label_descriptor[k]


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

    for k in dict_expected.keys():
        dict_expected[k] == ldm_relabelled.dict_label_descriptor[k]


@write_and_erase_temporary_folder_with_dummy_labels_descriptor
def test_permute_labels_from_descriptor_wrong_input_permutation():
    ldm = LabelsDescriptorManager(jph(pfo_tmp_test, 'labels_descriptor.txt'))
    perm = [[1, 2, 3], [1, 1]]
    with pytest.raises(IOError):
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

    for k in dict_expected.keys():
        dict_expected[k] == ldm_relabelled.dict_label_descriptor[k]


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

    for k in dict_expected.keys():
        assert dict_expected[k] == ldm_relabelled.dict_label_descriptor[k]


# -> multi-labels dict


@write_and_erase_temporary_folder_with_left_right_dummy_labels_descriptor
def test_save_multi_labels_descriptor_custom():
    # load it into a labels descriptor manager
    ldm_lr = LabelsDescriptorManager(jph(pfo_tmp_test, 'labels_descriptor_RL.txt'))

    # save it as labels descriptor text file
    pfi_multi_ld = jph(pfo_tmp_test, 'multi_labels_descriptor_LR.txt')
    ldm_lr.save_as_multi_label_descriptor(pfi_multi_ld)

    # expected lines:
    expected_lines = [['background', 0],
                      ['label A Left', 1], ['label A Right', 2], ['label A', 1, 2],
                      ['label B Left', 3], ['label B Right', 4], ['label B', 3, 4],
                      ['label C', 5], ['label D', 6],
                      ['label E Left', 7], ['label E Right', 8], ['label E', 7, 8]]

    # load saved labels descriptor
    with open(pfi_multi_ld, 'r') as g:
        multi_ld_lines = g.readlines()

    # modify as list of lists as the expected lines.
    multi_ld_lines_a_list_of_lists = [[int(a) if a.isdigit() else a
                                         for a in [n.strip() for n in m.split('&') if not n.startswith('#')]]
                                         for m in multi_ld_lines]
    # Compare:
    for li1, li2 in zip(expected_lines, multi_ld_lines_a_list_of_lists):
        assert li1 == li2


@write_and_erase_temporary_folder_with_left_right_dummy_labels_descriptor
def test_get_multi_label_dict_standard_combine():
    ldm_lr = LabelsDescriptorManager(jph(pfo_tmp_test, 'labels_descriptor_RL.txt'))

    multi_labels_dict_from_ldm = ldm_lr.get_multi_label_dict(combine_right_left=True)

    expected_multi_labels_dict = collections.OrderedDict()
    expected_multi_labels_dict.update({'background': [0]})
    expected_multi_labels_dict.update({'label A Left': [1]})
    expected_multi_labels_dict.update({'label A Right': [2]})
    expected_multi_labels_dict.update({'label A': [1, 2]})
    expected_multi_labels_dict.update({'label B Left': [3]})
    expected_multi_labels_dict.update({'label B Right': [4]})
    expected_multi_labels_dict.update({'label B': [3, 4]})
    expected_multi_labels_dict.update({'label C': [5]})
    expected_multi_labels_dict.update({'label D': [6]})
    expected_multi_labels_dict.update({'label E Left': [7]})
    expected_multi_labels_dict.update({'label E Right': [8]})
    expected_multi_labels_dict.update({'label E': [7, 8]})

    for k1, k2 in zip(multi_labels_dict_from_ldm.keys(), expected_multi_labels_dict.keys()):
        assert k1 == k2
        assert multi_labels_dict_from_ldm[k1] == expected_multi_labels_dict[k2]


@write_and_erase_temporary_folder_with_left_right_dummy_labels_descriptor
def test_get_multi_label_dict_standard_not_combine():
    ldm_lr = LabelsDescriptorManager(jph(pfo_tmp_test, 'labels_descriptor_RL.txt'))

    multi_labels_dict_from_ldm = ldm_lr.get_multi_label_dict(combine_right_left=False)

    expected_multi_labels_dict = collections.OrderedDict()
    expected_multi_labels_dict.update({'background': [0]})
    expected_multi_labels_dict.update({'label A Left': [1]})
    expected_multi_labels_dict.update({'label A Right': [2]})
    expected_multi_labels_dict.update({'label B Left': [3]})
    expected_multi_labels_dict.update({'label B Right': [4]})
    expected_multi_labels_dict.update({'label C': [5]})
    expected_multi_labels_dict.update({'label D': [6]})
    expected_multi_labels_dict.update({'label E Left': [7]})
    expected_multi_labels_dict.update({'label E Right': [8]})

    for k1, k2 in zip(multi_labels_dict_from_ldm.keys(), expected_multi_labels_dict.keys()):
        assert k1 == k2
        assert multi_labels_dict_from_ldm[k1] == expected_multi_labels_dict[k2]


@write_and_erase_temporary_folder
def test_save_multi_labels_descriptor_custom_test_robustness():

    # save this as file multi labels descriptor then read and check that it went in order!
    d = collections.OrderedDict()
    d.update({0: [[0, 0, 0],       [0, 0, 0], 'background']})
    d.update({1: [[255, 0, 0],     [1, 1, 1], 'label A Right']})
    d.update({2: [[204, 0, 0],     [1, 1, 1], 'label A Left']})
    d.update({3: [[51, 51, 255],   [1, 1, 1], 'label B left']})
    d.update({4: [[102, 102, 255], [1, 1, 1], 'label B Right']})
    d.update({5: [[0, 204, 51],    [1, 1, 1], 'label C ']})
    d.update({6: [[51, 255, 102],  [1, 1, 1], 'label D Right']})  # unpaired label
    d.update({7: [[255, 255, 0],   [1, 1, 1], 'label E right  ']})  # small r and spaces
    d.update({8: [[255, 50, 50],   [1, 1, 1], 'label E Left  ']})  # ... paired with small l and spaces

    with open(jph(pfo_tmp_test, 'labels_descriptor_RL.txt'), 'w+') as f:
        for j in d.keys():
            line = '{0: >5}{1: >6}{2: >6}{3: >6}{4: >9}{5: >6}{6: >6}    "{7}"\n'.format(
                j, d[j][0][0], d[j][0][1], d[j][0][2], d[j][1][0], d[j][1][1], d[j][1][2], d[j][2])
            f.write(line)

    # load it with an instance of LabelsDescriptorManager
    ldm_lr = LabelsDescriptorManager(jph(pfo_tmp_test, 'labels_descriptor_RL.txt'))
    multi_labels_dict_from_ldm = ldm_lr.get_multi_label_dict(combine_right_left=True)

    expected_multi_labels_dict = collections.OrderedDict()
    expected_multi_labels_dict.update({'background': [0]})
    expected_multi_labels_dict.update({'label A Right': [1]})
    expected_multi_labels_dict.update({'label A Left': [2]})
    expected_multi_labels_dict.update({'label A': [1, 2]})
    expected_multi_labels_dict.update({'label B left': [3]})
    expected_multi_labels_dict.update({'label B Right': [4]})
    expected_multi_labels_dict.update({'label C': [5]})
    expected_multi_labels_dict.update({'label D Right': [6]})
    expected_multi_labels_dict.update({'label E right': [7]})
    expected_multi_labels_dict.update({'label E Left': [8]})

    for k1, k2 in zip(multi_labels_dict_from_ldm.keys(), expected_multi_labels_dict.keys()):
        assert k1 == k2
        assert multi_labels_dict_from_ldm[k1] == expected_multi_labels_dict[k2]


# -> erase, assign and keep only one label relabeller.


@write_and_erase_temporary_folder_with_dummy_labels_descriptor
def test_relabel_standard():
    dict_expected = collections.OrderedDict()
    dict_expected.update({0: [[0, 0, 0], [0, 0, 0], 'background']})
    dict_expected.update({1: [[255, 0, 0], [1, 1, 1], 'label one (l1)']})
    dict_expected.update({9: [[204, 0, 0], [1, 1, 1], 'label two (l2)']})
    dict_expected.update({3: [[51, 51, 255], [1, 1, 1], 'label three']})
    dict_expected.update({10: [[102, 102, 255], [1, 1, 1], 'label four']})
    dict_expected.update({5: [[0, 204, 51], [1, 1, 1], 'label five (l5)']})
    dict_expected.update({6: [[51, 255, 102], [1, 1, 1], 'label six']})
    dict_expected.update({7: [[255, 255, 0], [1, 1, 1], 'label seven']})
    dict_expected.update({8: [[255, 50, 50], [1, 1, 1], 'label eight']})

    ldm_original = LabelsDescriptorManager(jph(pfo_tmp_test, 'labels_descriptor.txt'))
    old_labels = [2, 4]
    new_labels = [9, 10]
    ldm_relabelled = ldm_original.relabel(old_labels, new_labels)

    for k in dict_expected.keys():
        assert dict_expected[k] == ldm_relabelled.dict_label_descriptor[k]


@write_and_erase_temporary_folder_with_dummy_labels_descriptor
def test_relabel_bad_input():

    ldm_original = LabelsDescriptorManager(jph(pfo_tmp_test, 'labels_descriptor.txt'))
    old_labels = [2, 4, 180]
    new_labels = [9, 10, 12]

    with pytest.raises(IOError):
        ldm_original.relabel(old_labels, new_labels)


@write_and_erase_temporary_folder_with_dummy_labels_descriptor
def test_erase_labels_unexisting_labels():
    dict_expected = collections.OrderedDict()
    dict_expected.update({0: [[0, 0, 0], [0, 0, 0], 'background']})
    dict_expected.update({1: [[255, 0, 0], [1, 1, 1], 'label one (l1)']})
    dict_expected.update({3: [[51, 51, 255], [1, 1, 1], 'label three']})
    dict_expected.update({5: [[0, 204, 51], [1, 1, 1], 'label five (l5)']})
    dict_expected.update({6: [[51, 255, 102], [1, 1, 1], 'label six']})
    dict_expected.update({7: [[255, 255, 0], [1, 1, 1], 'label seven']})
    dict_expected.update({8: [[255, 50, 50], [1, 1, 1], 'label eight']})

    ldm_original = LabelsDescriptorManager(jph(pfo_tmp_test, 'labels_descriptor.txt'))
    labels_to_erase = [2, 4, 16, 32]
    ldm_relabelled = ldm_original.erase_labels(labels_to_erase)

    for k in dict_expected.keys():
        assert dict_expected[k] == ldm_relabelled.dict_label_descriptor[k]


@write_and_erase_temporary_folder_with_dummy_labels_descriptor
def test_assign_all_other_labels_the_same_value():
    dict_expected = collections.OrderedDict()
    dict_expected.update({0: [[0, 0, 0], [0, 0, 0], 'background']})  # Possible bug
    dict_expected.update({1: [[255, 0, 0], [1, 1, 1], 'label one (l1)']})  # copied over label two
    dict_expected.update({4: [[102, 102, 255], [1, 1, 1], 'label four']})
    dict_expected.update({7: [[255, 255, 0], [1, 1, 1], 'label seven']})

    ldm_original = LabelsDescriptorManager(jph(pfo_tmp_test, 'labels_descriptor.txt'))
    labels_to_keep = [0, 1, 4, 7]
    other_value = 12
    ldm_relabelled = ldm_original.assign_all_other_labels_the_same_value(labels_to_keep, other_value)

    print(dict_expected)
    print(ldm_relabelled.dict_label_descriptor)
    for k in dict_expected.keys():
        print()
        print(dict_expected[k])
        print(ldm_relabelled.dict_label_descriptor[k])
        assert dict_expected[k] == ldm_relabelled.dict_label_descriptor[k]


@write_and_erase_temporary_folder_with_dummy_labels_descriptor
def test_keep_one_label():
    dict_expected = collections.OrderedDict()
    dict_expected.update({3: [[51, 51, 255], [1, 1, 1], 'label three']})

    ldm_original = LabelsDescriptorManager(jph(pfo_tmp_test, 'labels_descriptor.txt'))
    label_to_keep = 3
    ldm_relabelled = ldm_original.keep_one_label(label_to_keep)

    for k in dict_expected.keys():
        assert dict_expected[k] == ldm_relabelled.dict_label_descriptor[k]


if __name__ == '__main__':
    test_generate_dummy_labels_descriptor_wrong_input1()
    test_generate_dummy_labels_descriptor_wrong_input2()
    test_generate_labels_descriptor_list_roi_names_None()
    test_generate_labels_descriptor_list_colors_triplets_None()
    test_generate_none_list_colour_triples()
    test_generate_labels_descriptor_general()

    test_basics_methods_labels_descriptor_manager_wrong_input_path()
    test_basics_methods_labels_descriptor_manager_wrong_input_convention()
    test_basic_dict_input()
    test_load_save_and_compare()
    test_save_in_fsl_convention_reload_as_dict_and_compare()
    test_signature_for_variable_convention_wrong_input()
    test_signature_for_variable_convention_wrong_input_after_initialisation()

    test_relabel_labels_descriptor()
    test_relabel_labels_descriptor_with_merging()
    test_permute_labels_from_descriptor_wrong_input_permutation()
    test_permute_labels_from_descriptor_check()
    test_erase_labels()

    test_save_multi_labels_descriptor_custom()

    test_get_multi_label_dict_standard_combine()
    test_get_multi_label_dict_standard_not_combine()
    test_save_multi_labels_descriptor_custom_test_robustness()

    test_relabel_standard()
    test_relabel_bad_input()
    test_erase_labels_unexisting_labels()
    test_assign_all_other_labels_the_same_value()
    test_keep_one_label()
