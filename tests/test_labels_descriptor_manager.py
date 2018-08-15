import os
import collections
from os.path import join as jph
from nose.tools import assert_raises

from LABelsToolkit.tools.descriptions.label_descriptor_manager import LabelsDescriptorManager, \
    generate_dummy_label_descriptor
from LABelsToolkit.tools.phantoms_generator import local_data_generator as ldg


def _create_data_set_for_tests():
    if not os.path.exists(jph(ldg.pfo_target_atlas, 'label_descriptor.txt')):
        print('Generating testing dataset. May take a while, but it is done only once!')
        ldg.generate_atlas_at_specified_folder()


def check_list_equal(l1, l2):
    return len(l1) == len(l2) and sorted(l1) == sorted(l2)


def test_basics_methods_labels_descriptor_manager_wrong_input_path():
    _create_data_set_for_tests()
    pfi_unexisting_label_descriptor_manager = 'zzz_path_to_spam'
    with assert_raises(IOError):
        LabelsDescriptorManager(pfi_unexisting_label_descriptor_manager)


def test_basics_methods_labels_descriptor_manager_wrong_input_convention():
    _create_data_set_for_tests()
    not_allowed_convention_name = 'just_spam'
    with assert_raises(IOError):
        LabelsDescriptorManager(jph(ldg.pfo_target_atlas, 'label_descriptor.txt'), not_allowed_convention_name)


def test_basic_dict_input():
    _create_data_set_for_tests()
    dict_ld = collections.OrderedDict()
    dict_ld.update({0: [[0, 0, 0],   [1.0, 1.0, 1.0], 'Bkg']})
    dict_ld.update({1: [[255, 0, 0], [1.0, 1.0, 1.0], 'Skull']})
    dict_ld.update({2: [[0, 255, 0], [1.0, 1.0, 1.0], 'WM']})
    dict_ld.update({3: [[0, 0, 255], [1.0, 1.0, 1.0], 'GM']})
    dict_ld.update({4: [[255, 0, 255], [1.0, 1.0, 1.0], 'CSF']})

    ldm = LabelsDescriptorManager(jph(ldg.pfo_target_atlas, 'label_descriptor.txt'))

    check_list_equal(ldm.dict_label_descriptor.keys(), dict_ld.keys())
    for k in ldm.dict_label_descriptor.keys():
        check_list_equal(ldm.dict_label_descriptor[k], dict_ld[k])


def test_save_in_itk_snap_convention():
    _create_data_set_for_tests()
    ldm = LabelsDescriptorManager(jph(ldg.pfo_target_atlas, 'label_descriptor.txt'))
    ldm.save_label_descriptor(jph(ldg.pfo_target_atlas, 'label_descriptor2.txt'))

    f1 = open(jph(ldg.pfo_target_atlas, 'label_descriptor.txt'), 'r')
    f2 = open(jph(ldg.pfo_target_atlas, 'label_descriptor2.txt'), 'r')

    for l1, l2 in zip(f1.readlines(), f2.readlines()):
        assert l1 == l2

    os.system('rm {}'.format(jph(ldg.pfo_target_atlas, 'label_descriptor2.txt')))


def test_save_in_fsl_convention_reload_as_dict_and_compare():
    _create_data_set_for_tests()
    ldm_itk = LabelsDescriptorManager(jph(ldg.pfo_target_atlas, 'label_descriptor.txt'))
    # change convention
    ldm_itk.convention = 'fsl'
    ldm_itk.save_label_descriptor(jph(ldg.pfo_target_atlas, 'label_descriptor_fsl.txt'))

    ldm_fsl = LabelsDescriptorManager(jph(ldg.pfo_target_atlas, 'label_descriptor_fsl.txt'), convention='fsl')

    # NOTE: test works only with default 1.0 values - fsl convention is less informative than itk-snap..
    check_list_equal(ldm_itk.dict_label_descriptor.keys(), ldm_fsl.dict_label_descriptor.keys())
    for k in ldm_itk.dict_label_descriptor.keys():
        check_list_equal(ldm_itk.dict_label_descriptor[k], ldm_fsl.dict_label_descriptor[k])

    os.system('rm {}'.format(jph(ldg.pfo_target_atlas, 'label_descriptor_fsl.txt')))


# IN progress:

# TESTING: labels permutations - permute_labels_in_descriptor

def test_relabel():
    pass


def test_permute_labels_from_descriptor_wrong_input_permutation():
    pfi_input_label_descriptor = jph(ldg.pfo_target_atlas, 'label_descriptor.txt')
    perm = [[1, 2, 3], [1, 1]]
    ldm = LabelsDescriptorManager(pfi_input_label_descriptor)
    with assert_raises(IOError):
        ldm.permute_labels(perm)


def test_permute_labels_from_descriptor_check():
    pfi_input_label_descriptor = jph(ldg.pfo_target_atlas, 'label_descriptor.txt')
    perm = [[1, 2, 3], [1, 3, 2]]
    ldm = LabelsDescriptorManager(pfi_input_label_descriptor)
    ldm_new = ldm.permute_labels(perm)
    for k1, k2 in zip(perm[0], perm[1]):
        print k1, k2


# TESTING: Generate dummy descriptor - generate_dummy_label_descriptor


pfi_testing_output = jph(ldg.pfo_target_atlas, 'label_descriptor.txt')


def test_generate_dummy_labels_descriptor_wrong_input1():
    with assert_raises(IOError):
        generate_dummy_label_descriptor(pfi_testing_output, list_labels=range(5),
                                        list_roi_names=['1', '2'])


def test_generate_dummy_labels_descriptor_wrong_input2():
    with assert_raises(IOError):
        generate_dummy_label_descriptor(pfi_testing_output, list_labels=range(5),
                                        list_roi_names=['1', '2', '3', '4', '5'],
                                        list_colors_triplets=[[0,0,0], [1,1,1]])


def none():
    print jph(ldg.pfo_target_atlas, 'label_descriptor.txt')

