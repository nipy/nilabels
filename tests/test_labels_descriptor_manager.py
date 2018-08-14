import os
import collections
from os.path import join as jph

from nose.tools import assert_raises
from numpy.testing import assert_array_equal

from LABelsToolkit.tools.descriptions.label_descriptor_manager import LabelsDescriptorManager
from LABelsToolkit.tools.phantoms_generator import local_data_generator as ldg



def _create_data_set_for_tests():
    if not os.path.exists(jph(ldg.pfo_target_atlas, 'label_descriptor.txt')):
        print('Generating testing dataset. May take a while, but it is done only once!')
        ldg.generate_atlas_at_specified_folder()


def check_equal(l1, l2):
    return len(l1) == len(l2) and sorted(l1) == sorted(l2)


def test_basics_methods_labels_descriptor_manager_wrong_input_path():
    _create_data_set_for_tests()
    pfi_unexisting_label_descriptor_manager = 'zzz_path_to_spam'
    with assert_raises(IOError):
        LabelsDescriptorManager(pfi_unexisting_label_descriptor_manager)


def test_basics_methods_labels_descriptor_manager_wrong_input_convention():
    _create_data_set_for_tests()
    not_allowed_convention_name = 'jsut_spam'
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

    check_equal(ldm.dict_label_descriptor.keys(), dict_ld.keys())
    for k in ldm.dict_label_descriptor.keys():
        check_equal(ldm.dict_label_descriptor[k], dict_ld[k])


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

    # NOTE: works only with default 1.0 values, as fsl convention is less informative than itk-snap..
    check_equal(ldm_itk.dict_label_descriptor.keys(), ldm_fsl.dict_label_descriptor.keys())
    for k in ldm_itk.dict_label_descriptor.keys():
        check_equal(ldm_itk.dict_label_descriptor[k], ldm_fsl.dict_label_descriptor[k])

    os.system('rm {}'.format(jph(ldg.pfo_target_atlas, 'label_descriptor_fsl.txt')))
