import os
import collections
from os.path import join as jph

from nose.tools import assert_equals, assert_raises
from numpy.testing import assert_array_equal

from LABelsToolkit.tools.descriptions.label_descriptor_manager import LabelsDescriptorManager
from LABelsToolkit.tools.descriptions.manipulate_descriptor import generate_dummy_label_descriptor
from LABelsToolkit.tools.phantoms_generator import local_data_generator as ldg


def _create_data_set_for_tests():
    if not os.path.exists(jph(ldg.pfo_target_atlas, 'label_descriptor.txt')):
        print('Generating testing dataset. May take a while, but it is done only once!')
        ldg.generate_atlas_at_specified_folder()
        assert os.path.exists(jph(ldg.pfo_target_atlas, 'label_descriptor.txt'))


def test_basics_methods_labels_descriptor_manager_wrong_input_path():
    _create_data_set_for_tests()
    pfi_unexisting_label_descriptor_manager = 'zzz_path_to_spam'
    with assert_raises(IOError):
        LabelsDescriptorManager(pfi_unexisting_label_descriptor_manager)


def test_basics_methods_labels_descriptor_manager_wrong_input_convention():
    _create_data_set_for_tests()
    not_allowed_convention_name = 'jsut_spam'
    with assert_raises(IOError):
        LabelsDescriptorManager(ldg.pfo_target_atlas, not_allowed_convention_name)


def test_basic_dict_input():
    _create_data_set_for_tests()
    dict_ld = collections.OrderedDict()
    dict_ld.update({0: [[0, 0, 0], [1.0, 1.0, 1.0], 'Bkg']})
    dict_ld.update({1: [[255, 0, 0], [1.0, 1.0, 1.0], 'Skull']})
    dict_ld.update({2: [[0, 255, 0], [1.0, 1.0, 1.0], 'WM']})
    dict_ld.update({3: [[0, 0, 255], [1.0, 1.0, 1.0], 'GM']})
    dict_ld.update({4: [[255, 0, 255], [1.0, 1.0, 1.0], 'CSF']})



