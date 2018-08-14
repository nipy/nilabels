import os
from os.path import join as jph
import collections


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


def test_none():
    pass




