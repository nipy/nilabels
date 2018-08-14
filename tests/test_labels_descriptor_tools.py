import os
from os.path import join as jph
import collections


from nose.tools import assert_equals, assert_raises
from numpy.testing import assert_array_equal

from LABelsToolkit.tools.descriptions.label_descriptor_manager import LabelsDescriptorManager
from LABelsToolkit.tools.descriptions.manipulate_descriptor import generate_dummy_label_descriptor, permute_labels_from_descriptor
from LABelsToolkit.tools.phantoms_generator import local_data_generator as ldg


# TESTING: labels permutations - permute_labels_from_descriptor


def test_permute_labels_from_descriptor_wrong_permutation():
    pass


def test_permute_labels_from_descriptor_check():
    pass


# TESTING: Generate dummy descriptor - generate_dummy_label_descriptor


pfi_testing_output = jph(ldg.pfo_target_atlas, 'label_descriptor.txt')


def test_generate_dummy_labels_descriptor_wrong_input1():
    with assert_raises(IOError):
        generate_dummy_label_descriptor(pfi_testing_output, list_labels=range(5), list_roi_names=['1', '2'])


def test_generate_dummy_labels_descriptor_wrong_input2():
    with assert_raises(IOError):
        generate_dummy_label_descriptor(pfi_testing_output, list_labels=range(5),
                                        list_roi_names=['1', '2', '3', '4', '5'], list_colors_triplets=[[0,0,0],
                                                                                                        [1,1,1]])




def none():
    print jph(ldg.pfo_target_atlas, 'label_descriptor.txt')