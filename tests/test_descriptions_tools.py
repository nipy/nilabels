import numpy as np
import nibabel as nib
from os.path import join as jph

from definitions import root_dir

from nose.tools import assert_equals, assert_raises
from numpy.testing import assert_array_equal

''' Test descriptions.manipulate_descriptors.py'''
from labels_manager.tools.descriptions.manipulate_descriptors import *
# TODO