import os
from os.path import join as jph

import numpy as np
import nibabel as nib
from nose.tools import assert_raises, assert_almost_equal, assert_equal, assert_equals
from numpy.testing import assert_array_equal

from nilabel.tools.defs import root_dir
from nilabel.main import Nilabel as NiL
from nilabel.tools.aux_methods.label_descriptor_manager import LabelsDescriptorManager
from nilabel.tools.detections.check_imperfections import check_missing_labels


def test_check_missing_labels_paired():
    array = np.array([[[0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 7, 0, 0, 0, 0],
                       [0, 1, 0, 0, 7, 0, 0, 0, 0],
                       [0, 1, 0, 6, 7, 0, 0, 0, 0],
                       [0, 1, 0, 6, 0, 0, 2, 0, 0],
                       [0, 1, 0, 6, 0, 0, 2, 0, 0],
                       [0, 1, 0, 0, 0, 0, 2, 0, 0],
                       [0, 0, 0, 0, 0, 0, 2, 0, 0],
                       [0, 0, 0, 0, 0, 0, 2, 0, 0]],

                      [[0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 1, 0, 0, 0, 0, 0, 0, 0],
                       [0, 1, 0, 0, 0, 0, 0, 0, 0],
                       [0, 1, 0, 0, 0, 0, 2, 0, 0],
                       [0, 1, 0, 5, 0, 0, 2, 0, 0],
                       [0, 1, 0, 5, 0, 0, 2, 0, 0],
                       [0, 0, 0, 5, 0, 0, 2, 0, 0],
                       [0, 0, 0, 0, 0, 0, 2, 0, 0]]
                      ])
    im = nib.Nifti1Image(array, np.eye(4))


def test_check_missing_labels_unpaired():
    pass


def test_check_number_connected_components():
    pass
