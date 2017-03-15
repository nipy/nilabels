import numpy as np
import os
import nibabel as nib

from definitions import root_dir
from numpy.testing import assert_array_almost_equal

from nose.tools import assert_equals, assert_raises, assert_almost_equals
from numpy.testing import assert_array_equal, assert_almost_equal

from labels_manager.tools.fusing_labels.weighted_sum_method import get_values_below_label
from labels_manager.tools.fusing_labels.weighted_sum_method import weighting_for_distance_from_trusted_label, \
    weighting_for_whole_label_values


def test_get_values_below_label():

    image = np.array(range(8*8)).reshape(8,8)
    mask = np.zeros_like(image)
    mask[2,2] = 1
    mask[2,3] = 1
    mask[3,2] = 1
    mask[3,3] = 1
    vals = get_values_below_label(image, mask, 1)
    assert_array_equal([image[2,2], image[2,3], image[3,2], image[3,3]], vals)


def test_weighting_for_whole_label_values():
    pass
