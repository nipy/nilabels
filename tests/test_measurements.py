import numpy as np
import os
import nibabel as nib

from definitions import root_dir
from numpy.testing import assert_array_almost_equal

from nose.tools import assert_equals, assert_raises, assert_almost_equals
from numpy.testing import assert_array_equal, assert_almost_equal


from labels_manager.tools.measurements.distances import lncc_distance


def test_simple_patches_values_lncc():

    patch1 = np.array([1, 2, 3])
    patch2 = np.array([1, 2, 3])

    assert lncc_distance(patch1, patch2) == 1.0

    patch1 = np.array([1, 2, 3])
    patch2 = np.array([0, 0, 0])

    assert lncc_distance(patch1, patch2) == 0.0

    patch1 = np.array([0, 0, 0])
    patch2 = np.array([1, 2, 3])

    assert lncc_distance(patch1, patch2) == 0.0

    patch1 = np.array([1, 0, 0])
    patch2 = np.array([0, 1, 0])

    assert lncc_distance(patch1, patch2) == 0.0

