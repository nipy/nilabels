import numpy as np
import os
import nibabel as nib

from definitions import root_dir
from numpy.testing import assert_array_almost_equal

from nose.tools import assert_equals, assert_raises, assert_almost_equals
from numpy.testing import assert_array_equal, assert_almost_equal


from labels_manager.tools.aux_methods.morpological_tools import get_shell_for_given_radius, get_morphological_patch, \
    get_morphological_mask, get_patch_values


def test_get_patch_values_simple():
    # simple homemade mask:
    image = np.random.randint(0,10,(7,7))
    patch = np.zeros_like(image).astype(np.bool)
    patch[2,2] = True
    patch[2,3] = True
    patch[3,2] = True
    patch[3,3] = True

    vals = get_patch_values([2,2,2], image, morfo_mask=patch)
    assert_array_equal([image[2,2], image[2,3], image[3,2], image[3,3]], vals)
