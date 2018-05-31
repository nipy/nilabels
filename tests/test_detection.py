from os.path import join as jph

import nibabel as nib
import numpy as np
from nose.tools import assert_equals, assert_raises
from numpy.testing import assert_array_equal


from LABelsToolkit.tools.detections.island_detection import island_for_label
