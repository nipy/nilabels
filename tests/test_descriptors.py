import numpy as np
import os
import nibabel as nib

from definitions import root_dir
from numpy.testing import assert_array_almost_equal
import matplotlib.pyplot as plt

from nose.tools import assert_equals, assert_raises, assert_almost_equals
from numpy.testing import assert_array_equal, assert_almost_equal


from labels_managers.tools.measurements.linear import box_sides
from labels_managers.tools.aux_methods.utils import generate_o, generate_c

c = generate_c(omega=(25, 256, 256))
print box_sides(c)
