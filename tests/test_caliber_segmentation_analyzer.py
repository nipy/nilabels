import os
import numpy as np
from os.path import join as jph

from definitions import root_dir
from labels_manager.caliber.segmentation_analyzer import SegmentationAnalyzer as SA

# global paths variable:

examples_folder = jph(root_dir, 'images_examples')

pfi_im = jph(examples_folder, 'cubes_in_space.nii.gz')
pfi_im_bin = jph(examples_folder, 'cubes_in_space_bin.nii.gz')

# segmentation analyzer loader:

def load_sa_scalar_binary():

    global examples_folder, pfi_im_bin, pfi_im

    for p in [pfi_im, pfi_im_bin, examples_folder]:
        if not os.path.exists(p):
            raise IOError('Run generate_images_examples.py to create the images examples before this, please.')

    # instantiate the SegmentationAnalyzer from caliber: scalar is the binary.
    sa = SA(pfi_segmentation=pfi_im, pfi_scalar_im=pfi_im_bin)

    return sa


def load_sa_segmentation_as_image():

    global examples_folder, pfi_im_bin, pfi_im

    for p in [pfi_im, pfi_im_bin, examples_folder]:
        if not os.path.exists(p):
            raise IOError('Run generate_images_examples.py to create the images examples before this, please.')

    # instantiate the SegmentationAnalyzer from caliber: scalar is the binary.
    sa = SA(pfi_segmentation=pfi_im, pfi_scalar_im=pfi_im)

    return sa


# -- First simple tests:


def test_get_total_volume_simple():

    sa = load_sa_scalar_binary()
    # The image contains 4 cubes of sides 11, 17, 19 and 9
    assert sa.get_total_volume() == 11 ** 3 + 17 ** 3 + 19 ** 3 + 9 ** 3


def get_volumes_per_label():

    sa = load_sa_scalar_binary()
    # The image contains 4 cubes of sides 11, 17, 19 and 9
    assert sa.get_volumes_per_label([[1,2]])[0] == 11 ** 3 + 17 ** 3


def get_average_below_volume():

    sa = load_sa_segmentation_as_image()
    assert sa.get_average_below_labels(1) == 11 ** 3 * 1 / float(11 ** 3)
    assert sa.get_average_below_labels(2) == 17 ** 3 * 2 / float(17 ** 3)
    assert sa.get_average_below_labels(3) == 19 ** 3 * 3 / float(19 ** 3)
    assert sa.get_average_below_labels(4) == 9 ** 3 * 4 / float(9 ** 3)
    np.testing.assert_array_equal(sa.get_average_below_labels([1, 3]),
                                  [11 ** 3 * 1 / float(11 ** 3), 19 ** 3 * 3 / float(19 ** 3)])
    assert sa.get_average_below_labels([[1, 3]]) == (11 ** 3 * 1 + 19 ** 3 * 3) / float(11 ** 3 + 19 ** 3)




get_volumes_per_label()
get_average_below_volume()


# -- More challenging tests:

