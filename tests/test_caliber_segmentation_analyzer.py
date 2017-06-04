import os
import numpy as np
from os.path import join as jph
from nose.tools import assert_raises, assert_almost_equal

from defs import root_dir
from labels_manager.caliber.segmentation_analyzer import SegmentationAnalyzer as SA


# global paths variable:

examples_folder = jph(root_dir, 'images_examples')

pfi_cubes = jph(examples_folder, 'cubes_in_space.nii.gz')
pfi_cubes_bin = jph(examples_folder, 'cubes_in_space_bin.nii.gz')

pfi_sandwich = jph(examples_folder, 'sandwich.nii.gz')


# segmentation analyzer loader:

def load_sa_scalar_binary():

    global examples_folder, pfi_cubes_bin, pfi_cubes

    for p in [pfi_cubes, pfi_cubes_bin, examples_folder]:
        if not os.path.exists(p):
            raise IOError('Run generate_images_examples.py to create the images examples before this, please.')

    # instantiate the SegmentationAnalyzer from caliber: scalar is the binary.
    sa = SA(pfi_segmentation=pfi_cubes, pfi_scalar_im=pfi_cubes_bin)

    return sa


def load_sa_segmentation_as_image():

    global examples_folder, pfi_cubes_bin, pfi_cubes

    for p in [pfi_cubes, pfi_cubes_bin, examples_folder]:
        if not os.path.exists(p):
            raise IOError('Run generate_images_examples.py to create the images examples before this, please.')

    # instantiate the SegmentationAnalyzer from caliber: scalar is the binary.
    sa = SA(pfi_segmentation=pfi_cubes, pfi_scalar_im=pfi_cubes)

    return sa


# -- First simple tests:


def test_get_total_volume_simple():

    sa = load_sa_scalar_binary()
    # The image contains 4 cubes of sides 11, 17, 19 and 9
    assert sa.get_total_volume() == 11 ** 3 + 17 ** 3 + 19 ** 3 + 9 ** 3


def test_get_volumes_per_label():

    sa = load_sa_scalar_binary()
    # The image contains 4 cubes of sides 11, 17, 19 and 9
    assert sa.get_volumes_per_label([[1,2]])[0] == 11 ** 3 + 17 ** 3


def test_get_average_below_volume():

    sa = load_sa_segmentation_as_image()

    sa.get_average_below_labels(1)

    assert sa.get_average_below_labels(1) == 11 ** 3 * 1 / float(11 ** 3)
    assert sa.get_average_below_labels(2) == 17 ** 3 * 2 / float(17 ** 3)
    assert sa.get_average_below_labels(3) == 19 ** 3 * 3 / float(19 ** 3)
    assert sa.get_average_below_labels(4) == 9 ** 3 * 4 / float(9 ** 3)
    np.testing.assert_array_equal(sa.get_average_below_labels([1, 3]),
                                  [11 ** 3 * 1 / float(11 ** 3), 19 ** 3 * 3 / float(19 ** 3)])
    assert sa.get_average_below_labels([[1, 3]]) == (11 ** 3 * 1 + 19 ** 3 * 3) / float(11 ** 3 + 19 ** 3)


# -- More challenging tests:


def test_input_error():
    with assert_raises(IOError):
        SA(pfi_segmentation=pfi_cubes, pfi_scalar_im=pfi_sandwich)


def test_modulating_affine_sandwich_total_volume():

    sa = SA(pfi_segmentation=pfi_sandwich, pfi_scalar_im=pfi_sandwich)
    sa.return_mm3 = True

    vol = sa.get_total_volume()
    expected_vol = 9 * 9 * 10 * 0.1 * 0.2 * 0.3
    assert_almost_equal(vol, expected_vol)

    sa.return_mm3 = False
    vol = sa.get_total_volume()
    expected_vol = 9 * 9 * 10
    assert_almost_equal(vol, expected_vol)



def test_modulating_affine_sandwich_volume_per_label():

    sa = SA(pfi_segmentation=pfi_sandwich, pfi_scalar_im=pfi_sandwich)
    sa.return_mm3 = True

    vol1 = sa.get_volumes_per_label([1])[0][0]
    expected_vol1 = 0
    assert_almost_equal(vol1, expected_vol1, places = 6)

    vol2 = sa.get_volumes_per_label([2])[0][0]
    expected_vol2 = 9 * 2 * 10 * 0.1 * 0.2 * 0.3
    assert_almost_equal(vol2, expected_vol2, places = 6)

    vol3 = sa.get_volumes_per_label([3])[0][0]
    expected_vol3 = 9 * 3 * 10 * 0.1 * 0.2 * 0.3
    assert_almost_equal(vol3, expected_vol3, places = 6)

    vol4 = sa.get_volumes_per_label([4])[0][0]
    expected_vol4 = 9 * 4 * 10 * 0.1 * 0.2 * 0.3
    assert_almost_equal(vol4, expected_vol4, places = 6)

    assert_almost_equal(sa.get_volumes_per_label([[2,3,4]])[0][0], sa.get_total_volume(), places = 6)

    vol2_3_2and3 = sa.get_volumes_per_label([2, 3, [2, 3]])[0][:]
    np.testing.assert_array_almost_equal(vol2_3_2and3,
                                         0.1 * 0.2 * 0.3 * np.array([2 * 9 * 10, 3 * 9 * 10, (2 + 3) * 9 * 10]))

    sa.return_mm3 = False

    vol1 = sa.get_volumes_per_label([1])[0][0]
    expected_vol1 = 0
    assert_almost_equal(vol1, expected_vol1, places=6)

    vol2 = sa.get_volumes_per_label([2])[0][0]
    expected_vol2 = 9 * 2 * 10
    assert_almost_equal(vol2, expected_vol2, places=6)

    vol3 = sa.get_volumes_per_label([3])[0][0]
    expected_vol3 = 9 * 3 * 10
    assert_almost_equal(vol3, expected_vol3, places=6)

    vol4 = sa.get_volumes_per_label([4])[0][0]
    expected_vol4 = 9 * 4 * 10
    assert_almost_equal(vol4, expected_vol4, places=6)

    assert_almost_equal(sa.get_volumes_per_label([[2, 3, 4]])[0][0], sa.get_total_volume(), places = 6)

    vol2_3_2and3 = sa.get_volumes_per_label([2, 3, [2, 3]])[0][:]
    np.testing.assert_array_equal(vol2_3_2and3, [2*9*10, 3*9*10, (2+3)*9*10])


def test_modulating_affine_sandwich_volume_below_label():

    sa = SA(pfi_segmentation=pfi_sandwich, pfi_scalar_im=pfi_sandwich)
    sa.return_mm3 = True

    av1 = sa.get_average_below_labels([1])[0]
    expected_av1 = 0
    assert_almost_equal(av1, expected_av1, places=6)

    av2 = sa.get_average_below_labels([2])[0]
    expected_av2 = 2.0  # 2.0 in each voxel: the mean over the voxels is still 2.0
    assert_almost_equal(av2, expected_av2, places=6)

    av2_3_2and3 = sa.get_average_below_labels([2, 3, [2, 3]])
    np.testing.assert_array_equal(av2_3_2and3, [2.0 * 2 * 9 * 10 / (2 * 9 * 10) ,
                                                3.0 * 3 * 9 * 10 / (3 * 9 * 10) ,
                                                (2.0 * 2 * 9 * 10 + 3.0 * 3 * 9 * 10) /  ((2 + 3) * 9 * 10) ])

test_modulating_affine_sandwich_volume_below_label()
