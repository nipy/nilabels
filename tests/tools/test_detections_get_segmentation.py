import numpy as np
from numpy.testing import assert_array_equal

from nilabels.tools.detections.get_segmentation import intensity_segmentation, otsu_threshold, MoG_array


#  ----- Test get segmentation ----


def test_intensity_segmentation_1():
    im_array = np.random.randint(0, 5, [10, 10], np.uint8)
    output_segm = intensity_segmentation(im_array)
    # if the input is a segmentation with 5 labels, the segmentation is the input.
    assert_array_equal(im_array, output_segm)


def test_intensity_segmentation_2():

    seed_segm  = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5])
    seed_image = np.linspace(0, 5, len(seed_segm))

    segm  = np.stack([seed_segm, ]*6)
    image = np.stack([seed_image, ]*6)

    output_segm = intensity_segmentation(image, num_levels=6)
    assert_array_equal(segm, output_segm)

    segm_transposed  = segm.T
    image_transposed = image.T

    output_segm_transposed = intensity_segmentation(image_transposed, num_levels=6)
    assert_array_equal(segm_transposed, output_segm_transposed)


def test_otsu_threshold_bad_input():
    # TODO
    pass

def test_otsu_threshold_1():
    # TODO
    pass


def test_otsu_threshold_2():
    # TODO
    pass


def test_MoG_array_bad_input():
    # TODO
    pass


def test_MoG_array_1():
    # TODO
    pass


def test_MoG_array_2():
    # TODO
    pass
