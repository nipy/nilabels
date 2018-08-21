import numpy as np
from numpy.testing import assert_array_equal

from nilabel.tools.detections.island_detection import island_for_label
from nilabel.tools.detections.get_segmentation import intensity_segmentation, otsu_threshold, MoG_array


def test_island_for_label_ok_input():
    in_data = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                        [0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0],
                        [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                        [0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

    expected_ans_False = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 3, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0],
                          [0, 3, 0, 0, 1, 1, 0, 0, 2, 0, 0, 0],
                          [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                          [0, 5, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]

    expected_ans_True = [[0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                         [0, -1,  0,  0,  0,  0,  0,  0, -1,  0,  0,  0],
                         [0, -1,  0,  0,  1,  1,  0,  0, -1,  0,  0,  0],
                         [0,  0,  0,  1,  1,  1,  0,  0,  0,  0,  0,  0],
                         [0, -1,  0,  1,  0,  1,  0,  0,  0,  0,  0,  0],
                         [0,  0,  0,  1,  1,  1,  1,  0,  0,  0,  0,  0],
                         [0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                         [0, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                         [0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0]]

    ans_False = island_for_label(in_data, 1)

    ans_True = island_for_label(in_data, 1, m=1)

    assert_array_equal(expected_ans_False, ans_False)
    assert_array_equal(expected_ans_True, ans_True)


def test_island_for_label_no_label_in_input():
    in_data = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                        [0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0],
                        [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                        [0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

    bypassed_ans = island_for_label(in_data, 2)
    assert_array_equal(bypassed_ans, in_data)


def test_island_for_label_multiple_components_for_more_than_one_m():
    in_data = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                        [0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                        [0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1],
                        [0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1],
                        [0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1],
                        [0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0],
                        [0, 0, 0, 2, 2, 0, 0, 0, 0, 0, 0, 0]])

    expected_output = np.array(
                       [[0,  0, 0, 0, 0, 0, 0, 0,  0,  0, -1, 0],
                        [0,  2, 0, 0, 0, 0, 0, 0, -1,  0,  0, 0],
                        [0,  2, 0, 0, 1, 1, 0, 0, -1,  0,  0, 3],
                        [0,  2, 0, 1, 1, 1, 0, 0,  0,  0,  0, 3],
                        [0,  2, 0, 1, 0, 1, 0, 0,  0,  0,  0, 3],
                        [0,  2, 0, 1, 1, 1, 1, 0,  0,  0,  0, 3],
                        [0,  0, 0, 0, 0, 0, 0, 0,  0,  0,  0, 0],
                        [0, -1, 0, 0, 0, 0, 0, 0, -1, -1, -1, 0],
                        [0,  0, 0, 0, 0, 0, 0, 0,  0,  0,  0, 0]])

    ans = island_for_label(in_data, 1, m=3, special_label=-1)

    assert_array_equal(expected_output, ans)


def test_island_for_label_multiple_components_for_more_than_one_m_again():
    in_data = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                        [0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                        [0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1],
                        [0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1],
                        [0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1],
                        [0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0],
                        [0, 0, 0, 2, 2, 0, 0, 0, 0, 0, 0, 0]])

    expected_output = np.array(
                       [[0,  0, 0, 0, 0, 0, 0, 0,  0,  0, -1, 0],
                        [0,  2, 0, 0, 0, 0, 0, 0, -1,  0,  0, 0],
                        [0,  2, 0, 0, 1, 1, 0, 0, -1,  0,  0, 3],
                        [0,  2, 0, 1, 1, 1, 0, 0,  0,  0,  0, 3],
                        [0,  2, 0, 1, 0, 1, 0, 0,  0,  0,  0, 3],
                        [0,  2, 0, 1, 1, 1, 1, 0,  0,  0,  0, 3],
                        [0,  0, 0, 0, 0, 0, 0, 0,  0,  0,  0, 0],
                        [0, -1, 0, 0, 0, 0, 0, 0,  4,  4,  4, 0],
                        [0,  0, 0, 0, 0, 0, 0, 0,  0,  0,  0, 0]])

    ans = island_for_label(in_data, 1, m=4, special_label=-1)

    assert_array_equal(expected_output, ans)


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
    pass



def test_otsu_threshold_1():
    pass


def test_otsu_threshold_2():
    pass


def test_MoG_array_bad_input():
    pass


def test_MoG_array_1():
    pass


def test_MoG_array_2():
    pass
