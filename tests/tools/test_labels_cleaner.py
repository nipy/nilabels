import numpy as np
from numpy.testing import assert_array_equal
from scipy import ndimage

from nilabels.tools.cleaning.labels_cleaner import multi_lab_segmentation_dilate_1_above_selected_label, \
    holes_filler, clean_semgentation


# TESTING multi_lab_segmentation_dilate_1_above_selected_label

def test_multi_lab_segmentation_dilate_1_above_selected_label_on_input_1():
    c = np.array([[0,   0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                  [0,   0,  0,  0,  0,  0, -1, -1,  2,  2,  2,  0],
                  [0,   0,  0,  1,  1,  1, -1, -1,  2,  2,  2,  0],
                  [0,   0,  0,  1, -1,  1, -1,  2,  2,  2,  2,  0],
                  [0,   0,  0,  1,  1,  1,  0,  0,  2,  2,  2,  0],
                  [0,   0,  0,  1,  1,  1,  1,  0,  2, -1,  2,  0],
                  [-1, -1,  0,  0,  0,  0,  0,  0,  0, -1,  0,  0],
                  [-1, -1, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                  [ 0, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0]])

    b1 = multi_lab_segmentation_dilate_1_above_selected_label(c, selected_label=-1, labels_to_dilate=(1, ))

    expected_b1 = np.array(
                  [[ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                   [ 0,  0,  0,  0,  0,  0, -1, -1,  2,  2,  2,  0],
                   [ 0,  0,  0,  1,  1,  1,  1, -1,  2,  2,  2,  0],
                   [ 0,  0,  0,  1,  1,  1,  1,  2,  2,  2,  2,  0],
                   [ 0,  0,  0,  1,  1,  1,  0,  0,  2,  2,  2,  0],
                   [ 0,  0,  0,  1,  1,  1,  1,  0,  2, -1,  2,  0],
                   [-1, -1,  0,  0,  0,  0,  0,  0,  0, -1,  0,  0],
                   [-1, -1, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                   [ 0, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0]])

    b2 = multi_lab_segmentation_dilate_1_above_selected_label(c, selected_label=-1, labels_to_dilate=(2,))

    expected_b2 = np.array(
                  [[ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                   [ 0,  0,  0,  0,  0,  0, -1,  2,  2,  2,  2,  0],
                   [ 0,  0,  0,  1,  1,  1, -1,  2,  2,  2,  2,  0],
                   [ 0,  0,  0,  1, -1,  1,  2,  2,  2,  2,  2,  0],
                   [ 0,  0,  0,  1,  1,  1,  0,  0,  2,  2,  2,  0],
                   [ 0,  0,  0,  1,  1,  1,  1,  0,  2,  2,  2,  0],
                   [-1, -1,  0,  0,  0,  0,  0,  0,  0, -1,  0,  0],
                   [-1, -1, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                   [ 0, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0]])

    b3 = multi_lab_segmentation_dilate_1_above_selected_label(c, selected_label=-1, labels_to_dilate=(0, 1, 2))

    expected_b3 = np.array(
                  [[0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                   [0,  0,  0,  0,  0,  0,  0,  0,  2,  2,  2,  0],
                   [0,  0,  0,  1,  1,  1,  1,  2,  2,  2,  2,  0],
                   [0,  0,  0,  1,  1,  1,  0,  2,  2,  2,  2,  0],
                   [0,  0,  0,  1,  1,  1,  0,  0,  2,  2,  2,  0],
                   [0,  0,  0,  1,  1,  1,  1,  0,  2,  2,  2,  0],
                   [0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                   [0, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                   [0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0]])

    b4 = multi_lab_segmentation_dilate_1_above_selected_label(c, selected_label=-1, labels_to_dilate=(2, 1, 0))

    expected_b4 = np.array(
        [[ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
         [ 0,  0,  0,  0,  0,  0,  0,  2,  2,  2,  2,  0],
         [ 0,  0,  0,  1,  1,  1,  1,  2,  2,  2,  2,  0],
         [ 0,  0,  0,  1,  1,  1,  2,  2,  2,  2,  2,  0],
         [ 0,  0,  0,  1,  1,  1,  0,  0,  2,  2,  2,  0],
         [ 0,  0,  0,  1,  1,  1,  1,  0,  2,  2,  2,  0],
         [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
         [ 0, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
         [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0]])

    b5 = multi_lab_segmentation_dilate_1_above_selected_label(c, selected_label=-1, labels_to_dilate=())

    assert_array_equal(b1, expected_b1)
    assert_array_equal(b2, expected_b2)
    assert_array_equal(b3, expected_b3)
    assert_array_equal(b4, expected_b4)
    assert_array_equal(b5, b3)


def test_multi_lab_segmentation_dilate_1_above_selected_label_on_input_2():

    c = np.array([[0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                  [0,  0,  0,  1,  1,  1,  1,  1,  0,  2,  0,  0],
                  [0,  0,  0,  1,  1, -1,  1,  1,  0,  2,  0,  0],
                  [0,  0,  0,  1, -1, -1, -1,  1,  0,  2, -1,  0],
                  [0,  0,  0,  1,  1, -1,  1,  1,  0,  2,  0,  0],
                  [0,  0,  0,  1,  1, -1,  1,  1,  0,  2,  0,  0],
                  [0,  0,  0,  1,  1, -1,  1,  1,  0,  2,  0,  0],
                  [0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                  [0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0]])

    b1 = multi_lab_segmentation_dilate_1_above_selected_label(c, selected_label=-1, labels_to_dilate=())

    expected_b1 = np.array([[0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                            [0,  0,  0,  1,  1,  1,  1,  1,  0,  2,  0,  0],
                            [0,  0,  0,  1,  1,  1,  1,  1,  0,  2,  0,  0],
                            [0,  0,  0,  1,  1, -1,  1,  1,  0,  2,  0,  0],
                            [0,  0,  0,  1,  1,  1,  1,  1,  0,  2,  0,  0],
                            [0,  0,  0,  1,  1,  1,  1,  1,  0,  2,  0,  0],
                            [0,  0,  0,  1,  1,  0,  1,  1,  0,  2,  0,  0],
                            [0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                            [0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0]])

    b2 = multi_lab_segmentation_dilate_1_above_selected_label(c, selected_label=-1, labels_to_dilate=(1, 2))

    expected_b2 = np.array([[0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                            [0,  0,  0,  1,  1,  1,  1,  1,  0,  2,  0,  0],
                            [0,  0,  0,  1,  1,  1,  1,  1,  0,  2,  0,  0],
                            [0,  0,  0,  1,  1, -1,  1,  1,  0,  2,  2,  0],
                            [0,  0,  0,  1,  1,  1,  1,  1,  0,  2,  0,  0],
                            [0,  0,  0,  1,  1,  1,  1,  1,  0,  2,  0,  0],
                            [0,  0,  0,  1,  1,  1,  1,  1,  0,  2,  0,  0],
                            [0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                            [0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0]])

    b3 = multi_lab_segmentation_dilate_1_above_selected_label(c, selected_label=-1, labels_to_dilate=(2, 1))

    assert_array_equal(b1, expected_b1)
    assert_array_equal(b2, expected_b2)
    assert_array_equal(b2, b3)  # for this particular case only!


def test_multi_lab_segmentation_dilate_1_above_selected_label_on_input_3():

    c = np.array([[0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                  [0,  0,  0,  1, -1, -1,  2, -1, -1,  3,  3,  0],
                  [0,  0,  0,  1,  1, -1,  2,  2, -1,  3,  3,  0],
                  [0,  0,  0,  1,  1, -1,  2,  2, -1,  3,  3,  0],
                  [0,  0,  0,  1,  1, -1,  2,  2, -1,  3,  3,  0],
                  [0,  0,  0,  1,  1, -1,  2,  2, -1,  3,  3,  0],
                  [0,  0,  0,  1,  1, -1, -1,  2, -1, -1,  3,  0],
                  [0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                  [0, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0]])

    b123 = multi_lab_segmentation_dilate_1_above_selected_label(c, selected_label=-1, labels_to_dilate=(1, 2, 3))

    expected_b123 = np.array(
        [[0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
         [0,  0,  0,  1,  1,  2,  2,  2,  3,  3,  3,  0],
         [0,  0,  0,  1,  1,  1,  2,  2,  2,  3,  3,  0],
         [0,  0,  0,  1,  1,  1,  2,  2,  2,  3,  3,  0],
         [0,  0,  0,  1,  1,  1,  2,  2,  2,  3,  3,  0],
         [0,  0,  0,  1,  1,  1,  2,  2,  2,  3,  3,  0],
         [0,  0,  0,  1,  1,  1,  2,  2,  2,  3,  3,  0],
         [0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
         [0, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0]])

    b231 = multi_lab_segmentation_dilate_1_above_selected_label(c, selected_label=-1, labels_to_dilate=(2, 3, 1))

    expected_b231 = np.array(
                 [[0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                  [0,  0,  0,  1,  1,  2,  2,  2,  3,  3,  3,  0],
                  [0,  0,  0,  1,  1,  2,  2,  2,  2,  3,  3,  0],
                  [0,  0,  0,  1,  1,  2,  2,  2,  2,  3,  3,  0],
                  [0,  0,  0,  1,  1,  2,  2,  2,  2,  3,  3,  0],
                  [0,  0,  0,  1,  1,  2,  2,  2,  2,  3,  3,  0],
                  [0,  0,  0,  1,  1,  1,  2,  2,  2,  3,  3,  0],
                  [0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                  [0, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0]])

    ball = multi_lab_segmentation_dilate_1_above_selected_label(c, selected_label=-1, labels_to_dilate=())

    expected_ball = np.array([[0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                              [0,  0,  0,  1,  0,  0,  2,  0,  0,  3,  3,  0],
                              [0,  0,  0,  1,  1,  1,  2,  2,  2,  3,  3,  0],
                              [0,  0,  0,  1,  1,  1,  2,  2,  2,  3,  3,  0],
                              [0,  0,  0,  1,  1,  1,  2,  2,  2,  3,  3,  0],
                              [0,  0,  0,  1,  1,  1,  2,  2,  2,  3,  3,  0],
                              [0,  0,  0,  1,  1,  0,  0,  2,  0,  0,  3,  0],
                              [0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                              [0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0]])

    assert_array_equal(b123, expected_b123)
    assert_array_equal(b231, expected_b231)
    assert_array_equal(ball, expected_ball)


# TESTING holes_filler


def test_hole_filler_bypass_expected():
    # segm with no holes
    c = np.array([[0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                  [0,  0,  0,  1,  1,  1,  1,  1,  0,  2,  0,  0],
                  [0,  0,  0,  1,  1,  1,  1,  1,  0,  2,  0,  0],
                  [0,  0,  0,  1,  1,  1,  1,  1,  0,  2,  0,  0],
                  [0,  0,  0,  1,  1,  1,  1,  1,  0,  2,  0,  0],
                  [0,  0,  0,  1,  1,  1,  1,  1,  0,  2,  0,  0],
                  [0,  0,  0,  1,  1,  0,  1,  1,  0,  2,  0,  0],
                  [0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                  [0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0]])

    a = holes_filler(c, holes_label=-1, labels_sequence=())

    assert_array_equal(a, c)


def test_hole_filler_example_1():

    c = np.array([[0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                  [0,  0,  1,  1,  1,  1,  1,  2,  2,  2,  2,  2],
                  [0,  0,  1,  1, -1,  1,  1,  2, -1, -1, -1,  2],
                  [0,  0,  1, -1, -1, -1,  1,  2, -1, -1, -1,  2],
                  [0,  0,  1, -1, -1, -1,  1,  2, -1, -1, -1,  2],
                  [0,  0,  1,  1, -1, -1,  1,  2, -1, -1, -1,  2],
                  [0,  0,  1, -1, -1, -1,  1,  2,  2,  2,  2,  2],
                  [0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                  [0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0]])

    a = holes_filler(c, holes_label=-1, labels_sequence=())

    expected_a = np.array(
                 [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2],
                  [0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2],
                  [0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2],
                  [0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2],
                  [0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2],
                  [0, 0, 1, 0, 0, 0, 1, 2, 2, 2, 2, 2],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

    assert_array_equal(a, expected_a)

    b = holes_filler(c, holes_label=-1, labels_sequence=(1, 2))

    expected_b = np.array(
                 [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2],
                  [0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2],
                  [0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2],
                  [0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2],
                  [0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2],
                  [0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

    assert_array_equal(b, expected_b)

    assert_array_equal(b, expected_b)


def test_hole_filler_example_2():

    c = np.array([[0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                  [0,  0,  1,  1,  1,  1,  1,  2,  2,  2,  2,  2],
                  [0,  0,  1,  1, -1,  1,  1,  2, -1, -1, -1,  2],
                  [0,  0,  1, -1, -1, -1,  1,  2, -1, -1, -1,  2],
                  [0,  0,  1, -1, -1, -1,  1,  2, -1, -1, -1,  2],
                  [0,  0,  1, -1, -1, -1,  1,  2, -1, -1, -1,  2],
                  [0,  0,  1, -1, -1, -1,  1,  2,  2,  2,  2,  2],
                  [0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                  [0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0]])

    a = holes_filler(c, holes_label=-1, labels_sequence=())

    expected_a = np.array(
                 [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2],
                  [0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2],
                  [0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2],
                  [0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2],
                  [0, 0, 1, 1, 0, 1, 1, 2, 2, 2, 2, 2],
                  [0, 0, 1, 0, 0, 0, 1, 2, 2, 2, 2, 2],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

    assert_array_equal(a, expected_a)


# TESTING clean segmentation

def test_clean_segmentation_simple_example():

    c = np.array([[0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                  [1,  0,  1,  1,  1,  1,  1,  2,  2,  4,  2,  2],
                  [0,  0,  1,  3,  2,  1,  1,  2,  4,  3,  4,  4],
                  [0,  0,  1,  1,  2,  2,  1,  2,  4,  4,  4,  2],
                  [3,  3,  1,  1,  2,  2,  1,  2,  4,  4,  4,  4],
                  [3,  3,  1,  1,  2,  2,  1,  2,  4,  4,  4,  4],
                  [3,  3,  1,  1,  2,  2,  2,  2,  2,  2,  4,  2],
                  [3,  4,  3,  3,  0,  0,  0,  4,  0,  0,  0,  0],
                  [3,  3,  3,  3,  0,  0,  0,  0,  0,  1,  0,  1]])

    b = clean_semgentation(c)

    for l in sorted(list(set(c.flat))):
        assert ndimage.label(b == l)[1] == 1


if __name__ == '__main__':
    test_multi_lab_segmentation_dilate_1_above_selected_label_on_input_1()
    test_multi_lab_segmentation_dilate_1_above_selected_label_on_input_2()
    test_multi_lab_segmentation_dilate_1_above_selected_label_on_input_3()
