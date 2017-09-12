import numpy as np

from numpy.testing import assert_array_equal, assert_equal

from labels_manager.tools.caliber.distances import centroid_array



def test_centroid_array():
    test_arr = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 1, 1, 1, 0, 0, 0, 0, 0],
                         [0, 1, 1, 1, 0, 0, 0, 0, 0],
                         [0, 1, 1, 1, 0, 2, 2, 2, 0],
                         [0, 0, 0, 0, 0, 2, 2, 2, 0],
                         [0, 0, 0, 0, 0, 2, 2, 2, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0]])

    ans = centroid_array(test_arr, labels=[1, 2, 3])
    assert_array_equal(ans[0], np.array([3,2]))
    assert_array_equal(ans[1], np.array([5,6]))
    assert_equal(ans[2], np.nan)


def test_centroid_array_2():
    test_arr = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 1, 0, 0, 0, 0, 0, 0, 0],
                         [0, 1, 0, 0, 0, 0, 0, 0, 0],
                         [0, 1, 0, 0, 0, 0, 2, 0, 0],
                         [0, 1, 0, 0, 0, 0, 2, 0, 0],
                         [0, 1, 0, 0, 0, 0, 2, 0, 0],
                         [0, 0, 0, 0, 0, 0, 2, 0, 0],
                         [0, 0, 0, 0, 0, 0, 2, 0, 0]])

    ans = centroid_array(test_arr, labels=[1, 2, 7])
    assert_array_equal(ans[0], np.array([4,1]))
    assert_array_equal(ans[1], np.array([6,6]))
    assert_equal(ans[2], np.nan)


def test_centroid():
    pass


#
#
# ''' Test measurements.linear.py'''
# from labels_manager.tools.caliber.distances import centroid_array
#
#
# def test_simple_centroid():
#
#     test_image = np.array([[[0, 0, 0, 0, 0, 0, 0, 0, 0],
#                             [0, 0, 0, 0, 0, 0, 0, 0, 0],
#                             [0, 1, 1, 1, 0, 0, 0, 0, 0],
#                             [0, 0, 0, 1, 1, 1, 1, 1, 0],
#                             [0, 0, 0, 0, 0, 0, 0, 0, 0],
#                             [0, 0, 2, 2, 0, 0, 0, 0, 0],
#                             [0, 0, 2, 2, 0, 0, 0, 0, 0],
#                             [0, 0, 0, 0, 0, 0, 0, 0, 0],
#                             [0, 0, 0, 0, 0, 0, 0, 0, 0]],
#
#                            [[0, 0, 0, 0, 0, 0, 0, 0, 0],
#                             [0, 0, 0, 0, 0, 0, 0, 0, 0],
#                             [0, 1, 1, 1, 0, 0, 0, 0, 0],
#                             [0, 0, 0, 1, 1, 1, 1, 1, 0],
#                             [0, 0, 0, 0, 0, 0, 0, 0, 0],
#                             [0, 0, 2, 2, 0, 0, 0, 0, 0],
#                             [0, 0, 2, 2, 0, 0, 0, 0, 0],
#                             [0, 0, 0, 0, 0, 0, 0, 0, 0],
#                             [0, 0, 0, 0, 0, 0, 0, 0, 0]]])
#
#     ans = centroid_array(test_image, labels=[1,2])
#     assert_array_equal(ans[0], np.array([ 0.5  ,  2.625,  3.875]))
#     assert_array_equal(ans[1], np.array([0.5, 5.5, 2.5]))
#



