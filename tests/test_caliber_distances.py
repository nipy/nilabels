import numpy as np
import nibabel as nib
import os
from scipy import ndimage as nd

from numpy.testing import assert_array_equal, assert_equal, assert_almost_equal
from LABelsToolkit.tools.aux_methods.utils_nib import set_new_data
from LABelsToolkit.tools.detections.contours import contour_from_segmentation
from LABelsToolkit.tools.phantoms_generator.shapes_for_phantoms import o_shape, cube_shape

from LABelsToolkit.tools.caliber.distances import centroid_array, centroid, dice_score, global_dice_score, \
    global_outline_error, covariance_matrices, covariance_distance, hausdorff_distance, \
    normalised_symmetric_contour_distance, covariance_distance_between_matrices, dice_score_one_label, \
    d_H, hausdorff_distance_one_label, box_sides_length


# --- Auxiliaries


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
    test_arr = np.array([[[0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 1, 0, 0, 0, 0, 0, 0, 0],
                          [0, 1, 0, 0, 0, 0, 0, 0, 0],
                          [0, 1, 0, 0, 0, 0, 2, 0, 0],
                          [0, 1, 0, 0, 0, 0, 2, 0, 0],
                          [0, 1, 0, 0, 0, 0, 2, 0, 0],
                          [0, 0, 0, 0, 0, 0, 2, 0, 0],
                          [0, 0, 0, 0, 0, 0, 2, 0, 0]],

                          [[0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 1, 0, 0, 0, 0, 0, 0, 0],
                           [0, 1, 0, 0, 0, 0, 0, 0, 0],
                           [0, 1, 0, 0, 0, 0, 2, 0, 0],
                           [0, 1, 0, 0, 0, 0, 2, 0, 0],
                           [0, 1, 0, 0, 0, 0, 2, 0, 0],
                           [0, 0, 0, 0, 0, 0, 2, 0, 0],
                           [0, 0, 0, 0, 0, 0, 2, 0, 0]]
                         ])

    im = nib.Nifti1Image(test_arr, 0.5 * np.eye(4))
    ans_v = centroid(im, labels=[1, 2, 7], return_mm3=False)

    assert_array_equal(ans_v[0], np.array([0, 4, 1]))
    assert_array_equal(ans_v[1], np.array([0, 6, 6]))
    assert ans_v[2] == 0 or np.isnan(ans_v[2])

    ans_mm = centroid(im, labels=[1, 2, 7], return_mm3=True)

    assert_array_equal(ans_mm[0], .5 * np.array([0.5, 4, 1]))
    assert_array_equal(ans_mm[1], .5 * np.array([0.5, 6, 6]))
    assert ans_mm[2] == 0 or np.isnan(ans_mm[2])


def test_covariance_matrices():
    arr_1 = np.array([[[0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 1, 1, 1, 1, 1, 0, 0, 0],
                       [0, 1, 1, 1, 1, 1, 0, 0, 0],
                       [0, 1, 1, 1, 1, 1, 2, 2, 0],
                       [0, 1, 1, 1, 1, 1, 2, 2, 0],
                       [0, 1, 1, 1, 1, 1, 0, 3, 3],
                       [0, 0, 0, 0, 0, 0, 0, 3, 3],
                       [0, 0, 0, 0, 0, 3, 3, 3, 3]],

                      [[0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 1, 1, 1, 1, 1, 0, 0, 0],
                       [0, 1, 1, 1, 1, 1, 0, 0, 0],
                       [0, 1, 1, 1, 1, 1, 2, 2, 0],
                       [0, 1, 1, 1, 1, 1, 2, 2, 0],
                       [0, 1, 1, 1, 1, 1, 0, 0, 3],
                       [0, 0, 0, 0, 0, 0, 0, 3, 3],
                       [0, 0, 0, 0, 0, 0, 3, 3, 3]]
                      ])

    im1 = nib.Nifti1Image(arr_1, np.eye(4))

    cov =  covariance_matrices(im1, [1, 2, 3])
    assert len(cov) == 3
    for i in cov:
        assert_array_equal(i.shape, [3, 3])
        if np.count_nonzero(i - np.diag(np.diagonal(i))) == 0:
            assert_array_equal(np.diag(np.diag(i)), i)

    cov1 = covariance_matrices(im1, [1, 2, 3, 4])
    assert_array_equal(cov1[-1], np.nan * np.ones([3, 3]))


def test_covariance_distance_between_matrices_simple_case():
    m1 = np.random.randn(4, 4)
    m2 = np.random.randn(4, 4)

    mul_factor = 1
    expected_ans = \
        mul_factor * (1 - (np.trace(m1.dot(m2)) / (np.linalg.norm(m1, ord='fro') * np.linalg.norm(m2, ord='fro'))))

    assert_equal(expected_ans, covariance_distance_between_matrices(m1, m2))

    m3 = np.random.randn(4, 4)
    m4 = np.random.randn(4, 4)
    m3[1, 1] = np.nan

    assert_equal(np.nan, covariance_distance_between_matrices(m3, m4))


# --- global distances: (segm, segm) |-> real


def test_dice_score():
    arr_1 = np.array([[[0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 1, 0, 0, 0, 0, 0, 0, 0],
                       [0, 1, 0, 0, 0, 0, 0, 0, 0],
                       [0, 1, 0, 0, 0, 0, 2, 0, 0],
                       [0, 1, 0, 0, 0, 0, 2, 0, 0],
                       [0, 1, 0, 0, 0, 0, 2, 0, 0],
                       [0, 0, 0, 0, 0, 0, 2, 0, 0],
                       [0, 0, 0, 0, 0, 0, 2, 0, 0]],

                      [[0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 1, 0, 0, 0, 0, 0, 0, 0],
                       [0, 1, 0, 0, 0, 0, 0, 0, 0],
                       [0, 1, 0, 0, 0, 0, 2, 0, 0],
                       [0, 1, 0, 0, 0, 0, 2, 0, 0],
                       [0, 1, 0, 0, 0, 0, 2, 0, 0],
                       [0, 0, 0, 0, 0, 0, 2, 0, 0],
                       [0, 0, 0, 0, 0, 0, 2, 0, 0]]
                      ])

    arr_2 = np.array([[[0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 1, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 1, 0, 0, 0, 0, 2, 0, 0],
                       [0, 1, 0, 0, 0, 0, 2, 0, 0],
                       [0, 0, 0, 0, 0, 0, 2, 0, 0],
                       [0, 0, 0, 0, 0, 0, 2, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0]],

                      [[0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 1, 0, 0, 0, 0, 0, 0, 0],
                       [0, 1, 0, 0, 0, 0, 0, 0, 0],
                       [0, 1, 0, 0, 0, 0, 2, 0, 0],
                       [0, 1, 0, 0, 0, 0, 2, 0, 0],
                       [0, 1, 0, 0, 0, 0, 2, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0]]
                      ])

    arr_3 = np.array([[[0, 0, 0, 0, 1, 1, 0, 0, 0],
                       [0, 0, 0, 0, 1, 1, 0, 0, 0],
                       [0, 0, 0, 0, 1, 1, 0, 0, 0],
                       [0, 0, 0, 0, 1, 1, 0, 0, 0],
                       [0, 0, 0, 0, 1, 1, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0]],

                      [[0, 0, 0, 0, 1, 1, 0, 0, 0],
                       [0, 0, 0, 0, 1, 1, 0, 0, 0],
                       [0, 0, 0, 0, 1, 1, 0, 0, 0],
                       [2, 2, 2, 0, 1, 1, 0, 0, 0],
                       [2, 2, 2, 0, 1, 1, 0, 0, 0],
                       [2, 2, 2, 0, 0, 0, 0, 0, 0],
                       [2, 2, 2, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0]]
                      ])

    arr_void = np.zeros_like(arr_3)

    im1 = nib.Nifti1Image(arr_1, np.eye(4))
    im2 = nib.Nifti1Image(arr_2, np.eye(4))
    im3 = nib.Nifti1Image(arr_3, np.eye(4))
    im_void = nib.Nifti1Image(arr_void, np.eye(4))

    dice_1_1 = dice_score(im1, im1, [1, 2], ['lab1', 'lab2'])
    dice_1_2 = dice_score(im1, im2, [1, 2], ['lab1', 'lab2'])
    dice_1_3 = dice_score(im1, im3, [1, 2], ['lab1', 'lab2'])

    dice_1_3_extra_lab = dice_score(im1, im3, [1, 2, 5], ['lab1', 'lab2', 'lab5'])

    dice_1_void = dice_score(im1, im_void, [1, 2], ['lab1', 'lab2'])

    assert_equal(dice_1_1['lab1'], 1)
    assert_equal(dice_1_1['lab2'], 1)

    assert_equal(dice_1_2['lab1'], 16/18.)
    assert_equal(dice_1_2['lab2'], 14/17.)

    assert_equal(dice_1_3['lab1'], 0)
    assert_equal(dice_1_3['lab2'], 0)

    assert_equal(dice_1_3_extra_lab['lab2'], 0)
    assert_equal(dice_1_3_extra_lab['lab1'], 0)
    assert_equal(dice_1_3_extra_lab['lab5'], np.nan)

    assert_equal(dice_1_void['lab2'], 0)
    assert_equal(dice_1_void['lab1'], 0)


def test_global_dice_score():
    arr_1 = np.array([[[0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 1, 0, 0, 0, 0, 0, 0, 0],
                       [0, 1, 0, 0, 0, 0, 0, 0, 0],
                       [0, 1, 0, 0, 0, 0, 2, 0, 0],
                       [0, 1, 0, 0, 0, 0, 2, 0, 0],
                       [0, 1, 0, 0, 0, 0, 2, 0, 0],
                       [0, 0, 0, 0, 0, 0, 2, 0, 0],
                       [0, 0, 0, 0, 0, 0, 2, 0, 0]],

                      [[0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 1, 0, 0, 0, 0, 0, 0, 0],
                       [0, 1, 0, 0, 0, 0, 0, 0, 0],
                       [0, 1, 0, 0, 0, 0, 2, 0, 0],
                       [0, 1, 0, 0, 0, 0, 2, 0, 0],
                       [0, 1, 0, 0, 0, 0, 2, 0, 0],
                       [0, 0, 0, 0, 0, 0, 2, 0, 0],
                       [0, 0, 0, 0, 0, 0, 2, 0, 0]]])

    arr_2 = np.array([[[0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 1, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 1, 0, 0, 0, 0, 2, 0, 0],
                       [0, 1, 0, 0, 0, 0, 2, 0, 0],
                       [0, 0, 0, 0, 0, 0, 2, 0, 0],
                       [0, 0, 0, 0, 0, 0, 2, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0]],

                      [[0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 1, 0, 0, 0, 0, 0, 0, 0],
                       [0, 1, 0, 0, 0, 0, 0, 0, 0],
                       [0, 1, 0, 0, 0, 0, 2, 0, 0],
                       [0, 1, 0, 0, 0, 0, 2, 0, 0],
                       [0, 1, 0, 0, 0, 0, 2, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0]]])

    arr_3 = np.array([[[0, 0, 0, 0, 1, 1, 0, 0, 0],
                       [0, 0, 0, 0, 1, 1, 0, 0, 0],
                       [0, 0, 0, 0, 1, 1, 0, 0, 0],
                       [0, 0, 0, 0, 1, 1, 0, 0, 0],
                       [0, 0, 0, 0, 1, 1, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0]],

                      [[0, 0, 0, 0, 1, 1, 0, 0, 0],
                       [0, 0, 0, 0, 1, 1, 0, 0, 0],
                       [0, 0, 0, 0, 1, 1, 0, 0, 0],
                       [2, 2, 2, 0, 1, 1, 0, 0, 0],
                       [2, 2, 2, 0, 1, 1, 0, 0, 0],
                       [2, 2, 2, 0, 0, 0, 0, 0, 0],
                       [2, 2, 2, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0]]])

    arr_void = np.zeros_like(arr_3)

    im1 = nib.Nifti1Image(arr_1, np.eye(4))
    im2 = nib.Nifti1Image(arr_2, np.eye(4))
    im3 = nib.Nifti1Image(arr_3, np.eye(4))
    im_void = nib.Nifti1Image(arr_void, np.eye(4))

    g_dice_1_1 = global_dice_score(im1, im1)
    g_dice_1_2 = global_dice_score(im1, im2)
    g_dice_1_3 = global_dice_score(im1, im3)
    g_dice_1_void = global_dice_score(im1, im_void)

    assert_equal(g_dice_1_1, 1)
    assert_equal(g_dice_1_2, (16 + 14) / (18. + 17.))
    assert_equal(g_dice_1_3, 0)
    assert_equal(g_dice_1_void, 0)


def test_global_outline_error():
    arr_1 = np.array([[[0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 1, 0, 0, 0, 0, 0, 0, 0],
                       [0, 1, 0, 0, 0, 0, 0, 0, 0],
                       [0, 1, 0, 0, 0, 0, 2, 0, 0],
                       [0, 1, 0, 0, 0, 0, 2, 0, 0],
                       [0, 1, 0, 0, 0, 0, 2, 0, 0],
                       [0, 0, 0, 0, 0, 0, 2, 0, 0],
                       [0, 0, 0, 0, 0, 0, 2, 0, 0]],

                      [[0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 1, 0, 0, 0, 0, 0, 0, 0],
                       [0, 1, 0, 0, 0, 0, 0, 0, 0],
                       [0, 1, 0, 0, 0, 0, 2, 0, 0],
                       [0, 1, 0, 0, 0, 0, 2, 0, 0],
                       [0, 1, 0, 0, 0, 0, 2, 0, 0],
                       [0, 0, 0, 0, 0, 0, 2, 0, 0],
                       [0, 0, 0, 0, 0, 0, 2, 0, 0]]])

    arr_2 = np.array([[[0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 1, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 1, 0, 0, 0, 0, 2, 0, 0],
                       [0, 1, 0, 0, 0, 0, 2, 0, 0],
                       [0, 0, 0, 0, 0, 0, 2, 0, 0],
                       [0, 0, 0, 0, 0, 0, 2, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0]],

                      [[0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 1, 0, 0, 0, 0, 0, 0, 0],
                       [0, 1, 0, 0, 0, 0, 0, 0, 0],
                       [0, 1, 0, 0, 0, 0, 2, 0, 0],
                       [0, 1, 0, 0, 0, 0, 2, 0, 0],
                       [0, 1, 0, 0, 0, 0, 2, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0]]])

    arr_3 = np.array([[[0, 0, 0, 0, 1, 1, 0, 0, 0],
                       [0, 0, 0, 0, 1, 1, 0, 0, 0],
                       [0, 0, 0, 0, 1, 1, 0, 0, 0],
                       [0, 0, 0, 0, 1, 1, 0, 0, 0],
                       [0, 0, 0, 0, 1, 1, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0]],

                      [[0, 0, 0, 0, 1, 1, 0, 0, 0],
                       [0, 0, 0, 0, 1, 1, 0, 0, 0],
                       [0, 0, 0, 0, 1, 1, 0, 0, 0],
                       [2, 2, 2, 0, 1, 1, 0, 0, 0],
                       [2, 2, 2, 0, 1, 1, 0, 0, 0],
                       [2, 2, 2, 0, 0, 0, 0, 0, 0],
                       [2, 2, 2, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0]]])

    arr_void = np.zeros_like(arr_3)

    im1 = nib.Nifti1Image(arr_1, np.eye(4))
    im2 = nib.Nifti1Image(arr_2, np.eye(4))
    im3 = nib.Nifti1Image(arr_3, np.eye(4))
    im_void = nib.Nifti1Image(arr_void, np.eye(4))

    goe_1_1 = global_outline_error(im1, im1)
    goe_1_2 = global_outline_error(im1, im2)
    goe_1_3 = global_outline_error(im1, im3)
    goe_1_void = global_outline_error(im1, im_void)

    assert_equal(goe_1_1, 0)
    assert_almost_equal(goe_1_2, 5 / (.5 * (20 + 15)))
    assert_almost_equal(goe_1_3, 48 / (.5 * (20 + 32)))
    assert_almost_equal(goe_1_void, 2)  # interesting case!


# --- Single labels distances (segm, segm, label) |-> real


def test_dice_score_one_label():
    arr_1 = np.array([[[0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 2, 2, 0],
                       [0, 0, 0, 0, 0, 0, 2, 2, 0],
                       [0, 0, 0, 0, 0, 0, 2, 2, 0],
                       [0, 0, 0, 0, 0, 0, 2, 2, 0],
                       [0, 0, 0, 0, 0, 0, 2, 2, 0]],

                      [[0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 1, 0, 0, 0, 0, 0, 0, 0],
                       [0, 1, 0, 0, 0, 0, 0, 0, 0],
                       [0, 1, 0, 0, 0, 0, 2, 2, 0],
                       [0, 1, 0, 0, 0, 0, 2, 2, 0],
                       [0, 1, 0, 0, 0, 0, 2, 2, 0],
                       [0, 0, 0, 0, 0, 0, 2, 2, 0],
                       [0, 0, 0, 0, 0, 0, 2, 2, 0]]])

    arr_2 = np.array([[[0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 2, 2, 0],
                       [0, 0, 0, 0, 0, 0, 2, 2, 0],
                       [0, 0, 0, 0, 0, 0, 2, 2, 0],
                       [0, 0, 0, 0, 0, 0, 2, 2, 0],
                       [0, 0, 0, 0, 0, 2, 2, 2, 0]],

                      [[0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 1, 0, 0, 0, 0, 0, 0, 0],
                       [0, 1, 0, 0, 0, 0, 0, 0, 0],
                       [0, 1, 0, 0, 0, 0, 2, 1, 0],
                       [0, 1, 0, 0, 0, 2, 2, 0, 0],
                       [0, 1, 0, 0, 0, 0, 2, 2, 0],
                       [0, 0, 0, 0, 0, 0, 2, 2, 0],
                       [0, 0, 0, 0, 0, 0, 2, 2, 0]]])

    im1 = nib.Nifti1Image(arr_1, np.eye(4))
    im2 = nib.Nifti1Image(arr_2, np.eye(4))
    assert_equal(36./40., dice_score_one_label(im1, im2, 2))
    assert_equal(np.nan, dice_score_one_label(im1, im2, 5))


def test_asymmetric_component_Hausdorff_distance_H_d_and_Hausdorff_distance():
    arr_1 = np.array([[[0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0]],

                      [[0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 2, 2, 0, 0, 0, 0, 0, 0],
                       [0, 2, 2, 2, 2, 2, 2, 2, 0],
                       [0, 2, 2, 2, 2, 2, 2, 2, 0],
                       [0, 2, 2, 2, 2, 2, 2, 2, 0],
                       [0, 2, 2, 2, 2, 2, 2, 2, 0],
                       [0, 2, 2, 2, 2, 2, 2, 2, 0],
                       [2, 0, 0, 0, 0, 0, 2, 2, 0]]])

    arr_2 = np.array([[[0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0]],

                      [[0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 2, 2, 2, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0]]])

    im1 = nib.Nifti1Image(arr_1, np.eye(4))
    im2 = nib.Nifti1Image(arr_2, np.eye(4))

    # In label in image 2 is embedded in label in image 1. We expect the Hd to be zero.
    assert_equal(d_H(im2, im1, 2, True), 0)
    assert_equal(d_H(im1, im2, 2, True), np.sqrt(3**2 + 3**2))

    # Test symmetry
    assert_equal(hausdorff_distance_one_label(im1, im2, 2, True), hausdorff_distance_one_label(im1, im2, 2, True))
    assert_equal(hausdorff_distance_one_label(im1, im2, 2, True), np.max((d_H(im2, im1, 2, True),
                                                                          (d_H(im1, im2, 2, True)))))


# --- distances - (segm, segm) |-> pandas.Series (indexed by labels)


def test_dice_score_multiple_labels():
    arr_1 = np.array([[[0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [1, 1, 0, 0, 0, 0, 0, 0, 0],
                       [1, 1, 0, 0, 0, 0, 0, 0, 0],
                       [1, 1, 0, 0, 0, 0, 0, 0, 0],
                       [1, 1, 0, 0, 0, 0, 0, 0, 0]],

                      [[0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 2, 2, 0, 0, 0, 0, 0, 0],
                       [0, 2, 2, 2, 2, 2, 2, 2, 0],
                       [0, 2, 2, 2, 2, 2, 2, 2, 0],
                       [0, 2, 2, 2, 2, 2, 2, 2, 0],
                       [0, 2, 2, 2, 2, 2, 2, 2, 0],
                       [0, 2, 2, 2, 2, 2, 2, 2, 0],
                       [2, 0, 0, 0, 0, 0, 2, 2, 0]]])

    arr_2 = np.array([[[0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 1, 1, 0, 0, 0, 0, 0, 0],
                       [0, 1, 1, 0, 0, 0, 0, 0, 0],
                       [0, 1, 1, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0]],

                      [[0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 2, 2, 2, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0]]])

    im1 = nib.Nifti1Image(arr_1, np.eye(4))
    im2 = nib.Nifti1Image(arr_2, np.eye(4))

    res = dice_score(im1, im2, [0,1,2,3], ['back', 'one', 'two', 'non existing'])

    assert_almost_equal(res['back'], 0.823970037453)
    assert_almost_equal(res['one'],  0.2857142857142857)
    assert_almost_equal(res['two'],  0.13953488372093023)
    assert_almost_equal(res['non existing'], np.nan)


def test_covariance_distance():
    arr_1 = np.array([[[0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 2, 2, 0],
                       [0, 0, 0, 0, 0, 0, 2, 2, 0],
                       [0, 0, 0, 0, 0, 0, 2, 2, 0],
                       [0, 0, 0, 0, 0, 0, 2, 2, 0],
                       [0, 0, 0, 0, 0, 0, 2, 2, 0]],

                      [[0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 1, 0, 0, 0, 0, 0, 0, 0],
                       [0, 1, 0, 0, 0, 0, 0, 0, 0],
                       [0, 1, 0, 0, 0, 0, 2, 2, 0],
                       [0, 1, 0, 0, 0, 0, 2, 2, 0],
                       [0, 1, 0, 0, 0, 0, 2, 2, 0],
                       [0, 0, 0, 0, 0, 0, 2, 2, 0],
                       [0, 0, 0, 0, 0, 0, 2, 2, 0]]
                      ])

    arr_2 = np.array([[[0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 2, 2, 0, 0, 0],
                       [0, 0, 0, 0, 2, 2, 0, 0, 0],
                       [0, 0, 0, 0, 2, 2, 0, 0, 0],
                       [0, 0, 0, 0, 2, 2, 0, 0, 0],
                       [0, 0, 0, 0, 2, 2, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0]],

                      [[0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 1, 0, 0, 0, 0, 0, 0],
                       [0, 0, 1, 0, 0, 0, 0, 0, 0],
                       [0, 0, 1, 0, 2, 2, 0, 0, 0],
                       [0, 0, 1, 0, 2, 2, 0, 0, 0],
                       [0, 0, 1, 0, 2, 2, 0, 0, 0],
                       [0, 0, 0, 0, 2, 2, 0, 0, 0],
                       [0, 0, 0, 0, 2, 2, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0]]])

    arr_3 = np.array([[[0, 1, 1, 1, 1, 1, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0]],

                      [[0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [2, 2, 2, 2, 2, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0]]])

    im1 = nib.Nifti1Image(arr_1, np.eye(4))
    im2 = nib.Nifti1Image(arr_2, np.eye(4))
    im3 = nib.Nifti1Image(arr_3, np.eye(4))

    cd_1_1 = covariance_distance(im1, im1, [1, 2], ['label1', 'label2'], factor=1)
    cd_1_2 = covariance_distance(im1, im2, [1, 2], ['label1', 'label2'], factor=1)
    cd_1_3 = covariance_distance(im1, im3, [1, 2], ['label1', 'label2'], factor=1)
    cd_1_2_extra_label = covariance_distance(im1, im2, [1, 2, 4], ['label1', 'label2', 'label4'], factor=1)

    assert_almost_equal(cd_1_1['label1'], 0)
    assert_almost_equal(cd_1_1['label2'], 0)
    # insensitive to shifts
    assert_almost_equal(cd_1_2['label1'], 0)
    assert_almost_equal(cd_1_2['label2'], 0)
    # maximised for 90deg linear structures.
    assert_almost_equal(cd_1_3['label1'], 1)
    assert_almost_equal(cd_1_2_extra_label['label4'], np.nan)


def test_covariance_distance_range():
    factor = 10
    m1 = np.random.randint(3, size=[20, 20, 20])
    im1 = nib.Nifti1Image(m1, np.eye(4))
    for _ in range(20):
        m2 = np.random.randint(3, size=[20, 20, 20])
        im2 = nib.Nifti1Image(m2, np.eye(4))
        cd = covariance_distance(im1, im2, [1, 2], ['label1', 'label2'], factor=factor)
        assert 0 <= cd['label1'] <= factor
        assert 0 <= cd['label2'] <= factor


def test_hausdorff_distance():
    arr_1 = np.array([[[0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [1, 1, 0, 0, 0, 0, 0, 0, 0],
                       [1, 1, 0, 0, 0, 0, 0, 0, 0],
                       [1, 1, 0, 0, 0, 0, 0, 0, 0],
                       [1, 1, 0, 0, 0, 0, 2, 2, 0],
                       [0, 0, 0, 0, 0, 0, 2, 2, 0],
                       [0, 0, 0, 0, 0, 0, 2, 2, 0],
                       [0, 0, 0, 0, 0, 0, 2, 2, 0],
                       [0, 0, 0, 0, 0, 0, 2, 2, 0]],

                      [[0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [1, 1, 0, 0, 0, 0, 0, 0, 0],
                       [1, 1, 0, 0, 0, 0, 0, 0, 0],
                       [1, 1, 0, 0, 0, 0, 0, 0, 0],
                       [1, 1, 0, 0, 0, 0, 2, 2, 0],
                       [0, 0, 0, 0, 0, 0, 2, 2, 0],
                       [0, 0, 0, 0, 0, 0, 2, 2, 0],
                       [0, 0, 0, 0, 0, 0, 2, 2, 0],
                       [0, 0, 0, 0, 0, 0, 2, 2, 0]]
                      ])

    arr_2 = np.array([[[0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [1, 1, 0, 0, 0, 0, 0, 0, 0],
                       [1, 1, 0, 0, 0, 0, 0, 0, 0],
                       [1, 1, 0, 0, 0, 0, 0, 0, 0],
                       [1, 1, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 2, 2, 2, 2, 2, 0],
                       [0, 0, 0, 2, 2, 2, 2, 2, 0],
                       [0, 0, 0, 2, 2, 2, 2, 2, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0]],

                      [[0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [1, 1, 0, 0, 0, 0, 0, 0, 0],
                       [1, 1, 1, 1, 1, 1, 1, 1, 0],
                       [1, 1, 0, 0, 0, 0, 0, 0, 0],
                       [1, 1, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 2, 2, 2, 2, 2, 0],
                       [0, 0, 0, 2, 2, 2, 2, 2, 0],
                       [0, 0, 0, 2, 2, 2, 2, 2, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0]]
                      ])

    arr_void = np.zeros_like(arr_2)

    im1 = nib.Nifti1Image(arr_1, np.eye(4))
    im2 = nib.Nifti1Image(arr_2, np.eye(4))
    im_void = nib.Nifti1Image(arr_void, np.eye(4))

    hd_1_1 = hausdorff_distance(im1, im1, [1, 2], ['label1', 'label2'])
    hd_1_2 = hausdorff_distance(im1, im2, [1, 2], ['label1', 'label2'])
    hd_1_2_extra = hausdorff_distance(im1, im2, [1, 2, 3], ['label1', 'label2', 'label3'])
    hd_1_void = hausdorff_distance(im1, im_void, [1, 2], ['label1', 'label2'])

    assert_almost_equal(hd_1_1['label1'], 0)
    assert_almost_equal(hd_1_1['label2'], 0)

    assert_almost_equal(hd_1_2['label1'], 6)
    assert_almost_equal(hd_1_2['label2'], 3)

    assert_almost_equal(hd_1_2_extra['label1'], 6)
    assert_almost_equal(hd_1_2_extra['label2'], 3)
    assert_almost_equal(hd_1_2_extra['label3'], np.nan)

    assert_almost_equal(hd_1_void['label1'], np.nan)
    assert_almost_equal(hd_1_void['label2'], np.nan)



# test symmetric_contour_distance


def test_normalised_symetric_contour_distance(save_data_path='/Users/aaabbbccc/Desktop', verbose=False):
    o1 = o_shape(omega=(19, 19, 19), radius=7, background_intensity=0, foreground_intensity=1, dtype=np.uint8)
    o2 = 2 * o1
    arr1 = np.concatenate([o1, o2], axis=2)

    half_o1 = np.zeros_like(o1)
    half_o1[:, :10, :] = o1[:, :10, :]
    half_o2 = 2 * half_o1
    arr2 = np.concatenate([half_o1, half_o2], axis=2)

    c1 = cube_shape(omega=(19, 19, 19), center=(9, 9, 9), side_length=13, background_intensity=0, foreground_intensity=1, dtype=np.uint8)
    c2 = 2 * c1
    arr3 = np.concatenate([c1, c2], axis=2)

    arr_void = np.zeros_like(arr1)

    im1 = nib.Nifti1Image(arr1, np.eye(4))
    im2 = nib.Nifti1Image(arr2, np.eye(4))
    im3 = nib.Nifti1Image(arr3, np.eye(4))

    im_void = nib.Nifti1Image(arr_void, np.eye(4))

    if os.path.exists(save_data_path):

        nib.save(im1, os.path.join(save_data_path, 'zzz1.nii.gz'))
        nib.save(im2, os.path.join(save_data_path, 'zzz2.nii.gz'))
        nib.save(im3, os.path.join(save_data_path, 'zzz3.nii.gz'))

        im1_cont = contour_from_segmentation(im1)
        im2_cont = contour_from_segmentation(im2)
        im3_cont = contour_from_segmentation(im3)

        nib.save(im1_cont, os.path.join(save_data_path, 'zzz1_contour.nii.gz'))
        nib.save(im2_cont, os.path.join(save_data_path, 'zzz2_contour.nii.gz'))
        nib.save(im3_cont, os.path.join(save_data_path, 'zzz3_contour.nii.gz'))

        im_dtb1_l1 = set_new_data(im1_cont, nd.distance_transform_edt(1 - (im1_cont.get_data() == 1)))
        im_dtb2_l1 = set_new_data(im2_cont, nd.distance_transform_edt(1 - (im2_cont.get_data() == 1)))
        im_dtb3_l1 = set_new_data(im3_cont, nd.distance_transform_edt(1 - (im3_cont.get_data() == 1)))

        im_dtb1_l2 = set_new_data(im1_cont, nd.distance_transform_edt(1 - (im1_cont.get_data() == 2)))
        im_dtb2_l2 = set_new_data(im2_cont, nd.distance_transform_edt(1 - (im2_cont.get_data() == 2)))
        im_dtb3_l2 = set_new_data(im3_cont, nd.distance_transform_edt(1 - (im3_cont.get_data() == 2)))

        nib.save(im_dtb1_l1, os.path.join(save_data_path, 'zzz1_contour_dist_label1.nii.gz'))
        nib.save(im_dtb2_l1, os.path.join(save_data_path, 'zzz2_contour_dist_label1.nii.gz'))
        nib.save(im_dtb3_l1, os.path.join(save_data_path, 'zzz3_contour_dist_label1.nii.gz'))

        nib.save(im_dtb1_l2, os.path.join(save_data_path, 'zzz1_contour_dist_label2.nii.gz'))
        nib.save(im_dtb2_l2, os.path.join(save_data_path, 'zzz2_contour_dist_label2.nii.gz'))
        nib.save(im_dtb3_l2, os.path.join(save_data_path, 'zzz3_contour_dist_label2.nii.gz'))

        print('Images testing saved in {}'.format(save_data_path))

    ascd_1_1       = normalised_symmetric_contour_distance(im1, im1, [1, 2], ['label1', 'label2'])
    ascd_1_2       = normalised_symmetric_contour_distance(im1, im2, [1, 2], ['label1', 'label2'])
    ascd_1_3       = normalised_symmetric_contour_distance(im1, im3, [1, 2], ['label1', 'label2'])
    ascd_1_2_extra = normalised_symmetric_contour_distance(im1, im2, [1, 2, 3], ['label1', 'label2', 'label3'])
    ascd_1_void    = normalised_symmetric_contour_distance(im1, im_void, [1, 2], ['label1', 'label2'])

    if verbose:
        print('\n 1 1')
        print(ascd_1_1)
        print('\n 1 2')
        print(ascd_1_2)
        print('\n 1 3')
        print(ascd_1_3)
        print('\n 1 2 extra')
        print(ascd_1_2_extra)
        print('\n 1 void')
        print(ascd_1_void)

    assert_almost_equal(ascd_1_1['label1'], 0)
    assert_almost_equal(ascd_1_1['label2'], 0)

    assert_almost_equal(ascd_1_2['label1'], 1.01226261459, decimal=4)
    assert_almost_equal(ascd_1_2['label2'], 1.01226261459, decimal=4)

    assert_almost_equal(ascd_1_3['label1'], 0.866793279546 , decimal=4)
    assert_almost_equal(ascd_1_3['label2'], 0.866793279546, decimal=4)

    assert_almost_equal(ascd_1_2_extra['label3'], np.nan)

    assert_almost_equal(ascd_1_void['label1'], np.nan)
    assert_almost_equal(ascd_1_void['label2'], np.nan)


# --- extra:


def test_box_side_lenght():
    arr_1 = np.array([[[0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [1, 1, 0, 0, 0, 0, 0, 0, 0],
                       [1, 1, 0, 0, 0, 0, 0, 0, 0],
                       [1, 1, 0, 0, 0, 0, 0, 0, 0],
                       [1, 1, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 2, 2, 2, 2, 2, 0],
                       [0, 0, 0, 2, 2, 2, 2, 2, 0],
                       [0, 0, 0, 2, 2, 2, 2, 2, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0]],

                      [[0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [1, 1, 0, 0, 0, 0, 0, 0, 0],
                       [1, 1, 1, 1, 1, 1, 1, 1, 0],
                       [1, 1, 0, 0, 0, 0, 0, 0, 0],
                       [1, 1, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 2, 2, 2, 2, 2, 0],
                       [0, 0, 0, 2, 2, 2, 2, 2, 0],
                       [0, 0, 0, 2, 2, 2, 2, 2, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0]]
                      ])
    im1 = nib.Nifti1Image(arr_1, np.eye(4))

    se_answer = box_sides_length(im1, [0, 1, 2, 3], ['0', '1', '2', '3'])

    assert_array_equal(se_answer['0'], [1., 8., 8.])
    assert_array_equal(se_answer['1'], [1., 3., 7.])
    assert_array_equal(se_answer['2'], [1., 2., 4.])
    assert_equal(se_answer['3'], np.nan)

