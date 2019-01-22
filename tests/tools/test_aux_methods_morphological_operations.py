import numpy as np
from numpy.testing import assert_array_equal, assert_raises

from nilabels.tools.aux_methods.morpological_operations import get_morphological_patch, get_morphological_mask, \
    get_values_below_patch, get_circle_shell_for_given_radius


# TEST aux_methods.morphological.py


def test_get_morpological_patch():
    expected = np.ones([3, 3]).astype(np.bool)
    expected[0, 0] = False
    expected[0, 2] = False
    expected[2, 0] = False
    expected[2, 2] = False
    assert_array_equal(get_morphological_patch(2, 'circle'), expected)
    assert_array_equal(get_morphological_patch(2, 'square'), np.ones([3, 3]).astype(np.bool))


def test_get_morpological_patch_not_allowed_input():
    with assert_raises(IOError):
        get_morphological_patch(2, 'spam')


def test_get_morphological_mask_not_allowed_input():
    with assert_raises(IOError):
        get_morphological_mask((5, 5), (11, 11), radius=2, shape='spam')


def test_get_morphological_mask_with_morpho_patch():
    morpho_patch = np.array([[[False,  True, False],
                              [True,  True,  True],
                              [False,  True, False]],

                             [[True,  True,  True],
                              [True,  False,  True],
                              [True,  True,  True]],

                             [[False,  True, False],
                              [True,  True,  True],
                              [False,  True, False]]])

    arr_mask = get_morphological_mask((2, 2, 2), (4, 4, 4), radius=1, shape='unused', morpho_patch=morpho_patch)

    expected_arr_mask = np.array([[[False, False,  False, False],
                                   [False, False,  False, False],
                                   [False, False,  False,  False],
                                   [False, False,  False, False]],

                                  [[False, False,  False, False],
                                   [False, False,  True, False],
                                   [False, True,  True,  True],
                                   [False, False,  True, False]],

                                  [[False, False, False, False],
                                   [False, True,  True,  True],
                                   [False, True,  False,  True],
                                   [False, True,  True,  True]],

                                  [[False, False,  False, False],
                                   [False, False,  True, False],
                                   [False, True,  True,  True],
                                   [False, False,  True, False]]])

    assert_array_equal(arr_mask, expected_arr_mask)


def test_get_morphological_mask_with_zero_radius():
    arr_mask = get_morphological_mask((2, 2, 2), (5, 5, 5), radius=0, shape='circle')

    expected_arr_mask = np.zeros((5, 5, 5), dtype=np.bool)
    expected_arr_mask[2, 2, 2] = 1

    assert_array_equal(arr_mask, expected_arr_mask)


def test_get_morphological_mask_without_morpho_patch():
    arr_mask = get_morphological_mask((2, 2), (5, 5), radius=2, shape='circle')
    expected_arr_mask = np.array([[False, False,  True, False, False],
                                  [False,  True,  True,  True, False],
                                  [True,  True,  True,  True,  True],
                                  [False,  True,  True,  True, False],
                                  [False, False,  True, False, False]])
    assert_array_equal(arr_mask, expected_arr_mask)


def test_get_patch_values_simple():
    # toy mask on a simple image:
    image = np.random.randint(0, 10, (7, 7))
    patch = np.zeros_like(image).astype(np.bool)
    patch[2, 2] = True
    patch[2, 3] = True
    patch[3, 2] = True
    patch[3, 3] = True

    vals = get_values_below_patch([2, 2, 2], image, morpho_mask=patch)
    assert_array_equal([image[2, 2], image[2, 3], image[3, 2], image[3, 3]], vals)


def test_get_values_below_patch_no_morpho_mask():
    image = np.ones((7, 7))
    vals = get_values_below_patch([3, 3], image, radius=1, shape='square')

    assert_array_equal([1.0, ] * 9, vals)


def test_get_shell_for_given_radius():
    expected_ans = [(-2, 0, 0), (-1, -1, -1), (-1, -1, 0), (-1, -1, 1), (-1, 0, -1), (-1, 0, 1), (-1, 1, -1),
                    (-1, 1, 0), (-1, 1, 1), (0, -2, 0), (0, -1, -1), (0, -1, 1), (0, 0, -2), (0, 0, 2), (0, 1, -1),
                    (0, 1, 1), (0, 2, 0), (1, -1, -1), (1, -1, 0), (1, -1, 1), (1, 0, -1), (1, 0, 1), (1, 1, -1),
                    (1, 1, 0), (1, 1, 1), (2, 0, 0)]
    computed_ans = get_circle_shell_for_given_radius(2)

    assert_array_equal(expected_ans, computed_ans)


def get_circle_shell_for_given_radius_2d():
    expected_ans = [(-2, 0), (-1, -1), (-1, 1), (0, -2), (0, 2), (1, -1), (1, 1), (2, 0)]
    computed_ans = get_circle_shell_for_given_radius(2, dimension=2)
    np.testing.assert_array_equal(expected_ans, computed_ans)


def get_circle_shell_for_given_radius_3_2d():
    expected_ans = [(-3, 0), (-2, -2), (-2, -1), (-2, 1), (-2, 2), (-1, -2), (-1, 2), (0, -3), (0, 3), (1, -2),
                    (1, 2), (2, -2), (2, -1), (2, 1), (2, 2), (3, 0)]
    computed_ans = get_circle_shell_for_given_radius(3, dimension=2)
    assert_array_equal(expected_ans, computed_ans)


def get_circle_shell_for_given_radius_wrong_input_nd():
    with assert_raises(IOError):
        get_circle_shell_for_given_radius(2, dimension=4)
    with assert_raises(IOError):
        get_circle_shell_for_given_radius(2, dimension=1)


if __name__ == '__main__':
    test_get_morpological_patch()
    test_get_morpological_patch_not_allowed_input()
    test_get_morphological_mask_not_allowed_input()
    test_get_morphological_mask_with_morpho_patch()
    test_get_morphological_mask_with_zero_radius()
    test_get_morphological_mask_without_morpho_patch()
    test_get_values_below_patch_no_morpho_mask()
    test_get_patch_values_simple()
    test_get_shell_for_given_radius()
    get_circle_shell_for_given_radius_2d()
    get_circle_shell_for_given_radius_3_2d()
    get_circle_shell_for_given_radius_wrong_input_nd()
