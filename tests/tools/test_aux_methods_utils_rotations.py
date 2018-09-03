import numpy as np
from numpy.testing import assert_array_equal, assert_raises

from nilabels.tools.aux_methods.utils_rotations import get_small_orthogonal_rotation, get_roto_translation_matrix, \
    basic_90_rot_ax, axial_90_rotations, flip_data, symmetrise_data


# TEST aux_methods.utils_rotations : get_small_orthogonal_rotation_yaw


def test_get_small_orthogonal_rotation_yaw():
    theta = np.pi / 8
    expected_rot = np.array([[np.cos(theta), -np.sin(theta), 0, 0],
                             [np.sin(theta), np.cos(theta), 0, 0],
                             [0, 0, 1, 0],
                             [0, 0, 0, 1]])
    rot = get_small_orthogonal_rotation(theta, 'yaw')
    assert_array_equal(rot, expected_rot)

    theta = - np.pi / 12
    expected_rot = np.array([[np.cos(theta), -np.sin(theta), 0, 0],
                             [np.sin(theta), np.cos(theta), 0, 0],
                             [0, 0, 1, 0],
                             [0, 0, 0, 1]])
    rot = get_small_orthogonal_rotation(theta, 'yaw')
    assert_array_equal(rot, expected_rot)


def test_get_small_orthogonal_rotation_pitch():
    theta = np.pi / 8
    expected_rot = np.array([[1,            0,           0,       0],
                             [0,  np.cos(theta),  -np.sin(theta), 0],
                             [0,  np.sin(theta), np.cos(theta),   0],
                             [0,             0,          0,       1]])
    rot = get_small_orthogonal_rotation(theta, 'pitch')
    assert_array_equal(rot, expected_rot)

    theta = - np.pi / 6
    expected_rot = np.array([[1,            0,           0,       0],
                             [0,  np.cos(theta),  -np.sin(theta), 0],
                             [0,  np.sin(theta), np.cos(theta),   0],
                             [0,             0,          0,       1]])
    rot = get_small_orthogonal_rotation(theta, 'pitch')
    assert_array_equal(rot, expected_rot)


def test_get_small_orthogonal_rotation_roll():
    theta = np.pi / 9
    expected_rot = np.array([[np.cos(theta), 0, np.sin(theta),  0],
                             [0,             1,      0,         0],
                             [-np.sin(theta), 0, np.cos(theta), 0],
                             [0,             0,      0,         1]])
    rot = get_small_orthogonal_rotation(theta, 'roll')
    assert_array_equal(rot, expected_rot)

    theta = - np.pi / 11
    expected_rot = np.array([[np.cos(theta), 0, np.sin(theta),  0],
                             [0,             1,      0,         0],
                             [-np.sin(theta), 0, np.cos(theta), 0],
                             [0,             0,      0,         1]])
    rot = get_small_orthogonal_rotation(theta, 'roll')
    assert_array_equal(rot, expected_rot)


def test_get_small_orthogonal_rotation_zeros_theta():
    assert_array_equal(get_small_orthogonal_rotation(0, 'yaw'), np.eye(4))
    assert_array_equal(get_small_orthogonal_rotation(0, 'pitch'), np.eye(4))
    assert_array_equal(get_small_orthogonal_rotation(0, 'roll'), np.eye(4))


def test_get_small_orthogonal_rotation_unkonwn_axis():
    with assert_raises(IOError):
        get_small_orthogonal_rotation(0, 'spam')


# TEST aux_methods.utils_rotations : get_roto_translation_matrix


def test_get_roto_translation_matrix_too_small_rotation_axis():
    with assert_raises(IOError):
        get_roto_translation_matrix(np.pi/4, rotation_axis=np.array([0.0009, 0, 0]), translation=np.array([0, 0, 0]))


def test_get_roto_translation_matrix_get_check_shape_and_translation():
    rt = get_roto_translation_matrix(np.pi/8, rotation_axis=np.array([1, 0, 0]), translation=np.array([1, 2, 3]))
    assert cmp(rt.shape, (4, 4)) == 0
    assert_array_equal(rt[3, :], np.array([0, 0, 0, 1]))
    assert_array_equal(rt[:, 3], np.array([1, 2, 3, 1]))


def test_get_roto_translation_matrix_around_x_axis():
    # standard nifti convention (RAS) x -> pitch. y -> roll, z -> yaw
    theta    = np.pi / 9
    rot_axis = np.array([1, 0, 0])
    transl   = np.array([1, 2, 3])
    expected_rot = np.array([[1,            0,           0],
                             [0,  np.cos(theta),  -np.sin(theta)],
                             [0,  np.sin(theta), np.cos(theta)]])

    rt = get_roto_translation_matrix(theta, rot_axis, transl)
    expected_rt = np.eye(4)
    expected_rt[:3, :3] = expected_rot
    expected_rt[:3, 3] = transl

    assert_array_equal(rt, expected_rt)


def test_get_roto_translation_matrix_around_y_axis():
    # standard nifti convention (RAS) x -> pitch. y -> roll, z -> yaw
    theta    = np.pi / 9
    rot_axis = np.array([0, 1, 0])
    transl   = np.array([1, 2, 3])
    expected_rot = np.array([[np.cos(theta), 0, np.sin(theta)],
                             [0,             1,      0],
                             [-np.sin(theta), 0, np.cos(theta)]])

    rt = get_roto_translation_matrix(theta, rot_axis, transl)
    expected_rt = np.eye(4)
    expected_rt[:3, :3] = expected_rot
    expected_rt[:3, 3] = transl

    assert_array_equal(rt, expected_rt)


def test_get_roto_translation_matrix_around_z_axis():
    # standard nifti convention (RAS) x -> pitch, y -> roll, z -> yaw.
    theta    = np.pi / 9
    rot_axis = np.array([0, 0, 1])
    transl   = np.array([1, 2, 3])
    expected_rot = np.array([[np.cos(theta), -np.sin(theta), 0],
                             [np.sin(theta), np.cos(theta),  0],
                             [0,             0,              1]])

    rt = get_roto_translation_matrix(theta, rot_axis, transl)
    expected_rt = np.eye(4)
    expected_rt[:3, :3] = expected_rot
    expected_rt[:3, 3] = transl

    assert_array_equal(rt, expected_rt)


def test_basic_rotation_ax_simple_and_visual():

    cube_id = np.array([[[0, 1, 2, 3],
                         [4, 5, 6, 7],
                         [8, 9, 10, 11]],

                        [[12, 13, 14, 15],
                         [16, 17, 18, 19],
                         [20, 21, 22, 23]]])

    # counterclockwise looking front face axis
    cube_ax0 = np.array([[[3,  7, 11],
                          [2,  6, 10],
                          [1,  5,  9],
                          [0,  4,  8]],

                         [[15, 19, 23],
                          [14, 18, 22],
                          [13, 17, 21],
                          [12, 16, 20]]])

    # clockwise looking upper face axis
    cube_ax1 = np.array([[[3, 15],
                          [7, 19],
                          [11, 23]],

                         [[2, 14],
                          [6, 18],
                          [10, 22]],

                         [[1, 13],
                          [5, 17],
                          [9, 21]],

                         [[0, 12],
                          [4, 16],
                          [8, 20]]])

    # clockwise looking right face axis
    cube_ax2 = np.array([[[8,  9, 10, 11],
                          [20, 21, 22, 23]],

                         [[4,  5,  6,  7],
                          [16, 17, 18, 19]],

                         [[0,  1,  2,  3],
                          [12, 13, 14, 15]]])

    assert_array_equal(basic_90_rot_ax(cube_id, ax=0), cube_ax0)
    assert_array_equal(basic_90_rot_ax(cube_id, ax=1), cube_ax1)
    assert_array_equal(basic_90_rot_ax(cube_id, ax=2), cube_ax2)


def test_axial_90_rotations_4_rotations_invariance():
    cube = np.array(range(2 * 3 * 4)).reshape(2, 3, 4)
    for x in range(3):
        # rot = 4 rotates four times around the same axis (no changes) must give the input matrix.
        assert_array_equal(axial_90_rotations(cube, rot=4, ax=x), cube)


def test_axial_90_rotations_wrong_input_dimensions():
    with assert_raises(IOError):
        axial_90_rotations(np.ones([5, 5, 5, 5]), rot=1, ax=2)


def test_axial_90_rotations_around_x_standard_input_data():
    # input
    cube = np.array([[[0, 1],
                      [2, 3]],
                     [[4, 5],
                      [6, 7]]])
    # around front face (0)
    cube_rot_1_axis0 = np.array([[[1, 3],
                                  [0, 2]],
                                 [[5, 7],
                                  [4, 6]]])
    cube_rot_2_axis0 = np.array([[[3, 2],
                                  [1, 0]],
                                 [[7, 6],
                                  [5, 4]]])
    cube_rot_3_axis0 = np.array([[[2, 0],
                                  [3, 1]],
                                 [[6, 4],
                                  [7, 5]]])

    assert_array_equal(axial_90_rotations(cube, rot=3, ax=0), cube_rot_3_axis0)
    assert_array_equal(axial_90_rotations(cube, rot=2, ax=0), cube_rot_2_axis0)
    assert_array_equal(axial_90_rotations(cube, rot=1, ax=0), cube_rot_1_axis0)

    # around top-bottom face (1)
    cube_rot_1_axis1 = np.array([[[1, 5],
                                  [3, 7]],
                                 [[0, 4],
                                  [2, 6]]])
    cube_rot_2_axis1 = np.array([[[5, 4],
                                  [7, 6]],
                                 [[1, 0],
                                  [3, 2]]])
    cube_rot_3_axis1 = np.array([[[4, 0],
                                  [6, 2]],
                                 [[5, 1],
                                  [7, 3]]])

    assert_array_equal(axial_90_rotations(cube, rot=1, ax=1), cube_rot_1_axis1)
    assert_array_equal(axial_90_rotations(cube, rot=2, ax=1), cube_rot_2_axis1)
    assert_array_equal(axial_90_rotations(cube, rot=3, ax=1), cube_rot_3_axis1)

    # around front face (2)
    cube_rot_1_axis2 = np.array([[[2, 3],
                                  [6, 7]],
                                 [[0, 1],
                                  [4, 5]]])
    cube_rot_2_axis2 = np.array([[[6, 7],
                                  [4, 5]],
                                 [[2, 3],
                                  [0, 1]]])
    cube_rot_3_axis2 = np.array([[[4, 5],
                                  [0, 1]],
                                 [[6, 7],
                                  [2, 3]]])

    assert_array_equal(axial_90_rotations(cube, rot=1, ax=2), cube_rot_1_axis2)
    assert_array_equal(axial_90_rotations(cube, rot=2, ax=2), cube_rot_2_axis2)
    assert_array_equal(axial_90_rotations(cube, rot=3, ax=2), cube_rot_3_axis2)


def test_flip_data():

    cube_id = np.array([[[0, 1, 2, 3],
                         [4, 5, 6, 7],
                         [8, 9, 10, 11]],

                        [[12, 13, 14, 15],
                         [16, 17, 18, 19],
                         [20, 21, 22, 23]]])

    cube_flip_x = np.array([[[8, 9, 10, 11],
                             [4, 5, 6, 7],
                             [0, 1, 2, 3]],

                            [[20, 21, 22, 23],
                             [16, 17, 18, 19],
                             [12, 13, 14, 15]]])

    cube_flip_y = np.array([[[3, 2, 1, 0],
                             [7, 6, 5, 4],
                             [11, 10, 9, 8]],

                            [[15, 14, 13, 12],
                             [19, 18, 17, 16],
                             [23, 22, 21, 20]]])

    cube_flip_z = np.array([[[12, 13, 14, 15],
                              [16, 17, 18, 19],
                              [20, 21, 22, 23]],

                             [[0, 1, 2, 3],
                              [4, 5, 6, 7],
                              [8, 9, 10, 11]]])

    assert_array_equal(flip_data(cube_id, axis_direction='x'), cube_flip_x)
    assert_array_equal(flip_data(cube_id, axis_direction='y'), cube_flip_y)
    assert_array_equal(flip_data(cube_id, axis_direction='z'), cube_flip_z)


def test_flip_data_error_input():
    # TODO
    pass


def test_symmetrise_data_():

    cube_id = np.array([[[0, 1, 2, 3],
                         [4, 5, 6, 7],
                         [8, 9, 10, 11],
                         [12, 13, 14, 15]],
                        [[16, 17, 18, 19],
                         [20, 21, 22, 23],
                         [24, 25, 26, 27],
                         [28, 29, 30, 31]]])

    cube_sym_x2_ab_T = np.array([[[0,  1,  2,  3],
                                  [4,  5,  6,  7],
                                  [4,  5,  6,  7],
                                  [0,  1,  2,  3]],
                                 [[16, 17, 18, 19],
                                  [20, 21, 22, 23],
                                  [20, 21, 22, 23],
                                  [16, 17, 18, 19]]])

    cube_sym_x3_ab_F = np.array([[[0,  1,  2,  3],
                                  [4,  5,  6,  7],
                                  [8,  9, 10, 11],
                                  [8,  9, 10, 11],
                                  [4,  5,  6,  7],
                                  [0,  1,  2,  3]],
                                 [[16, 17, 18, 19],
                                  [20, 21, 22, 23],
                                  [24, 25, 26, 27],
                                  [24, 25, 26, 27],
                                  [20, 21, 22, 23],
                                  [16, 17, 18, 19]]])

    cube_sym_x3_ab_T = np.array([[[0, 1, 2, 3],
                                  [4, 5, 6, 7],
                                  [8, 9, 10, 11],
                                  [8, 9, 10, 11]],
                                 [[16, 17, 18, 19],
                                  [20, 21, 22, 23],
                                  [24, 25, 26, 27],
                                  [24, 25, 26, 27]]])

    assert_array_equal(symmetrise_data(cube_id, axis_direction='x', plane_intercept=2, side_to_copy='below',
                                       keep_in_data_dimensions=True), cube_sym_x2_ab_T)
    assert_array_equal(symmetrise_data(cube_id, axis_direction='x', plane_intercept=3, side_to_copy='below',
                                       keep_in_data_dimensions=False), cube_sym_x3_ab_F)
    assert_array_equal(symmetrise_data(cube_id, axis_direction='x', plane_intercept=3, side_to_copy='below',
                                       keep_in_data_dimensions=True), cube_sym_x3_ab_T)















if __name__ == '__main__':
    test_get_small_orthogonal_rotation_yaw()
    test_get_small_orthogonal_rotation_pitch()
    test_get_small_orthogonal_rotation_roll()
    test_get_small_orthogonal_rotation_zeros_theta()
    test_get_small_orthogonal_rotation_unkonwn_axis()

    test_get_roto_translation_matrix_too_small_rotation_axis()
    test_get_roto_translation_matrix_get_check_shape_and_translation()

    test_get_roto_translation_matrix_around_x_axis()
    test_get_roto_translation_matrix_around_y_axis()
    test_get_roto_translation_matrix_around_z_axis()

    test_basic_rotation_ax_simple_and_visual()

    test_axial_90_rotations_4_rotations_invariance()
    test_axial_90_rotations_around_x_standard_input_data()


    test_flip_data()
    test_symmetrise_data_()
