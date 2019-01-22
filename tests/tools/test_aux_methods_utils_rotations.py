import os
import numpy as np
from numpy.testing import assert_array_equal, assert_raises, assert_array_almost_equal

from nilabels.tools.aux_methods.utils_rotations import get_small_orthogonal_rotation, get_roto_translation_matrix, \
    basic_90_rot_ax, axial_90_rotations, flip_data, symmetrise_data, reorient_b_vect, reorient_b_vect_from_files, matrix_vector_field_product

from tests.tools.decorators_tools import create_and_erase_temporary_folder_with_a_dummy_b_vectors_list, pfo_tmp_test


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
    assert rt.shape == (4, 4)
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


def test_flip_data_error_input_direction():
    in_data = np.zeros([10, 10, 10])
    with assert_raises(IOError):
        flip_data(in_data, axis_direction='s')


def test_flip_data_error_dimension():
    in_data = np.zeros([10, 10, 10, 10])
    with assert_raises(IOError):
        flip_data(in_data, axis_direction='x')


def test_symmetrise_data_x_axis():

    cube_id = np.array([[[0, 1, 2, 3],
                         [4, 5, 6, 7],
                         [8, 9, 10, 11],
                         [12, 13, 14, 15]],
                        [[16, 17, 18, 19],
                         [20, 21, 22, 23],
                         [24, 25, 26, 27],
                         [28, 29, 30, 31]]])

    cube_sym_x2_be_T = np.array([[[0,  1,  2,  3],
                                  [4,  5,  6,  7],
                                  [4,  5,  6,  7],
                                  [0,  1,  2,  3]],
                                 [[16, 17, 18, 19],
                                  [20, 21, 22, 23],
                                  [20, 21, 22, 23],
                                  [16, 17, 18, 19]]])

    cube_sym_x2_ab_T = np.array([[[12, 13, 14, 15],
                                  [8, 9, 10, 11],
                                  [8, 9, 10, 11],
                                  [12, 13, 14, 15]],
                                 [[28, 29, 30, 31],
                                  [24, 25, 26, 27],
                                  [24, 25, 26, 27],
                                  [28, 29, 30, 31]]])

    cube_sym_x3_be_F = np.array([[[0,  1,  2,  3],
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

    cube_sym_x3_be_T = np.array([[[0, 1, 2, 3],
                                  [4, 5, 6, 7],
                                  [8, 9, 10, 11],
                                  [8, 9, 10, 11]],
                                 [[16, 17, 18, 19],
                                  [20, 21, 22, 23],
                                  [24, 25, 26, 27],
                                  [24, 25, 26, 27]]])

    assert_array_equal(symmetrise_data(cube_id, axis_direction='x', plane_intercept=2, side_to_copy='below',
                                       keep_in_data_dimensions_boundaries=True), cube_sym_x2_be_T)
    assert_array_equal(symmetrise_data(cube_id, axis_direction='x', plane_intercept=2, side_to_copy='above',
                                       keep_in_data_dimensions_boundaries=True), cube_sym_x2_ab_T)
    assert_array_equal(symmetrise_data(cube_id, axis_direction='x', plane_intercept=3, side_to_copy='below',
                                       keep_in_data_dimensions_boundaries=False), cube_sym_x3_be_F)
    assert_array_equal(symmetrise_data(cube_id, axis_direction='x', plane_intercept=3, side_to_copy='below',
                                       keep_in_data_dimensions_boundaries=True), cube_sym_x3_be_T)


def test_symmetrise_data_y_axis():

    cube_id = np.array([[[0, 1, 2, 3],
                         [4, 5, 6, 7],
                         [8, 9, 10, 11],
                         [12, 13, 14, 15]],
                        [[16, 17, 18, 19],
                         [20, 21, 22, 23],
                         [24, 25, 26, 27],
                         [28, 29, 30, 31]]])

    cube_sym_y2_be_T = np.array([[[0, 1, 1, 0],
                                  [4, 5, 5, 4],
                                  [8, 9, 9, 8],
                                  [12, 13, 13, 12]],
                                 [[16, 17, 17, 16],
                                  [20, 21, 21, 20],
                                  [24, 25, 25, 24],
                                  [28, 29, 29, 28]]])

    cube_sym_y2_ab_T = np.array([[[3, 2, 2, 3],
                                  [7, 6, 6, 7],
                                  [11, 10, 10, 11],
                                  [15, 14, 14, 15]],
                                 [[19, 18, 18, 19],
                                  [23, 22, 22, 23],
                                  [27, 26, 26, 27],
                                  [31, 30, 30, 31]]])

    cube_sym_y3_be_F = np.array([[[0, 1, 2, 2, 1, 0],
                                  [4, 5, 6, 6, 5, 4],
                                  [8, 9, 10, 10, 9, 8],
                                  [12, 13, 14, 14, 13, 12]],
                                 [[16, 17, 18, 18, 17, 16],
                                  [20, 21, 22, 22, 21, 20],
                                  [24, 25, 26, 26, 25, 24],
                                  [28, 29, 30, 30, 29, 28]]])

    cube_sym_y3_be_T = np.array([[[0, 1, 2, 2],
                                  [4, 5, 6, 6],
                                  [8, 9, 10, 10],
                                  [12, 13, 14, 14]],
                                 [[16, 17, 18, 18],
                                  [20, 21, 22, 22],
                                  [24, 25, 26, 26],
                                  [28, 29, 30, 30]]])

    assert_array_equal(symmetrise_data(cube_id, axis_direction='y', plane_intercept=2, side_to_copy='below',
                                       keep_in_data_dimensions_boundaries=True), cube_sym_y2_be_T)
    assert_array_equal(symmetrise_data(cube_id, axis_direction='y', plane_intercept=2, side_to_copy='above',
                                       keep_in_data_dimensions_boundaries=True), cube_sym_y2_ab_T)
    assert_array_equal(symmetrise_data(cube_id, axis_direction='y', plane_intercept=3, side_to_copy='below',
                                       keep_in_data_dimensions_boundaries=False), cube_sym_y3_be_F)
    assert_array_equal(symmetrise_data(cube_id, axis_direction='y', plane_intercept=3, side_to_copy='below',
                                       keep_in_data_dimensions_boundaries=True), cube_sym_y3_be_T)


def test_symmetrise_data_z_axis():

    cube_id = np.array([[[0, 1, 2, 3],
                         [4, 5, 6, 7],
                         [8, 9, 10, 11],
                         [12, 13, 14, 15]],
                        [[16, 17, 18, 19],
                         [20, 21, 22, 23],
                         [24, 25, 26, 27],
                         [28, 29, 30, 31]],
                        [[32, 33, 34, 35],
                         [36, 37, 38, 39],
                         [40, 41, 42, 43],
                         [44, 45, 46, 47]]])

    cube_sym_z2_be_T = np.array([[[0, 1, 2, 3],
                                  [4, 5, 6, 7],
                                  [8, 9, 10, 11],
                                  [12, 13, 14, 15]],
                                 [[16, 17, 18, 19],
                                  [20, 21, 22, 23],
                                  [24, 25, 26, 27],
                                  [28, 29, 30, 31]],
                                 [[16, 17, 18, 19],
                                  [20, 21, 22, 23],
                                  [24, 25, 26, 27],
                                  [28, 29, 30, 31]]])

    cube_sym_z1_ab_T = np.array([[[16, 17, 18, 19],
                                  [20, 21, 22, 23],
                                  [24, 25, 26, 27],
                                  [28, 29, 30, 31]],
                                 [[16, 17, 18, 19],
                                  [20, 21, 22, 23],
                                  [24, 25, 26, 27],
                                  [28, 29, 30, 31]],
                                 [[32, 33, 34, 35],
                                  [36, 37, 38, 39],
                                  [40, 41, 42, 43],
                                  [44, 45, 46, 47]]])

    cube_sym_z1_ab_F = np.array([[[32, 33, 34, 35],
                                  [36, 37, 38, 39],
                                  [40, 41, 42, 43],
                                  [44, 45, 46, 47]],
                                 [[16, 17, 18, 19],
                                  [20, 21, 22, 23],
                                  [24, 25, 26, 27],
                                  [28, 29, 30, 31]],
                                 [[16, 17, 18, 19],
                                  [20, 21, 22, 23],
                                  [24, 25, 26, 27],
                                  [28, 29, 30, 31]],
                                 [[32, 33, 34, 35],
                                  [36, 37, 38, 39],
                                  [40, 41, 42, 43],
                                  [44, 45, 46, 47]]])

    assert_array_equal(symmetrise_data(cube_id, axis_direction='z', plane_intercept=2, side_to_copy='below',
                                       keep_in_data_dimensions_boundaries=True), cube_sym_z2_be_T)
    assert_array_equal(symmetrise_data(cube_id, axis_direction='z', plane_intercept=1, side_to_copy='above',
                                       keep_in_data_dimensions_boundaries=True), cube_sym_z1_ab_T)
    assert_array_equal(symmetrise_data(cube_id, axis_direction='z', plane_intercept=1, side_to_copy='above',
                                       keep_in_data_dimensions_boundaries=False), cube_sym_z1_ab_F)


def test_symmetrise_data_error_input_ndim():
    with assert_raises(IOError):
        symmetrise_data(np.ones([5, 5, 5, 5]))


def test_symmetrise_data_error_input_side_to_copy():
    with assert_raises(IOError):
        symmetrise_data(np.ones([5, 5, 5]), side_to_copy='spam')


def test_symmetrise_data_error_input_axis_direction():
    with assert_raises(IOError):
        symmetrise_data(np.ones([5, 5, 5]), axis_direction='s')


def test_reorient_b_vect():
    n = 10  # 10 random b-vectors
    b_vects = np.random.randn(n, 3)
    transformation = np.random.randn(3, 3)

    expected_answer = np.zeros_like(b_vects)

    for r in range(n):
        expected_answer[r, :] = transformation.dot(b_vects[r, :])

    assert_array_almost_equal(reorient_b_vect(b_vects, transformation), expected_answer)


@create_and_erase_temporary_folder_with_a_dummy_b_vectors_list
def test_reorient_b_vect_from_files():
    in_b_vects = np.loadtxt(os.path.join(pfo_tmp_test, 'b_vects_file.txt'))

    transformation = np.random.randn(3, 3)

    expected_saved_answer = np.zeros_like(in_b_vects)

    for r in range(in_b_vects.shape[0]):
        expected_saved_answer[r, :] = transformation.dot(in_b_vects[r, :])

    reorient_b_vect_from_files(os.path.join(pfo_tmp_test, 'b_vects_file.txt'),
                               os.path.join(pfo_tmp_test, 'b_vects_file_reoriented.txt'),
                               transformation)

    loaded_answer = np.loadtxt(os.path.join(pfo_tmp_test, 'b_vects_file_reoriented.txt'))

    assert_array_almost_equal(loaded_answer, expected_saved_answer)


def test_matrix_vector_fields_product():
    j_input = np.random.randn(10, 15, 4)
    v_input = np.random.randn(10, 15, 2)

    d = v_input.shape[-1]
    vol = list(v_input.shape[:-1])
    v = np.tile(v_input, [1] * d + [d])
    j_times_v = np.multiply(j_input, v)

    expected_answer = np.sum(j_times_v.reshape(vol + [d, d]), axis=d + 1).reshape(vol + [d])

    obtained_answer = matrix_vector_field_product(j_input, v_input)

    assert_array_almost_equal(expected_answer, obtained_answer)


def test_matrix_vector_fields_product_3d():
    j_input = np.random.randn(10, 15, 18, 9)
    v_input = np.random.randn(10, 15, 18, 3)

    d = v_input.shape[-1]
    vol = list(v_input.shape[:-1])
    v = np.tile(v_input, [1] * d + [d])
    j_times_v = np.multiply(j_input, v)

    expected_answer = np.sum(j_times_v.reshape(vol + [d, d]), axis=d + 1).reshape(vol + [d])

    obtained_answer = matrix_vector_field_product(j_input, v_input)

    assert_array_almost_equal(expected_answer, obtained_answer)


def test_matrix_vector_fields_product_3d_bad_input():
    j_input = np.random.randn(10, 15, 3, 9)
    v_input = np.random.randn(10, 15, 3)
    with assert_raises(IOError):
        matrix_vector_field_product(j_input, v_input)

    j_input = np.random.randn(10, 15, 9)
    v_input = np.random.randn(10, 14, 3)
    with assert_raises(IOError):
        matrix_vector_field_product(j_input, v_input)


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
    test_flip_data_error_input_direction()
    test_flip_data_error_dimension()

    test_symmetrise_data_x_axis()
    test_symmetrise_data_y_axis()
    test_symmetrise_data_z_axis()
    test_symmetrise_data_error_input_ndim()
    test_symmetrise_data_error_input_side_to_copy()
    test_symmetrise_data_error_input_axis_direction()

    test_reorient_b_vect()
    test_reorient_b_vect_from_files()

    test_matrix_vector_fields_product()
    test_matrix_vector_fields_product_3d()
    test_matrix_vector_fields_product_3d_bad_input()

