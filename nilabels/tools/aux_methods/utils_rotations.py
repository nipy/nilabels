import numpy as np


def get_small_orthogonal_rotation(theta, principal_axis='pitch'):
    if principal_axis == 'yaw':
        rot = np.array([[np.cos(theta), -np.sin(theta), 0, 0],
                        [np.sin(theta), np.cos(theta),  0, 0],
                        [0,             0,              1, 0],
                        [0,             0,              0, 1]])
    elif principal_axis == 'pitch':
        rot = np.array([[1,            0,           0,       0],
                        [0,  np.cos(theta),  -np.sin(theta), 0],
                        [0,  np.sin(theta), np.cos(theta),   0],
                        [0,             0,          0,       1]])
    elif principal_axis == 'roll':
        rot = np.array([[np.cos(theta), 0, np.sin(theta),  0],
                        [0,             1,      0,         0],
                        [-np.sin(theta), 0, np.cos(theta), 0],
                        [0,             0,      0,         1]])
    else:
        raise IOError('principal_axis parameter can be pitch, roll or yaw')

    return rot  # to be multiplied on the right side as im_input.affine.dot(rot)


def get_roto_translation_matrix(theta, rotation_axis=np.array([1, 0, 0]),  translation=np.array([0, 0, 0])):
    """
    Exploit the fact that every rotation is uniquely defined by an angle and a rotation direction.
    :param theta: rotation parameter
    :param rotation_axis: rotation axis (3d vector)
    :param translation: tranlsational part.
    :return: the conventional nifti header roto-translational matrix [[Rot | Transl][ 0, 0, 0 | 1]].
    """
    n = np.linalg.norm(rotation_axis)
    if np.abs(n) < 0.001:
        raise IOError('Input rotation axis too close to zero.')
    rot_versor = rotation_axis / n

    # Rodriguez magic formula for rotation part:
    skew_rot_versor = np.array([[0, -rot_versor[2], rot_versor[1]],
                                [rot_versor[2], 0, -rot_versor[0]],
                                [-rot_versor[1], rot_versor[0], 0]])
    rot_part = np.eye(3) + np.sin(theta) * skew_rot_versor + (1 - np.cos(theta)) * skew_rot_versor.dot(skew_rot_versor)

    # transformations parameters
    rot_transl = np.identity(4)
    rot_transl[:3, :3] = rot_part
    rot_transl[:3, 3] = translation

    return rot_transl


def basic_90_rot_ax(m, ax=0):
    """
    Basic rotations of a 3d matrix. Ingredient of the method axial_rotations.
    ----------
    Example:

    cube = array([[[0, 1],
                   [2, 3]],

                  [[4, 5],
                   [6, 7]]])

    axis 0: perpendicular to the face [[0,1],[2,3]] (front-rear)
    axis 1: perpendicular to the face [[0,1],[5,4]] (top-bottom)
    axis 2: perpendicular to the face [[1,5],[3,7]] (right-left)
    ----------
    Note: the command m[:, ::-1, :].swapaxes(0, 1)[::-1, :, :].swapaxes(0, 2) rotates the cube m
    around the diagonal axis 0-7.
    ----------
    Note: avoid reorienting the data if you can reorient the header instead.
    :param m: 3d matrix
    :param ax: axis of rotation
    :return: rotate the cube around axis ax, perpendicular to the face [[0,1],[2,3]]
    """
    ax %= 3

    if ax == 0:
        return np.rot90(m[:, ::-1, :].swapaxes(0, 1)[::-1, :, :].swapaxes(0, 2), 3)
    if ax == 1:
        return m.swapaxes(0, 2)[::-1, :, :]
    if ax == 2:
        return np.rot90(m, 1)


def axial_90_rotations(m, rot=1, ax=2):
    """
    :param m: 3d matrix
    :param rot: number of rotations
    :param ax: axis of rotation
    :return: m rotate rot times around axis ax, according to convention.
    """

    if m.ndim is not 3:
        raise IOError('Input matrix must be a 3d volume.')

    rot %= 4

    if rot == 0:
        return m

    for _ in range(rot):
        m = basic_90_rot_ax(m, ax=ax)

    return m


def flip_data(in_data, axis_direction='x'):
    """
    Flip an array along one dimension and respect to one orthogonal axis
    :param in_data: input array
    :param axis_direction: axis for the flipping
    :return: in_data flipped respect to the axis_direction.
    """
    if not in_data.ndim == 3:
        msg = 'Input array must be 3-dimensional.'
        raise IOError(msg)

    if axis_direction == 'x':
        out_data = in_data[:, ::-1, :]
    elif axis_direction == 'y':
        out_data = in_data[:, :, ::-1]
    elif axis_direction == 'z':
        out_data = in_data[::-1, :, :]
    else:
        msg = 'axis variable must be one of the following: {}.'.format(['x', 'y', 'z'])
        raise IOError(msg)

    return out_data


def symmetrise_data(in_data, axis_direction='x', plane_intercept=10, side_to_copy='below',
                    keep_in_data_dimensions_boundaries=True):
    """
    Symmetrise the input_array according to the axial plane
      axis = plane_intercept
    the copied part can be 'below' or 'above' the axes, following the ordering.

    :param in_data: (Z, X, Y) C convention input data
    :param axis_direction:
    :param plane_intercept:
    :param side_to_copy:
    :param keep_in_data_dimensions_boundaries:
    :return:
    """

    # Sanity check:

    if not in_data.ndim == 3:
        raise IOError('Input array must be 3-dimensional.')

    if side_to_copy not in ['below', 'above']:
        raise IOError('side_to_copy must be one of the two {}.'.format(['below', 'above']))

    if axis_direction not in ['x', 'y', 'z']:
        raise IOError('axis variable must be one of the following: {}.'.format(['x', 'y', 'z']))

    # step 1: find the block to symmetrise.
    # step 2: create the symmetric and glue it to the block.
    # step 3: add or remove a patch of slices if required to keep the in_data dimension.

    out_data = None

    if axis_direction == 'x':

        if side_to_copy == 'below':
            s_block = in_data[:, :plane_intercept, :]
            s_block_symmetric = s_block[:, ::-1, :]
            out_data = np.concatenate((s_block, s_block_symmetric), axis=1)

        if side_to_copy == 'above':
            s_block = in_data[:, plane_intercept:, :]
            s_block_symmetric = s_block[:, ::-1, :]
            out_data = np.concatenate((s_block_symmetric, s_block), axis=1)

    if axis_direction == 'y':

        if side_to_copy == 'below':
            s_block = in_data[:, :, :plane_intercept]
            s_block_symmetric = s_block[:, :, ::-1]
            out_data = np.concatenate((s_block, s_block_symmetric), axis=2)

        if side_to_copy == 'above':
            s_block = in_data[:, :, plane_intercept:]
            s_block_symmetric = s_block[:, :, ::-1]
            out_data = np.concatenate((s_block_symmetric, s_block), axis=2)

    if axis_direction == 'z':

        if side_to_copy == 'below':
            s_block = in_data[:plane_intercept, :, :]
            s_block_symmetric = s_block[::-1, :, :]
            out_data = np.concatenate((s_block, s_block_symmetric), axis=0)

        if side_to_copy == 'above':
            s_block = in_data[plane_intercept:, :, :]
            s_block_symmetric = s_block[::-1, :, :]
            out_data = np.concatenate((s_block_symmetric, s_block), axis=0)

    if keep_in_data_dimensions_boundaries:

        if side_to_copy == 'below':
            out_data = out_data[:in_data.shape[0], :in_data.shape[1], :in_data.shape[2]]

        if side_to_copy == 'above':
            out_data = out_data[out_data.shape[0] - in_data.shape[0]:,
                                out_data.shape[1] - in_data.shape[1]:,
                                out_data.shape[2] - in_data.shape[2]:]

    return out_data


def reorient_b_vect(bvect_array, transform):
    """
    Reorient b-vectors of a DWI scan.
    :param bvect_array: array of b-vectors column convention n x 3.
    :param transform: transformation to be applied to each b-vector.
    :return:
    """
    return np.einsum('...kl,...l->...k', bvect_array, transform).T


def reorient_b_vect_from_files(pfi_input, pfi_output, transform, fmt='%.14f'):
    """
    Reorient b-vectors of a DWI scan from textfiles.
    :param pfi_input: input to txt data with b-vectors column convention 'x y z \\'.
    :param pfi_output: output b-vectors after transformation in a new .txt file.
    :param transform: transformation to be applied to each b-vector.
    :param fmt:
    :return:
    """
    transformed_vect = reorient_b_vect(np.loadtxt(pfi_input), transform)
    # noinspection PyTypeChecker
    np.savetxt(pfi_output, transformed_vect, fmt=fmt)


def matrix_vector_field_product(j_input, v_input):
    """
    :param j_input: matrix m x n x (4 or 9) as for example a jacobian column major
    :param v_input: matrix m x n x (2 or 3) to be multiplied by the matrix point-wise.
    :return: m x n  x (2 or 3) whose each element is the result of the product of the
     matrix (i,j,:) multiplied by the corresponding element in the vector v (i,j,:).

    In tensor notation for n = 1: R_{i,j,k} = \sum_{l=0}^{2} M_{i,j,l+3k} v_{i,j,l}

    ### equivalent code, more readable and less efficient:

    # dimensions of the problem:
    d = v_input.shape[-1]
    vol = list(v_input.shape[:-1])

    # repeat v input 3 times, one for each row of the input matrix 3x3 or 2x2 in corresponding position:
    v = np.tile(v_input, [1]*d + [d])

    # element-wise product:
    j_times_v = np.multiply(j_input, v)

    # Sum the three blocks in the third dimension:
    return np.sum(j_times_v.reshape(vol + [d, d]), axis=d+1).reshape(vol + [d])

    """
    if not len(j_input.shape) == len(v_input.shape):
        raise IOError
    if not j_input.shape[:-1] == v_input.shape[:-1]:
        raise IOError

    d = v_input.shape[-1]
    vol = list(v_input.shape[:d])
    extra_ones = len(v_input.shape) - (len(vol) + 1)

    temp = j_input.reshape(vol + [1] * extra_ones + [d, d])  # transform in squared block with additional ones
    return np.einsum('...kl,...l->...k', temp, v_input)
