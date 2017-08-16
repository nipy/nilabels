import nibabel as nib
import numpy as np
import os

from labels_manager.tools.aux_methods.utils_nib import set_new_data
from labels_manager.tools.aux_methods.utils import print_and_run


def adjust_nifti_image_type_path(pfi_nifti_input, new_dtype, pfi_nifti_output, update_description=None, verbose=1):
    # TODO expose in facade
    im_input = nib.load(pfi_nifti_input)
    if update_description is not None:
        if not isinstance(update_description, str):
            raise IOError('update_description must be a string')
        hd = im_input.header
        hd['descrip'] = update_description
        im_input.update_header()
    new_im = set_new_data(im_input, im_input.get_data().astype(new_dtype), new_dtype=new_dtype, remove_nan=True)
    if verbose > 0:
        print('Data type before {}'.format(im_input.get_data_dtype()))
        print('Data type after {}'.format(new_im.get_data_dtype()))
    nib.save(new_im, pfi_nifti_output)


def adjust_affine_header(pfi_input, pfi_output, theta, trasl=np.array([0, 0, 0])):
    # TODO expose in facade

    if theta != 0:
        # transformations parameters
        rot_x = np.array([[1,            0,           0,      trasl[0]],
                         [0,  np.cos(theta),  -np.sin(theta), trasl[1]],
                         [0,  np.sin(theta), np.cos(theta),   trasl[2]],
                         [0,             0,          0,       1]])

        # Load input image:
        im_input = nib.load(pfi_input)

        # generate new affine transformation (from bicommissural to histological)
        new_transf = rot_x.dot(im_input.get_affine())

        # create output image on the input
        if im_input.header['sizeof_hdr'] == 348:
            new_image = nib.Nifti1Image(im_input.get_data(), new_transf, header=im_input.get_header())
        # if nifty2
        elif im_input.header['sizeof_hdr'] == 540:
            new_image = nib.Nifti2Image(im_input.get_data(), new_transf, header=im_input.get_header())
        else:
            raise IOError

        # print intermediate results
        print('Affine input image: \n')
        print(im_input.get_affine())
        print('Affine after transformation: \n')
        print(new_image.get_affine())

        # sanity check
        np.testing.assert_almost_equal(np.linalg.det(new_transf), np.linalg.det(im_input.get_affine()))

        # save output image
        nib.save(new_image, pfi_output)
    else:
        if not pfi_input == pfi_output:
            print_and_run('cp {0} {1}'.format(pfi_input, pfi_output))


def adjust_nifti_translation_path(pfi_nifti_input, new_traslation, pfi_nifti_output, q_form=True, s_form=True,
                                  verbose=1):
    # TODO expose in facade
    """
    Change q_form or s_form or both translational part.
    :param pfi_nifti_input: path to file of the input image
    :param new_traslation: 3dim array, affine coordinates, will be the future translational part of the affine.
    :param pfi_nifti_output: path to file of the image with the modifed translation. Try not to be destructive, unless you do not really want.
    :param q_form: [True] affect q_form
    :param s_form: [True] affect s_form
    :param verbose:
    :return: None. It creates a new image in pfi_nifti_output with defined translational part.
    """
    im_input = nib.load(pfi_nifti_input)

    # generate new affine transformation (from bicommissural to histological)
    aff = im_input.affine
    # create output image on the input
    if im_input.header['sizeof_hdr'] == 348:
        new_image = nib.Nifti1Image(im_input.get_data(), aff, header=im_input.header)
    # if nifty2
    elif im_input.header['sizeof_hdr'] == 540:
        new_image = nib.Nifti2Image(im_input.get_data(), aff, header=im_input.header)
    else:
        raise IOError

    new_transf = np.copy(aff)
    if len(new_traslation) == 4 and new_traslation[-1] == 1:
        new_transf[:, 3] = new_traslation
    elif len(new_traslation) == 3:
        new_transf[:3, 3] = new_traslation
    else:
        raise IOError

    if q_form:
        new_image.set_qform(new_transf)

    if s_form:
        new_image.set_sform(new_transf)

    new_image.update_header()

    if verbose > 0:
        # print intermediate results
        print('Affine input image:')
        print(im_input.get_affine())
        print('Affine after update:')
        print(new_image.get_affine())
        print('Q-form after update:')
        print(new_image.get_qform())
        print('S-form after update:')
        print(new_image.get_sform())

    # save output image
    nib.save(new_image, pfi_nifti_output)


def apply_orientation_matrix_to_image(pfi_nifti_image, affine_transformation_left,
                                      pfo_output=None, pfi_b_vects=None, suffix='new', verbose=1):
    """

    :param pfi_nifti_image: path to file to a nifti image.
    :param affine_transformation_left: a reorientation matrix.
    :param pfo_output: path to folder where the output will be stored
    :param pfi_b_vects: path to file to b-vectors.
    :param suffix: added at the end of the new files.
    :param verbose:
    :return: None. It saves the input image and the optional input b-vectors transformed according to a matrix.
    """
    # TODO expose in facade
    assert isinstance(affine_transformation_left, np.ndarray)

    if pfo_output is None:
        pfo_output = os.path.dirname(pfi_nifti_image)

    pfi_new_image = os.path.join(pfo_output,
                                 os.path.basename(pfi_nifti_image).split('.')[0] + '{}.nii.gz'.format(suffix))

    im = nib.load(pfi_nifti_image)

    new_affine = im.affine.dot(affine_transformation_left)

    if im.header['sizeof_hdr'] == 348:
        new_im = nib.Nifti1Image(im.get_data(), new_affine, header=im.get_header())
    elif im.header['sizeof_hdr'] == 540:
        new_im = nib.Nifti2Image(im.get_data(), new_affine, header=im.get_header())
    else:
        raise IOError

    # sanity check
    msg = 'Is the input matrix a re-orientation matrix?'
    np.testing.assert_almost_equal(np.linalg.det(new_affine), np.linalg.det(im.get_affine()), err_msg=msg)

    # save output image
    nib.save(new_im, pfi_new_image)
    if pfi_b_vects is not None:
        bvects_name = os.path.basename(pfi_b_vects).split('.')[0]
        bvects_ext = os.path.basename(pfi_b_vects).split('.')[-1]
        pfi_new_bvects = os.path.join(pfo_output, '{0}_{1}.{2}'.format(bvects_name, suffix, bvects_ext))

        if bvects_ext == 'txt':
            b_vects = np.loadtxt(pfi_b_vects)
            new_bvects = np.einsum('ij, kj -> ki', affine_transformation_left[:3, :3], b_vects)
            np.savetxt(pfi_new_bvects, new_bvects, fmt='%10.14f')
        else:
            b_vects = np.load(pfi_b_vects)
            new_bvects = np.einsum('ij, kj -> ki', affine_transformation_left[:3, :3], b_vects)
            np.save(pfi_new_bvects, new_bvects)

    if verbose > 0:
        # print intermediate results
        print('Affine input image: \n')
        print(im.get_affine())
        print('Affine after transformation: \n')
        print(new_im.get_affine())


def basic_rot_ax(m, ax=0):
    """
    Basic rotations of a 3d matrix. Ingredient of the method axial_rotations.
    ----------
    Example:

    cube = array([[[0, 1],
                   [2, 3]],

                  [[4, 5],
                   [6, 7]]])

    axis 0: perpendicular to the face [[0,1],[2,3]] (front-rear)
    axis 1: perpendicular to the face [[1,5],[3,7]] (lateral right-left)
    axis 2: perpendicular to the face [[0,1],[5,4]] (top-bottom)
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


def axial_rotations(m, rot=1, ax=2):
    """
    :param m: 3d matrix
    :param rot: number of rotations
    :param ax: axis of rotation
    :return: m rotate rot times around axis ax, according to convention.
    """

    if m.ndim is not 3:
        assert IOError

    rot %= 4

    if rot == 0:
        return m

    for _ in range(rot):
        m = basic_rot_ax(m, ax=ax)

    return m


def flip_data(in_data, axis='x'):
    msg = 'Input array must be 3-dimensional.'
    assert in_data.ndim == 3, msg

    msg = 'axis variable must be one of the following: {}.'.format(['x', 'y', 'z'])
    assert axis in ['x', 'y', 'z'], msg

    if axis == 'x':
        out_data = in_data[:, ::-1, :]
    elif axis == 'y':
        out_data = in_data[:, :, ::-1]
    elif axis == 'z':
        out_data = in_data[::-1, :, :]
    else:
        raise IOError

    return out_data


def symmetrise_data(in_data, axis='x', plane_intercept=10, side_to_copy='below', keep_in_data_dimensions=True):
    """
    Symmetrise the input_array according to the axial plane
      axis = plane_intercept
    the copied part can be 'below' or 'above' the axes, following the ordering.

    :param in_data: (Z, X, Y) C convention input data
    :param axis:
    :param plane_intercept:
    :param side_to_copy:
    :param keep_in_data_dimensions:
    :return:
    """

    # Sanity check:

    msg = 'Input array must be 3-dimensional.'
    assert in_data.ndim == 3, msg

    msg = 'side_to_copy must be one of the two {}.'.format(['below', 'above'])
    assert side_to_copy in ['below', 'above'], msg

    msg = 'axis variable must be one of the following: {}.'.format(['x', 'y', 'z'])
    assert axis in ['x', 'y', 'z'], msg

    # step 1: find the block to symmetrise.
    # step 2: create the symmetric and glue it to the block.
    # step 3: add or remove a patch of slices if required to keep the in_data dimension.

    out_data = 0

    if axis == 'x':

        if side_to_copy == 'below':
            s_block = in_data[:, :plane_intercept, :]
            s_block_symmetric = s_block[:, ::-1, :]
            out_data = np.concatenate((s_block, s_block_symmetric), axis=1)

        if side_to_copy == 'above':
            s_block = in_data[:, plane_intercept:, :]
            s_block_symmetric = s_block[:, ::-1, :]
            out_data = np.concatenate((s_block_symmetric, s_block), axis=1)

    if axis == 'y':

        if side_to_copy == 'below':
            s_block = in_data[:, :, :plane_intercept]
            s_block_symmetric = s_block[:, :, ::-1]
            out_data = np.concatenate((s_block, s_block_symmetric), axis=2)

        if side_to_copy == 'above':
            s_block = in_data[:, :, plane_intercept:]
            s_block_symmetric = s_block[:, :, ::-1]
            out_data = np.concatenate((s_block_symmetric, s_block), axis=2)

    if axis == 'z':

        if side_to_copy == 'below':
            s_block = in_data[:plane_intercept, :, :]
            s_block_symmetric = s_block[::-1, :, :]
            out_data = np.concatenate((s_block, s_block_symmetric), axis=0)

        if side_to_copy == 'above':
            s_block = in_data[plane_intercept:, :, :]
            s_block_symmetric = s_block[::-1, :, :]
            out_data = np.concatenate((s_block_symmetric, s_block), axis=0)

    if keep_in_data_dimensions:
        out_data = out_data[:in_data.shape[0], :in_data.shape[1], :in_data.shape[2]]

    return out_data





