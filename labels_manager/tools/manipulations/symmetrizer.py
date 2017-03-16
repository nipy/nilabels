import os
import numpy as np

import nibabel as nib
from labels_manager.tools.aux_methods.utils import set_new_data
from labels_manager.tools.manipulations.relabeller import relabeller


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

    if len(m.shape) is not 3:
        assert IOError

    rot %= 4

    if rot == 0:
        return m

    for _ in range(rot):
        m = basic_rot_ax(m, ax=ax)

    return m


def flip_data(in_data, axis='x'):

    msg = 'Input array must be 3-dimensional.'
    assert len(in_data.shape) == 3, msg

    msg = 'axis variable must be one of the following: {}.'.format(['x', 'y', 'z'])
    assert axis in ['x', 'y', 'z'], msg

    if axis == 'x':
        out_data = in_data[:, ::-1, :]
    if axis == 'y':
        out_data = in_data[:, :, ::-1]
    if axis == 'z':
        out_data = in_data[::-1, :, :]

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
    assert len(in_data.shape) == 3, msg

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


def sym_labels(pfi_anatomy,
               pfi_segmentation,
               pfo_results,
               pfi_result_segmentation,
               list_labels_input,
               list_labels_transformed=None,
               reuse_registration=False,
               coord='z'):
    """
    Symmetrise a segmentation with registration: it uses NiftyReg.
    The old side is symmetrised in the new side, with new relabelling.
    :param pfi_anatomy: Path to FIle anatomical image
    :param pfi_segmentation: Path to FIle segmentation of the anatomical image
    :param pfo_results: Path to FOlder where intermediate results are stored
    :param pfi_result_segmentation: Path to FIle symmetrised segmentation
    :param list_labels_input: labels that will be taken into account in the symmetrisation from the old side.
    :param list_labels_transformed: corresponding labels in the same order. If None, labels of the new side
    will be kept with the same numbering in the new side.
    :param reuse_registration: if a registration is already present in the pfo_results, and you only need to change the
    labels value, it will spare you some time when set to True.
    :param coord: coordinate of the registration: in RAS, 'z' will symmetrise Left on Right.
    :return: symmetrised segmentation.
    """

    def flip_data_path(input_im_path, output_im_path, axis='x'):
        # wrap flip data, having path for inputs and outputs.
        if not os.path.isfile(input_im_path):
            raise IOError('input image file does not exist.')

        im_labels = nib.load(input_im_path)
        data_labels = im_labels.get_data()
        data_flipped = flip_data(data_labels, axis=axis)

        im_relabelled = set_new_data(im_labels, data_flipped)
        nib.save(im_relabelled, output_im_path)

    # side A is the input, side B is the one where we want to symmetrise.
    # --- Initialisation  --- #

    # check input:
    if not os.path.isfile(pfi_anatomy):
        raise IOError('input image file {} does not exist.'.format(pfi_anatomy))
    if not os.path.isfile(pfi_segmentation):
        raise IOError('input segmentation file {} does not exist.'.format(pfi_segmentation))

    # erase labels that are not in the list from image and descriptor

    out_labels_side_A_path = os.path.join(pfo_results, 'z_labels_side_A.nii.gz')
    labels_im = nib.load(pfi_segmentation)
    labels_data = labels_im.get_data()
    labels_to_erase = list(set(labels_data.flat) - set(list_labels_input + [0]))

    # Relabel: from pfi_segmentation to out_labels_side_A_path
    im_pfi_segmentation = nib.load(pfi_segmentation)

    segmentation_data_relabelled = relabeller(im_pfi_segmentation.get_data(), list_old_labels=labels_to_erase,
                                              list_new_labels=[0, ] * len(labels_to_erase))
    nib_labels_side_A_path = set_new_data(im_pfi_segmentation, segmentation_data_relabelled)
    nib.save(nib_labels_side_A_path, out_labels_side_A_path)

    # --- Create side B  --- #

    # flip anatomical image and register it over the non flipped
    out_anatomical_flipped_path = os.path.join(pfo_results, 'z_anatomical_flipped.nii.gz')
    flip_data_path(pfi_anatomy, out_anatomical_flipped_path, axis=coord)

    # flip the labels
    out_labels_flipped_path = os.path.join(pfo_results, 'z_labels_flipped.nii.gz')
    flip_data_path(out_labels_side_A_path, out_labels_flipped_path, axis=coord)

    # register anatomical flipped over non flipped
    out_anatomical_flipped_warped_path = os.path.join(pfo_results, 'z_anatomical_flipped_warped.nii.gz')
    out_affine_transf_path = os.path.join(pfo_results, 'z_affine_transformation.txt')

    if not reuse_registration:
        cmd = 'reg_aladin -ref {0} -flo {1} -aff {2} -res {3}'.format(pfi_anatomy,
                                                                      out_anatomical_flipped_path,
                                                                      out_affine_transf_path,
                                                                      out_anatomical_flipped_warped_path)
        print('Registration started!\n')
        os.system(cmd)

        # propagate the registration to the flipped labels
        out_labels_side_B_path = os.path.join(pfo_results, 'z_labels_side_B.nii.gz')
        cmd = 'reg_resample -ref {0} -flo {1} ' \
          '-res {2} -trans {3} -inter {4}'.format(out_labels_side_A_path,
                                                   out_labels_flipped_path,
                                                   out_labels_side_B_path,
                                                   out_affine_transf_path,
                                                   0)

        print('Resampling started!\n')
        os.system(cmd)
    else:
        out_labels_side_B_path = os.path.join(pfo_results, 'z_labels_side_B.nii.gz')

    # update labels of the side B if necessarily
    if list_labels_transformed is not None:

        print('relabelling step!')

        assert len(list_labels_transformed) == len(list_labels_input)

        # relabel from out_labels_side_B_path to out_labels_side_B_path
        im_segmentation_side_B = nib.load(out_labels_side_B_path)

        data_segmentation_side_B_new = relabeller(im_segmentation_side_B.get_data(), list_old_labels=list_labels_input,
                                                list_new_labels=list_labels_transformed)
        nib_segmentation_side_B_new = set_new_data(im_segmentation_side_B, data_segmentation_side_B_new)
        nib.save(nib_segmentation_side_B_new, out_labels_side_B_path)

    # --- Merge side A and side B in a single volume according to a criteria --- #
    # out_labels_side_A_path,  out_labels_side_B_path --> result_path.nii.gz

    nib_side_A = nib.load(out_labels_side_A_path)
    nib_side_B = nib.load(out_labels_side_B_path)

    data_side_A = nib_side_A.get_data()
    data_side_B = nib_side_B.get_data()

    symmetrised_data = np.zeros_like(data_side_A)

    # To manage the intersections of labels between old and new side. Vectorize later...
    dims = data_side_A.shape

    print('Pointwise symmetrisation started!')

    for z in xrange(dims[0]):
        for x in xrange(dims[1]):
            for y in xrange(dims[2]):
                if (data_side_A[z, x, y] == 0 and data_side_B[z, x, y] != 0) or \
                   (data_side_A[z, x, y] != 0 and data_side_B[z, x, y] == 0):
                    symmetrised_data[z, x, y] = np.max([data_side_A[z, x, y], data_side_B[z, x, y]])
                elif data_side_A[z, x, y] != 0 and data_side_B[z, x, y] != 0:
                    if data_side_A[z, x, y] == data_side_B[z, x, y]:
                        symmetrised_data[z, x, y] = data_side_A[z, x, y]
                    else:
                        symmetrised_data[z, x, y] = 255  # devil label!

    im_symmetrised = set_new_data(nib_side_A, symmetrised_data)
    nib.save(im_symmetrised, pfi_result_segmentation)
