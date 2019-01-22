import numpy as np

from nilabels.tools.aux_methods.utils_nib import set_new_data


def merge_labels_from_4d(in_data, keep_original_values=True):
    """
    Can be the inverse function of split label with default parameters.
    The labels are assuming to have no overlaps.
    :param in_data: 4d volume
    :param keep_original_values: merge the labels with their values, otherwise it uses the values of the slice numbering (including zero label!).
    :return:
    """
    if not in_data.ndim == 4:
        msg = 'Input array must be 4-dimensional.'
        raise IOError(msg)

    in_data_shape = in_data.shape
    out_data = np.zeros(in_data_shape[:3], dtype=in_data.dtype)

    for t in range(in_data.shape[3]):
        slice_t = in_data[..., t]
        if keep_original_values:
            out_data = out_data + slice_t
        else:
            out_data = out_data + ((t + 1) * slice_t.astype(np.bool)).astype(in_data.dtype)
    return out_data


def stack_images(list_images):
    """
    From a list of images of the same shape, the stack of these images in the new dimension.
    :param list_images:
    :return: stack image of the input list
    """
    msg = 'input images shapes are not all of the same dimension'
    assert False not in [list_images[0].shape == im.shape for im in list_images[1:]], msg
    new_data = np.stack([nib_image.get_data() for nib_image in list_images] , axis=len(list_images[0].shape))
    stack_im = set_new_data(list_images[0], new_data)
    return stack_im


def reproduce_slice_fourth_dimension(nib_image, num_slices=10, repetition_axis=3):
    im_sh = nib_image.shape
    if not (len(im_sh) == 2 or len(im_sh) == 3):
        raise IOError('Methods can be used only for 2 or 3 dim images. No conflicts with existing multi, slices')

    new_data = np.stack([nib_image.get_data(), ] * num_slices, axis=repetition_axis)
    output_im = set_new_data(nib_image, new_data)
    return output_im


def grafting(im_hosting, im_patch, im_patch_mask=None):
    """
    Takes an hosting image, an image patch and a patch mask (optional) of the same dimension and in the same real space.
    It crops the patch (or patch mask if present) on the hosting image, and substitute the value from the patch.
    :param im_hosting: Mould or holder of the patch
    :param im_patch: patch to add.
    :param im_patch_mask: mask in case the mould is not zero in the region where the patch goes.
    :return:
    """
    np.testing.assert_array_equal(im_hosting.affine, im_patch.affine)

    if im_patch_mask is None:
        patch_region = im_patch.get_data().astype(np.bool)
    else:
        np.testing.assert_array_equal(im_hosting.affine, im_patch_mask.affine)
        np.testing.assert_array_equal(im_hosting.shape, im_patch_mask.shape)

        patch_region = im_patch_mask.get_data().astype(np.bool)

    patch_inverted = np.invert(patch_region)
    new_data = im_hosting.get_data() * patch_inverted + im_patch.get_data() * patch_region

    return set_new_data(im_hosting, new_data)


def from_segmentations_stack_to_probabilistic_segmentation(arr_labels_stack):
    """
    A probabilistic atlas has at each time-point a different label (a mapping is provided as well in
    the conversion with correspondence time-point<->label number). Each time point has the normalised average
    of each label.
    :param arr_labels_stack: stack of 1D arrays (segmentations x num_voxels) containing a different discrete segmentation.
    Values of labels needs to be consecutive, or there will be empty images in the result.
    :return:
    N number of voxels, J number of segmentations, K number of labels.
    """
    J, K = arr_labels_stack.shape[0], np.max(arr_labels_stack) + 1
    return 1/float(J) * np.stack([np.sum(arr_labels_stack == k, axis=0).astype(np.float64) for k in range(K)], axis=0)


def substitute_volume_at_timepoint(im_input_4d, im_input_3d, timepoint):
    """
    Substitute the im_input_3d image at the time point timepoint of the im_input_4d.
    :param im_input_4d: 4d image
    :param im_input_3d: 3d image whose shape is compatible with the fist 3 dimensions of im_input_4d
    :param timepoint: a timepoint in the 4th dimension of the im_input_4d
    :return: im_input_4d whose at the timepoint-th time point the data of im_input_3d are stored.
    Handle base case: If the input 4d volume is actually a 3d and timepoint is 0, then just return the same volume.
    """
    if len(im_input_4d.shape) == 3 and timepoint == 0:
        return im_input_3d
    elif len(im_input_4d.shape) == 4 and timepoint < im_input_4d.shape[-1]:
        new_data = im_input_4d.get_data()[:]
        new_data[..., timepoint] = im_input_3d.get_data()[:]
        return set_new_data(im_input_4d, new_data)
    else:
        raise IOError('Incompatible shape input volume.')
