import numpy as np
import nibabel as nib

from labels_manager.tools.aux_methods.utils_nib import set_new_data


def merge_labels_from_4d(in_data, keep_original_values=True):
    """
    Can be the inverse function of split label with default parameters.
    The labels are assuming to have no overlaps.
    :param in_data: 4d volume
    :param keep_original_values: merge the labels with their values, otherwise it uses the values of the slice numbering (including zero label!).
    :return:
    """
    msg = 'Input array must be 4-dimensional.'
    assert in_data.ndim == 4, msg

    in_data_shape = in_data.shape
    out_data = np.zeros(in_data_shape[:3], dtype=in_data.dtype)

    for t in xrange(in_data.shape[3]):
        slice_t = in_data[...,t]
        if keep_original_values:
            out_data = out_data + slice_t
        else:
            out_data = out_data + ((t + 1) * slice_t.astype(np.bool)).astype(in_data.dtype)
    return out_data


def stack_images(list_images):
    """
    From a list of image of the same shape, the stack of these images in the new dimension.
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
    Take an hosting image, an image patch and a patch mask (optional) of the same dimension and in the same real space.
    It crops the patch (or patch mask if present) on the hosting image, and substitute the value from the patch.
    :param im_hosting:
    :param im_patch:
    :param im_patch_mask:
    :return:
    """
    np.testing.assert_array_equal(im_hosting.affine, im_patch.affine)
    if im_patch_mask is not None:
        np.testing.assert_array_equal(im_hosting.affine, im_patch_mask.affine)

    if im_patch_mask is None:
        patch_region = im_patch.get_data().astype(np.bool)
    else:
        patch_region = im_patch_mask.get_data().astype(np.bool)
    new_data = np.copy(im_hosting.get_data())
    new_data[patch_region] = im_patch.get_data()[patch_region]
    # np.place(new_data, patch_region, im_patch.get_data())

    return set_new_data(im_hosting, new_data)


def from_segmentation_stack_to_probabilistic_atlas(im_stack_label):
    """
    A probabilistic atlas has at each time-point a different label (a mapping is provided as well in
    the conversion with correspondence time-point<->label number). Each time point has the normalised average
    of each label.
    :return:
    """
    assert len(im_stack_label.shape) == 4
    
    labels_list = 0



