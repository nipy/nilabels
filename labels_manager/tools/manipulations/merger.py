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


def reproduce_slice_fourth_dimension_path(pfi_input_image, pfi_output_image, num_slices=10, repetition_axis=3):
    # TODO expose in facade
    old_im = nib.load(pfi_input_image)
    new_im = reproduce_slice_fourth_dimension(old_im, num_slices=num_slices, repetition_axis=repetition_axis)
    nib.save(new_im, pfi_output_image)
    print 'New image created and saved in {0}'.format(pfi_output_image)