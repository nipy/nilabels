import numpy as np

from nilabels.tools.aux_methods.utils_nib import set_new_data


def cut_4d_volume_with_a_1_slice_mask(data_4d, data_mask):
    """
    Fist slice maks is applied to all the timepoints of the volume.
    :param data_4d:
    :param data_mask:
    :return:
    """
    assert data_4d.shape[:3] == data_mask.shape

    if len(data_4d.shape) == 3:  # 4d data is actually a 3d data
        data_masked_4d = np.multiply(data_mask, data_4d)
    else:
        data_masked_4d = np.zeros_like(data_4d)
        for t in range(data_4d.shape[-1]):
            data_masked_4d[..., t] = np.multiply(data_mask, data_4d[..., t])

    return data_masked_4d


def cut_4d_volume_with_a_1_slice_mask_nib(input_4d_nib, input_mask_nib):
    """
    Fist slice maks is applied to all the timepoints of the nibabel image.
    :param input_4d_nib: input 4d nifty image
    :param input_mask_nib: input mask
    :return:
    """

    data_4d  = input_4d_nib.get_data()
    data_mask = input_mask_nib.get_data()
    ans = cut_4d_volume_with_a_1_slice_mask(data_4d, data_mask)

    return set_new_data(input_4d_nib, ans)


def apply_a_mask_nib(im_input, im_mask):
    """
    Set to zero all the values outside the mask.
    From nibabel input and output.
    Adaptative - if the mask is 3D and the image is 4D, will create a temporary mask,
    generate the stack of masks, and apply the stacks to the image.
    :param im_input: nibabel image to be masked
    :param im_mask: nibabel image with the mask
    :return: im_input with intensities cropped after im_mask.
    """
    assert len(im_mask.shape) == 3

    # TODO correct this: merge the cut_4d_volume_with_a_1_slice_mask here
    if not im_mask.shape == im_input.shape[:3]:
        msg = 'Provided mask and image does not have compatible dimension: {0} and {1}'.format(
            im_input.shape, im_mask.shape)
        raise IOError(msg)

    if len(im_input.shape) == 3:
        new_data = im_input.get_data() * im_mask.get_data().astype(np.bool)
    else:
        new_data = np.zeros_like(im_input.get_data())
        for t in range(im_input.shape[3]):
            new_data[..., t] = im_input.get_data()[..., t] * im_mask.get_data().astype(np.bool)

    return set_new_data(image=im_input, new_data=new_data)
