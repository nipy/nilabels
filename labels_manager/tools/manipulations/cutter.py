import numpy as np
import nibabel as nib

from labels_manager.tools.aux_methods.utils_nib import set_new_data


def cut_4d_volume_with_a_1_slice_mask(data_4d, data_mask):
    """
    Fist slice maks is applied to all the timepoints of the volume.
    :param data_4d:
    :param data_mask:
    :return:
    """
    assert data_4d.shape[:3] == data_mask.shape
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
    ans = cut_4d_volume_with_a_1_slice_mask(data_4d.get_data(), data_mask.get_data())

    return set_new_data(input_4d_nib, ans)


def apply_a_mask_path(pfi_input, pfi_mask, pfi_output):
    """
    Adaptative - if the mask is 3D and the image is 4D, will create a temporary mask,
    generate the stack of masks, and apply the stacks to the image.
    :param pfi_input: path to file 3d x T image
    :param pfi_mask: 3d mask same dimension as the 3d of the pfi_input
    :param pfi_output: apply the mask to each time point T in the fourth dimension if any.
    :return: None, save the output in pfi_output.
    """
    im_input = nib.load(pfi_input)
    im_mask = nib.load(pfi_mask)

    assert len(im_mask.shape) == 3

    if not im_mask.shape == im_input.shape[:3]:
        msg = 'Mask {0} and image {1} does not have compatible dimension: {2} and {3}'.format(
            pfi_input, pfi_mask, im_input, im_mask.shape)
        raise IOError(msg)

    if len(im_input.shape) == 3:
        new_data = im_input.get_data() * im_mask.get_data().astype(np.bool)
    else:
        new_data = np.zeros_like(im_input.get_data())
        for t in range(im_input.shape[3]):
            new_data[..., t] = im_input.get_data()[..., t] * im_mask.get_data().astype(np.bool)

    new_im = set_new_data(image=im_input, new_data=new_data)

    nib.save(new_im, filename=pfi_output)
