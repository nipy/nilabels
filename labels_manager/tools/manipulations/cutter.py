import numpy as np

from labels_manager.tools.aux_methods.utils import set_new_data


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
