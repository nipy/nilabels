import numpy as np


def reproduce_slice_fourth_dimension(in_data, num_slices=10, new_axis=3):

    sh = in_data.shape
    if not (len(sh) == 2 or len(sh) == 3):
        raise IOError('Methods can be used only for 2 or 3 dim images. No conflicts with existing multi, slices')

    new_data = np.stack([in_data, ] * num_slices, axis=new_axis)
    return new_data


def cut_4d_volume_with_a_1_slice_mask(data_4d, data_mask):

    assert data_4d.shape[:3] == data_mask.shape

    data_masked_4d = np.zeros_like(data_4d)

    for t in range(data_4d.shape[-1]):
        data_masked_4d[..., t] = np.multiply(data_mask, data_4d[..., t])

    # image with header of the dwi and values under the mask for each slice:
    return data_masked_4d
