import numpy as np
import nibabel as nib
from numpy.testing import assert_array_equal

from nilabels.tools.image_colors_manipulations.cutter import cut_4d_volume_with_a_1_slice_mask, \
    cut_4d_volume_with_a_1_slice_mask_nib, apply_a_mask_nib


def test_cut_4d_volume_with_a_1_slice_mask():
    data_0 = np.stack([np.array(range(5*5*5)).reshape(5, 5, 5)] * 4, axis=3)
    mask = np.zeros([5, 5, 5])
    for k in range(5):
        mask[k, k, k] = 1
    expected_answer_for_each_slice = np.zeros([5, 5, 5])
    for k in range(5):
        expected_answer_for_each_slice[k, k, k] = 30 * k + k
    ans = cut_4d_volume_with_a_1_slice_mask(data_0, mask)

    for k in range(4):
        assert_array_equal(ans[..., k], expected_answer_for_each_slice)


def test_cut_4d_volume_with_a_1_slice_mask_if_input_3d():
    data_0 = np.array(range(5*5*5)).reshape(5, 5, 5)
    mask = np.zeros([5, 5, 5])
    mask[:3, :3, :] = 1

    ans = cut_4d_volume_with_a_1_slice_mask(data_0, mask)

    assert_array_equal(ans, mask * data_0)


def test_cut_4d_volume_with_a_1_slice_mask_nib():
    data_0 = np.stack([np.array(range(5 * 5 * 5)).reshape(5, 5, 5)] * 4, axis=3)
    mask = np.zeros([5, 5, 5])
    for k in range(5):
        mask[k, k, k] = 1
    expected_answer_for_each_slice = np.zeros([5, 5, 5])
    for k in range(5):
        expected_answer_for_each_slice[k, k, k] = 30 * k + k

    im_data0 = nib.Nifti1Image(data_0, affine=np.eye(4))
    im_mask = nib.Nifti1Image(mask, affine=np.eye(4))

    im_ans = cut_4d_volume_with_a_1_slice_mask_nib(im_data0, im_mask)

    for k in range(4):
        assert_array_equal(im_ans.get_data()[..., k], expected_answer_for_each_slice)


def test_apply_a_mask_nib_wrong_input():
    data_0 = np.array(range(5 * 5 * 5)).reshape(5, 5, 5)
    mask = np.zeros([5, 5, 4])
    mask[:3, :3, :] = 1

    im_data0 = nib.Nifti1Image(data_0, affine=np.eye(4))
    im_mask = nib.Nifti1Image(mask, affine=np.eye(4))
    with np.testing.assert_raises(IOError):
        apply_a_mask_nib(im_data0, im_mask)


def test_apply_a_mask_nib_ok_input():
    data_0 = np.array(range(5 * 5 * 5)).reshape(5, 5, 5)
    mask = np.zeros([5, 5, 5])
    mask[:3, :3, :] = 1

    expected_data = data_0 * mask

    im_data0 = nib.Nifti1Image(data_0, affine=np.eye(4))
    im_mask = nib.Nifti1Image(mask, affine=np.eye(4))

    im_masked = apply_a_mask_nib(im_data0, im_mask)

    np.testing.assert_array_equal(im_masked.get_data(), expected_data)


def test_apply_a_mask_nib_4d_input():
    data_0 = np.stack([np.array(range(5 * 5 * 5)).reshape(5, 5, 5)] * 4, axis=3)
    mask = np.zeros([5, 5, 5])
    for k in range(5):
        mask[k, k, k] = 1
    expected_answer_for_each_slice = np.zeros([5, 5, 5])
    for k in range(5):
        expected_answer_for_each_slice[k, k, k] = 30 * k + k

    im_data0 = nib.Nifti1Image(data_0, affine=np.eye(4))
    im_mask = nib.Nifti1Image(mask, affine=np.eye(4))

    im_ans = apply_a_mask_nib(im_data0, im_mask)

    for k in range(4):
        assert_array_equal(im_ans.get_data()[..., k], expected_answer_for_each_slice)


if __name__ == '__main__':
    test_cut_4d_volume_with_a_1_slice_mask()
    test_cut_4d_volume_with_a_1_slice_mask_if_input_3d()

    test_cut_4d_volume_with_a_1_slice_mask_nib()

    test_apply_a_mask_nib_wrong_input()
    test_apply_a_mask_nib_ok_input()
    test_apply_a_mask_nib_4d_input()
