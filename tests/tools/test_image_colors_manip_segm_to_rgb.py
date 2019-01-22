from os.path import join as jph
import numpy as np
import nibabel as nib

from nilabels.tools.aux_methods.label_descriptor_manager import LabelsDescriptorManager
from nilabels.tools.image_colors_manipulations.segmentation_to_rgb import \
    get_rgb_image_from_segmentation_and_label_descriptor

from tests.tools.decorators_tools import pfo_tmp_test, \
    write_and_erase_temporary_folder_with_dummy_labels_descriptor


@write_and_erase_temporary_folder_with_dummy_labels_descriptor
def test_get_rgb_image_from_segmentation_and_label_descriptor_simple():
    ldm = LabelsDescriptorManager(jph(pfo_tmp_test, 'labels_descriptor.txt'))

    segm_data = np.zeros([20, 20, 20])  # block diagonal dummy segmentation
    segm_data[:5, :5, :5]          = 1
    segm_data[5:10, 5:10, 5:10]    = 2
    segm_data[10:15, 10:15, 10:15] = 3
    segm_data[15:, 15:, 15:]       = 4

    im_segm = nib.Nifti1Image(segm_data, affine=np.eye(4))

    im_segm_rgb = get_rgb_image_from_segmentation_and_label_descriptor(im_segm, ldm)

    segm_rgb_expected = np.zeros([20, 20, 20, 3])
    segm_rgb_expected[:5, :5, :5, :]          = np.array([255, 0, 0])
    segm_rgb_expected[5:10, 5:10, 5:10, :]    = np.array([204, 0, 0])
    segm_rgb_expected[10:15, 10:15, 10:15, :] = np.array([51, 51, 255])
    segm_rgb_expected[15:, 15:, 15:, :]       = np.array([102, 102, 255])

    np.testing.assert_equal(im_segm_rgb.get_data(), segm_rgb_expected)


@write_and_erase_temporary_folder_with_dummy_labels_descriptor
def test_get_rgb_image_from_segmentation_and_label_descriptor_simple_invert_bl_wh():
    ldm = LabelsDescriptorManager(jph(pfo_tmp_test, 'labels_descriptor.txt'))

    segm_data = np.zeros([20, 20, 20])  # block diagonal dummy segmentation
    segm_data[:5, :5, :5]          = 1
    segm_data[5:10, 5:10, 5:10]    = 2
    segm_data[10:15, 10:15, 10:15] = 3
    segm_data[15:, 15:, 15:]       = 4

    im_segm = nib.Nifti1Image(segm_data, affine=np.eye(4))

    im_segm_rgb = get_rgb_image_from_segmentation_and_label_descriptor(im_segm, ldm, invert_black_white=True)

    segm_rgb_expected = np.zeros([20, 20, 20, 3])
    segm_rgb_expected[..., :]                 = np.array([255, 255, 255])
    segm_rgb_expected[:5, :5, :5, :]          = np.array([255, 0, 0])
    segm_rgb_expected[5:10, 5:10, 5:10, :]    = np.array([204, 0, 0])
    segm_rgb_expected[10:15, 10:15, 10:15, :] = np.array([51, 51, 255])
    segm_rgb_expected[15:, 15:, 15:, :]       = np.array([102, 102, 255])

    np.testing.assert_equal(im_segm_rgb.get_data(), segm_rgb_expected)


@write_and_erase_temporary_folder_with_dummy_labels_descriptor
def test_get_rgb_image_from_segmentation_and_label_descriptor_wrong_input_dimension():
    ldm = LabelsDescriptorManager(jph(pfo_tmp_test, 'labels_descriptor.txt'))
    im_segm = nib.Nifti1Image(np.zeros([4, 4, 4, 4]), affine=np.eye(4))
    with np.testing.assert_raises(IOError):
        get_rgb_image_from_segmentation_and_label_descriptor(im_segm, ldm)


if __name__ == "__main__":
    test_get_rgb_image_from_segmentation_and_label_descriptor_simple()
    test_get_rgb_image_from_segmentation_and_label_descriptor_simple_invert_bl_wh()
    test_get_rgb_image_from_segmentation_and_label_descriptor_wrong_input_dimension()