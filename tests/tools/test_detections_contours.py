import numpy as np
import nibabel as nib

from nilabels.tools.detections.contours import get_xyz_borders_of_a_label, \
    get_internal_contour_with_erosion_at_label, contour_from_array_at_label, \
    contour_from_segmentation


def test_contour_from_array_at_label_empty_image():
    arr = np.zeros([50, 50, 50])
    arr_contour = contour_from_array_at_label(arr, 1)

    np.testing.assert_array_equal(arr_contour, np.zeros_like(arr_contour))


def test_contour_from_array_at_label_simple_border():
    arr = np.zeros([50, 50, 50])
    arr[:, :, 25:] = 1

    arr_contour = contour_from_array_at_label(arr, 1)

    im_expected = np.zeros([50, 50, 50])
    im_expected[:, :, 24:26] = 1

    np.testing.assert_array_equal(arr_contour, im_expected)


def test_contour_from_array_at_label_simple_border_omit_axis_x():
    arr = np.zeros([50, 50, 50])
    arr[:, :, 25:] = 1

    arr_contour = contour_from_array_at_label(arr, 1, omit_axis='x')

    im_expected = np.zeros([50, 50, 50])
    im_expected[:, :, 24:26] = 1

    np.testing.assert_array_equal(arr_contour, im_expected)


def test_contour_from_array_at_label_simple_border_omit_axis_y():
    arr = np.zeros([50, 50, 50])
    arr[:, :, 25:] = 1

    arr_contour = contour_from_array_at_label(arr, 1, omit_axis='y')

    im_expected = np.zeros([50, 50, 50])
    im_expected[:, :, 24:26] = 1

    np.testing.assert_array_equal(arr_contour, im_expected)


def test_contour_from_array_at_label_simple_border_omit_axis_z():
    arr = np.zeros([50, 50, 50])
    arr[:, :, 25:] = 1

    arr_contour = contour_from_array_at_label(arr, 1, omit_axis='z')

    np.testing.assert_array_equal(arr_contour, np.zeros_like(arr_contour))


def test_contour_from_array_at_label_error():
    arr = np.zeros([50, 50, 50])

    with np.testing.assert_raises(IOError):
        contour_from_array_at_label(arr, 1, omit_axis='spam')


def test_contour_from_segmentation():
    im_data = np.zeros([50, 50, 50])
    im_data[:, :, 20:] = 1
    im_data[:, :, 30:] = 2

    im = nib.Nifti1Image(im_data, affine=np.eye(4))

    im_contour = contour_from_segmentation(im, omit_axis=None, verbose=1)

    im_data_expected = np.zeros([50, 50, 50])
    im_data_expected[:, :, 20:21] = 1

    im_data_expected[:, :, 29:30] = 1
    im_data_expected[:, :, 30:31] = 2
    np.testing.assert_array_equal(im_contour.get_data(), im_data_expected)


def test_get_xyz_borders_of_a_label():
    arr = np.zeros([10, 10, 10])

    arr[1, 1, 1] = 1
    arr[3, 3, 2] = 1
    out_coords = get_xyz_borders_of_a_label(arr, 1)
    np.testing.assert_equal(out_coords, [1, 3, 1, 3, 1, 2])


def test_get_xyz_borders_of_a_label_no_labels_found():
    arr = np.zeros([10, 10, 10])
    out_coords = get_xyz_borders_of_a_label(arr, 3)
    np.testing.assert_equal(out_coords is None, True)


def test_get_internal_contour_with_erosion_at_label():

    arr = np.zeros([10, 10, 10])
    arr[2:-2, 2:-2, 2:-2] = 1

    expected_output = np.zeros([10, 10, 10])
    expected_output[2:-2, 2:-2, 2:-2] = 1
    expected_output[3:-3, 3:-3, 3:-3] = 0

    im_cont = get_internal_contour_with_erosion_at_label(arr, 1)

    np.testing.assert_array_equal(im_cont, expected_output)


if __name__ == "__main__":
    test_contour_from_array_at_label_empty_image()

    test_contour_from_array_at_label_simple_border()
    test_contour_from_array_at_label_simple_border_omit_axis_x()
    test_contour_from_array_at_label_simple_border_omit_axis_y()
    test_contour_from_array_at_label_simple_border_omit_axis_z()

    test_contour_from_array_at_label_error()

    test_contour_from_segmentation()

    test_get_xyz_borders_of_a_label()
    test_get_xyz_borders_of_a_label_no_labels_found()
    test_get_internal_contour_with_erosion_at_label()
