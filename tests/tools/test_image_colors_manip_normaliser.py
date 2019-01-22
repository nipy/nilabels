import numpy as np
import nibabel as nib
from numpy.testing import assert_array_equal

from nilabels.tools.image_colors_manipulations.normaliser import normalise_below_labels, \
    intensities_normalisation_linear, mahalanobis_distance_map


def test_normalise_below_labels():
    arr_data = np.ones([20, 21, 22])
    arr_segm = np.zeros([20, 21, 22])

    arr_data[5:10, 1, 1] = np.array([1, 2, 3, 4, 5])

    arr_segm[5:10, 1, 1] = np.ones([5])
    factor = np.median(np.array([1, 2, 3, 4, 5]))

    im_data = nib.Nifti1Image(arr_data, affine=np.eye(4))
    im_segm = nib.Nifti1Image(arr_segm, affine=np.eye(4))

    expected_array_normalised = arr_data / factor

    im_normalised_below = normalise_below_labels(im_data, im_segm)

    np.testing.assert_array_almost_equal(im_normalised_below.get_data(), expected_array_normalised)


def test_normalise_below_labels_specified_list():
    arr_data = np.ones([20, 21, 22])
    arr_segm = np.zeros([20, 21, 22])

    arr_data[5:10, 1, 1] = np.array([1, 2, 3, 4, 5])
    arr_data[5:10, 2, 1] = np.array([3, 6, 9, 12, 15])

    arr_segm[5:10, 1, 1] = np.ones([5])
    arr_segm[5:10, 2, 1] = 2 * np.ones([5])

    factor_1 = np.median(np.array([1, 2, 3, 4, 5]))
    factor_2 = np.median(np.array([3, 6, 9, 12, 15]))
    factor_1_2 = np.median(np.array([1, 2, 3, 4, 5, 3, 6, 9, 12, 15]))

    im_data = nib.Nifti1Image(arr_data, affine=np.eye(4))
    im_segm = nib.Nifti1Image(arr_segm, affine=np.eye(4))

    # No labels indicated:
    expected_array_normalised = arr_data / factor_1_2
    im_normalised_below = normalise_below_labels(im_data, im_segm, labels_list=None, exclude_first_label=False)
    np.testing.assert_array_almost_equal(im_normalised_below.get_data(), expected_array_normalised)

    # asking only for label 2
    expected_array_normalised = arr_data / factor_2
    im_normalised_below = normalise_below_labels(im_data, im_segm, labels_list=[2], exclude_first_label=False)
    np.testing.assert_array_almost_equal(im_normalised_below.get_data(), expected_array_normalised)

    # asking only for label 1
    expected_array_normalised = arr_data / factor_1
    im_normalised_below = normalise_below_labels(im_data, im_segm, labels_list=[1], exclude_first_label=False)
    np.testing.assert_array_almost_equal(im_normalised_below.get_data(), expected_array_normalised)


def test_normalise_below_labels_specified_list_exclude_first():
    arr_data = np.ones([20, 21, 22])
    arr_segm = np.zeros([20, 21, 22])

    arr_data[5:10, 1, 1] = np.array([1, 2, 3, 4, 5])
    arr_data[5:10, 2, 1] = np.array([3, 6, 9, 12, 15])

    arr_segm[5:10, 1, 1] = np.ones([5])
    arr_segm[5:10, 2, 1] = 2 * np.ones([5])

    factor_2 = np.median(np.array([3, 6, 9, 12, 15]))

    im_data = nib.Nifti1Image(arr_data, affine=np.eye(4))
    im_segm = nib.Nifti1Image(arr_segm, affine=np.eye(4))

    expected_array_normalised = arr_data / factor_2
    im_normalised_below = normalise_below_labels(im_data, im_segm, labels_list=[1, 2], exclude_first_label=True)
    np.testing.assert_array_almost_equal(im_normalised_below.get_data(), expected_array_normalised)


def test_intensities_normalisation():
    arr_data = np.zeros([20, 20, 20])
    arr_segm = np.zeros([20, 20, 20])

    arr_data[:5, :5, :5]          = 2
    arr_data[5:10, 5:10, 5:10]    = 4
    arr_data[10:15, 10:15, 10:15] = 6
    arr_data[15:, 15:, 15:]       = 8

    arr_segm[arr_data > 1] = 1

    im_data = nib.Nifti1Image(arr_data, affine=np.eye(4))
    im_segm = nib.Nifti1Image(arr_segm, affine=np.eye(4))

    im_normalised = intensities_normalisation_linear(im_data, im_segm, im_mask_foreground=im_segm)
    np.testing.assert_almost_equal(np.min(im_normalised.get_data()), 0.0)
    np.testing.assert_almost_equal(np.max(im_normalised.get_data()), 10.0)

    im_normalised = intensities_normalisation_linear(im_data, im_segm)
    np.testing.assert_almost_equal(np.min(im_normalised.get_data()), -3.2)
    np.testing.assert_almost_equal(np.max(im_normalised.get_data()), 10.0)


def test_mahalanobis_distance_map():
    data = np.zeros([10, 10, 10])
    im = nib.Nifti1Image(data, affine=np.eye(4))
    md_im = mahalanobis_distance_map(im)
    np.testing.assert_array_equal(md_im.get_data(), np.zeros_like(md_im.get_data()))

    data = np.ones([10, 10, 10])
    im = nib.Nifti1Image(data, affine=np.eye(4))
    md_im = mahalanobis_distance_map(im)
    np.testing.assert_array_equal(md_im.get_data(), np.zeros_like(md_im.get_data()))


def test_mahalanobis_distance_map_with_mask():
    data = np.random.randn(10, 10, 10)
    mask = np.zeros_like(data)
    mask[2:-2, 2:-2, 2:-2] = 1

    mu = np.mean(data.flatten() * mask.flatten())
    sigma2 = np.std(data.flatten() * mask.flatten())
    mn_data = np.sqrt((data - mu) * sigma2 ** (-1) * (data - mu))

    im = nib.Nifti1Image(data, affine=np.eye(4))
    im_mask = nib.Nifti1Image(mask, affine=np.eye(4))

    md_im = mahalanobis_distance_map(im, im_mask)
    np.testing.assert_array_equal(md_im.get_data(), mn_data)

    mn_data_trimmed = mn_data * mask.astype(np.bool)
    md_im = mahalanobis_distance_map(im, im_mask, trim=True)
    np.testing.assert_array_equal(md_im.get_data(), mn_data_trimmed)


if __name__ == '__main__':
    test_normalise_below_labels()
    test_normalise_below_labels_specified_list()
    test_normalise_below_labels_specified_list_exclude_first()

    test_intensities_normalisation()

    test_mahalanobis_distance_map()
    test_mahalanobis_distance_map_with_mask()