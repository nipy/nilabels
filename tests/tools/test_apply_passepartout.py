import nibabel as nib
import numpy as np
from numpy.testing import assert_array_equal

from nilabels.tools.image_shape_manipulations.apply_passepartout import crop_with_passepartout, \
    crop_with_passepartout_based_on_label_segmentation


def test_crop_with_passepartout_simple():
    data = np.random.randint(0, 100, [10, 10, 10])
    im = nib.Nifti1Image(data, np.eye(4))

    x_min, x_max  = 2, 2
    y_min, y_max = 3, 1
    z_min, z_max = 4, 5

    new_im = crop_with_passepartout(im, [x_min, x_max, y_min, y_max, z_min, z_max])

    assert_array_equal(data[x_min:-x_max, y_min:-y_max, z_min:-z_max], new_im.get_data())


def test_crop_with_passepartout_based_on_label_segmentation_simple():

    arr = np.array([[[0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 2, 0, 0],
                     [0, 0, 0, 0, 0, 0, 2, 0, 0],
                     [0, 0, 0, 0, 0, 0, 2, 0, 0],
                     [0, 0, 0, 0, 0, 0, 2, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0]],

                    [[0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 2, 2, 0, 0],
                     [0, 0, 0, 0, 2, 2, 2, 0, 0],
                     [0, 0, 0, 0, 0, 2, 2, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0]]])

    arr_expected = np.array([[[0, 0, 2],
                              [0, 0, 2],
                              [0, 0, 2],
                              [0, 0, 2]],
                             [[0, 2, 2],
                              [2, 2, 2],
                              [0, 2, 2],
                              [0, 0, 0]]])

    im_intput = nib.Nifti1Image(arr, np.eye(4))

    im_cropped = crop_with_passepartout_based_on_label_segmentation(im_intput, im_intput, [0, 0, 0], 2)

    assert_array_equal(arr_expected, im_cropped.get_data())


def test_crop_with_passepartout_based_on_label_segmentation_with_im():
    dat = np.array([[[1, 1, 1, 1, 1, 1, 1, 1, 1],
                     [2, 2, 2, 2, 2, 2, 2, 2, 2],
                     [3, 3, 3, 3, 3, 3, 3, 3, 3],
                     [4, 4, 4, 4, 4, 4, 4, 4, 4],
                     [5, 5, 5, 5, 5, 5, 5, 5, 5],
                     [6, 6, 6, 6, 6, 6, 6, 6, 6],
                     [7, 7, 7, 7, 7, 7, 7, 7, 7],
                     [8, 8, 8, 8, 8, 8, 8, 8, 8],
                     [9, 9, 9, 9, 9, 9, 9, 9, 9]],

                    [[9, 9, 9, 9, 9, 9, 9, 9, 9],
                     [8, 8, 8, 8, 8, 8, 8, 8, 8],
                     [7, 7, 7, 7, 7, 7, 7, 7, 7],
                     [6, 6, 6, 6, 6, 6, 6, 6, 6],
                     [5, 5, 5, 5, 5, 5, 5, 5, 5],
                     [4, 4, 4, 4, 4, 4, 4, 4, 4],
                     [3, 3, 3, 3, 3, 3, 3, 3, 3],
                     [2, 2, 2, 2, 2, 2, 2, 2, 2],
                     [1, 1, 1, 1, 1, 1, 1, 1, 1]]])

    sgm = np.array([[[0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 2, 0, 0],
                     [0, 0, 0, 0, 0, 0, 2, 0, 0],
                     [0, 0, 0, 0, 0, 0, 2, 0, 0],
                     [0, 0, 0, 0, 0, 0, 2, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0]],

                    [[0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 2, 2, 0, 0],
                     [0, 0, 0, 0, 2, 2, 2, 0, 0],
                     [0, 0, 0, 0, 0, 2, 2, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0]]])

    arr_expected = np.array([[[5, 5, 5],
                              [6, 6, 6],
                              [7, 7, 7],
                              [8, 8, 8]],
                             [[5, 5, 5],
                              [4, 4, 4],
                              [3, 3, 3],
                              [2, 2, 2]]])

    img_input = nib.Nifti1Image(dat, np.eye(4))
    segm_intput = nib.Nifti1Image(sgm, np.eye(4))

    im_cropped = crop_with_passepartout_based_on_label_segmentation(img_input, segm_intput, [0, 0, 0], 2)

    assert_array_equal(arr_expected, im_cropped.get_data())


if __name__ == '__main__':
    test_crop_with_passepartout_simple()
    test_crop_with_passepartout_based_on_label_segmentation_simple()
    test_crop_with_passepartout_based_on_label_segmentation_with_im()
