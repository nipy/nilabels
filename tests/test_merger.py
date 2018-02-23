import nibabel as nib
import numpy as np
from nose.tools import assert_raises
from numpy.testing import assert_array_equal


''' From manipulations.merger.py'''
from labels_manager.tools.image_shape_manipulations.merger import stack_images, merge_labels_from_4d, grafting


def test_merge_labels_from_4d_fake_input():

    data = np.zeros([3,3,3])
    with assert_raises(AssertionError):
        merge_labels_from_4d(data)


def test_merge_labels_from_4d_shape_output():

    data000 = np.zeros([3, 3, 3])
    data111 = np.zeros([3, 3, 3])
    data222 = np.zeros([3, 3, 3])
    data000[0,0,0] = 1
    data111[1,1,1] = 2
    data222[2,2,2] = 4
    data = np.stack([data000, data111, data222], axis=3)

    out = merge_labels_from_4d(data)
    assert_array_equal([out[0,0,0], out[1,1,1], out[2,2,2]], [1, 2, 4])

    out = merge_labels_from_4d(data, keep_original_values=False)
    assert_array_equal([out[0,0,0], out[1,1,1], out[2,2,2]], [1, 2, 3])


def test_stack_images_cascade():

    d = 2
    im1 = nib.Nifti1Image(np.zeros([d,d]), affine=np.eye(4))
    assert_array_equal(im1.shape, (d, d))

    list_images1 = [im1] * d
    im2 = stack_images(list_images1)
    assert_array_equal(im2.shape, (d,d,d))

    list_images2 = [im2] * d
    im3 = stack_images(list_images2)
    assert_array_equal(im3.shape, (d,d,d,d))

    list_images3 = [im3] * d
    im4 = stack_images(list_images3)
    assert_array_equal(im4.shape, (d, d, d, d, d))

def test_grafting_ok_input_output():

    array_host = np.array([
                           [[1, 2, 3, 4, 5, 6, 7],
                            [1, 2, 3, 4, 5, 6, 7],
                            [1, 2, 3, 4, 5, 6, 7]],

                           [[1, 2, 3, 4, 5, 6, 7],
                            [1, 2, 3, 4, 5, 6, 7],
                            [1, 2, 3, 4, 5, 6, 7]],

                           [[1, 2, 3, 4, 5, 6, 7],
                            [1, 2, 3, 4, 5, 6, 7],
                            [1, 2, 3, 4, 5, 6, 7]],
                           ])

    array_patch = np.array([
                           [[0, 0, 4, 4, 4, 0, 0],
                            [0, 0, 3, 3, 3, 0, 0],
                            [0, 0, 2, 2, 2, 0, 0]],

                           [[0, 0, 8, 7, 9, 0, 0],
                            [0, 0, 8, 7, 9, 0, 0],
                            [0, 0, 8, 7, 9, 0, 0]],

                           [[0, 0, 9, 1, 5, 0, 0],
                            [0, 0, 9, 1, 5, 0, 0],
                            [0, 0, 9, 1, 5, 0, 0]],
                           ])

    expected_no_mask = np.array([
                           [[1, 2, 4, 4, 4, 6, 7],
                            [1, 2, 3, 3, 3, 6, 7],
                            [1, 2, 2, 2, 2, 6, 7]],

                           [[1, 2, 8, 7, 9, 6, 7],
                            [1, 2, 8, 7, 9, 6, 7],
                            [1, 2, 8, 7, 9, 6, 7]],

                           [[1, 2, 9, 1, 5, 6, 7],
                            [1, 2, 9, 1, 5, 6, 7],
                            [1, 2, 9, 1, 5, 6, 7]],
                           ])

    array_mask = np.array([
                           [[0, 0, 1, 1, 1, 0, 0],
                            [0, 0, 1, 1, 1, 0, 0],
                            [0, 0, 1, 1, 1, 0, 0]],

                           [[0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 1, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0]],

                           [[0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 1, 0, 0, 0],
                            [0, 0, 0, 1, 0, 0, 0]],
                           ])

    expected_mask = np.array([
                            [[1, 2, 4, 4, 4, 6, 7],
                             [1, 2, 3, 3, 3, 6, 7],
                             [1, 2, 2, 2, 2, 6, 7]],

                            [[1, 2, 3, 4, 5, 6, 7],
                             [1, 2, 3, 7, 5, 6, 7],
                             [1, 2, 3, 4, 5, 6, 7]],

                            [[1, 2, 3, 4, 5, 6, 7],
                             [1, 2, 3, 1, 5, 6, 7],
                             [1, 2, 3, 1, 5, 6, 7]],
                            ])

    affine = np.array([[0, 0.5, 0, 0],[0.5, 0, 0, 0],[0, 0, 0.5, 0], [0, 0, 0, 1]])

    # get images
    im_array_host = nib.Nifti1Image(array_host, affine)
    im_array_patch = nib.Nifti1Image(array_patch, affine)
    im_array_mask = nib.Nifti1Image(array_mask, affine)

    # TODO
    # test with no mask
    # im_output_no_mask = grafting(im_array_host, im_array_patch)
    # assert_array_equal(im_output_no_mask.get_data(), expected_no_mask)

    # test with mask
    # im_output_mask = grafting(im_array_host, im_array_patch, im_array_mask)
    # assert_array_equal(im_output_mask.get_data(), expected_mask)
