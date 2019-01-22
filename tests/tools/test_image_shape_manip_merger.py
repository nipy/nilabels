import numpy as np
import nibabel as nib

from nilabels.tools.image_shape_manipulations.merger import reproduce_slice_fourth_dimension, \
    grafting, substitute_volume_at_timepoint,  stack_images, merge_labels_from_4d, \
    from_segmentations_stack_to_probabilistic_segmentation


def test_merge_labels_from_4d_fake_input():

    data = np.zeros([3, 3, 3])
    with np.testing.assert_raises(IOError):
        merge_labels_from_4d(data)


def test_merge_labels_from_4d_shape_output():

    data000 = np.zeros([3, 3, 3])
    data111 = np.zeros([3, 3, 3])
    data222 = np.zeros([3, 3, 3])
    data000[0, 0, 0] = 1
    data111[1, 1, 1] = 2
    data222[2, 2, 2] = 4
    data = np.stack([data000, data111, data222], axis=3)

    out = merge_labels_from_4d(data)
    np.testing.assert_array_equal([out[0, 0, 0], out[1, 1, 1], out[2, 2, 2]], [1, 2, 4])

    out = merge_labels_from_4d(data, keep_original_values=False)
    np.testing.assert_array_equal([out[0, 0, 0], out[1, 1, 1], out[2, 2, 2]], [1, 2, 3])


def test_stack_images_cascade():

    d = 2
    im1 = nib.Nifti1Image(np.zeros([d, d]), affine=np.eye(4))
    np.testing.assert_array_equal(im1.shape, (d, d))

    list_images1 = [im1] * d
    im2 = stack_images(list_images1)
    np.testing.assert_array_equal(im2.shape, (d, d, d))

    list_images2 = [im2] * d
    im3 = stack_images(list_images2)
    np.testing.assert_array_equal(im3.shape, (d, d, d, d))

    list_images3 = [im3] * d
    im4 = stack_images(list_images3)
    np.testing.assert_array_equal(im4.shape, (d, d, d, d, d))


def test_reproduce_slice_fourth_dimension_wrong_input():
    im_test = nib.Nifti1Image(np.zeros([5, 5, 5, 5]), affine=np.eye(4))
    with np.testing.assert_raises(IOError):
        reproduce_slice_fourth_dimension(im_test)


def test_reproduce_slice_fourth_dimension_simple():
    data_test = np.arange(16).reshape(4, 4)
    num_slices = 4
    repetition_axis = 2

    im_reproduced = reproduce_slice_fourth_dimension(nib.Nifti1Image(data_test, affine=np.eye(4)),
                                                     num_slices=4, repetition_axis=repetition_axis)

    data_expected = np.stack([data_test, ] * num_slices, axis=repetition_axis)

    np.testing.assert_array_equal(im_reproduced.get_data(), data_expected)


def test_grafting_simple():
    data_hosting = 3 * np.ones([5, 5, 5])
    data_patch = np.zeros([5, 5, 5])
    data_patch[2:4, 2:4, 2:4] = 7

    data_expected = 3 * np.ones([5, 5, 5])
    data_expected[2:4, 2:4, 2:4] = 7

    im_hosting = nib.Nifti1Image(data_hosting, affine=np.eye(4))
    im_patch = nib.Nifti1Image(data_patch, affine=np.eye(4))

    im_grafted = grafting(im_hosting, im_patch)

    np.testing.assert_array_equal(im_grafted.get_data(), data_expected)


def test_from_segmentations_stack_to_probabilistic_segmentation_simple():

    # Generate initial 1D segmentations:
    #     1  2  3  4  5  6  7  8  9  0  1  2  3  4  5  6  7  8  9
    a1 = [0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 4]
    a2 = [0, 0, 1, 1, 1, 1, 1, 2, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4]
    a3 = [0, 0, 0, 1, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4]
    a4 = [0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 4]
    a5 = [0, 0, 1, 1, 1, 1, 1, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4]
    a6 = [0, 0, 1, 1, 1, 1, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4]

    stack = np.stack([np.array(a) for a in [a1, a2, a3, a4, a5, a6]])

    prob = from_segmentations_stack_to_probabilistic_segmentation(stack)

    # expected output probability for each class, computed manually(!)
    k0 = [6, 5, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    k1 = [0, 1, 4, 6, 5, 5, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    k2 = [0, 0, 0, 0, 1, 1, 4, 6, 3, 3, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    k3 = [0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 5, 6, 5, 5, 3, 3, 2, 1, 0]
    k4 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 3, 3, 4, 5, 6]

    k0 = 1 / 6. * np.array(k0)
    k1 = 1 / 6. * np.array(k1)
    k2 = 1 / 6. * np.array(k2)
    k3 = 1 / 6. * np.array(k3)
    k4 = 1 / 6. * np.array(k4)

    prob_expected = np.stack([k0, k1, k2, k3, k4], axis=0)
    np.testing.assert_array_equal(prob, prob_expected)


def test_from_segmentations_stack_to_probabilistic_segmentation_random_sum_rows_to_get_one():
    J = 12
    N = 120
    K = 7
    stack = np.stack([np.random.choice(range(K), N) for _ in range(J)])
    prob = from_segmentations_stack_to_probabilistic_segmentation(stack)
    s = np.sum(prob, axis=0)
    np.testing.assert_array_almost_equal(s, np.ones(N))


def test_substitute_volume_at_timepoint_wrong_input():
    im_4d = nib.Nifti1Image(np.zeros([5, 5, 5, 3]), affine=np.eye(4))
    im_3d = nib.Nifti1Image(np.ones([5, 5, 5]), affine=np.eye(4))
    tp = 7
    with np.testing.assert_raises(IOError):
        substitute_volume_at_timepoint(im_4d, im_3d, tp)


def test_substitute_volume_at_timepoint_simple():
    im_4d = nib.Nifti1Image(np.zeros([5, 5, 5, 4]), affine=np.eye(4))
    im_3d = nib.Nifti1Image(np.ones([5, 5, 5]), affine=np.eye(4))
    tp = 2
    expected_data = np.stack([np.zeros([5, 5, 5]), np.zeros([5, 5, 5]), np.ones([5, 5, 5]), np.zeros([5, 5, 5])],
                             axis=3)
    im_subs = substitute_volume_at_timepoint(im_4d, im_3d, tp)

    np.testing.assert_array_equal(im_subs.get_data(), expected_data)


if __name__ == '__main__':
    test_merge_labels_from_4d_fake_input()
    test_merge_labels_from_4d_shape_output()

    test_stack_images_cascade()

    test_reproduce_slice_fourth_dimension_wrong_input()
    test_reproduce_slice_fourth_dimension_simple()

    test_grafting_simple()

    test_substitute_volume_at_timepoint_wrong_input()
    test_substitute_volume_at_timepoint_simple()
