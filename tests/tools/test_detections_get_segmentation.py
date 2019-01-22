import numpy as np

from nilabels.tools.detections.get_segmentation import intensity_segmentation, otsu_threshold, MoG_array


#  ----- Test get segmentation ----


def test_intensity_segmentation_1():
    im_array = np.random.randint(0, 5, [10, 10], np.uint8)
    output_segm = intensity_segmentation(im_array)
    # if the input is a segmentation with 5 labels, the segmentation is the input.
    np.testing.assert_array_equal(im_array, output_segm)


def test_intensity_segmentation_2():

    seed_segm  = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5])
    seed_image = np.linspace(0, 5, len(seed_segm))

    segm  = np.stack([seed_segm, ]*6)
    image = np.stack([seed_image, ]*6)

    output_segm = intensity_segmentation(image, num_levels=6)
    np.testing.assert_array_equal(segm, output_segm)

    segm_transposed  = segm.T
    image_transposed = image.T

    output_segm_transposed = intensity_segmentation(image_transposed, num_levels=6)
    np.testing.assert_array_equal(segm_transposed, output_segm_transposed)


def test_otsu_threshold_bad_input():
    with np.testing.assert_raises(IOError):
        otsu_threshold(np.random.rand(40, 40), side='spam')


def test_otsu_threshold_side_above():
    arr = np.zeros([20, 20])
    arr[:10, :] = 1
    arr[10:, :] = 2
    arr_thr = otsu_threshold(arr, side='above', return_as_mask=False)

    expected_arr_thr = np.zeros([20, 20])
    expected_arr_thr[10:, :] = 2

    np.testing.assert_array_equal(arr_thr, expected_arr_thr)


def test_otsu_threshold_side_below():
    arr = np.zeros([20, 20])
    arr[:10, :] = 1
    arr[10:, :] = 2
    arr_thr = otsu_threshold(arr, side='below', return_as_mask=False)

    expected_arr_thr = np.zeros([20, 20])
    expected_arr_thr[:10, :] = 1

    np.testing.assert_array_equal(arr_thr, expected_arr_thr)


def test_otsu_threshold_as_mask():
    arr = np.zeros([20, 20])
    arr[:10, :] = 1
    arr[10:, :] = 2
    arr_thr = otsu_threshold(arr, side='above', return_as_mask=True)

    expected_arr_thr = np.zeros([20, 20])
    expected_arr_thr[10:, :] = 1

    np.testing.assert_array_equal(arr_thr, expected_arr_thr)


def test_MoG_array_1():
    arr = np.zeros([20, 20, 20])
    arr[:10, ...] = 1
    arr[10:, ...] = 2
    crisp, prob = MoG_array(arr, K=2)

    expected_crisp = np.zeros([20, 20, 20])
    expected_crisp[:10, ...] = 0
    expected_crisp[10:, ...] = 1

    expected_prob = np.zeros([20, 20, 20, 2])
    expected_prob[:10, ..., 0] = 1
    expected_prob[10:, ..., 1] = 1

    np.testing.assert_array_equal(crisp, expected_crisp)
    np.testing.assert_array_equal(prob, expected_prob)


if __name__ == '__main__':
    test_intensity_segmentation_1()
    test_intensity_segmentation_2()

    test_otsu_threshold_bad_input()
    test_otsu_threshold_side_above()
    test_otsu_threshold_side_below()
    test_otsu_threshold_as_mask()

    test_MoG_array_1()
