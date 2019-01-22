import numpy as np

from nilabels.tools.image_shape_manipulations.splitter import split_labels_to_4d


def test_split_labels_to_4d_True_False():
    data = np.array(range(8)).reshape(2, 2, 2)
    splitted_4d = split_labels_to_4d(data, list_labels=range(8))
    for t in range(8):
        expected_slice = np.zeros(8)
        expected_slice[t] = t
        np.testing.assert_array_equal(splitted_4d[..., t], expected_slice.reshape(2, 2, 2))

    splitted_4d = split_labels_to_4d(data, list_labels=range(8), keep_original_values=False)
    expected_ans = [[[[1, 0, 0, 0, 0, 0, 0, 0],
                      [0, 1, 0, 0, 0, 0, 0, 0]],
                     [[0, 0, 1, 0, 0, 0, 0, 0],
                      [0, 0, 0, 1, 0, 0, 0, 0]]],
                    [[[0, 0, 0, 0, 1, 0, 0, 0],
                      [0, 0, 0, 0, 0, 1, 0, 0]],
                     [[0, 0, 0, 0, 0, 0, 1, 0],
                      [0, 0, 0, 0, 0, 0, 0, 1]]]]
    np.testing.assert_array_equal(splitted_4d, expected_ans)


if __name__ == '__main__':
    test_split_labels_to_4d_True_False()
