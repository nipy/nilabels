import nibabel as nib
import numpy as np


def test_check_missing_labels_paired():
    array = np.array([[[0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 7, 0, 0, 0, 0],
                       [0, 1, 0, 0, 7, 0, 0, 0, 0],
                       [0, 1, 0, 6, 7, 0, 0, 0, 0],
                       [0, 1, 0, 6, 0, 0, 2, 0, 0],
                       [0, 1, 0, 6, 0, 0, 2, 0, 0],
                       [0, 1, 0, 0, 0, 0, 2, 0, 0],
                       [0, 0, 0, 0, 0, 0, 2, 0, 0],
                       [0, 0, 0, 0, 0, 0, 2, 0, 0]],

                      [[0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 1, 0, 0, 0, 0, 0, 0, 0],
                       [0, 1, 0, 0, 0, 0, 0, 0, 0],
                       [0, 1, 0, 0, 0, 0, 2, 0, 0],
                       [0, 1, 0, 5, 0, 0, 2, 0, 0],
                       [0, 1, 0, 5, 0, 0, 2, 0, 0],
                       [0, 0, 0, 5, 0, 0, 2, 0, 0],
                       [0, 0, 0, 0, 0, 0, 2, 0, 0]]
                      ])
    im = nib.Nifti1Image(array, np.eye(4))
    # TODO


def test_check_missing_labels_unpaired():
    # TODO
    pass


def test_check_number_connected_components():
    # TODO
    pass
