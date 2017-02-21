import numpy as np

from tools.labels.relabeller import permute_labels


def test_permute_labels_basic():

    a = np.array([[1, 3, 1, 2], [3, 1, 2, 1], [1, 3, 1, 2]])
    b = np.zeros([3, 4, 3])
    for i in range(3):
        b[..., i] = a

    print b
    c = permute_labels(b, permutation=[[2, 3], [3, 2]])
    print c


def test_permute_labels_basic_2():
    a = np.array([[5, 4, 1, 8, 3, 8, 0, 9, 2, 9],
                   [1, 9, 7, 0, 2, 5, 2, 2, 9, 0],
                   [6, 9, 7, 3, 2, 5, 7, 6, 2, 0],
                   [3, 4, 7, 1, 9, 5, 9, 6, 2, 5],
                   [0, 8, 7, 0, 3, 5, 2, 2, 2, 3],
                   [4, 8, 7, 7, 0, 5, 3, 5, 2, 3],
                   [4, 8, 3, 0, 1, 5, 1, 7, 2, 3],
                   [4, 8, 2, 0, 5, 5, 1, 9, 5, 3],
                   [4, 8, 6, 4, 9, 5, 6, 0, 4, 3],
                   [4, 8, 7, 2, 0, 5, 5, 9, 8, 0]])
    print a
    c = permute_labels(a, permutation=[[5, 7, 8, 3], [7, 8, 3, 5]])
    print
    print c

