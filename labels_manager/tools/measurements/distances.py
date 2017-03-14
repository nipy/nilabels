import numpy as np


def lncc_distance(values_patch1, values_patch2):
    """
    Import values below the patches, containing the same number of eolem
    :param values_patch1:
    :param values_patch2:
    :return:
    """
    patches = [values_patch1.flatten(), values_patch2.flatten()]
    np.testing.assert_array_equal(patches[0].shape, patches[1].shape)

    for index_p, p in enumerate(patches):
        den = float(np.linalg.norm(p))
        if den == 0: patches[index_p] = np.zeros_like(p)
        else: patches[index_p] = patches[index_p] / den

    return patches[0].dot(patches[1])
