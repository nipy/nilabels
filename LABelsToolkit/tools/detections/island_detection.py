import numpy as np
from scipy import ndimage


def island_for_label(array_segm, label, emphasis_max=False):
    """
    As ndimage.label, with output ordered by the size of the connected component.
    :param array_segm:
    :param label:
    :param emphasis_max: if True, the max component will be one and all the others will be -1.
    :return:
    """
    if label not in array_segm:
        print('Label {} not in the provided array.'.format(label))

    binary_segm_components, num_components = ndimage.label(array_segm == label)
    voxels_per_components = np.array([np.count_nonzero(binary_segm_components == l + 1) for l in range(num_components)])
    scores = voxels_per_components.argsort()[::-1]
    binary_segm_components_sorted = np.zeros_like(binary_segm_components)

    for l in range(num_components):
        binary_segm_components_sorted[binary_segm_components == l + 1] = scores[l] + 1

    if emphasis_max:
        binary_segm_components_sorted[binary_segm_components_sorted > 1] = -1

    return binary_segm_components_sorted

if __name__ == '__main__':

    # TODO move in examples/test

    a = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                  [0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0],
                  [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                  [0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

    print(island_for_label(a, 1, emphasis_max=False))
    print(island_for_label(a, 1, emphasis_max=True))
