import numpy as np

from labels_manager.tools.manipulations.relabeller import keep_only_one_label
from labels_manager.tools.aux_methods.utils import binarise_a_matrix


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



def centroid(im, labels, affine=np.eye(3)):
    """

    :param im:
    :param labels: list of labels, even with a single element!
    :param affine: transformation matrix, from matrix space to real world coordinates. Identity by default
    :return: centroid of the labels of an image, in the order of the labels
    """
    centers_of_mass = [np.array([0, 0, 0]).astype(np.uint64), ] * len(labels)
    num_voxel_per_label = [0, ]*len(labels)
    for i in xrange(im.shape[0]):
        for j in xrange(im.shape[1]):
            for k in xrange(im.shape[2]):
                if im[i, j, k] in labels:
                    label_index = labels.index(im[i, j, k])
                    centers_of_mass[label_index] = centers_of_mass[label_index] +  np.array([i, j, k]).astype(np.uint64)
                    num_voxel_per_label[label_index] += 1
    for n_index, n in enumerate(num_voxel_per_label):
        centers_of_mass[n_index] = (1 / float(n)) * affine.dot(centers_of_mass[n_index].astype(np.float64))

    return centers_of_mass


def box_sides(in_segmentation, label_to_box=1, affine=np.eye(4), dtype_output=np.float64):
    """
    We assume the component with label equals to label_to_box is connected
    :return:
    """
    one_label_data = keep_only_one_label(in_segmentation, label_to_keep=label_to_box)
    ans = []
    for d in range(len(one_label_data.shape)):
        ans.append(np.sum(binarise_a_matrix(np.sum(one_label_data, axis=d), dtype=np.int)))
    return ans
