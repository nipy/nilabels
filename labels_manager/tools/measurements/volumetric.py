import numpy as np


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

