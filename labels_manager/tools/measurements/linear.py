import numpy as np


def box_sides(in_data, label_to_box=1, affine=np.eye(4), precision_output=np.float32):
    """
    We assume the component with label equals to label_to_box is connected
    :return:
    """
    dim = len(in_data.shape)

    voxel_per_sides = [0, ] * dim
    for d in xrange(dim):
        for x in xrange(in_data.shape[d]):
            if label_to_box in in_data[x, :, :]:
                voxel_per_sides[d] += 1

    ans = np.array(dim, dtype=precision_output)
    for i in range(dim):
        ans[i] = affine[i, i, i] * voxel_per_sides[i]

    return ans