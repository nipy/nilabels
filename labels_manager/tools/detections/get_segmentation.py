import numpy as np


def intensity_segmentation(in_data, num_levels=5):
    """
    Very very very simple way of getting an intensity based segmentation.
    :param in_data: image data in a numpy array.
    :param num_levels: maximum allowed 65535 - 1.
    :return: segmentation of the result in levels levels based on the intensities of the in_data.
    """
    # NOTE: right extreme is excluded, must be considered in outside the for loop.
    segm = np.zeros_like(in_data, dtype=np.uint16)
    min_data = np.min(in_data)
    max_data = np.max(in_data)
    h = (max_data - min_data) / float(int(num_levels))

    for k in xrange(0, num_levels):
        places = (min_data + k * h <= in_data) * (in_data < min_data + (k + 1) * h)
        np.place(segm, places, k)

    places = in_data == max_data
    np.place(segm, places, num_levels-1)

    return segm
