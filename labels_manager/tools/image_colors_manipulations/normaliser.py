import numpy as np
try:
    from skimage import filters
except ImportError:
    from skimage import filter as filters

from labels_manager.tools.aux_methods.utils_nib import set_new_data


def normalise_below_labels(im_input, im_segm, labels_list, stats=np.median, exclude_first_label=True):
    """
    Normalise the intensities of im_input for the stats obtained of the values found under input labels.
    :param im_input:
    :param im_segm:
    :param labels_list:
    :param stats:
    :return:
    """
    if exclude_first_label:
        labels_list = labels_list[1:]
    mask_data = np.zeros_like(im_segm.get_data(), dtype=np.bool)
    for label_k in labels_list[1:]:  # want to exclude the first one!
        mask_data += im_segm.get_data() == label_k

    masked_im_data = np.nan_to_num((mask_data.astype(np.float64) * im_input.get_data().astype(np.float64)).flatten())
    non_zero_masked_im_data = masked_im_data[np.where(masked_im_data > 1e-6)]
    s = stats(non_zero_masked_im_data)
    assert isinstance(s, float)
    output_im = set_new_data(im_input, (1 / float(s))* im_input.get_data())

    return output_im


def otsu_threshold(im):
    """
    :param im:
    :return:
    """
    val = filters.threshold_otsu(im.get_data())
    return set_new_data(im, im.get_data() * (im.get_data() > val))
