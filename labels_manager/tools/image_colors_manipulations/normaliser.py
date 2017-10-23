import numpy as np
try:
    from skimage import filters
except ImportError:
    from skimage import filter as filters

from labels_manager.tools.aux_methods.utils_nib import set_new_data


def normalise_below_labels(im_input, im_segm, labels_list=None, stats=np.median, exclude_first_label=True):
    """
    Normalise the intensities of im_input for the stats obtained of the values found under input labels.
    :param im_input:
    :param im_segm:
    :param labels_list:
    :param stats:
    :return:
    """
    if labels_list is not None:
        if exclude_first_label:
            labels_list = labels_list[1:]
        mask_data = np.zeros_like(im_segm.get_data(), dtype=np.bool)
        for label_k in labels_list:
            mask_data += im_segm.get_data() == label_k
    else:
        mask_data = np.zeros_like(im_segm.get_data(), dtype=np.bool)
        mask_data[im_segm.get_data()>0] = 1

    masked_im_data = np.nan_to_num((mask_data.astype(np.float64) * im_input.get_data().astype(np.float64)).flatten())
    non_zero_masked_im_data = masked_im_data[np.where(masked_im_data > 1e-6)]
    s = stats(non_zero_masked_im_data)
    assert isinstance(s, float)
    output_im = set_new_data(im_input, (1 / float(s))* im_input.get_data())

    return output_im


def intensties_normalisation_linear(im_input, im_segm, im_mask_foreground=None,
                                    toll=1e-12, percentile_range=(1, 99), output_range=(0.1, 10)):
    """
    Normalise the values below the binarised segmentation so that it will be between 0 and 1, based on a linear
    transformation whose parameters are learned from the values below the segmentation.
    E.G.

    min_intensities = 1% of the intensities below the mask
    max_intensities = 99% of the intensities above the mask

    compute the parameters a and b so that:

    a * min_intensities  + b = 0.1
    a * max_intensities  + b = 10

    To all the values in the image is then applied the linear function:
    f(x) = a * x  + b in the foreground,
    f(x) = 0 in the background.
    :return:
    """
    mask_data = np.zeros_like(im_segm.get_data(), dtype=np.bool)
    mask_data[im_segm.get_data() > 0] = 1

    non_zero_below_mask = im_input.get_data()[np.where(mask_data > toll)].flatten()

    min_intensities, max_intensities = np.percentile(non_zero_below_mask, percentile_range[0], percentile_range[1])

    a = (output_range[1] - output_range[0]) / (max_intensities - min_intensities)
    b = output_range[0] - a * min_intensities

    if im_mask_foreground is None:
        im_mask_foreground_data = np.ones_like(im_input.get_data())
    else:
        im_mask_foreground_data = im_mask_foreground.get_data()

    return set_new_data(im_input, im_mask_foreground_data * (a * im_input.get_data() + b))


def otsu_threshold(im):
    """
    :param im:
    :return:
    """
    val = filters.threshold_otsu(im.get_data())
    return set_new_data(im, im.get_data() * (im.get_data() > val))
