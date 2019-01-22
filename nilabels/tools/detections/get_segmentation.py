import numpy as np
import matplotlib.mlab as mlab
from matplotlib import pyplot as plt

from sklearn.mixture import GaussianMixture
from scipy.signal import medfilt

try:
    from skimage import filters
except ImportError:
    from skimage import filter as filters

from nilabels.tools.image_colors_manipulations.relabeller import relabeller


def intensity_segmentation(in_array, num_levels=5):
    """
    Simplest way of getting an intensity based segmentation.
    :param in_array: image data in a numpy array.
    :param num_levels: maximum allowed 65535 - 1.
    :return: segmentation of the result in levels levels based on the intensities of the in_data.
    """
    segm = np.zeros_like(in_array, dtype=np.uint16)
    min_data = np.min(in_array)
    max_data = np.max(in_array)
    h = (max_data - min_data) / float(int(num_levels))
    for k in range(0, num_levels):
        places = (min_data + k * h <= in_array) * (in_array < min_data + (k + 1) * h)
        np.place(segm, places, k)
    places = in_array == max_data
    np.place(segm, places, num_levels-1)
    return segm


def otsu_threshold(in_array, side='above', return_as_mask=True):
    """
    Segmentation of an array with Otsu thresholding parameters from skimage filters.
    :param in_array: input array representing an rgb image.
    :param side: must be 'above' or 'below', representing the side of the image thresholded after Otsu response.
    :param return_as_mask: the output can be a boolean mask if True.
    :return: thresholded input image according to Otsu and input parameters.
    """
    otsu_thr = filters.threshold_otsu(in_array)
    if side == 'above':
        new_data = in_array * (in_array >= otsu_thr)
    elif side == 'below':
        new_data = in_array * (in_array < otsu_thr)
    else:
        raise IOError("Parameter side must be 'above' or 'below'.")
    if return_as_mask:
        new_data = new_data.astype(np.bool)
    return new_data


def MoG_array(in_array, K=None, mask_array=None, pre_process_median_filter=False,
              output_gmm_class=False, pre_process_only_interquartile=False,
              see_histogram=None, reorder_mus=True):
    """
    Mixture of gaussians for medical images. A simple wrap of
    sklearn.mixture.GaussianMixture to get a mog-based segmentation of an input
    nibabel image.
    :param in_array: input array format to be segmented with a MOG method.
    :param K: number of classes, if None, it is estimated with a BIC criterion (may take a while)
    :param mask_array: nibabel mask if you want to consider only a subset of the masked data.
    :param pre_process_median_filter: apply a median filter before pre-processing (reduce salt and pepper noise).
    :param pre_process_only_interquartile: set to zero above and below interquartile (below mask if any) in the data.
    :param output_gmm_class: return only the gmm sklearn class instance.
    :param see_histogram: can be True, False (or None) or a string (with a path where to save the plotted histogram).
    :param reorder_mus: only if output_gmm_class=False, reorder labels from smallest to bigger means.
    :return: [c, p] crisp and probabilistic segmentation OR gmm, instance of the class sklearn.mixture.GaussianMixture.
    """
    if pre_process_median_filter:
        print('Pre-process with a median filter.')
        data = medfilt(in_array)
    else:
        data = in_array
    data = np.copy(data.flatten().reshape(-1, 1))

    if mask_array is not None:
        mask_data = np.copy(mask_array.flatten().astype(np.bool_).reshape(-1, 1))
        data = mask_data * data

    if pre_process_only_interquartile:
        print('Get only interquartile data.')
        non_zero_data = data[np.where(np.nan_to_num(data) > 1e-6)]
        low_p = np.percentile(non_zero_data, 25)
        high_p = np.percentile(non_zero_data, 75)
        data = (data > low_p) * (data < high_p) * data

    if K is None:
        print('Estimating numbers of components with BIC criterion... may take some minutes')
        n_components = range(3, 15)
        models = [GaussianMixture(n_components=k, random_state=0).fit(data) for k in n_components]
        K = np.min([m.bic(data) for m in models])
        print('Estimated number of classes according to BIC: {}'.format(K))

    gmm = GaussianMixture(n_components=K).fit(data)

    if output_gmm_class:
        return gmm
    else:
        crisp = gmm.predict(data).reshape(in_array.shape)
        prob = gmm.predict_proba(data).reshape(list(in_array.shape) + [K])

        if reorder_mus:
            mu = gmm.means_.reshape(-1)
            p = list(np.argsort(mu))

            old_labels = list(range(K))
            new_labels = [p.index(l) for l in old_labels]  # the inverse of p

            crisp = np.copy(relabeller(crisp, old_labels, new_labels))
            prob = np.stack([prob[..., t] for t in new_labels], axis=3)

        if see_histogram is not None and see_histogram is not False:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.set_aspect(1)
            ax.hist(crisp.flatten(), bins=50, normed=True)
            lx = ax.get_xlim()
            x = np.arange(lx[0], lx[1], (lx[1] - lx[0]) / 1000.)
            for m, s in zip(gmm.means_, gmm.precisions_.reshape(-1)):
                ax.plot(x, mlab.normpdf(x, m, s))

            if isinstance(see_histogram, str):
                plt.savefig(see_histogram)
            else:
                plt.show()

        return crisp, prob
