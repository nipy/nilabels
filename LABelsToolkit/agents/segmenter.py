import nibabel as nib
from LABelsToolkit.tools.aux_methods.utils_path import connect_path_tail_head

from LABelsToolkit.tools.aux_methods.utils_nib import set_new_data
from LABelsToolkit.tools.detections.get_segmentation import intensity_segmentation, otsu_threshold, \
    MoG_array




class LABelsToolkitSegmenter(object):
    """
    Facade for the simple segmentation methods based on intensities, Otsu thresholding and
    mixture of gaussians.
    """
    def __init__(self, input_data_folder=None, output_data_folder=None):
        self.pfo_in = input_data_folder
        self.pfo_out = output_data_folder

    def simple_intensities_thresholding(self):
        # TODO
        intensity_segmentation
        pass

    def otsu_thresholding(self):
        otsu_threshold
        # TODO
        pass

    def get_MoG(self, path_to_input_image, path_to_output_segmentation, K=None, mask_im=None, pre_process_median_filter=False,
            pre_process_only_interquartile=False, see_histogram=None, reorder_mus=True):
        """
        Wrap of MoG_array for nibabel images.
        -----
        :param input_im: nibabel input image format to be segmented with a MOG method.
        :param K: number of classes, if None, it is estimated with a BIC criterion (may take a while)
        :param mask_im: nibabel mask if you want to consider only a subset of the masked data.
        :param pre_process_median_filter: apply a median filter before pre-processing (reduce salt and pepper noise).
        :param pre_process_only_interquartile: set to zero above and below interquartile in the data.
        :param see_histogram: can be True, False (or None) or a string (with a path where to save the plotted histogram).
        :param reorder_mus: only if output_gmm_class=False, reorder labels from smallest to bigger means.
        :return: [c, p] crisp and probabilistic segmentation OR gmm, instance of the class sklearn.mixture.GaussianMixture.
        """

        pfi_segm = connect_path_tail_head(self.pfo_in, path_to_input_image)

        input_im = nib.load(pfi_segm)
        if mask_im is not None:
            mask_array = mask_im.get_data()
        else:
            mask_array = None

        ans = MoG_array(input_im.get_data(), K=K, mask_array=mask_array,
                        pre_process_median_filter=pre_process_median_filter,
                        pre_process_only_interquartile=pre_process_only_interquartile,
                        output_gmm_class=False, see_histogram=see_histogram, reorder_mus=reorder_mus)

        crisp, prob = ans[0], ans[1]

        im_crisp = set_new_data(input_im, crisp, new_dtype=np.uint8)
        im_prob = set_new_data(input_im, prob, new_dtype=np.float64)

        return im_crisp, im_prob