import numpy as np
import nibabel as nib

from nilabels.tools.aux_methods.utils_path import connect_path_tail_head
from nilabels.tools.aux_methods.utils_nib import set_new_data
from nilabels.tools.detections.get_segmentation import intensity_segmentation, otsu_threshold, \
    MoG_array


class LabelsSegmenter(object):
    """
    Facade for the simple segmentation methods based on intensities, Otsu thresholding and
    mixture of gaussians.
    """
    def __init__(self, input_data_folder=None, output_data_folder=None):
        self.pfo_in = input_data_folder
        self.pfo_out = output_data_folder

    def simple_intensities_thresholding(self, path_to_input_image, path_to_output_segmentation, number_of_levels=5,
                                        output_dtype=np.uint16):
        """
        Simple level intensity-based segmentation.
        :param path_to_input_image:
        :param path_to_output_segmentation:
        :param number_of_levels: number of levels in the output segmentations
        :param output_dtype: data type output crisp segmentation (uint16)
        :return: the method saves crisp segmentation based on intensities at the specified path.
        """
        pfi_input_image = connect_path_tail_head(self.pfo_in, path_to_input_image)

        input_im = nib.load(pfi_input_image)
        output_array = intensity_segmentation(input_im.get_data(), num_levels=number_of_levels)
        output_im = set_new_data(input_im, output_array, new_dtype=output_dtype)

        pfi_output_segm = connect_path_tail_head(self.pfo_out, path_to_output_segmentation)
        nib.save(output_im, pfi_output_segm)

    def otsu_thresholding(self, path_to_input_image, path_to_output_segmentation, side='above', return_as_mask=True):
        """
        Binary segmentation with Otsu thresholding parameters from skimage filters.
        :param path_to_input_image:
        :param path_to_output_segmentation:
        :param side: can be 'above' or 'below' if the user requires to mask the values above the Otsu threshold or
        below.
        :param return_as_mask: if False it returns the thresholded input image.
        :return: the method saves the crisp binary segmentation (or the thresholded input image) according to Otsu
        threshold.
        """
        pfi_input_image = connect_path_tail_head(self.pfo_in, path_to_input_image)

        input_im = nib.load(pfi_input_image)
        output_array = otsu_threshold(input_im.get_data(), side=side, return_as_mask=return_as_mask)
        output_im = set_new_data(input_im, output_array, new_dtype=output_array.dtype)

        pfi_output_segm = connect_path_tail_head(self.pfo_out, path_to_output_segmentation)
        nib.save(output_im, pfi_output_segm)

    def mixture_of_gaussians(self, path_to_input_image, path_to_output_segmentation_crisp,
                             path_to_output_segmentation_prob, K=None, mask_im=None, pre_process_median_filter=False,
                             pre_process_only_interquartile=False, see_histogram=None, reorder_mus=True,
                             output_dtype_crisp=np.uint16, output_dtype_prob=np.float32):
        """
        Wrap of MoG_array for nibabel images.
        -----
        :param path_to_input_image: path to input image format to be segmented with a MOG method.
        :param path_to_output_segmentation_crisp: path to output crisp segmentation
        :param path_to_output_segmentation_prob: path to probabilistic output segmentation
        :param K: number of classes, if None, it is estimated with a BIC criterion (may take a while)
        :param mask_im: nibabel mask if you want to consider only a subset of the masked data.
        :param pre_process_median_filter: apply a median filter before pre-processing (reduce salt and pepper noise).
        :param pre_process_only_interquartile: set to zero above and below interquartile in the data.
        :param see_histogram: can be True, False (or None) or a string (with a path where to save the plotted
                              histogram).
        :param reorder_mus: only if output_gmm_class=False, reorder labels from smallest to bigger means.
        :param output_dtype_crisp: data type output crisp segmentation (uint16)
        :param output_dtype_prob: data type output probabilistic segmentation (float32)
        :return: save crisp and probabilistic segmentation at the specified files after sklearn.mixture.GaussianMixture
        """

        pfi_input_image = connect_path_tail_head(self.pfo_in, path_to_input_image)

        input_im = nib.load(pfi_input_image)
        if mask_im is not None:
            mask_array = mask_im.get_data()
        else:
            mask_array = None

        ans = MoG_array(input_im.get_data(), K=K, mask_array=mask_array,
                        pre_process_median_filter=pre_process_median_filter,
                        pre_process_only_interquartile=pre_process_only_interquartile,
                        output_gmm_class=False, see_histogram=see_histogram, reorder_mus=reorder_mus)

        crisp, prob = ans[0], ans[1]

        im_crisp = set_new_data(input_im, crisp, new_dtype=output_dtype_crisp)
        im_prob = set_new_data(input_im, prob, new_dtype=output_dtype_prob)

        pfi_im_crisp = connect_path_tail_head(self.pfo_out, path_to_output_segmentation_crisp)
        pfi_im_prob = connect_path_tail_head(self.pfo_out, path_to_output_segmentation_prob)

        nib.save(im_crisp, pfi_im_crisp)
        nib.save(im_prob, pfi_im_prob)
