import nibabel as nib
import numpy as np

from nilabels.tools.aux_methods.utils_path import get_pfi_in_pfi_out, connect_path_tail_head
from nilabels.tools.aux_methods.utils import labels_query
from nilabels.tools.image_colors_manipulations.normaliser import normalise_below_labels
from nilabels.tools.detections.contours import contour_from_segmentation
from nilabels.tools.image_shape_manipulations.merger import grafting
from nilabels.tools.image_colors_manipulations.cutter import apply_a_mask_nib


class IntensitiesManipulator(object):
    """
    Facade of the methods in tools, for work with paths to images rather than
    with data. Methods under LabelsManagerManipulate are taking in general
    one or more input manipulate them according to some rule and save the
    output in the output_data_folder or in the specified paths.
    """
    def __init__(self, input_data_folder=None, output_data_folder=None, path_label_descriptor=None):
        self.pfo_in = input_data_folder
        self.pfo_out = output_data_folder
        self.path_label_descriptor = path_label_descriptor

    def normalise_below_label(self, filename_image_in, filename_image_out, filename_segm, labels, stats=np.median):
        """

        :param filename_image_in: path to image input
        :param filename_image_out: path to image output
        :param filename_segm: path to segmentation
        :param labels: list of labels below which the voxels are collected
        :param stats: a statistics (by default the median).
        :return: a new image with the intensites normalised according to the proposed statistics computed on the
        intensities below the provided labels.
        """
        pfi_in, pfi_out = get_pfi_in_pfi_out(filename_image_in, filename_image_out, self.pfo_in, self.pfo_out)
        pfi_segm = connect_path_tail_head(self.pfo_in, filename_segm)

        im_input = nib.load(pfi_in)
        im_segm = nib.load(pfi_segm)

        labels_list, labels_names = labels_query(labels, im_segm.get_data())
        im_out = normalise_below_labels(im_input, labels_list, labels, stats=stats, exclude_first_label=True)

        nib.save(im_out, pfi_out)

    def get_contour_from_segmentation(self, filename_input_segmentation, filename_output_contour,
                                      omit_axis=None, verbose=0):
        """
        Get the contour from a segmentation.
        :param filename_input_segmentation: input segmentation
        :param filename_output_contour: output contour
        :param omit_axis: meant to avoid "walls" in the output segmentation
        :param verbose:
        :return:
        """

        pfi_in, pfi_out = get_pfi_in_pfi_out(filename_input_segmentation, filename_output_contour,
                                             self.pfo_in, self.pfo_out)
        im_segm = nib.load(pfi_in)

        im_contour = contour_from_segmentation(im_segm, omit_axis=omit_axis, verbose=verbose)

        nib.save(im_contour, filename_output_contour)

    def get_grafting(self, pfi_input_hosting_mould, pfi_input_patch, pfi_output_grafted, pfi_input_patch_mask=None):
        """
        :param pfi_input_hosting_mould: base image where the grafting should happen
        :param pfi_input_patch: patch to be grafted in the hosting_mould
        :param pfi_output_grafted: output image with the grafting
        :param pfi_input_patch_mask: optional additional mask, where the grafting will take place.
        :return:
        """
        pfi_hosting = connect_path_tail_head(self.pfo_in, pfi_input_hosting_mould)
        pfi_patch = connect_path_tail_head(self.pfo_in, pfi_input_patch)

        im_hosting = nib.load(pfi_hosting)
        im_patch   = nib.load(pfi_patch)
        im_mask    = None

        if pfi_input_patch_mask is not None:
            pfi_mask = connect_path_tail_head(self.pfo_in, pfi_input_patch_mask)
            im_mask  = nib.load(pfi_mask)

        im_grafted = grafting(im_hosting, im_patch, im_patch_mask=im_mask)

        pfi_output = connect_path_tail_head(self.pfo_out, pfi_output_grafted)
        nib.save(im_grafted, pfi_output)

    def crop_outside_mask(self, filename_input_image, filename_mask, filename_output_image_masked):
        """
        Set to zero all the values outside the mask.
        Adaptative - if the mask is 3D and the image is 4D, will create a temporary mask,
        generate the stack of masks, and apply the stacks to the image.
        :param filename_input_image: path to file 3d x T image
        :param filename_mask: 3d mask same dimension as the 3d of the pfi_input
        :param filename_output_image_masked: apply the mask to each time point T in the fourth dimension if any.
        :return: None, it saves the output in pfi_output.
        """
        pfi_in, pfi_out = get_pfi_in_pfi_out(filename_input_image, filename_output_image_masked,
                                             self.pfo_in, self.pfo_out)

        pfi_mask = connect_path_tail_head(self.pfo_in, filename_mask)

        im_in, im_mask = nib.load(pfi_in), nib.load(pfi_mask)
        im_masked = apply_a_mask_nib(im_in, im_mask)

        pfi_output = connect_path_tail_head(self.pfo_out, filename_output_image_masked)
        nib.save(im_masked, pfi_output)
