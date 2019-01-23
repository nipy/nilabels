import os

import nibabel as nib
import numpy as np

from nilabels.tools.aux_methods.label_descriptor_manager import LabelsDescriptorManager as LdM
from nilabels.tools.aux_methods.utils_nib import set_new_data
from nilabels.tools.aux_methods.utils_path import get_pfi_in_pfi_out, connect_path_tail_head
from nilabels.tools.cleaning.labels_cleaner import clean_semgentation
from nilabels.tools.image_colors_manipulations.relabeller import relabeller, \
    permute_labels, erase_labels, assign_all_other_labels_the_same_value, keep_only_one_label
from nilabels.tools.image_colors_manipulations.segmentation_to_rgb import \
    get_rgb_image_from_segmentation_and_label_descriptor
from nilabels.tools.image_shape_manipulations.merger import from_segmentations_stack_to_probabilistic_segmentation


class LabelsManipulator(object):
    """
    Facade of the methods in tools, for work with paths to images rather than
    with data. Methods under LabelsManagerManipulate are taking in general
    one or more input manipulate them according to some rule and save the
    output in the output_data_folder or in the specified paths.
    """
    def __init__(self, input_data_folder=None, output_data_folder=None):
        self.pfo_in  = input_data_folder
        self.pfo_out = output_data_folder
        self.verbose = True
        self.labels_descriptor_convention = 'itk-snap'

    def relabel(self, path_to_input_segmentation, path_to_output_segmentation=None,
                list_old_labels=(), list_new_labels=(), path_to_input_labels_descriptor=None,
                path_to_output_labels_descriptor=None):
        """
        Masks of :func:`labels_manager.tools.manipulations.relabeller.relabeller` using filenames
        """

        pfi_in, pfi_out = get_pfi_in_pfi_out(path_to_input_segmentation, path_to_output_segmentation, self.pfo_in,
                                             self.pfo_out)
        im_labels = nib.load(pfi_in)
        data_labels = im_labels.get_data()
        data_relabelled = relabeller(data_labels, list_old_labels=list_old_labels,
                                     list_new_labels=list_new_labels)

        im_relabelled = set_new_data(im_labels, data_relabelled)

        nib.save(im_relabelled, pfi_out)

        if path_to_input_labels_descriptor is not None:
            pfi_in_ld = connect_path_tail_head(self.pfo_in, path_to_input_labels_descriptor)

            ldm_input = LdM(pfi_in_ld, labels_descriptor_convention=self.labels_descriptor_convention)
            ldm_relabelled = ldm_input.relabel(list_old_labels=list_old_labels, list_new_labels=list_new_labels)

            if path_to_output_labels_descriptor is None:
                ldm_relabelled.save_label_descriptor(pfi_in_ld)
            else:
                pfi_out_ld = connect_path_tail_head(self.pfo_out, path_to_output_labels_descriptor)
                ldm_relabelled.save_label_descriptor(pfi_out_ld)

        print('Relabelled image {0} saved in {1}.'.format(pfi_in, pfi_out))
        return pfi_out

    def permute_labels(self, path_to_input_segmentation, path_to_output_segmentation=None, permutation=(),
                       path_to_input_labels_descriptor=None, path_to_output_labels_descriptor=None):

        pfi_in, pfi_out = get_pfi_in_pfi_out(path_to_input_segmentation, path_to_output_segmentation,
                                             self.pfo_in, self.pfo_out)

        im_labels = nib.load(pfi_in)
        data_labels = im_labels.get_data()
        data_permuted = permute_labels(data_labels, permutation=permutation)

        im_permuted = set_new_data(im_labels, data_permuted)
        nib.save(im_permuted, pfi_out)

        if path_to_input_labels_descriptor is not None:
            pfi_in_ld = connect_path_tail_head(self.pfo_in, path_to_input_labels_descriptor)

            ldm_input = LdM(pfi_in_ld, labels_descriptor_convention=self.labels_descriptor_convention)
            ldm_relabelled = ldm_input.permute_labels(permutation)

            if path_to_output_labels_descriptor is None:
                ldm_relabelled.save_label_descriptor(pfi_in_ld)
            else:
                pfi_out_ld = connect_path_tail_head(self.pfo_out, path_to_output_labels_descriptor)
                ldm_relabelled.save_label_descriptor(pfi_out_ld)

        print('Permuted labels from image {0} saved in {1}.'.format(pfi_in, pfi_out))
        return pfi_out

    def erase_labels(self, path_to_input_segmentation, path_to_output_segmentation=None, labels_to_erase=(),
                     path_to_input_labels_descriptor=None, path_to_output_labels_descriptor=None):

        pfi_in, pfi_out = get_pfi_in_pfi_out(path_to_input_segmentation, path_to_output_segmentation,
                                             self.pfo_in, self.pfo_out)

        im_labels = nib.load(pfi_in)
        data_labels = im_labels.get_data()
        data_erased = erase_labels(data_labels, labels_to_erase=labels_to_erase)

        im_erased = set_new_data(im_labels, data_erased)
        nib.save(im_erased, pfi_out)

        if path_to_input_labels_descriptor is not None:
            pfi_in_ld = connect_path_tail_head(self.pfo_in, path_to_input_labels_descriptor)

            ldm_input = LdM(pfi_in_ld, labels_descriptor_convention=self.labels_descriptor_convention)
            ldm_relabelled = ldm_input.erase_labels(labels_to_erase, verbose=self.verbose)

            if path_to_output_labels_descriptor is None:
                ldm_relabelled.save_label_descriptor(pfi_in_ld)
            else:
                pfi_out_ld = connect_path_tail_head(self.pfo_out, path_to_output_labels_descriptor)
                ldm_relabelled.save_label_descriptor(pfi_out_ld)

        print('Erased labels from image {0} saved in {1}.'.format(pfi_in, pfi_out))
        return pfi_out

    def assign_all_other_labels_the_same_value(self, path_to_input_segmentation, path_to_output_segmentation=None,
                                               labels_to_keep=(), same_value_label=255,
                                               path_to_input_labels_descriptor=None,
                                               path_to_output_labels_descriptor=None):
        """
        :param path_to_input_segmentation:
        :param path_to_output_segmentation:
        :param labels_to_keep:
        :param same_value_label:
        :param path_to_input_labels_descriptor:
        :param path_to_output_labels_descriptor:
        :return:
        """
        pfi_in, pfi_out = get_pfi_in_pfi_out(path_to_input_segmentation, path_to_output_segmentation,
                                             self.pfo_in, self.pfo_out)

        im_labels = nib.load(pfi_in)
        data_labels = im_labels.get_data()
        data_reassigned = assign_all_other_labels_the_same_value(data_labels,
                                labels_to_keep=labels_to_keep, same_value_label=same_value_label)

        im_reassigned = set_new_data(im_labels, data_reassigned)
        nib.save(im_reassigned, pfi_out)

        if path_to_input_labels_descriptor is not None:
            pfi_in_ld = connect_path_tail_head(self.pfo_in, path_to_input_labels_descriptor)

            ldm_input = LdM(pfi_in_ld, labels_descriptor_convention=self.labels_descriptor_convention)
            ldm_relabelled = ldm_input.assign_all_other_labels_the_same_value(labels_to_keep=labels_to_keep,
                                                                              same_value_label=same_value_label)

            if path_to_output_labels_descriptor is None:
                ldm_relabelled.save_label_descriptor(pfi_in_ld)
            else:
                pfi_out_ld = connect_path_tail_head(self.pfo_out, path_to_output_labels_descriptor)
                ldm_relabelled.save_label_descriptor(pfi_out_ld)

        print('Reassigned labels from image {0} saved in {1}.'.format(pfi_in, pfi_out))
        return pfi_out

    def keep_one_label(self, path_to_input_segmentation, path_to_output_segmentation=None, label_to_keep=1,
                       path_to_input_labels_descriptor=None, path_to_output_labels_descriptor=None):
        pfi_in, pfi_out = get_pfi_in_pfi_out(path_to_input_segmentation, path_to_output_segmentation,
                                             self.pfo_in, self.pfo_out)

        im_labels      = nib.load(pfi_in)
        data_labels    = im_labels.get_data()
        data_one_label = keep_only_one_label(data_labels, label_to_keep)

        im_one_label = set_new_data(im_labels, data_one_label)
        nib.save(im_one_label, pfi_out)

        if path_to_input_labels_descriptor is not None:
            pfi_in_ld = connect_path_tail_head(self.pfo_in, path_to_input_labels_descriptor)

            ldm_input = LdM(pfi_in_ld, labels_descriptor_convention=self.labels_descriptor_convention)
            ldm_relabelled = ldm_input.keep_one_label(label_to_keep)

            if path_to_output_labels_descriptor is None:
                ldm_relabelled.save_label_descriptor(pfi_in_ld)
            else:
                pfi_out_ld = connect_path_tail_head(self.pfo_out, path_to_output_labels_descriptor)
                ldm_relabelled.save_label_descriptor(pfi_out_ld)

        print('Label {0} kept from image {1} and saved in {2}.'.format(label_to_keep, pfi_in, pfi_out))
        return pfi_out

    def get_probabilistic_prior_from_stack_segmentations(self, path_to_stack_crisp_segm, path_to_fuzzy_output):
        pfi_in, pfi_out = get_pfi_in_pfi_out(path_to_stack_crisp_segm, path_to_fuzzy_output, self.pfo_in, self.pfo_out)

        im_stack_crisp = nib.load(path_to_stack_crisp_segm)

        dims = im_stack_crisp.shape

        vec = [np.prod(im_stack_crisp.shape[:3])] + [dims[3]]

        array_output = from_segmentations_stack_to_probabilistic_segmentation(
            im_stack_crisp.get_data().reshape(vec).T).T

        data_output = array_output.reshape(list(im_stack_crisp.shape[:3]) + [array_output.shape[1]])

        new_im = set_new_data(im_stack_crisp, data_output, new_dtype=np.float64)
        nib.save(new_im, pfi_out)

        return pfi_out

    def clean_segmentation(self, path_to_input_segmentation, path_to_output_cleaned_segmentation, labels_to_clean=(),
                           verbose=1, special_label=None, force_overwriting=False):
        """
        Clean the segmentation merging the small connected components with the surrounding tissue.
        :param path_to_input_segmentation: path to the input segmentation
        :param path_to_output_cleaned_segmentation: path to the output cleaned segmentation. For safety this file can
        not exist before calling this method.
        :param labels_to_clean: list of binaries lists. [[z_1, zc_1], ... , [z_J, zc_J]] where z_j is the label you want
        to clean and zc_1 is the number of components you want to keep. If empty tuple, by default cleans all the labels
        keeping only one component.
        :param special_label: internal variable for the dummy labels that will be used for the 'holes'. This must
        not be a label already present in the segmentation. If None, it is the max label + 1.
        :param verbose:
        :return: it saves the cleaned segmentation at the specified output path.
        Note: as a feature (really!) after the holes identification, all labels, and not only the ones indicated in the
        input dilate iteratively over the 'holes'.
        """
        pfi_in, pfi_out = get_pfi_in_pfi_out(path_to_input_segmentation, path_to_output_cleaned_segmentation,
                                             self.pfo_in, self.pfo_out)
        if not force_overwriting:
            if os.path.exists(pfi_out):
                raise IOError('File {} already exists. Cleaner can not overwrite a segmentation'.format(pfi_out))

        im_segm = nib.load(pfi_in)

        if special_label is None:
            special_label = np.max(im_segm.get_data()) + 1

        new_segm_data = clean_semgentation(im_segm.get_data(), labels_to_clean=labels_to_clean,
                                           label_for_holes=special_label)

        im_segm_cleaned = set_new_data(im_segm, new_segm_data.astype(im_segm.get_data_dtype()))
        nib.save(im_segm_cleaned, pfi_out)
        if verbose:
            print('Segmentation {} cleaned saved to {}'.format(pfi_in, pfi_out))

    def from_segmentation_and_labels_descriptor_to_rgb(self, path_to_input_segmentation,
                                                       path_to_input_txt_labels_descriptor,
                                                       path_to_output_4d_rgb_image,
                                                       invert_black_white=False, dtype_output=np.int32):
        """
        + Masks of :func:`nilabel.tools.image_colors_manipulations.segmentation_to_rgb.
        get_rgb_image_from_segmentation_and_label_descriptor` using filenames.
        From a segmentation and its label descriptro in itk-snap convention or in fsl convention
        it creates a corresponding 4d image with the 3 R G B channels in the fourth dimension.
        :param path_to_input_segmentation: path to the input segmentation
        :param path_to_input_txt_labels_descriptor: path to the txt labels descriptor
        :param path_to_output_4d_rgb_image: path where to save the output
        :param invert_black_white: changes the background colour, if black to white.
        :param dtype_output: data type of the output np.int32
        :return:
        """
        pfi_segm = connect_path_tail_head(self.pfo_in, path_to_input_segmentation)
        pfi_ld = connect_path_tail_head(self.pfo_in, path_to_input_txt_labels_descriptor)

        im_segm = nib.load(pfi_segm)
        ldm = LdM(pfi_ld, labels_descriptor_convention=self.labels_descriptor_convention)
        im_rgb_4d_image = get_rgb_image_from_segmentation_and_label_descriptor(im_segm, ldm,
                                                                               invert_black_white=invert_black_white,
                                                                               dtype_output=dtype_output)

        pfi_out = connect_path_tail_head(self.pfo_out, path_to_output_4d_rgb_image)
        nib.save(im_rgb_4d_image, pfi_out)
