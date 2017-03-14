import os
import nibabel as nib
import numpy as np

from labels_manager.tools.aux_methods.utils import set_new_data
from labels_manager.tools.manipulations.relabeller import relabeller, \
        permute_labels, erase_labels, assign_all_other_labels_the_same_value, keep_only_one_label
from labels_manager.tools.manipulations.splitter import split_labels_to_4d
from labels_manager.tools.manipulations.merger import merge_labels_from_4d
from labels_manager.tools.manipulations.symmetrizer import symmetrise_data, sym_labels
from labels_manager.tools.manipulations.propagators import simple_propagator
from labels_manager.tools.aux_methods.sanity_checks import get_pfi_in_pfi_out,\
     connect_tail_head_path
from labels_manager.tools.manipulations.slicing import reproduce_slice_fourth_dimension


class LabelsManagerManipulate(object):
    """
    Facade of the methods in tools, for work with paths to images rather than
    with data. Methods under LabelsManagerManipulate are taking in general
    one or more input manipulate them according to some rule and save the
    output in the output_data_folder or in the specified paths.
    """
    # TODO add filename for labels descriptors and manipulations of labels descriptors

    def __init__(self, input_data_folder=None, output_data_folder=None):
        self.pfo_in = input_data_folder
        self.pfo_out = output_data_folder

    def relabel(self, filename_in, filename_out=None, list_old_labels=(),
                list_new_labels=()):
        """
        Masks of :func:`labels_manager.tools.manipulations.relabeller.relabeller` using filename
        """

        pfi_in, pfi_out = get_pfi_in_pfi_out(filename_in, filename_out, self.pfo_in,
                                             self.pfo_out)
        im_labels = nib.load(pfi_in)
        data_labels = im_labels.get_data()
        data_relabelled = relabeller(data_labels, list_old_labels=list_old_labels,
                                     list_new_labels=list_new_labels)

        im_relabelled = set_new_data(im_labels, data_relabelled)

        nib.save(im_relabelled, pfi_out)
        print('Relabelled image {0} saved in {1}.'.format(pfi_in, pfi_out))
        return pfi_out

    def permute(self, filename_in, filename_out=None, permutation=()):

        pfi_in, pfi_out = get_pfi_in_pfi_out(filename_in, filename_out, self.pfo_in, self.pfo_out)

        im_labels = nib.load(pfi_in)
        data_labels = im_labels.get_data()
        data_permuted = permute_labels(data_labels, permutation=permutation)

        im_permuted = set_new_data(im_labels, data_permuted)
        nib.save(im_permuted, pfi_out)
        print('Permuted labels from image {0} saved in {1}.'.format(pfi_in, pfi_out))
        return pfi_out

    def erase(self, filename_in, filename_out=None, labels_to_erase=()):

        pfi_in, pfi_out = get_pfi_in_pfi_out(filename_in, filename_out, self.pfo_in, self.pfo_out)

        im_labels = nib.load(pfi_in)
        data_labels = im_labels.get_data()
        data_erased = erase_labels(data_labels, labels_to_erase=labels_to_erase)

        im_erased = set_new_data(im_labels, data_erased)
        nib.save(im_erased, pfi_out)
        print('Erased labels from image {0} saved in {1}.'.format(pfi_in, pfi_out))
        return pfi_out

    def assign_all_other_labels_the_same_value(self, filename_in, filename_out=None,
        labels_to_keep=(), same_value_label=255):

        pfi_in, pfi_out = get_pfi_in_pfi_out(filename_in, filename_out, self.pfo_in, self.pfo_out)

        im_labels = nib.load(pfi_in)
        data_labels = im_labels.get_data()
        data_reassigned = assign_all_other_labels_the_same_value(data_labels,
                                labels_to_keep=labels_to_keep, same_value_label=same_value_label)

        im_reassigned = set_new_data(im_labels, data_reassigned)
        nib.save(im_reassigned, pfi_out)
        print('Reassigned labels from image {0} saved in {1}.'.format(pfi_in, pfi_out))
        return pfi_out

    def keep_one_label(self, filename_in, filename_out=None, label_to_keep=1):

        pfi_in, pfi_out = get_pfi_in_pfi_out(filename_in, filename_out, self.pfo_in, self.pfo_out)

        im_labels = nib.load(pfi_in)
        data_labels = im_labels.get_data()
        data_one_label = keep_only_one_label(data_labels, label_to_keep)

        im_one_label = set_new_data(im_labels, data_one_label)
        nib.save(im_one_label, pfi_out)
        print('Label {0} kept from image {1} and saved in {2}.'.format(label_to_keep, pfi_in, pfi_out))
        return pfi_out

    def extend_slice_new_dimension(self, filename_in, filename_out=None, new_axis=3, num_slices=10):

        pfi_in, pfi_out = get_pfi_in_pfi_out(filename_in, filename_out, self.pfo_in, self.pfo_out)

        im_slice = nib.load(pfi_in)
        data_slice = im_slice.get_data()

        data_extended = reproduce_slice_fourth_dimension(data_slice,
                                num_slices=num_slices, new_axis=new_axis)

        im_extended = set_new_data(im_slice, data_extended)
        nib.save(im_extended, pfi_out)
        print('Extended image of {0} saved in {1}.'.format(pfi_in, pfi_out))
        return pfi_out

    def split_in_4d(self, filename_in, filename_out=None, list_labels=None, keep_original_values=True):

        pfi_in, pfi_out = get_pfi_in_pfi_out(filename_in, filename_out, self.pfo_in, self.pfo_out)

        im_labels_3d = nib.load(pfi_in)
        data_labels_3d = im_labels_3d.get_data()
        assert len(data_labels_3d.shape) == 3
        if list_labels is None:
            list_labels = list(np.sort(list(set(data_labels_3d.flat))))
        data_split_in_4d = split_labels_to_4d(data_labels_3d, list_labels=list_labels, keep_original_values=keep_original_values)

        im_split_in_4d = set_new_data(im_labels_3d, data_split_in_4d)
        nib.save(im_split_in_4d, pfi_out)
        print('Split labels from image {0} saved in {1}.'.format(pfi_in, pfi_out))
        return pfi_out

    def merge_from_4d(self, filename_in, filename_out=None):

        pfi_in, pfi_out = get_pfi_in_pfi_out(filename_in, filename_out, self.pfo_in, self.pfo_out)

        im_labels_4d = nib.load(pfi_in)
        data_labels_4d = im_labels_4d.get_data()
        assert len(data_labels_4d.shape) == 4
        data_merged_in_3d = merge_labels_from_4d(data_labels_4d)

        im_merged_in_3d = set_new_data(im_labels_4d, data_merged_in_3d)
        nib.save(im_merged_in_3d, pfi_out)
        print('Merged labels from 4d image {0} saved in {1}.'.format(pfi_in, pfi_out))
        return pfi_out

    def symmetrise_axial(self, filename_in, filename_out=None, axis='x', plane_intercept=10,
        side_to_copy='below', keep_in_data_dimensions=True):

        pfi_in, pfi_out = get_pfi_in_pfi_out(filename_in, filename_out, self.pfo_in, self.pfo_out)

        im_labels = nib.load(pfi_in)
        data_labels = im_labels.get_data()
        data_symmetrised = symmetrise_data(data_labels,
                                           axis=axis,
                                           plane_intercept=plane_intercept,
                                           side_to_copy=side_to_copy,
                                           keep_in_data_dimensions=keep_in_data_dimensions)

        im_symmetrised = set_new_data(im_labels, data_symmetrised)
        nib.save(im_symmetrised, pfi_out)
        print('Symmetrised axis {0}, plane_intercept {1}, image of {2} saved in {3}.'.format(axis, plane_intercept, pfi_in, pfi_out))
        return pfi_out

    def symmetrise_with_registration(self,
                                     filename_anatomy,
                                     filename_segmentation,
                                     list_labels_input,
                                     result_img_path,
                                     results_folder_path=None,
                                     list_labels_transformed=None,
                                     coord='z',
                                     reuse_registration=False):

        pfi_in_anatomy = connect_tail_head_path(self.pfo_in, filename_anatomy)
        pfi_in_segmentation = connect_tail_head_path(self.pfo_in, filename_segmentation)

        if results_folder_path is None:
            if self.pfo_out is not None:
                results_folder_path = self.pfo_out
            else:
                results_folder_path = self.pfo_in
        else:
            results_folder_path = os.path.dirname(pfi_in_segmentation)

        pfi_out_segmentation = connect_tail_head_path(results_folder_path, result_img_path)

        sym_labels(pfi_in_anatomy,
                   pfi_in_segmentation,
                   results_folder_path,
                   pfi_out_segmentation,
                   list_labels_input,
                   list_labels_transformed=list_labels_transformed,
                   coord=coord,
                   reuse_registration=reuse_registration)
