import os
import numpy as np
import nibabel as nib


from labels_manager.tools.aux_methods.utils import set_new_data
from labels_manager.tools.labels.relabeller import relabeller, permute_labels, erase_labels, assign_all_other_labels_the_same_value
from labels_manager.tools.labels.splitter import split_labels_to_4d
from labels_manager.tools.labels.merger import merge_labels_from_4d
from labels_manager.tools.labels.selector import keep_only_one_label
from labels_manager.tools.labels.symmetrizer import symmetrise_data, sym_labels
from labels_manager.tools.labels.propagators import simple_propagator
from labels_manager.tools.aux_methods.sanity_checks import arrange_path


class LabelsManagerManipulate(object):
    """
    Facade of the methods in tools, for work both with data (numpy array) or with paths.
    Additionally it will manipulate the labels descriptions in consequence.
    Use directly the methods in tools
    """

    def __init__(self, input_data_folder=None, output_data_folder=None):
        self.input_data_folder = input_data_folder
        self.output_data_folder = output_data_folder

    def relabel(self, data_in, data_out=None, list_old_labels=(), list_new_labels=(), label_description_path=None):

        if isinstance(data_in, np.ndarray):
            return relabeller(data_in, list_old_labels, list_new_labels)

        elif isinstance(data_in, str):
            input_im_path = arrange_path(self.input_data_folder, data_in)
            im_labels = nib.load(input_im_path)
            data_labels = im_labels.get_data()
            data_relabelled = relabeller(data_labels, list_old_labels=list_old_labels, list_new_labels=list_new_labels)

            if label_description_path is not None:
                pass  # TODO apply here the same to the description the same permutation!

            if data_out is None:
                return data_relabelled
            else:
                output_im_path = arrange_path(self.output_data_folder, data_out)
                im_relabelled = set_new_data(im_labels, data_relabelled)
                nib.save(im_relabelled, output_im_path)
                print 'Relabelled image {0} saved in {1}.'.format(input_im_path, output_im_path)
                return None
        else:
            raise IOError

    def permute(self, data_in, data_out=None, permutation=(), label_description_path=None):

        if isinstance(data_in, np.ndarray):
            return permute_labels(data_in, permutation=permutation)

        elif isinstance(data_in, str):
            input_im_path = arrange_path(self.input_data_folder, data_in)
            im_labels = nib.load(input_im_path)
            data_labels = im_labels.get_data()
            data_permuted = permute_labels(data_labels, permutation=permutation)

            if label_description_path is not None:
                pass  # TODO apply here to the description the same permutation!

            if data_out is None:
                return data_permuted
            else:
                output_im_path = arrange_path(self.output_data_folder, data_out)
                im_permuted = set_new_data(im_labels, data_permuted)
                nib.save(im_permuted, output_im_path)
                print 'Permuted labels from image {0} saved in {1}.'.format(input_im_path, output_im_path)
                return None
        else:
            raise IOError

    def erase(self, data_in, data_out=None, labels_to_erase=(), label_description_path=None):

        if isinstance(data_in, np.ndarray):
            return erase_labels(data_in, labels_to_erase=labels_to_erase)

        elif isinstance(data_in, str):
            input_im_path = arrange_path(self.input_data_folder, data_in)
            im_labels = nib.load(input_im_path)
            data_labels = im_labels.get_data()
            data_erased = erase_labels(data_labels, labels_to_erase=labels_to_erase)

            if label_description_path is not None:
                pass  # TODO apply to label descriptors

            if data_out is None:
                return data_erased
            else:
                output_im_path = arrange_path(self.output_data_folder, data_out)
                im_erased = set_new_data(im_labels, data_erased)
                nib.save(im_erased, output_im_path)
                print 'Erased labels from image {0} saved in {1}.'.format(input_im_path, output_im_path)
                return None
        else:
            raise IOError

    def assign_all_other_labels_the_same_value(self, data_in, data_out=None, labels_to_keep=(), same_value_label=255,
                                               label_description_path=None):

        if isinstance(data_in, np.ndarray):
            return assign_all_other_labels_the_same_value(data_in,
                                                          labels_to_keep=labels_to_keep,
                                                          same_value_label=same_value_label)

        elif isinstance(data_in, str):
            input_im_path = arrange_path(self.input_data_folder, data_in)
            im_labels = nib.load(input_im_path)
            data_labels = im_labels.get_data()
            data_reassigned = assign_all_other_labels_the_same_value(data_labels,
                                                                     labels_to_keep=labels_to_keep,
                                                                     same_value_label=same_value_label)

            if label_description_path is not None:
                pass  # TODO apply to label descriptors

            if data_out is None:
                return data_reassigned
            else:
                output_im_path = arrange_path(self.output_data_folder, data_out)
                im_reassigned = set_new_data(im_labels, data_reassigned)
                nib.save(im_reassigned, output_im_path)
                print 'Reassigned labels from image {0} saved in {1}.'.format(input_im_path, output_im_path)
                return None
        else:
            raise IOError

    def split_in_4d(self, data_in, data_out=None):

        if isinstance(data_in, np.ndarray):
            assert len(data_in.shape) == 3
            return split_labels_to_4d(data_in)

        elif isinstance(data_in, str):
            input_im_path = arrange_path(self.input_data_folder, data_in)
            im_labels_3d = nib.load(input_im_path)
            data_labels_3d = im_labels_3d.get_data()
            assert len(data_labels_3d.shape) == 3
            data_split_in_4d = split_labels_to_4d(data_labels_3d)

            if data_out is None:
                return data_split_in_4d
            else:
                output_im_path = arrange_path(self.output_data_folder, data_out)
                im_split_in_4d = set_new_data(im_labels_3d, data_split_in_4d)
                nib.save(im_split_in_4d, output_im_path)
                print 'Permuted labels from image {0} saved in {1}.'.format(input_im_path, output_im_path)
                return None
        else:
            raise IOError

    def merge_from_4d(self, data_in, data_out=None):

        if isinstance(data_in, np.ndarray):
            assert len(data_in.shape) == 4
            return merge_labels_from_4d(data_in)

        elif isinstance(data_in, str):
            input_im_path = arrange_path(self.input_data_folder, data_in)
            im_labels_4d = nib.load(input_im_path)
            data_labels_4d = im_labels_4d.get_data()
            assert len(data_labels_4d.shape) == 4
            data_merged_in_3d = merge_labels_from_4d(data_labels_4d)

            if data_out is None:
                return data_merged_in_3d
            else:
                output_im_path = arrange_path(self.output_data_folder, data_out)
                im_merged_in_3d = set_new_data(im_labels_4d, data_merged_in_3d)
                nib.save(im_merged_in_3d, output_im_path)
                print 'Permuted labels from image {0} saved in {1}.'.format(input_im_path, output_im_path)
                return None
        else:
            raise IOError

    def keep_one_label(self, data_in, data_out=None, labels_to_keep=1, label_description_path=None):

        if isinstance(data_in, np.ndarray):
            return keep_only_one_label(data_in, labels_to_keep)

        elif isinstance(data_in, str):
            input_im_path = arrange_path(self.input_data_folder, data_in)
            im_labels = nib.load(input_im_path)
            data_labels = im_labels.get_data()
            data_one_label = keep_only_one_label(data_labels, labels_to_keep)

            if label_description_path is not None:
                pass  # TODO apply to label descriptors

            if data_out is None:
                return data_one_label
            else:
                output_im_path = arrange_path(self.output_data_folder, data_out)
                im_one_label = set_new_data(im_labels, data_one_label)
                nib.save(im_one_label, output_im_path)
                print 'One label estracted from image {0} and saved in {1}.'.format(input_im_path, output_im_path)
                return None
        else:
            raise IOError

    def symmetrise_axial(self, data_in, data_out=None, axis='x', plane_intercept=10, side_to_copy='below',
                         keep_in_data_dimensions=True):

        if isinstance(data_in, np.ndarray):
            return symmetrise_data(data_in,
                                   axis=axis,
                                   plane_intercept=plane_intercept,
                                   side_to_copy=side_to_copy,
                                   keep_in_data_dimensions=keep_in_data_dimensions)

        elif isinstance(data_in, str):

            input_im_path = arrange_path(self.input_data_folder, data_in)
            im_labels = nib.load(input_im_path)
            data_labels = im_labels.get_data()
            data_symmetrised = symmetrise_data(data_labels,
                                               axis=axis,
                                               plane_intercept=plane_intercept,
                                               side_to_copy=side_to_copy,
                                               keep_in_data_dimensions=keep_in_data_dimensions)

            if data_out is None:
                return data_symmetrised

            else:
                output_im_path = arrange_path(self.output_data_folder, data_out)
                im_symmetrised = set_new_data(im_labels, data_symmetrised)
                nib.save(im_symmetrised, output_im_path)
                print 'Symmetrised image of {0} saved in {1}.'.format(input_im_path, output_im_path)
                return None
        else:
            raise IOError

    def symmetrise_with_registration(self,
                                     in_img_anatomy_path,
                                     in_img_labels_path,
                                     labels_input,
                                     result_img_path,
                                     results_folder_path=None,
                                     labels_transformed=None,
                                     coord='z',
                                     labels_descriptor_path=None
                                     ):

        # set paths - add the root if exists:
        in_img_anatomy_path = arrange_path(self.input_data_folder, in_img_anatomy_path)
        in_img_labels_path = arrange_path(self.input_data_folder, in_img_labels_path)

        if results_folder_path is None:
            if self.output_data_folder is not None:
                results_folder_path = self.output_data_folder
            else:
                results_folder_path = arrange_path(self.output_data_folder, results_folder_path)
        else:
            results_folder_path = os.path.dirname(in_img_labels_path)

        if labels_descriptor_path is not None:
            # TODO label descriptor manager for the symmetriser.
            pass

        sym_labels(in_img_anatomy_path,
                   in_img_labels_path,
                   labels_input,
                   result_img_path,
                   results_folder=results_folder_path,
                   labels_transformed=labels_transformed,
                   coord=coord)

    def propagator(self,
                   ref_in,
                   flo_in,
                   flo_mask_in,
                   flo_on_ref_img_out,
                   flo_on_ref_mask_out,
                   flo_on_ref_trans_out):

        ref_in               = arrange_path(self.input_data_folder, ref_in)
        flo_in               = arrange_path(self.input_data_folder, flo_in)
        flo_mask_in          = arrange_path(self.input_data_folder, flo_mask_in)
        flo_on_ref_img_out   = arrange_path(self.input_data_folder, flo_on_ref_img_out)
        flo_on_ref_mask_out  = arrange_path(self.input_data_folder, flo_on_ref_mask_out)
        flo_on_ref_trans_out = arrange_path(self.input_data_folder, flo_on_ref_trans_out)

        simple_propagator(ref_in, flo_in, flo_mask_in,
                          flo_on_ref_img_out, flo_on_ref_mask_out, flo_on_ref_trans_out,
                          settings_reg='', settings_interp=' -inter 0 ',
                          verbose_on=True, safety_on=False)

    def permutator(self, data_in, data_out=None, permutation=None, pfi_label_descriptor=None):

        if isinstance(data_in, np.ndarray):
            return keep_only_one_label(data_in)

        elif isinstance(data_in, str):
            input_im_path = arrange_path(self.input_data_folder, data_in)
            im_labels = nib.load(input_im_path)
            data_labels = im_labels.get_data()
            labels_permuted = permute_labels(data_labels, permutation)

            if pfi_label_descriptor is not None:
                pass  # TODO apply to label descriptors

            if data_out is None:
                return labels_permuted
            else:
                output_im_path = arrange_path(self.output_data_folder, data_out)
                im_one_label = set_new_data(im_labels, labels_permuted)
                nib.save(im_one_label, output_im_path)
                print 'One label estracted from image {0} and saved in {1}.'.format(input_im_path, output_im_path)
                return None
        else:
            raise IOError
