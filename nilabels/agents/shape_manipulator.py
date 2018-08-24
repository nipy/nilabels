import nibabel as nib
import numpy as np

from nilabels.tools.aux_methods.utils_path import get_pfi_in_pfi_out, connect_path_tail_head
from nilabels.tools.aux_methods.utils_nib import set_new_data
from nilabels.tools.image_colors_manipulations.cutter import cut_4d_volume_with_a_1_slice_mask_nib
from nilabels.tools.image_shape_manipulations.merger import merge_labels_from_4d, stack_images
from nilabels.tools.image_shape_manipulations.splitter import split_labels_to_4d


class ShapeManipulator(object):

    def __init__(self, input_data_folder=None, output_data_folder=None):
        self.pfo_in = input_data_folder
        self.pfo_out = output_data_folder

    def extend_slice_new_dimension(self, pfi_input, pfi_output=None, new_axis=3, num_slices=10):
        pfi_in, pfi_out = get_pfi_in_pfi_out(pfi_input, pfi_output, self.pfo_in, self.pfo_out)

        im_slice = nib.load(pfi_in)
        data_slice = im_slice.get_data()

        data_extended = np.stack([data_slice, ] * num_slices, axis=new_axis)

        im_extended = set_new_data(im_slice, data_extended)
        nib.save(im_extended, pfi_out)
        print('Extended image of {0} saved in {1}.'.format(pfi_in, pfi_out))
        return pfi_out

    def split_in_4d(self, pfi_input, pfi_output=None, list_labels=None, keep_original_values=True):
        pfi_in, pfi_out = get_pfi_in_pfi_out(pfi_input, pfi_output, self.pfo_in, self.pfo_out)

        im_labels_3d = nib.load(pfi_in)
        data_labels_3d = im_labels_3d.get_data()
        assert len(data_labels_3d.shape) == 3
        if list_labels is None:
            list_labels = list(np.sort(list(set(data_labels_3d.flat))))
        data_split_in_4d = split_labels_to_4d(data_labels_3d, list_labels=list_labels,
                                              keep_original_values=keep_original_values)

        im_split_in_4d = set_new_data(im_labels_3d, data_split_in_4d)
        nib.save(im_split_in_4d, pfi_out)
        print('Split labels from image {0} saved in {1}.'.format(pfi_in, pfi_out))
        return pfi_out

    def merge_from_4d(self, pfi_input, pfi_output=None):
        pfi_in, pfi_out = get_pfi_in_pfi_out(pfi_input, pfi_output, self.pfo_in, self.pfo_out)

        im_labels_4d = nib.load(pfi_in)
        data_labels_4d = im_labels_4d.get_data()
        assert len(data_labels_4d.shape) == 4
        data_merged_in_3d = merge_labels_from_4d(data_labels_4d)

        im_merged_in_3d = set_new_data(im_labels_4d, data_merged_in_3d)
        nib.save(im_merged_in_3d, pfi_out)
        print('Merged labels from 4d image {0} saved in {1}.'.format(pfi_in, pfi_out))
        return pfi_out

    def cut_4d_volume_with_a_1_slice_mask(self, pfi_input, filename_mask, pfi_output=None):
        pfi_in, pfi_out = get_pfi_in_pfi_out(pfi_input, pfi_output, self.pfo_in, self.pfo_out)
        pfi_mask = connect_path_tail_head(self.pfo_in, filename_mask)

        im_dwi = nib.load(pfi_in)
        im_mask = nib.load(pfi_mask)

        im_masked = cut_4d_volume_with_a_1_slice_mask_nib(im_dwi, im_mask)

        nib.save(im_masked, pfi_out)

    def stack_list_pfi_images(self, list_pfi_input, pfi_output):
        list_pfi_in = [connect_path_tail_head(self.pfo_in, p) for p in list_pfi_input]
        pfi_out = connect_path_tail_head(self.pfo_out, pfi_output)

        list_im = [nib.load(p) for p in list_pfi_in]
        stack_im = stack_images(list_im)

        nib.save(stack_im, pfi_out)
