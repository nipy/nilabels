import os
import numpy as np
import nibabel as nib

from nilabels.tools.aux_methods.utils_path import connect_path_tail_head
from nilabels.tools.aux_methods.utils_nib import set_new_data


class LabelsFuser(object):
    """
    Facade of the methods in tools, for work with paths to images rather than
    with data.
    """
    def __init__(self, input_data_folder=None, output_data_folder=None):
        self.pfo_in = input_data_folder
        self.pfo_out = output_data_folder

    def create_stack_for_labels_fusion(self, pfi_target, pfi_result, list_pfi_segmentations, list_pfi_warped=None,
                                       seg_output_name='res_4d_seg', warp_output_name='res_4d_warp', output_tag=''):
        """
        Stack and fuse anatomical images and segmentations in a single command.
        :param pfi_target: path to file to the target of the segmentation
        :param pfi_result: path to file where to store the result.
        :param list_pfi_segmentations: list of the segmentations to fuse
        :param list_pfi_warped: list of the warped images to fuse
        :param seg_output_name:
        :param warp_output_name:
        :param output_tag: additional tag output.
        :return: if prepare_data_only is True it returns the path to files prepared to be provided to nifty_seg,
                in the following order
                    [pfi_target, pfi_result, pfi_4d_seg, pfi_4d_warp]

        """
        pfi_target = connect_path_tail_head(self.pfo_in, pfi_target)
        pfi_result = connect_path_tail_head(self.pfo_out, pfi_result)
        # save 4d segmentations in stack_seg
        list_pfi_segmentations = [connect_path_tail_head(self.pfo_in, j) for j in list_pfi_segmentations]
        #
        list_stack_seg = [nib.load(pfi).get_data() for pfi in list_pfi_segmentations]
        stack_seg = np.stack(list_stack_seg, axis=3)
        del list_stack_seg
        im_4d_seg = set_new_data(nib.load(list_pfi_segmentations[0]), stack_seg)
        pfi_4d_seg = connect_path_tail_head(self.pfo_out, '{0}_{1}.nii.gz'.format(seg_output_name, output_tag))
        nib.save(im_4d_seg, pfi_4d_seg)

        # save 4d warped if available
        if list_pfi_warped is None:
            pfi_4d_warp = None
        else:
            list_pfi_warped = [connect_path_tail_head(self.pfo_in, j) for j in list_pfi_warped]
            #
            list_stack_warp = [nib.load(pfi).get_data() for pfi in list_pfi_warped]
            stack_warp = np.stack(list_stack_warp, axis=3)
            del list_stack_warp
            im_4d_warp = set_new_data(nib.load(list_pfi_warped[0]), stack_warp)
            pfi_4d_warp = connect_path_tail_head(self.pfo_out, '{0}_{1}.nii.gz'.format(warp_output_name, output_tag))
            nib.save(im_4d_warp, pfi_4d_warp)


        return pfi_target, pfi_result, pfi_4d_seg, pfi_4d_warp
