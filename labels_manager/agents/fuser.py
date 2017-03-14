import os
import numpy as np
import nibabel as nib

from labels_manager.tools.aux_methods.sanity_checks import get_pfi_in_pfi_out, connect_tail_head_path
from labels_manager.tools.aux_methods.utils import set_new_data


class LabelsManagerFuse(object):
    """
    Facade of the methods in tools, for work with paths to images rather than
    with data. Methods under LabelsManagerFuse access label fusion methods.
    """
    def __init__(self, input_data_folder=None, output_data_folder=None):
        self.pfo_in = input_data_folder
        self.pfo_out = output_data_folder

    def seg_LabFusion(self, pfi_target, pfi_result, list_pfi_segmentations, list_pfi_warped=None, options='MV'):

        # save 4d segmentations
        list_pfi_segmentations = [connect_tail_head_path(self.pfo_in, j) for j in list_pfi_segmentations]
        list_stack_seg = []
        for pfi_seg_j in list_pfi_segmentations:
            im_seg = nib.load(pfi_seg_j)
            list_stack_seg.append(im_seg.get_data())

        stack_seg = np.stack(list_stack_seg, axis=3)
        del list_stack_seg

        im_4d_seg = set_new_data(nib.load(list_pfi_segmentations[0]), stack_seg)
        nib.save(im_4d_seg, connect_tail_head_path(self.pfo_out, 'z_4d_seg.nii.gz'))

        # save 4d warped if available
        if list_pfi_warped is not None:

            list_pfi_warped = [connect_tail_head_path(self.pfo_in, j) for j in list_pfi_warped]
            list_stack_warp = []
            for pfi_warp_j in list_pfi_warped:

                im_warp = nib.load(pfi_warp_j)
                list_stack_warp.append(np.nan_to_num(im_warp.get_data().astype(np.float64)))
            stack_warp = np.stack(list_stack_warp, axis=3)
            del list_stack_warp

            im_4d_warp = set_new_data(nib.load(list_pfi_warped[0]), stack_warp)
            nib.save(im_4d_warp, connect_tail_head_path(self.pfo_out, 'z_4d_warp.nii.gz'))

        pfi_result = connect_tail_head_path(self.pfo_out, pfi_result)
        cmd = 'seg_LabFuse -in {0} -out {1} '.format()

        # create stack segmentation and warped



        # Create the stack image of seg


        pass
