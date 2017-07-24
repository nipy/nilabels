import os
import numpy as np
import nibabel as nib

from labels_manager.tools.aux_methods.sanity_checks import connect_tail_head_path
from labels_manager.tools.aux_methods.utils import set_new_data


class LabelsManagerFuse(object):
    """
    Facade of the methods in tools, for work with paths to images rather than
    with data. Methods under LabelsManagerFuse access label fusion methods.
    """
    def __init__(self, input_data_folder=None, output_data_folder=None):
        self.pfo_in = input_data_folder
        self.pfo_out = output_data_folder

    def seg_LabFusion(self, pfi_target, pfi_result, list_pfi_segmentations, list_pfi_warped=None, options='-MV', prepare_data_only=False):
        """
        Takes into account the
        :param pfi_target: path to file to the target of the segmentation
        :param pfi_result: path to file where to store the result.
        :param list_pfi_segmentations: list of the segmentations to fuse
        :param list_pfi_warped: list of the warped images to fuse
        :param options: simple option of NiftySeg, can be -MV -SBA or -STAPLE
        :param prepare_data_only: Return the paths to the stack images, and does not call seg_LabFusion.
             This can be set as True when some more sophistication on the methods is required.
        :return: if prepare_data_only is True it returns the path to files prepared to be used externally in nifty seg,
                in the following order
                    [pfi_target, pfi_result, pfi_4d_seg, pfi_4d_warp]

        """
        pfi_target = connect_tail_head_path(self.pfo_in, pfi_target)
        pfi_result = connect_tail_head_path(self.pfo_out, pfi_result)
        # save 4d segmentations in stack_seg
        list_pfi_segmentations = [connect_tail_head_path(self.pfo_in, j) for j in list_pfi_segmentations]
        #
        list_stack_seg = [nib.load(pfi).get_data() for pfi in list_pfi_segmentations]
        stack_seg = np.stack(list_stack_seg, axis=3)
        del list_stack_seg
        im_4d_seg = set_new_data(nib.load(list_pfi_segmentations[0]), stack_seg)
        pfi_4d_seg = connect_tail_head_path(self.pfo_out, 'res_4d_seg.nii.gz')
        nib.save(im_4d_seg, pfi_4d_seg)

        # save 4d warped if available
        if list_pfi_warped is None:
            pfi_4d_warp = None
        else:
            list_pfi_warped = [connect_tail_head_path(self.pfo_in, j) for j in list_pfi_warped]
            #
            list_stack_warp = [nib.load(pfi).get_data() for pfi in list_pfi_warped]
            stack_warp = np.stack(list_stack_warp, axis=3)
            del list_stack_warp
            im_4d_warp = set_new_data(nib.load(list_pfi_warped[0]), stack_warp)
            pfi_4d_warp = connect_tail_head_path(self.pfo_out, 'res_4d_warp.nii.gz')
            nib.save(im_4d_warp, pfi_4d_warp)

        if not prepare_data_only:
            cmd = 'seg_LabFusion -in {0} -out {1} {2}'.format(pfi_4d_seg, pfi_result, options)
            os.system(cmd)

        else:
            return pfi_target, pfi_result, pfi_4d_seg, pfi_4d_warp
