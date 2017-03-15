# Label fusion with four methods considered.
# Generate images examples first.
import os
from os.path import join as jph

import numpy as np
import nibabel as nib

from definitions import root_dir
from labels_manager.tools.aux_methods.utils import set_new_data
from labels_manager.tools.fusing_labels.lncc_based_method import simple_majority_voting_lncc
from labels_manager.tools.fusing_labels.weighted_sum_method import weighted_sum_label_fusion

if __name__ == '__main__':

    run_steps = {'Generate results folders' : False,
                 'Register and propagate'   : False,
                 'Create stack of images'   : False,
                 'Fuse LNCC'                : False,
                 'Fuse weighted'            : True,
                 'Fuse seg_LabFusion'       : False,
                 'Fuse staple cmtk'         : False}

    pfo_input_dataset = jph(root_dir, 'images_examples', 'ellipsoids_family')
    pfo_results_propagation = jph(root_dir, 'images_output', 'results_propagation')
    pfo_results_label_fusion = jph(root_dir, 'images_output', 'results_label_fusion')

    if run_steps['Generate results folders']:
        cmd0 = 'mkdir -p {0}'.format(pfo_results_propagation)
        cmd1 = 'mkdir -p {0}'.format(pfo_results_label_fusion)
        os.system(cmd0)
        os.system(cmd1)

    if run_steps['Register and propagate']:
        fin_target = 'target.nii.gz'
        fin_target_seg = 'target_seg.nii.gz'

        for k in range(1, 11):
            # affine registration
            cmd_aff = 'reg_aladin -ref {0} -flo {1} -aff {2} -res {3}'.format(jph(pfo_input_dataset, 'target.nii.gz'),
                                                                              jph(pfo_input_dataset, 'ellipsoid' + str(k) + '.nii.gz'),
                                                                              jph(pfo_results_propagation, 'aff_ellipsoid' + str(k) + '_on_target.txt'),
                                                                              jph(pfo_results_propagation, 'aff_ellipsoid' + str(k) + '_on_target.nii.gz'))
            os.system(cmd_aff)
            # non-rigid registration
            cmd_nrig = 'reg_f3d -ref {0} -flo {1} -cpp {2} -res {3}'.format(jph(pfo_input_dataset, 'target.nii.gz'),
                                                                           jph(pfo_results_propagation, 'aff_ellipsoid' + str(k) + '_on_target.nii.gz'),
                                                                           jph(pfo_results_propagation, 'cpp_ellipsoid' + str(k) + '_on_target.nii.gz'),
                                                                           jph(pfo_results_propagation, 'nrig_ellipsoid' + str(k) + '_on_target.nii.gz'))
            os.system(cmd_nrig)
            # affine propagation
            cmd_resample_aff = 'reg_resample -ref {0} -flo {1} -trans {2} -res {3} -inter 0'.format(
                jph(pfo_input_dataset, 'target.nii.gz'),
                jph(pfo_input_dataset, 'ellipsoid' + str(k) + '_seg.nii.gz'),
                jph(pfo_results_propagation, 'aff_ellipsoid' + str(k) + '_on_target.txt'),
                jph(pfo_results_propagation, 'seg_aff_ellipsoid' + str(k) + '_on_target.nii.gz')
                )
            os.system(cmd_resample_aff)
            # non-rigid propagation
            cmd_resample_nrig = 'reg_resample -ref {0} -flo {1} -trans {2} -res {3} -inter 0'.format(
                jph(pfo_input_dataset, 'target.nii.gz'),
                jph(pfo_results_propagation, 'seg_aff_ellipsoid' + str(k) + '_on_target.nii.gz'),
                jph(pfo_results_propagation, 'cpp_ellipsoid' + str(k) + '_on_target.nii.gz'),
                jph(pfo_results_propagation, 'seg_nrig_ellipsoid' + str(k) + '_on_target.nii.gz')
                )
            os.system(cmd_resample_nrig)

    if run_steps['Create stack of images']:

        # Segmentations:
        list_pfi_segmentations = [jph(pfo_results_propagation, 'seg_nrig_ellipsoid' + str(k) + '_on_target.nii.gz')
                                  for k in range(1, 11)]
        list_stack_seg = []
        for pfi_seg_j in list_pfi_segmentations:
            im_seg = nib.load(pfi_seg_j)
            list_stack_seg.append(im_seg.get_data())

        stack_seg = np.stack(list_stack_seg, axis=3)
        stack_seg[np.isnan(stack_seg)] = 0
        del list_stack_seg

        im_4d_seg = set_new_data(nib.load(list_pfi_segmentations[0]), stack_seg)
        nib.save(im_4d_seg, jph(pfo_results_label_fusion, '4d_seg.nii.gz'))

        # Warped:
        list_pfi_warped = [jph(pfo_results_propagation, 'nrig_ellipsoid' + str(k) + '_on_target.nii.gz')
                                  for k in range(1, 11)]
        list_stack_warp = []
        for pfi_warp_j in list_pfi_warped:
            im_warp = nib.load(pfi_warp_j)
            list_stack_warp.append(im_warp.get_data())

        stack_warp = np.stack(list_stack_warp, axis=3)
        stack_warp[np.isnan(stack_warp)] = 0
        del list_stack_warp

        im_4d_warp = set_new_data(nib.load(list_pfi_warped[0]), stack_warp)
        nib.save(im_4d_warp, jph(pfo_results_label_fusion, '4d_warp.nii.gz'))

        # clean extra data:
        del im_4d_seg, im_4d_warp

    if run_steps['Fuse LNCC']:

        stack_segmentations = nib.load(jph(pfo_results_label_fusion, '4d_seg.nii.gz'))
        stack_warped = nib.load(jph(pfo_results_label_fusion, '4d_warp.nii.gz'))
        target_image = nib.load(jph(pfo_input_dataset, 'target.nii.gz'))

        lab_fusion_lncc_data = simple_majority_voting_lncc(stack_segmentations.get_data(), stack_warped.get_data(), target_image.get_data())

        lab_fusion_lncc_im = set_new_data(target_image, lab_fusion_lncc_data)
        nib.save(lab_fusion_lncc_im, jph(pfo_results_label_fusion, 'fusion_lncc_test1.nii.gz'))

    if run_steps['Fuse weighted']:

        stack_segmentations = nib.load(jph(pfo_results_label_fusion, '4d_seg.nii.gz'))
        stack_warped = nib.load(jph(pfo_results_label_fusion, '4d_warp.nii.gz'))
        target_image = nib.load(jph(pfo_input_dataset, 'target.nii.gz'))

        lab_fusion_weighted_data = weighted_sum_label_fusion(stack_segmentations.get_data(), stack_warped.get_data(), target_image.get_data())

        lab_fusion_weighted_im = set_new_data(target_image, lab_fusion_weighted_data)
        nib.save(lab_fusion_weighted_im, jph(pfo_results_label_fusion, 'fusion_weighted_test1.nii.gz'))

    if run_steps['Fuse seg_LabFusion']:

        # simple majority voting
        cmd_mv = 'seg_LabFusion -in {0} -MV -out {1}  '.format(jph(pfo_results_label_fusion, '4d_seg.nii.gz'),
                                                        jph(pfo_results_label_fusion, 'fusion_seg_MV_test1.nii.gz'))
        print cmd_mv
        os.system(cmd_mv)

        # simple majority voting
        cmd_mv = 'seg_LabFusion -in {0} -STAPLE -out {1}  '.format(jph(pfo_results_label_fusion, '4d_seg.nii.gz'),
                                                        jph(pfo_results_label_fusion, 'fusion_seg_STAPLE_test1.nii.gz'))
        print cmd_mv
        os.system(cmd_mv)

        #

    if run_steps['Fuse staple cmtk']:

        path_to_antsJoinFusion = '~/sw_libraries/ANTS/ants-build/bin/antsJointFusion'
        cmd = '{0} -t {1} -g {2} -l {3} -o {4}'.format(path_to_antsJoinFusion,
                                                       jph(pfo_input_dataset, 'target.nii.gz'),
                                                       jph(pfo_results_label_fusion, '4d_warp.nii.gz'),
                                                       jph(pfo_results_label_fusion, '4d_seg.nii.gz'),
                                                       jph(pfo_results_label_fusion, 'fusion_seg_ANTS_test1.nii.gz'))
        os.system(cmd)

        #
