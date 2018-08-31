import os
from os.path import join as jph

from nilabels.agents.agents_controller import AgentsController as NiL
from nilabels.definitions import root_dir

if __name__ == '__main__':

    run_steps = {'Generate results folders': False,
                 'Register and propagate': False,
                 'Fuse seg_LabFusion': True}

    pfo_input_dataset = jph(root_dir, 'data_examples', 'ellipsoids_family')
    pfo_results_propagation = jph(root_dir, 'data_output', 'results_propagation')
    pfo_results_label_fusion = jph(root_dir, 'data_output', 'results_label_fusion')

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

    if run_steps['Fuse seg_LabFusion']:

        # instantiate a label manager:
        lm = NiL(jph(root_dir, 'data_examples', 'ellipsoids_family'), jph(root_dir, 'data_output'))

        # With majority voting
        options_seg = '-MV'

        list_pfi_segmentations = [jph(pfo_results_propagation, 'seg_nrig_ellipsoid' + str(k) + '_on_target.nii.gz') for k in range(1, 11)]
        list_pfi_warped = [jph(pfo_results_propagation, 'seg_nrig_ellipsoid' + str(k) + '_on_target.nii.gz') for k in range(1, 11)]

        lm.fuse.create_stack_for_labels_fusion('target.nii.gz', jph('results_label_fusion', 'output' + options_seg + '.nii.gz'),
                                               list_pfi_segmentations, options=options_seg)

        # If something more sophisticated needs to be done, it returns the paths to the stacks of images:
        options_seg = '_test2'
        list_paths = lm.fuse.create_stack_for_labels_fusion('target.nii.gz', jph('results_label_fusion', 'output' + options_seg + '.nii.gz'),
                                                            list_pfi_segmentations, list_pfi_warped, options=options_seg, prepare_data_only=True)

        print(list_paths)
