import os
from os.path import join as jph
import numpy as np

from examples.generate_headlike_phantoms import example_generate_multi_atlas_at_specified_folder
from LABelsToolkit.tools.defs import root_dir
from LABelsToolkit.main import LABelsToolkit


if __name__ == '__main__':

    controller = {'Get_initial_data'                   : False,
                  'Create_data_folders'                : False,
                  'Compute_groud_icv_and_m'            : True,
                  'Icv_estimation_transf'              : False,
                  'Estimate_m_and_compare_with_ground' : True,
                  'Icv_estimation'                     : True,
                  'Compare_output'                     : True}

    num_subjects = 7

    pfo_examples          = jph(root_dir, 'data_examples')
    pfo_dummy_multi_atlas = jph(pfo_examples, 'dummy_multi_atlas')

    pfo_icv_estimation     = jph(pfo_examples, 'icv_benchmarking')
    pfo_icv_brains         = jph(pfo_icv_estimation, 'brains')
    pfo_icv_segmentations  = jph(pfo_icv_estimation, 'segm')
    pfo_output             = jph(pfo_icv_estimation, 'output')

    if controller['Get_initial_data']:
        if not os.path.exists(pfo_dummy_multi_atlas):
            example_generate_multi_atlas_at_specified_folder()

    if controller['Create_data_folders']:
        # Prepare the folder with subjects to co-register: copy the main modality of each atlas of the generated synthetic
        # multi atlas created above.
        os.system('mkdir -p {}'.format(pfo_icv_brains))
        os.system('mkdir -p {}'.format(pfo_icv_segmentations))
        os.system('mkdir -p {}'.format(pfo_output))

        for j in range(1, num_subjects + 1):
            pfi_dummy_atlas = jph(pfo_dummy_multi_atlas, 'e00{}'.format(j))
            pfi_modGT  = jph(pfi_dummy_atlas, 'mod', 'e00{}_modGT.nii.gz'.format(j) )
            pfi_segmGT = jph(pfi_dummy_atlas, 'segm', 'e00{}_segmGT.nii.gz'.format(j))
            assert os.path.exists(pfi_modGT), pfi_modGT
            assert os.path.exists(pfi_segmGT), pfi_segmGT
            pfi_modGT_new  = jph(pfo_icv_brains, 'e00{}_modGT.nii.gz'.format(j) )
            pfi_segmGT_new = jph(pfo_icv_segmentations, 'e00{}_segmGT.nii.gz'.format(j))
            os.system('cp {} {}'.format(pfi_modGT, pfi_modGT_new))
            os.system('cp {} {}'.format(pfi_segmGT, pfi_segmGT_new))

    list_pfi_sj      = [jph(pfo_icv_brains, 'e00{}_modGT.nii.gz'.format(j + 1)) for j in range(num_subjects)]
    list_pfi_sj_segm = [jph(pfo_icv_segmentations, 'e00{}_segmGT.nii.gz'.format(j + 1)) for j in range(num_subjects)]

    if controller['Compute_groud_icv_and_m']:
        # Compute the ground truth icv (up to discretisation
        os.system('mkdir -p {}'.format(pfo_output))
        v_ground = np.zeros(num_subjects, dtype=np.float)
        lab = LABelsToolkit()
        for j in range(1, num_subjects + 1):
            pfi_segmGT = jph(pfo_icv_segmentations, 'e00{}_segmGT.nii.gz'.format(j))
            df_vols = lab.measure.volume(pfi_segmGT, labels='tot')
            v_ground[j - 1] = np.sum(df_vols['Volume'])
            print('Subject {}, volume: {}'.format(j, v_ground[j - 1]))
        np.savetxt(jph(pfo_output, 'v_ground.txt'), v_ground, fmt='%10.1f')
        print('Average volume {}'.format(np.mean(v_ground)))
        np.savetxt(jph(pfo_output, 'm.txt'), [np.mean(v_ground)], fmt='%10.1f')

    if controller['Icv_estimation_transf']:
        # compute transformations and S matrix
        lab = LABelsToolkit()
        icv_estimator = lab.icv(list_pfi_sj, pfo_output)
        # icv_estimator.generate_transformations()
        icv_estimator.compute_S()
        np.savetxt(jph(pfo_output, 'S.txt'), icv_estimator.S, fmt='%5.5f')
        del lab, icv_estimator

    if controller['Estimate_m_and_compare_with_ground']:
        lab = LABelsToolkit()
        icv_estimator = lab.icv(list_pfi_sj, pfo_output)
        icv_estimator.compute_m_from_list_masks(list_pfi_sj_segm, correction_volume_estimate=0)
        print('Average volume estimated with ICV {}'.format(icv_estimator.m))

    if controller['Icv_estimation']:
        # Estimate the icv with our method.
        m = np.loadtxt(jph(pfo_output, 'm.txt'))
        S = np.loadtxt(jph(pfo_output, 'S.txt'))
        lab = LABelsToolkit()
        icv_estimator = lab.icv(list_pfi_sj, pfo_output, m=m, S=S)
        v_est = icv_estimator.estimate_icv()
        np.savetxt(jph(pfo_output, 'v_est.txt'), v_est, fmt='%8.10f')

    if controller['Compare_output']:
        # Compare ground truth and estimated ground truth.
        v_ground = np.loadtxt(jph(pfo_output, 'v_ground.txt'))
        v_est   = np.loadtxt(jph(pfo_output, 'v_est.txt'))
        print(v_ground)
        print(v_est)
        print
        print(np.abs(v_ground  - v_est)/v_ground)
