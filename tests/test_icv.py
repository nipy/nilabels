import os
from os.path import join as jph

import numpy as np
from nose.tools import assert_equals

from nilabels.agents.agents_controller import AgentsController
from nilabels.tools.phantoms_generator import local_data_generator as ldg


num_subjects = 7

pfo_icv_estimation     = jph(ldg.pfo_examples, 'icv_benchmarking')
pfo_icv_brains         = jph(pfo_icv_estimation, 'brains')
pfo_icv_segmentations  = jph(pfo_icv_estimation, 'segm')
pfo_output             = jph(pfo_icv_estimation, 'output')

list_pfi_sj      = [jph(pfo_icv_brains, 'e00{}_modGT.nii.gz'.format(j + 1)) for j in range(num_subjects)]
list_pfi_sj_segm = [jph(pfo_icv_segmentations, 'e00{}_segmGT.nii.gz'.format(j + 1)) for j in range(num_subjects)]


def _create_data_set_for_tests():
    if not os.path.exists(ldg.pfo_multi_atlas):
        print('Generating dummy dataset for testing part 1. May take 10 min.')
        ldg.generate_multi_atlas_at_specified_folder()

    if not os.path.exists(pfo_icv_brains):
        print('generating dummy dataset for testing part 2. May take 5 min.')
        os.system('mkdir -p {}'.format(pfo_icv_brains))
        os.system('mkdir -p {}'.format(pfo_output))

        for j in range(1, num_subjects + 1):
            pfi_dummy_atlas = jph(ldg.pfo_multi_atlas, 'e00{}'.format(j))
            pfi_modGT  = jph(pfi_dummy_atlas, 'mod', 'e00{}_modGT.nii.gz'.format(j))
            pfi_segmGT = jph(pfi_dummy_atlas, 'segm', 'e00{}_segmGT.nii.gz'.format(j))
            assert os.path.exists(pfi_modGT), pfi_modGT
            assert os.path.exists(pfi_segmGT), pfi_segmGT
            pfi_modGT_new  = jph(pfo_icv_brains, 'e00{}_modGT.nii.gz'.format(j))
            pfi_segmGT_new = jph(pfo_icv_segmentations, 'e00{}_segmGT.nii.gz'.format(j))
            os.system('cp {} {}'.format(pfi_modGT, pfi_modGT_new))
            os.system('cp {} {}'.format(pfi_segmGT, pfi_segmGT_new))

    if not os.path.exists(pfo_icv_segmentations):
        print('Co-registering dummy dataset for testing part 3. May take 1 min.')
        os.system('mkdir -p {}'.format(pfo_icv_segmentations))
        lab = AgentsController()
        icv_estimator = lab.icv(list_pfi_sj, pfo_output)
        icv_estimator.generate_transformations()


def test_compute_ground_truth_m_and_estimated_m():
    # Ground:
    v_ground = np.zeros(num_subjects, dtype=np.float)
    lab = AgentsController()
    for j in range(1, num_subjects + 1):
        pfi_segmGT = jph(pfo_icv_segmentations, 'e00{}_segmGT.nii.gz'.format(j))
        df_vols    = lab.measure.volume(pfi_segmGT, labels='tot')
        v_ground[j - 1] = np.sum(df_vols['Volume'])
        print('Subject {}, volume: {}'.format(j, v_ground[j - 1]))
    m_ground = np.mean(v_ground)
    print('Average volume {}'.format(m_ground))
    # Estimated:
    lab = AgentsController()
    icv_estimator = lab.icv(list_pfi_sj, pfo_output)
    icv_estimator.compute_m_from_list_masks(list_pfi_sj_segm, correction_volume_estimate=0)
    print('Average volume estimated with ICV {}'.format(icv_estimator.m))
    # Comparison
    assert_equals(icv_estimator.m, m_ground)


def test_compute_ground_truth_v_and_estimated_v():
    # Ground:
    v_ground = np.zeros(num_subjects, dtype=np.float)
    lab = AgentsController()
    for j in range(1, num_subjects + 1):
        pfi_segmGT = jph(pfo_icv_segmentations, 'e00{}_segmGT.nii.gz'.format(j))
        df_vols = lab.measure.volume(pfi_segmGT, labels='tot')
        v_ground[j - 1] = np.sum(df_vols['Volume'])
        print('Subject {}, volume: {}'.format(j, v_ground[j - 1]))
    # Estimated:
    lab = AgentsController()
    icv_estimator = lab.icv(list_pfi_sj, pfo_output)
    icv_estimator.compute_S()
    icv_estimator.compute_m_from_list_masks(list_pfi_sj_segm, correction_volume_estimate=0)
    v_est = icv_estimator.estimate_icv()
    # Comparison
    av = (np.abs(v_ground + v_est) / float(2))
    err = np.abs(v_ground - v_est)
    for e in list(err / av):
        assert e < 1.0  # assert the error in % is below 1%.
