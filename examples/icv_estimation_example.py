import os
from os.path import join as jph
import nibabel as nib
import numpy as np

from generate_headlike_phantoms import example_generate_multi_atlas_at_specified_folder
from LABelsToolkit.tools.defs import root_dir
from LABelsToolkit.main import LABelsToolkit

if __name__ == '__main__':

    controller = {'Get_initial_data'      : True,
                  'Create_data_folders'   : True,
                  'Compute_groud_icv'     : True,
                  'Icv_estimation_transf' : True,
                  'Icv_estimation'        : True,
                  'Compare_output'        : True}

    num_subjects = 7

    pfo_examples          = jph(root_dir, 'data_examples')
    pfo_dummy_multi_atlas = jph(pfo_examples, 'dummy_multi_atlas')

    pfo_icv_estimation     = jph(pfo_examples, 'icv_benchmarking')
    pfo_icv_brains         = jph(pfo_icv_estimation, 'brains')
    pfo_icv_segmentations  = jph(pfo_icv_estimation, 'segm')
    pfo_intermediate       = jph(pfo_icv_estimation, 'intermediate')

    if controller['Get_initial_data']:
        if not os.path.exists(pfo_dummy_multi_atlas):
            example_generate_multi_atlas_at_specified_folder()

    if controller['Create_data_folders']:
        # Prepare the folder with subjects to co-register: copy the main modality of each atlas of the generated synthetic
        # multi atlas created above.
        os.system('mkdir -p {}'.format(pfo_icv_brains))
        os.system('mkdir -p {}'.format(pfo_icv_segmentations))
        os.system('mkdir -p {}'.format(pfo_intermediate))

        for j in range(num_subjects):
            pfi_dummy_atlas = jph(pfo_dummy_multi_atlas, 'e00{}'.format(j))
            pfi_modGT  = jph(pfi_dummy_atlas, 'mod', 'e00{}_modGT.nii.gz'.format(j) )
            pfi_segmGT = jph(pfi_dummy_atlas, 'segm', 'e00{}_segmGT.nii.gz'.format(j))
            assert os.path.exists(pfi_modGT), pfi_modGT
            assert os.path.exists(pfi_segmGT), pfi_segmGT
            pfi_modGT_new  = jph(pfo_icv_brains, 'e00{}_modGT.nii.gz'.format(j) )
            pfi_segmGT_new = jph(pfo_icv_segmentations, 'e00{}_segmGT.nii.gz'.format(j))
            os.system('cp {} {}'.format(pfi_modGT, pfi_modGT_new))
            os.system('cp {} {}'.format(pfi_segmGT, pfi_segmGT_new))

    if controller['Compute_groud_icv']:
        # Compute the ground truth icv (up to discretisation)
        v_tilda = np.zeros(num_subjects, dtype=np.float)
        lab = LABelsToolkit()

        for j in range(num_subjects):
            pfi_segmGT = jph(pfo_icv_segmentations, 'e00{}_segmGT.nii.gz'.format(j))
            df_vols = lab.measure.volume(pfi_segmGT, labels=[1, ])

            v_tilda[j] = ''



    if controller['Icv_estimation_transf']:
        # Estimate the icv with our method.
        v = ''

    if controller['Icv_estimation']:
        # Compare ground truth and estimated ground truth.
        pass

    if controller['Compare_output']:
        # Compare ground truth and estimated ground truth.
        pass