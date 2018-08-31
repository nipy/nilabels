import os
from os.path import join as jph

import a_generate_phantoms_for_examples as gen
from nilabels.agents.agents_controller import AgentsController as NiL
from nilabels.definitions import root_dir

# ---- GENERATE DATA ----


if not os.path.exists(jph(root_dir, 'data_examples', 'ellipsoids.nii.gz')):

    creation_list = {'Examples folder'    : True,
                      'Punt e mes'        : False,
                      'C'                 : False,
                      'Planetaruim'       : False,
                      'Buckle ellipsoids' : True,
                      'Ellipsoids family' : False,
                      'Cubes in the sky'  : False,
                      'Sandwich'          : False,
                      'Four-folds'        : False}

    gen.generate_figures(creation_list)


# ---- PATH MANAGER ----

# input:
pfi_input_anatomy = jph(root_dir, 'data_examples', 'ellipsoids.nii.gz')
pfo_output_folder = jph(root_dir, 'data_output')

assert os.path.exists(pfi_input_anatomy), pfi_input_anatomy
assert os.path.exists(pfo_output_folder)

# output:
pfi_intensities_segmentation = jph(pfo_output_folder, 'ellipsoids_segm_int.nii.gz')
pfi_otsu_segmentation        = jph(pfo_output_folder, 'ellipsoids_segm_otsu.nii.gz')
pfi_mog_segmentation_crisp   = jph(pfo_output_folder, 'ellipsoids_segm_mog_crisp.nii.gz')
pfi_mog_segmentation_prob    = jph(pfo_output_folder, 'ellipsoids_segm_mog_prob.nii.gz')

print('---- PROCESS 1: intensities segmentation ----')

la = NiL()
la.segment.simple_intensities_thresholding(pfi_input_anatomy, pfi_intensities_segmentation, number_of_levels=5)

print('---- PROCESS 2: Otsu ----')

la = NiL()
la.segment.otsu_thresholding(pfi_input_anatomy, pfi_otsu_segmentation, side='above', return_as_mask=False)

print('---- PROCESS 2: MoG ----')

la = NiL()
la.segment.mixture_of_gaussians(pfi_input_anatomy, pfi_mog_segmentation_crisp, pfi_mog_segmentation_prob,
                                K=5, see_histogram=True)

print('---- VIEW ----')

opener1 = 'itksnap -g {} -s {}'.format(pfi_input_anatomy, pfi_intensities_segmentation)
opener2 = 'itksnap -g {} -s {}'.format(pfi_input_anatomy, pfi_otsu_segmentation)
opener3 = 'itksnap -g {} -s {}'.format(pfi_input_anatomy, pfi_mog_segmentation_crisp)
opener4 = 'itksnap -g {} -o {}'.format(pfi_input_anatomy, pfi_mog_segmentation_prob)

os.system(opener1)
os.system(opener2)
os.system(opener3)
os.system(opener4)
