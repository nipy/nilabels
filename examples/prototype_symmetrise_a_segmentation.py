import os
from os.path import join as jph

import a_generate_phantoms_for_examples as gen
from nilabels.agents.agents_controller import AgentsController as NiL
from nilabels.definitions import root_dir

# ---- GENERATE DATA ----


if not os.path.exists(jph(root_dir, 'data_examples', 'ellipsoids.nii.gz')):

    creation_list = {'Examples folder'    : False,
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


# input
pfi_input_anatomy      = jph(root_dir, 'data_examples', 'ellipsoids.nii.gz')
pfi_input_segmentation = jph(root_dir, 'data_examples', 'ellipsoids_seg_half.nii.gz')
pfo_output_folder      = jph(root_dir, 'data_output')

assert os.path.exists(pfi_input_anatomy), pfi_input_anatomy
assert os.path.exists(pfi_input_segmentation), pfi_input_segmentation
assert os.path.exists(pfo_output_folder), pfo_output_folder

# output
pfi_output_segmentation = jph(root_dir, 'data_examples', 'ellipsoids_seg_symmetrised.nii.gz')


# ---- LABELS LIST ----


labels_central = []
labels_left    = [1, 2, 3, 4, 5, 6]
labels_right   = [a + 10 for a in labels_left]

labels_sym_left  = labels_left + labels_central
labels_sym_right = labels_right + labels_central


# --- EXECUTE ----


lt = NiL()
lt.symmetrize.symmetrise_with_registration(pfi_input_anatomy,
                                           pfi_input_segmentation,
                                           labels_sym_left,
                                           pfi_output_segmentation,
                                           results_folder_path=pfo_output_folder,
                                           list_labels_transformed=labels_sym_right,
                                           coord='z',
                                           reuse_registration=False)

# --- SEE RESULTS ----

opener1 = 'itksnap -g {} -s {}'.format(pfi_input_anatomy, pfi_input_segmentation)
opener2 = 'itksnap -g {} -s {}'.format(pfi_input_anatomy, pfi_output_segmentation)

os.system(opener1)
os.system(opener2)
