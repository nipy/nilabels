import os
from os.path import join as jph

import a_generate_phantoms_for_examples as gen
from nilabels.agents.agents_controller import AgentsController as NiL
from nilabels.definitions import root_dir
from nilabels.tools.aux_methods.label_descriptor_manager import generate_dummy_label_descriptor

# ---- GENERATE DATA ----


if not os.path.exists(jph(root_dir, 'data_examples', 'ellipsoids_seg.nii.gz')):

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

pfi_input_segm = jph(root_dir, 'data_examples', 'ellipsoids_seg.nii.gz')

# ---- CREATE LABELS DESCRIPTOR FOR PHANTOM ellipsoids with 0 to 6 labels ----

pfi_labels_descriptor = jph(root_dir, 'data_examples', 'labels_descriptor_ellipsoids.txt')
generate_dummy_label_descriptor(pfi_labels_descriptor, list_labels=[0, 1, 4, 5, 6, 7, 8])  # add extra labels to test

# ---- PERFORM the check ----

la = NiL()
in_descriptor_not_delineated, delineated_not_in_descriptor = la.check.missing_labels(pfi_input_segm, pfi_labels_descriptor, pfi_where_to_save_the_log_file=None)

# Print expected to be seen in the terminal: set([8, 7]) set([2, 3])
