
import os
from os.path import join as jph

from LABelsToolkit.main import LABelsToolkit as LT


# ---- PATH MANAGER ----

# input
pfi_input_anatomy      = '/Users/sebastiano/Dropbox/RabbitEOP-MRI/study/A_MultiAtlas_W8/12503/mod/12503_T1.nii.gz'
pfi_input_segmentation = '/Users/sebastiano/Dropbox/RabbitEOP-MRI/study/A_MultiAtlas_W8/12503/segm/automatic/12503_pre_sym_cleaned.nii.gz'
pfo_output_folder      = '/Users/sebastiano/Dropbox/RabbitEOP-MRI/study/A_MultiAtlas_W8/12503/segm/automatic/z_tmp'

assert os.path.exists(pfi_input_anatomy), pfi_input_anatomy
assert os.path.exists(pfi_input_segmentation), pfi_input_segmentation
assert os.path.exists(pfo_output_folder), pfo_output_folder

# output
pfi_output_segmentation = '/Users/sebastiano/Dropbox/RabbitEOP-MRI/study/A_MultiAtlas_W8/12503/segm/automatic/12503_SYM.nii.gz'

# ---- LABELS LIST ----


labels_central = [0, 77, 78,  121, 127, 151, 153, 161, 201, 215, 218, 233, 237, 253]

labels_left       = [5, 7, 9, 11,  13, 15, 17, 19, 25, 27, 31, 43, 45, 53, 55, 69, 71, 75, 83, 109, 129, 133, 135,
                     139, 141, 179, 203, 211, 219, 223, 225, 227, 229, 239, 241, 243, 247, 251]

labels_right = [a + 1 for a in labels_left]

labels_sym_left = labels_left + labels_central
labels_sym_right = labels_right + labels_central




# --- EXECUTE ----


lt = LT()
lt.symmetrize.symmetrise_with_registration(pfi_input_anatomy,
                                           pfi_input_segmentation,
                                           labels_sym_left,
                                           pfi_output_segmentation,
                                           results_folder_path=pfo_output_folder,
                                           list_labels_transformed=labels_sym_right,
                                           coord='z',
                                           reuse_registration=False)
