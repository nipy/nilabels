import os
from os.path import join as jph

import a_generate_phantoms_for_examples as gen
import nibabel as nib
import numpy as np

import nilabels as nil
from nilabels.definitions import root_dir

# ---- GENERATE DATA ----


if not os.path.exists(jph(root_dir, "data_examples", "ellipsoids.nii.gz")):
    creation_list = {
        "Examples folder": True,
        "Punt e mes": False,
        "C": False,
        "Planetaruim": False,
        "Buckle ellipsoids": True,
        "Ellipsoids family": False,
        "Cubes in the sky": False,
        "Sandwich": False,
        "Four-folds": False,
    }

    gen.generate_figures(creation_list)

# ---- PATH MANAGER ----

# input
pfi_input_anatomy = jph(root_dir, "data_examples", "ellipsoids.nii.gz")
pfi_input_segmentation_noisy = jph(root_dir, "data_examples", "ellipsoids_seg_noisy.nii.gz")
pfo_output_folder = jph(root_dir, "data_output")

assert os.path.exists(pfi_input_anatomy), pfi_input_anatomy
assert os.path.exists(pfi_input_segmentation_noisy), pfi_input_segmentation_noisy
assert os.path.exists(pfo_output_folder), pfo_output_folder

# Output
log_file_before_cleaning = jph(pfo_output_folder, "log_before_cleaning.txt")
pfi_output_cleaned_segmentation = jph(pfo_output_folder, "ellipsoids_segm_cleaned.nii.gz")
log_file_after_cleaning = jph(pfo_output_folder, "log_after_cleaning.txt")
pfi_differece_cleaned_non_cleaned = jph(pfo_output_folder, "difference_half_cleaned_uncleaned.nii.gz")


# ---- PROCESS ----

nl = nil.App()

# get the report before cleaning
nl.check.number_connected_components_per_label(
    pfi_input_segmentation_noisy, where_to_save_the_log_file=log_file_before_cleaning,
)

print("Wanted final number of components per label:")
im_input_segmentation_noisy = nib.load(pfi_input_segmentation_noisy)
correspondences_labels_components = [[k, 1] for k in range(np.max(im_input_segmentation_noisy.get_fdata()) + 1)]
print(correspondences_labels_components)

# get the cleaned segmentation
nl.manipulate_labels.clean_segmentation(
    pfi_input_segmentation_noisy,
    pfi_output_cleaned_segmentation,
    labels_to_clean=correspondences_labels_components,
    force_overwriting=True,
)

# get the report of the connected components afterwards
nl.check.number_connected_components_per_label(
    pfi_output_cleaned_segmentation, where_to_save_the_log_file=log_file_after_cleaning,
)

# ---- GET DIFFERENCE ----

cmd = f"seg_maths {pfi_input_segmentation_noisy} -sub {pfi_output_cleaned_segmentation} {pfi_differece_cleaned_non_cleaned}"
os.system(cmd)
cmd = f"seg_maths {pfi_differece_cleaned_non_cleaned} -bin {pfi_differece_cleaned_non_cleaned}"
os.system(cmd)

# ---- VISUALISE OUTPUT ----

opener1 = f"itksnap -g {pfi_input_anatomy} -s {pfi_input_segmentation_noisy}"
opener2 = f"itksnap -g {pfi_input_anatomy} -s {pfi_output_cleaned_segmentation}"
opener3 = f"itksnap -g {pfi_input_anatomy} -s {pfi_differece_cleaned_non_cleaned}"

os.system(opener1)
os.system(opener2)
os.system(opener3)
