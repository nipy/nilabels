import os
from os.path import join as jph

import numpy as np
import nibabel as nib

from labels_manager.main import LabelsManager
from definitions import root_dir
from labels_manager.tools.aux_methods.utils import set_new_data
from labels_manager.tools.manipulations.relabeller import relabeller, permute_labels


if __name__ == '__main__':

    # Run generate_images_examples.py first

    # Create output folder:
    cmd = 'mkdir -p {}'.format(jph(root_dir, 'images_output'))
    os.system(cmd)

    # Instantiate a manager from the class LabelsManager
    lm = LabelsManager(jph(root_dir, 'images_examples'), jph(root_dir, 'images_output'))
    print(lm.pfo_in)
    print(lm.pfo_out)

    examples = {'Example relabel': True,
                'Example permute': True}

    """
    Example 1:
    Relabel the segmentation punt_seg.nii.gz increasing the label values by 1.
    """
    if examples['Example relabel']:

        # data:
        pfi_punt_seg_original = 'punt_seg.nii.gz'
        pfi_punt_seg_new      = 'punt_seg.nii.gz'

        list_old_labels = [1, 2, 3, 4, 5, 6]
        list_new_labels = [2, 3, 4, 5, 6, 7]

        # Using the manager:
        lm.manipulate.relabel(pfi_punt_seg_original, pfi_punt_seg_new,
                              list_old_labels, list_new_labels)

        # Without using the managers: loading the data and applying the relabeller
        im_seg = nib.load(jph(root_dir, 'images_examples', pfi_punt_seg_original))
        data_seg = im_seg.get_data()
        seg_new = relabeller(data_seg, list_old_labels, list_new_labels)

        im_relabelled = set_new_data(im_seg, seg_new)

        # Results comparison:
        nib_seg_new = nib.load(jph(root_dir, 'images_output', pfi_punt_seg_new))
        nib_seg_new_data = nib_seg_new.get_data()
        np.testing.assert_array_equal(seg_new, nib_seg_new_data)

    """
    Example 2:
    Permute labels according to a given permutation
    """
    if examples['Example permute']:

        # data:
        pfi_punt_seg_original = 'punt_seg.nii.gz'
        pfi_punt_seg_new      = 'punt_seg.nii.gz'
        perm = [[1, 2, 3], [3, 1, 2]]

        # Using the manager:
        lm.manipulate.permute(pfi_punt_seg_original, pfi_punt_seg_new, perm)

        # without using the manager:
        im_seg = nib.load(jph(root_dir, 'images_examples', pfi_punt_seg_original))
        data_seg = im_seg.get_data()
        seg_new = permute_labels()

        # Results comparison: