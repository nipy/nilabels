import os
from os.path import join as jph

import numpy as np
import nibabel as nib

from labels_manager.main import LabelsManager
from definitions import root_dir
from labels_manager.tools.aux_methods.utils import set_new_data
from labels_manager.tools.manipulations.relabeller import relabeller, permute_labels


if __name__ == '__main__':

    # Run generate examples first

    # Create output folder:
    cmd = 'mkdir -p {}'.format(jph(root_dir, 'images_output'))
    os.system(cmd)

    # Instantiate a manager from the class LabelsManager
    lm = LabelsManager(jph(root_dir, 'images_examples'), jph(root_dir, 'images_output'))
    print(lm.pfo_in)
    print(lm.pfo_out)