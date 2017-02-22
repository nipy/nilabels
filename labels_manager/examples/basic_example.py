import os
from os.path import join as jph

import numpy as np

from labels_manager.main import LabelsManager

from definitions import root_dir
from labels_manager.examples import generate_images_examples


# generate_images_examples.generate_figures()


lm = LabelsManager(jph(root_dir, 'images_examples'))

print lm.input_data_folder
