import os
from os.path import join as jph

import numpy as np

from labels_managers.main import LabelsManager

from definitions import root_dir
from labels_managers.examples import generate_images_examples


print 'Generate figures for the examples, may take some seconds: '
generate_images_examples.generate_figures()


lm = LabelsManager(jph(root_dir, 'examples', 'data_for_examples'))

print lm.input_data_folder

lm.manipulate.erase

a = np.array([4], dtype=np.uint8)

