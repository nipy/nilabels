"""
Module to quickly generate dummy data set for tests and examples.
"""
import os
from os.path import join as jph

from nilabels.tools.defs import root_dir


from nilabels.tools.phantoms_generator.generate_phantom_multi_atlas import generate_atlas_at_folder, \
    generate_multi_atlas_at_folder


pfo_examples     = jph(root_dir, 'data_examples')
pfo_target_atlas = jph(pfo_examples, 'dummy_target')
pfo_multi_atlas  = jph(pfo_examples, 'dummy_multi_atlas')


def generate_atlas_at_specified_folder():
    os.system('mkdir -p {}'.format(pfo_target_atlas))
    generate_atlas_at_folder(pfo_target_atlas, atlas_name='t01', randomness_shape=0.3, randomness_noise=0.4,
                             get_labels_descriptor=True)


def generate_multi_atlas_at_specified_folder(n=10):
    os.system('mkdir -p {}'.format(pfo_multi_atlas))
    generate_multi_atlas_at_folder(pfo_multi_atlas, number_of_subjects=n,
                                   multi_atlas_root_name='e', randomness_shape=0.3, randomness_noise=0.4)
