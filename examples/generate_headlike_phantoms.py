import os
from os.path import join as jph

from LABelsToolkit.tools.defs import root_dir
from LABelsToolkit.tools.visualiser.see_volume import see_array
from LABelsToolkit.tools.phantoms_generator.shapes_for_headlike_phantoms import headlike_phantom
from LABelsToolkit.tools.phantoms_generator.generate_phantom_multi_atlas import generate_atlas_at_folder, \
    generate_multi_atlas_at_folder


def example_generate_and_visualise_headlike():
    print('Example to generate and visualise an headlike phantom. (phantom is not saved anywhere)')
    omega = (70, 70, 70)
    anatomy, segm = headlike_phantom(omega)
    see_array(anatomy, in_array_segm=segm)


def example_generate_atlas_at_specified_folder():
    print('Example generation of an atlas')
    pfo_examples = jph(root_dir, 'data_examples')
    pfo_target_atlas = jph(pfo_examples, 'dummy_target')
    os.system('mkdir -p {}'.format(pfo_target_atlas))
    generate_atlas_at_folder(pfo_target_atlas, atlas_name='t01', randomness_shape=0.3, randomness_noise=0.4)
    print('look under {}'.format(pfo_target_atlas))


def example_generate_multi_atlas_at_specified_folder():
    print('Example generation of a multi-atlas')
    pfo_examples = jph(root_dir, 'data_examples')
    pfo_multi_atlas = jph(pfo_examples, 'dummy_template')
    os.system('mkdir -p {}'.format(pfo_multi_atlas))
    generate_multi_atlas_at_folder(pfo_multi_atlas, number_of_subjects=10,
                                   multi_atlas_root_name='e', randomness_shape=0.3, randomness_noise=0.4)
    print('look under {}'.format(pfo_multi_atlas))


if __name__ == '__main__':
    # example_generate_and_visualise_headlike()
    example_generate_atlas_at_specified_folder()
    example_generate_multi_atlas_at_specified_folder()
