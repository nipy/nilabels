from nilabels.tools.phantoms_generator.shapes_for_headlike_phantoms import headlike_phantom
from nilabels.tools.visualiser.see_volume import see_array
from nilabels.tools.phantoms_generator import local_data_generator as ldg


def example_generate_and_visualise_headlike():
    print('Example to generate and visualise an headlike phantom. (phantom is not saved anywhere)')
    omega = (70, 70, 70)
    anatomy, segm = headlike_phantom(omega)
    see_array(anatomy, in_array_segm=segm)


def example_generate_atlas_at_specified_folder():
    print('Example generation of an atlas')
    ldg.generate_atlas_at_specified_folder()
    print('Please look under {}'.format(ldg.pfo_target_atlas))


def example_generate_multi_atlas_at_specified_folder():
    print('Example generation of a multi-atlas')
    ldg.generate_multi_atlas_at_specified_folder()
    print('Please look under {}'.format(ldg.pfo_multi_atlas))


if __name__ == '__main__':
    example_generate_and_visualise_headlike()
    example_generate_atlas_at_specified_folder()
    example_generate_multi_atlas_at_specified_folder()
