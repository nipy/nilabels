"""
Use ITK-snap and fls to obtain the surfaces and overlay the surface to the main image.
"""

import os
from os.path import join as jph
from LABelsToolkit.tools.descriptions.manipulate_descriptors import LabelsDescriptorManager


def fsl_surface_overlayed(pfi_anatomy, pfo_stl_surfaces, pfi_descriptor, convention_descriptor='itk-snap',
                          suffix_surf='surf', add_colors=True, labels_to_delineate='all'):
    """
    Manual step: from a segmentation export all the labels in stand-alone .stl files with ITK-snap, in a folder
    pfo_stl_surfaces, with suffix suffix_surf
    :param pfi_anatomy:
    :param pfo_stl_surfaces:
    :param pfi_descriptor:
    :param convention_descriptor:
    :param suffix_surf:
    :param add_colors:
    :param labels_to_delineate:
    :return:
    """
    ldm = LabelsDescriptorManager(pfi_descriptor, convention=convention_descriptor)

    cmd = 'source $FREESURFER_HOME/SetUpFreeSurfer.sh; freeview -v {0} -f '.format(pfi_anatomy)

    if labels_to_delineate:
        labels_to_delineate = ldm._dict_label_descriptor.keys()[1:-1]

    for k in labels_to_delineate:

        pfi_surface = os.path.join(pfo_stl_surfaces, '{0}{1:05d}.stl'.format(suffix_surf, k))
        assert os.path.exists(pfi_surface), pfi_surface
        if add_colors:
            triplet_rgb = '{0},{1},{2}'.format(ldm._dict_label_descriptor[k][0][0],
                                               ldm._dict_label_descriptor[k][0][1],
                                               ldm._dict_label_descriptor[k][0][2])

            cmd += ' {0}:edgecolor={1}:color={1} '.format(pfi_surface, triplet_rgb)
        else:
            cmd += ' {0} '.format(pfi_surface)
    os.system(cmd)


if __name__ == '__main__':
    # TODO add an example from the ellipsoids.
    pass

