"""
Use ITK-snap, its labels_descriptor.txt and freeview to get the surfaces and overlay the surface to the main image
directly in freeview with correct naming convention.
"""

import os

from nilabels.tools.aux_methods.label_descriptor_manager import LabelsDescriptorManager


def freesurfer_surface_overlayed(pfi_anatomy, pfo_stl_surfaces, pfi_descriptor, convention_descriptor='itk-snap',
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
    ldm = LabelsDescriptorManager(pfi_descriptor, labels_descriptor_convention=convention_descriptor)

    cmd = 'source $FREESURFER_HOME/SetUpFreeSurfer.sh; freeview -v {0} -f '.format(pfi_anatomy)

    if labels_to_delineate:
        labels_to_delineate = ldm.dict_label_descriptor.keys()[1:-1]

    for k in labels_to_delineate:

        pfi_surface = os.path.join(pfo_stl_surfaces, '{0}{1:05d}.stl'.format(suffix_surf, k))
        assert os.path.exists(pfi_surface), pfi_surface
        if add_colors:
            triplet_rgb = '{0},{1},{2}'.format(ldm.dict_label_descriptor[k][0][0],
                                               ldm.dict_label_descriptor[k][0][1],
                                               ldm.dict_label_descriptor[k][0][2])

            cmd += ' {0}:edgecolor={1}:color={1} '.format(pfi_surface, triplet_rgb)
        else:
            cmd += ' {0} '.format(pfi_surface)
    os.system(cmd)


if __name__ == '__main__':
    print('Step 0: create segmented atlas with phantom generator.')
    print('Step 1: Manual step - open the segmentation in ITK-Snap and export all the surfaces in .stl in the '
          'specified folder')
    print('Step 2: run freesurfer_surface_overlayed')
