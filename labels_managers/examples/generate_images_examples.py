import os
from os.path import join as jph

import numpy as np
import nibabel as nib
import scipy.ndimage.filters as fil

from definitions import root_dir
from labels_managers.tools.aux_methods.utils import generate_o, generate_c
from labels_managers.tools.detectors.segmentation import intensity_segmentation


def generate_figures():

    # create images examples folder:

    examples_folder = jph(root_dir, 'examples', 'images_examples')
    os.system('mkdir -p ' + examples_folder)

    segmentation_levels = 7
    sigma_smoothing = 15

    # create data and segmentation o (punt):

    data_o_punt = generate_o(omega=(256, 256, 256), foreground_intensity=100, dtype=np.float64)
    fil.gaussian_filter(data_o_punt, sigma=sigma_smoothing, output=data_o_punt)

    data_o_punt_segmentation = intensity_segmentation(data_o_punt, num_levels=segmentation_levels)

    nib_o_punt = nib.Nifti1Image(data_o_punt, affine=np.eye(4))
    nib_o_punt_seg = nib.Nifti1Image(data_o_punt_segmentation, affine=np.eye(4))

    nib.save(nib_o_punt, filename=jph(examples_folder, 'punt.nii.gz'))
    nib.save(nib_o_punt_seg, filename=jph(examples_folder, 'punt_seg.nii.gz'))

    del data_o_punt_segmentation, nib_o_punt, nib_o_punt_seg

    print 'punt generated'

    # create half o (... e mes):

    data_o_mes = np.zeros_like(data_o_punt)
    data_o_mes[:128, ...] = data_o_punt[:128, ...]  # R, A ,S

    data_o_mes_segmentation = intensity_segmentation(data_o_mes, num_levels=segmentation_levels)

    nib_o_mes = nib.Nifti1Image(data_o_mes, affine=np.eye(4))
    nib_o_mes_seg = nib.Nifti1Image(data_o_mes_segmentation, affine=np.eye(4))

    nib.save(nib_o_mes, filename=jph(examples_folder, 'mes.nii.gz'))
    nib.save(nib_o_mes_seg, filename=jph(examples_folder, 'mes_seg.nii.gz'))

    del data_o_punt, data_o_mes, data_o_mes_segmentation, nib_o_mes, nib_o_mes_seg

    print 'mes generated'

    # create a 'c' (acee):

    data_c_ = generate_c(omega=(256, 256, 256), foreground_intensity=100)
    data_c_ = fil.gaussian_filter(data_c_, sigma_smoothing)

    data_c = np.zeros_like(data_c_)
    data_c[..., 20:-20] = data_c_[..., 20:-20]

    data_c_segmentation = intensity_segmentation(data_c, num_levels=segmentation_levels)

    nib_c = nib.Nifti1Image(data_c, affine=np.eye(4))
    nib_c_seg = nib.Nifti1Image(data_c_segmentation, affine=np.eye(4))

    nib.save(nib_c, filename=jph(examples_folder, 'acee.nii.gz'))
    nib.save(nib_c_seg, filename=jph(examples_folder, 'acee_seg.nii.gz'))

    del data_c_, data_c, data_c_segmentation, nib_c, nib_c_seg

    print 'C generated'

    # create planetarium:

    omega = (256, 256, 256)
    centers = [(32, 36, 40),    (100, 61, 94),   (130, 140, 99),  (220, 110, 210),
               (224, 220, 216), (156, 195, 162), (126, 116, 157), (36, 146, 46)]
    radii = [21, 55, 34, 13, 21, 55, 34, 13]
    intensities = [100] * 8

    sky = np.zeros(omega, dtype=np.uint8)

    for center, radius, intensity in zip(centers, radii, intensities):
        for x in xrange(center[0] - radius - 1, center[0] + radius + 1):
            for y in xrange(center[1] - radius - 1, center[1] + radius + 1):
                for z in xrange(center[2] - radius - 1, center[2] + radius + 1):
                    if (center[0] - x) ** 2 + (center[1] - y) ** 2 + (center[2] - z) ** 2 <= radius:
                        sky[x, y, z] = intensity

    planetarium = fil.gaussian_filter(sky, np.min(radii)-np.std(radii))

    nib_planetarium = nib.Nifti1Image(planetarium, affine=np.eye(4))
    nib.save(nib_planetarium, filename=jph(examples_folder, 'planetarium.nii.gz'))

    print 'Planetarium generated'
    # create star-dust:

    num_particles = 100
    max_radius = ''
    min_radius = ''

    sky = np.zeros((256, 256, 256), dtype=np.uint8)

    print 'Figures generated'

    # TODO !!


    '''
    data_c_ = generate_c(omega=(256, 256, 256), foreground_intensity=100)
    data_c_ = fil.gaussian_filter(data_c_, sigma_smoothing)

    data_c = np.zeros_like(data_c_)
    data_c[..., 20:-20] = data_c_[..., 20:-20]

    data_c_segmentation = intensity_segmentation(data_c, num_levels=segmentation_levels)

    nib_c = nib.Nifti1Image(data_c, affine=np.eye(4))
    nib_c_seg = nib.Nifti1Image(data_c_segmentation, affine=np.eye(4))

    nib.save(nib_c, filename=jph(examples_folder, 'acee.nii.gz'))
    nib.save(nib_c_seg, filename=jph(examples_folder, 'acee_seg.nii.gz'))
    '''


if __name__ == '__main__':
    generate_figures()
