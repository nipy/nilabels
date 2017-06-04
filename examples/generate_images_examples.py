import os
from os.path import join as jph

import nibabel as nib
import numpy as np
import scipy.ndimage.filters as fil

from defs import root_dir
from labels_manager.tools.aux_methods.utils import generate_ellipsoid, binarise_a_matrix, generate_o, generate_c, \
    generate_cube
from labels_manager.tools.detections.get_segmentation import intensity_segmentation


def generate_figures(segmentation_levels=7, sigma_smoothing=6, foreground=10):

    create = {'Examples folder'   : True,
              'Punt e mes'        : True,
              'C'                 : True,
              'Planetaruim'       : True,
              'Buckle ellipsoids' : True,
              'Ellipsoids family' : True,
              'Cubes in the sky'  : True,
              'Sandwich'          : True
              }

    print('\n.\n.\n\nGenerate figures for the examples, may take some seconds, and will take approx 150MB.\n.\n.')
    examples_folder = jph(root_dir, 'images_examples')

    if create['Examples folder']:

        os.system('mkdir -p ' + examples_folder)

    if create['Punt e mes']:

        data_o_punt = generate_o(omega=(256, 256, 256), foreground_intensity=foreground, dtype=np.float64)
        data_o_punt = fil.gaussian_filter(data_o_punt, sigma=sigma_smoothing)

        data_o_punt_segmentation = intensity_segmentation(data_o_punt, num_levels=segmentation_levels)

        nib_o_punt = nib.Nifti1Image(data_o_punt, affine=np.eye(4))
        nib_o_punt_seg = nib.Nifti1Image(data_o_punt_segmentation, affine=np.eye(4))

        nib.save(nib_o_punt, filename=jph(examples_folder, 'punt.nii.gz'))
        nib.save(nib_o_punt_seg, filename=jph(examples_folder, 'punt_seg.nii.gz'))

        del data_o_punt_segmentation, nib_o_punt, nib_o_punt_seg

        print 'punt generated'

        data_o_mes = np.zeros_like(data_o_punt)
        data_o_mes[:,:128, :] = data_o_punt[:,:128, :]  # R, A ,S

        data_o_mes_segmentation = intensity_segmentation(data_o_mes, num_levels=segmentation_levels)

        nib_o_mes = nib.Nifti1Image(data_o_mes, affine=np.eye(4))
        nib_o_mes_seg = nib.Nifti1Image(data_o_mes_segmentation, affine=np.eye(4))

        nib.save(nib_o_mes, filename=jph(examples_folder, 'mes.nii.gz'))
        nib.save(nib_o_mes_seg, filename=jph(examples_folder, 'mes_seg.nii.gz'))

        del data_o_punt, data_o_mes, data_o_mes_segmentation, nib_o_mes, nib_o_mes_seg

        print 'mes generated'

    if create['C']:

        data_c_ = generate_c(omega=(256, 256, 256), foreground_intensity=foreground)
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

    if create['Planetaruim']:

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

    if create['Buckle ellipsoids']:
        # Toy example for symmetrisation with registration test

        omega = (120, 140, 160)
        epsilon = 5  # perturbation set to 0 to restore the axial symmetry
        foci_ellipses_left = [np.array([35, 70, 40]), np.array([50, 70, 120])]
        foci_ellipses_right = [np.array([85 + epsilon, 70 - epsilon, 40+epsilon]), np.array([70, 70, 120])]
        d_left = 95
        d_right = 95

        ellipsoid_left  = generate_ellipsoid(omega, foci_ellipses_left[0], foci_ellipses_left[1], d_left, foreground_intensity=foreground)
        ellipsoid_right = generate_ellipsoid(omega, foci_ellipses_right[0], foci_ellipses_right[1], d_right, foreground_intensity=foreground)

        two_ellipsoids = foreground * binarise_a_matrix(ellipsoid_left + ellipsoid_right, dtype=np.float64)

        two_ellipsoids = fil.gaussian_filter(two_ellipsoids, sigma=sigma_smoothing)

        nib_ellipsoids = nib.Nifti1Image(two_ellipsoids, affine=np.eye(4))
        nib.save(nib_ellipsoids, filename=jph(examples_folder, 'ellipsoids.nii.gz'))

        two_ellipsoids_segmentation = intensity_segmentation(two_ellipsoids, num_levels=segmentation_levels)

        nib_ellipsoids_seg = nib.Nifti1Image(two_ellipsoids_segmentation, affine=np.eye(4))
        nib.save(nib_ellipsoids_seg, filename=jph(examples_folder, 'ellipsoids_seg.nii.gz'))

        two_ellipsoids_half_segmentation = np.zeros_like(two_ellipsoids_segmentation)
        two_ellipsoids_half_segmentation[:60, :, :] = two_ellipsoids_segmentation[:60, :, :]

        nib_ellipsoids_seg_half = nib.Nifti1Image(two_ellipsoids_half_segmentation, affine=np.eye(4))
        nib.save(nib_ellipsoids_seg_half, filename=jph(examples_folder, 'ellipsoids_seg_half.nii.gz'))

        print 'Buckle ellipsoids half segmented and whole segmented generated'

    if create['Ellipsoids family']:
        # Toy example for registration propagation tests
        pfo_ellipsoids_family = jph(examples_folder, 'ellipsoids_family')
        os.system('mkdir -p ' + pfo_ellipsoids_family)

        omega = (100, 100, 100)
        num_ellipsoids = 10

        # Target image:
        target_data = generate_o(omega=omega, radius=25, foreground_intensity=foreground, dtype=np.float64)
        target_data = fil.gaussian_filter(target_data, sigma=sigma_smoothing)
        target_seg = intensity_segmentation(target_data, num_levels=segmentation_levels)

        nib_target = nib.Nifti1Image(target_data, affine=np.eye(4))
        nib.save(nib_target, filename=jph(pfo_ellipsoids_family, 'target.nii.gz'))

        nib_target_seg = nib.Nifti1Image(target_seg, affine=np.eye(4))
        nib.save(nib_target_seg, filename=jph(pfo_ellipsoids_family, 'target_seg.nii.gz'))

        # --- Generate random ellipsoids in omega: --
        for k in range(1, num_ellipsoids + 1):

            mid_point = np.array([50, 50, 50])
            # Get the first focus in a sphere at (25, 25, 50) with radius 15
            center_first_focus = np.array([25, 50, 50])
            radius_first_focus = 15 * np.random.uniform()

            u, v = np.random.uniform(), np.random.uniform()
            # sphere point picking (Wolfram)
            theta = 2 * np.pi * u
            phi = np.arccos(2*v - 1)
            first_focus = np.array([radius_first_focus * np.cos(theta) * np.sin(phi) + center_first_focus[0],
                                    radius_first_focus * np.sin(theta) * np.sin(phi) + center_first_focus[1],
                                    radius_first_focus * np.cos(phi) +  center_first_focus[2]])

            # get the second focus as symmetric of the fist focus respect to the midpoint
            second_focus = np.array([2 * mid_point[0] - first_focus[0],
                                     2 * mid_point[1] - first_focus[1],
                                     2 * mid_point[2] - first_focus[2]])

            np.testing.assert_almost_equal(mid_point, (first_focus + second_focus) / float(2) )

            # get the distance from the 2 focus as a random value, having as min the distance between
            dist_2_focus = np.linalg.norm(first_focus - second_focus)
            epsilon = 3
            dist = np.random.uniform(low=dist_2_focus + epsilon, high=dist_2_focus + epsilon + np.linalg.norm(first_focus - np.array([25, 25, 50])))
            # get the ellipsoid
            ellips_data = generate_ellipsoid(omega, first_focus, second_focus, dist, foreground_intensity=foreground, dtype=np.float64)
            ellips_data = fil.gaussian_filter(ellips_data, sigma=sigma_smoothing)

            nib_ellipsoids = nib.Nifti1Image(ellips_data, affine=np.eye(4))
            nib.save(nib_ellipsoids, filename=jph(pfo_ellipsoids_family, 'ellipsoid' + str(k) + '.nii.gz'))

            ellips_data_seg = intensity_segmentation(ellips_data, num_levels=segmentation_levels)

            nib_ellipsoids_seg = nib.Nifti1Image(ellips_data_seg, affine=np.eye(4))
            nib.save(nib_ellipsoids_seg, filename=jph(pfo_ellipsoids_family, 'ellipsoid' + str(k) + '_seg.nii.gz'))

        print 'Data ellipsoids generated'

        # open target and ellipsoids
        str_pfi_ellipsoids = ''
        for k in range(1, num_ellipsoids + 1):
            str_pfi_ellipsoids += jph(pfo_ellipsoids_family, 'ellipsoid' + str(k) + '.nii.gz') + ' '

        cmd = 'itksnap -g {0} -s {1} -o {2}'.format(jph(pfo_ellipsoids_family, 'target.nii.gz'),
                                                    jph(pfo_ellipsoids_family, 'target_seg.nii.gz'),
                                                    str_pfi_ellipsoids)
        os.system(cmd)

    if create['Cubes in the sky']:

        omega = [80, 80, 80]
        cube_a = [[10, 60, 55], 11, 1]
        cube_b = [[50, 55, 42], 17, 2]
        cube_c = [[25, 20, 20], 19, 3]
        cube_d = [[55, 16, 9], 9, 4]

        sky1 = generate_cube(omega, center=cube_a[0], side_length=cube_a[1], foreground_intensity=1) + \
               generate_cube(omega, center=cube_b[0], side_length=cube_b[1], foreground_intensity=1) + \
               generate_cube(omega, center=cube_c[0], side_length=cube_c[1], foreground_intensity=1) + \
               generate_cube(omega, center=cube_d[0], side_length=cube_d[1], foreground_intensity=1)
        im1 = nib.Nifti1Image(sky1, affine=np.eye(4))

        sky2 = generate_cube(omega, center=cube_a[0], side_length=cube_a[1], foreground_intensity=cube_a[2]) + \
               generate_cube(omega, center=cube_b[0], side_length=cube_b[1], foreground_intensity=cube_b[2]) + \
               generate_cube(omega, center=cube_c[0], side_length=cube_c[1], foreground_intensity=cube_c[2]) + \
               generate_cube(omega, center=cube_d[0], side_length=cube_d[1], foreground_intensity=cube_d[2])
        im2 = nib.Nifti1Image(sky2, affine=np.eye(4))

        nib.save(im1, filename=jph(examples_folder, 'cubes_in_space_bin.nii.gz'))
        nib.save(im2, filename=jph(examples_folder, 'cubes_in_space.nii.gz'))

    if create['Sandwich']:

        omega = [9, 9, 10]
        sandwich = np.zeros(omega)

        sandwich[:, :2, :]  = 2 * np.ones([9, 2, 10])
        sandwich[:, 2:5, :] = 3 * np.ones([9, 3, 10])
        sandwich[:, 5:, :]  = 4 * np.ones([9, 4, 10])
        im_sandwich = nib.Nifti1Image(sandwich, affine=np.diag([0.1, 0.2, 0.3, 1]))

        nib.save(im_sandwich, filename=jph(examples_folder, 'sandwich.nii.gz'))

if __name__ == '__main__':
    generate_figures()
