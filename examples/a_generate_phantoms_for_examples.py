import os
from os.path import join as jph

import nibabel as nib
import numpy as np
import scipy.ndimage.filters as fil

from nilabels.definitions import root_dir
from nilabels.tools.detections.get_segmentation import intensity_segmentation


# ---------- Simple shapes generators ---------------


def o_shape(omega=(250, 250), radius=50,
            background_intensity=0, foreground_intensity=20, dtype=np.uint8):
    m = background_intensity * np.ones(omega, dtype=dtype)

    if len(omega) == 2:
        c = [omega[j] / 2 for j in range(len(omega))]
        for x in range(omega[0]):
            for y in range(omega[1]):
                if (x - c[0]) ** 2 + (y - c[1]) ** 2 < radius ** 2:
                    m[x, y] = foreground_intensity
    elif len(omega) == 3:
        c = [omega[j] / 2 for j in range(len(omega))]
        for x in range(omega[0]):
            for y in range(omega[1]):
                for z in range(omega[2]):
                    if (x - c[0]) ** 2 + (y - c[1]) ** 2 + (z - c[2]) ** 2 < radius ** 2:
                        m[x, y, z] = foreground_intensity
    return m


def c_shape(omega=(250, 250), internal_radius=40, external_radius=60, opening_height=50,
            background_intensity=0, foreground_intensity=20, dtype=np.uint8, margin=None):

    def get_a_2d_c(omega2, internal_radius2d, external_radius2d, opening_height2d, background_intensity2d,
                   foreground_intensity2d, dtype2d):

        m = background_intensity2d * np.ones(omega2[:2], dtype=dtype2d)

        c = [omega2[j] / 2 for j in range(len(omega2))]
        # create the crown
        for x in range(omega2[0]):
            for y in range(omega2[1]):
                if internal_radius2d ** 2 < (x - c[0]) ** 2 + (y - c[1]) ** 2 < external_radius2d ** 2:
                    m[x, y] = foreground_intensity2d

        # open the c
        low_lim = int(omega2[0] / 2) - int(opening_height2d / 2)
        high_lim = int(omega2[0] / 2) + int(opening_height2d / 2)

        for x in range(omega2[0]):
            for y in range(int(omega2[1] / 2), omega2[1]):
                if low_lim < x < high_lim and m[x, y] == foreground_intensity2d:
                    m[x, y] = background_intensity2d

        return m

    c_2d = get_a_2d_c(omega2=omega[:2], internal_radius2d=internal_radius, external_radius2d=external_radius,
                      opening_height2d=opening_height, background_intensity2d=background_intensity,
                      foreground_intensity2d=foreground_intensity, dtype2d=dtype)

    if len(omega) == 2:
        return c_2d

    elif len(omega) == 3:
        if margin is None:
            return np.repeat(c_2d, omega[2]).reshape(omega)
        else:
            res = np.zeros(omega, dtype=dtype)
            for z in range(margin, omega[2] - 2 * margin):
                res[..., z] = c_2d
            return res


def ellipsoid_shape(omega, focus_1, focus_2, distance, background_intensity=0, foreground_intensity=100,
                    dtype=np.uint8):
    sky = background_intensity * np.ones(omega, dtype=dtype)
    for xi in range(omega[0]):
        for yi in range(omega[1]):
            for zi in range(omega[2]):
                if np.sqrt((focus_1[0] - xi) ** 2 + (focus_1[1] - yi) ** 2 + (focus_1[2] - zi) ** 2) + \
                        np.sqrt((focus_2[0] - xi) ** 2 + (focus_2[1] - yi) ** 2 + (focus_2[2] - zi) ** 2) <= distance:
                    sky[xi, yi, zi] = foreground_intensity
    return sky


def cube_shape(omega, center, side_length, background_intensity=0, foreground_intensity=100, dtype=np.uint8):
    sky = background_intensity * np.ones(omega, dtype=dtype)
    half_side_length = int(np.ceil(int(side_length / 2)))

    for lx in range(-half_side_length, half_side_length + 1):
        for ly in range(-half_side_length, half_side_length + 1):
            for lz in range(-half_side_length, half_side_length + 1):
                sky[center[0] + lx, center[1] + ly, center[2] + lz] = foreground_intensity
    return sky


def sphere_shape(omega, centre, radius, foreground_intensity=100, dtype=np.uint8):
    sky = np.zeros(omega, dtype=dtype)
    for xi in range(omega[0]):
        for yi in range(omega[1]):
            for zi in range(omega[2]):
                if np.sqrt((centre[0] - xi) ** 2 + (centre[1] - yi) ** 2 + (centre[2] - zi) ** 2) <= radius:
                    sky[xi, yi, zi] = foreground_intensity
    return sky


def generate_figures(creation_list, pfo_examples, segmentation_levels=7, sigma_smoothing=6, foreground=10):
    print('\n.\n.\n\nGenerate figures for the examples, may take some seconds, and will take approx 150MB.\n.\n.')

    if creation_list['Examples folder']:
        os.system('mkdir -p ' + pfo_examples)

    if creation_list['Punt e mes']:
        data_o_punt = o_shape(omega=(256, 256, 256), foreground_intensity=foreground, dtype=np.float64)
        data_o_punt = fil.gaussian_filter(data_o_punt, sigma=sigma_smoothing)

        data_o_punt_segmentation = intensity_segmentation(data_o_punt, num_levels=segmentation_levels)

        nib_o_punt = nib.Nifti1Image(data_o_punt, affine=np.eye(4))
        nib_o_punt_seg = nib.Nifti1Image(data_o_punt_segmentation, affine=np.eye(4))

        nib.save(nib_o_punt, filename=jph(pfo_examples, 'punt.nii.gz'))
        nib.save(nib_o_punt_seg, filename=jph(pfo_examples, 'punt_seg.nii.gz'))

        del data_o_punt_segmentation, nib_o_punt, nib_o_punt_seg

        print('punt generated')

        data_o_mes = np.zeros_like(data_o_punt)
        data_o_mes[:, :128, :] = data_o_punt[:, :128, :]  # R, A ,S

        data_o_mes_segmentation = intensity_segmentation(data_o_mes, num_levels=segmentation_levels)

        nib_o_mes = nib.Nifti1Image(data_o_mes, affine=np.eye(4))
        nib_o_mes_seg = nib.Nifti1Image(data_o_mes_segmentation, affine=np.eye(4))

        nib.save(nib_o_mes, filename=jph(pfo_examples, 'mes.nii.gz'))
        nib.save(nib_o_mes_seg, filename=jph(pfo_examples, 'mes_seg.nii.gz'))

        del data_o_punt, data_o_mes, data_o_mes_segmentation, nib_o_mes, nib_o_mes_seg

        print('mes generated')

    if creation_list['C']:
        data_c_ = c_shape(omega=(256, 256, 256), foreground_intensity=foreground)
        data_c_ = fil.gaussian_filter(data_c_, sigma_smoothing)

        data_c = np.zeros_like(data_c_)
        data_c[..., 20:-20] = data_c_[..., 20:-20]

        data_c_segmentation = intensity_segmentation(data_c, num_levels=segmentation_levels)

        nib_c = nib.Nifti1Image(data_c, affine=np.eye(4))
        nib_c_seg = nib.Nifti1Image(data_c_segmentation, affine=np.eye(4))

        nib.save(nib_c, filename=jph(pfo_examples, 'acee.nii.gz'))
        nib.save(nib_c_seg, filename=jph(pfo_examples, 'acee_seg.nii.gz'))

        del data_c_, data_c, data_c_segmentation, nib_c, nib_c_seg

        print('C generated')

    if creation_list['Planetaruim']:

        omega = (256, 256, 256)
        centers = [(32, 36, 40), (100, 61, 94), (130, 140, 99), (220, 110, 210),
                   (224, 220, 216), (156, 195, 162), (126, 116, 157), (36, 146, 46)]
        radii = [21, 55, 34, 13, 21, 55, 34, 13]
        intensities = [100] * 8

        sky = np.zeros(omega, dtype=np.uint8)

        for center, radius, intensity in zip(centers, radii, intensities):
            for x in range(center[0] - radius - 1, center[0] + radius + 1):
                for y in range(center[1] - radius - 1, center[1] + radius + 1):
                    for z in range(center[2] - radius - 1, center[2] + radius + 1):
                        if (center[0] - x) ** 2 + (center[1] - y) ** 2 + (center[2] - z) ** 2 <= radius ** 2:
                            sky[x, y, z] = intensity

        planetarium = fil.gaussian_filter(sky, np.min(radii) - np.std(radii))

        nib_planetarium = nib.Nifti1Image(planetarium, affine=np.eye(4))
        nib.save(nib_planetarium, filename=jph(pfo_examples, 'planetarium.nii.gz'))

        print('Planetarium generated')

    if creation_list['Buckle ellipsoids']:
        # Toy example for symmetrisation with registration test

        omega = (120, 140, 160)
        epsilon = 5  # perturbation set to 0 to restore the axial symmetry
        foci_ellipses_left = [np.array([35, 70, 40]), np.array([50, 70, 120])]
        foci_ellipses_right = [np.array([85 + epsilon, 70 - epsilon, 40 + epsilon]), np.array([70, 70, 120])]
        d_left = 95
        d_right = 95

        ellipsoid_left = ellipsoid_shape(omega, foci_ellipses_left[0], foci_ellipses_left[1], d_left,
                                         foreground_intensity=foreground)
        ellipsoid_right = ellipsoid_shape(omega, foci_ellipses_right[0], foci_ellipses_right[1], d_right,
                                          foreground_intensity=foreground)

        two_ellipsoids = foreground * (ellipsoid_left + ellipsoid_right).astype(np.bool).astype(np.float64)

        two_ellipsoids = fil.gaussian_filter(two_ellipsoids, sigma=sigma_smoothing)

        nib_ellipsoids = nib.Nifti1Image(two_ellipsoids, affine=np.eye(4))
        nib.save(nib_ellipsoids, filename=jph(pfo_examples, 'ellipsoids.nii.gz'))

        two_ellipsoids_segmentation = intensity_segmentation(two_ellipsoids, num_levels=segmentation_levels)

        nib_ellipsoids_seg = nib.Nifti1Image(two_ellipsoids_segmentation, affine=np.eye(4))
        nib.save(nib_ellipsoids_seg, filename=jph(pfo_examples, 'ellipsoids_seg.nii.gz'))

        two_ellipsoids_half_segmentation = np.zeros_like(two_ellipsoids_segmentation)
        two_ellipsoids_half_segmentation[:60, :, :] = two_ellipsoids_segmentation[:60, :, :]

        nib_ellipsoids_seg_half = nib.Nifti1Image(two_ellipsoids_half_segmentation, affine=np.eye(4))
        nib.save(nib_ellipsoids_seg_half, filename=jph(pfo_examples, 'ellipsoids_seg_half.nii.gz'))

        print('Buckle ellipsoids half segmented and whole segmented generated')

    if creation_list['Ellipsoids family']:
        # Toy example for registration propagation tests
        pfo_ellipsoids_family = jph(pfo_examples, 'ellipsoids_family')
        os.system('mkdir -p ' + pfo_ellipsoids_family)

        omega = (100, 100, 100)
        num_ellipsoids = 10

        # Target image:
        target_data = o_shape(omega=omega, radius=25, foreground_intensity=foreground, dtype=np.float64)
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
            phi = np.arccos(2 * v - 1)
            first_focus = np.array([radius_first_focus * np.cos(theta) * np.sin(phi) + center_first_focus[0],
                                    radius_first_focus * np.sin(theta) * np.sin(phi) + center_first_focus[1],
                                    radius_first_focus * np.cos(phi) + center_first_focus[2]])

            # get the second focus as symmetric of the fist focus respect to the midpoint
            second_focus = np.array([2 * mid_point[0] - first_focus[0],
                                     2 * mid_point[1] - first_focus[1],
                                     2 * mid_point[2] - first_focus[2]])

            np.testing.assert_almost_equal(mid_point, (first_focus + second_focus) / float(2))

            # get the twice_aance from the 2 focus as a random value, having as min the twice_aance between
            twice_a_2_focus = np.linalg.norm(first_focus - second_focus)
            epsilon = 3
            twice_a = np.random.uniform(low=twice_a_2_focus + epsilon,
                                        high=twice_a_2_focus + epsilon +
                                             np.linalg.norm(first_focus - np.array([25, 25, 50])))
            # get the ellipsoid
            ellips_data = ellipsoid_shape(omega, first_focus, second_focus, twice_a,
                                          foreground_intensity=foreground, dtype=np.float64)
            ellips_data = fil.gaussian_filter(ellips_data, sigma=sigma_smoothing)

            nib_ellipsoids = nib.Nifti1Image(ellips_data, affine=np.eye(4))
            nib.save(nib_ellipsoids, filename=jph(pfo_ellipsoids_family, 'ellipsoid' + str(k) + '.nii.gz'))

            ellips_data_seg = intensity_segmentation(ellips_data, num_levels=segmentation_levels)

            nib_ellipsoids_seg = nib.Nifti1Image(ellips_data_seg, affine=np.eye(4))
            nib.save(nib_ellipsoids_seg, filename=jph(pfo_ellipsoids_family, 'ellipsoid' + str(k) + '_seg.nii.gz'))

        print('Data ellipsoids generated')

        # open target and ellipsoids
        str_pfi_ellipsoids = ''
        for k in range(1, num_ellipsoids + 1):
            str_pfi_ellipsoids += jph(pfo_ellipsoids_family, 'ellipsoid' + str(k) + '.nii.gz') + ' '

        cmd = 'itksnap -g {0} -s {1} -o {2}'.format(jph(pfo_ellipsoids_family, 'target.nii.gz'),
                                                    jph(pfo_ellipsoids_family, 'target_seg.nii.gz'),
                                                    str_pfi_ellipsoids)
        os.system(cmd)

    if creation_list['Cubes in the sky']:
        omega = [80, 80, 80]
        cube_a = [[10, 60, 55], 11, 1]
        cube_b = [[50, 55, 42], 17, 2]
        cube_c = [[25, 20, 20], 19, 3]
        cube_d = [[55, 16, 9], 9, 4]

        sky1 = cube_shape(omega, center=cube_a[0], side_length=cube_a[1], foreground_intensity=1) + \
               cube_shape(omega, center=cube_b[0], side_length=cube_b[1], foreground_intensity=1) + \
               cube_shape(omega, center=cube_c[0], side_length=cube_c[1], foreground_intensity=1) + \
               cube_shape(omega, center=cube_d[0], side_length=cube_d[1], foreground_intensity=1)
        im1 = nib.Nifti1Image(sky1, affine=np.eye(4))

        sky2 = cube_shape(omega, center=cube_a[0], side_length=cube_a[1], foreground_intensity=cube_a[2]) + \
               cube_shape(omega, center=cube_b[0], side_length=cube_b[1], foreground_intensity=cube_b[2]) + \
               cube_shape(omega, center=cube_c[0], side_length=cube_c[1], foreground_intensity=cube_c[2]) + \
               cube_shape(omega, center=cube_d[0], side_length=cube_d[1], foreground_intensity=cube_d[2])
        im2 = nib.Nifti1Image(sky2, affine=np.eye(4))

        nib.save(im1, filename=jph(pfo_examples, 'cubes_in_space_bin.nii.gz'))
        nib.save(im2, filename=jph(pfo_examples, 'cubes_in_space.nii.gz'))

    if creation_list['Sandwich']:
        omega = [9, 9, 10]
        sandwich = np.zeros(omega)

        sandwich[:, :2, :] = 2 * np.ones([9, 2, 10])
        sandwich[:, 2:5, :] = 3 * np.ones([9, 3, 10])
        sandwich[:, 5:, :] = 4 * np.ones([9, 4, 10])
        im_sandwich = nib.Nifti1Image(sandwich, affine=np.diag([0.1, 0.2, 0.3, 1]))

        nib.save(im_sandwich, filename=jph(pfo_examples, 'sandwich.nii.gz'))

    if creation_list['Four-folds']:
        # Case A: best case, the two ellipsoids are overlapping - label id = 1
        # Case B: ellipsoids are only translated  - label id = 2
        # Case C: no dispersion, low precision  - label id = 3
        # Case D: worst case, ellipsoids not overlapping  - label id = 4
        # indexes image, quadrant - two images (1,2), four quadrant (1 2 3 4 for A B C D)

        omega = [100, 100, 50]
        foreground = 1

        # A) center 75,75,25
        first_focus_11 = [79, 79, 25]
        second_focus_11 = [72, 72, 23]
        twice_a_11 = 20

        elli_11 = ellipsoid_shape(omega, first_focus_11, second_focus_11, twice_a_11,
                                  foreground_intensity=foreground, dtype=np.float64)

        first_focus_21 = [79, 79, 25]
        second_focus_21 = [72, 72, 23]
        twice_a_21 = 20

        elli_21 = ellipsoid_shape(omega, first_focus_21, second_focus_21, twice_a_21,
                                  foreground_intensity=foreground, dtype=np.float64)

        # B) center 25, 75, 25
        first_focus_12 = [29, 79, 25]
        second_focus_12 = [24, 72, 23]
        twice_a_12 = 20

        elli_12 = ellipsoid_shape(omega, first_focus_12, second_focus_12, twice_a_12,
                                  foreground_intensity=foreground, dtype=np.float64)

        first_focus_22 = [24, 74, 20]  # - 5 respect to the 1
        second_focus_22 = [19, 67, 18]
        twice_a_22 = 20

        elli_22 = ellipsoid_shape(omega, first_focus_22, second_focus_22, twice_a_22,
                                  foreground_intensity=foreground, dtype=np.float64)

        # C) Center 75, 25, 25
        first_focus_13 = [75, 25, 25]  # shpere
        second_focus_13 = [75, 25, 25]
        twice_a_13 = 24

        elli_13 = ellipsoid_shape(omega, first_focus_13, second_focus_13, twice_a_13,
                                  foreground_intensity=foreground, dtype=np.float64)

        r = .5 * twice_a_13
        # f = 15
        # a = np.sqrt( (f ** 2 + np.sqrt(f**4 + 4*r**4)) / 2. )
        a = 22
        f = np.sqrt((a ** 4 - r ** 4) / float(a ** 2))
        assert f < a, [f, a]
        first_focus_23 = [75 - f, 25, 25]  # ellipsoid
        second_focus_23 = [75 + f, 25, 25]
        twice_a_23 = 2 * a

        elli_23 = ellipsoid_shape(omega, first_focus_23, second_focus_23, twice_a_23,
                                  foreground_intensity=foreground, dtype=np.float64)

        # D) Center (25, 25, 25)
        first_focus_14 = [30, 35, 25]
        second_focus_14 = [30, 30, 25]
        twice_a_14 = 15

        elli_14 = ellipsoid_shape(omega, first_focus_14, second_focus_14, twice_a_14,
                                  foreground_intensity=foreground, dtype=np.float64)

        first_focus_24 = [10, 12, 25]
        second_focus_24 = [9, 8, 25]
        twice_a_24 = 15

        elli_24 = ellipsoid_shape(omega, first_focus_24, second_focus_24, twice_a_24,
                                  foreground_intensity=foreground, dtype=np.float64)

        # ---------- #
        # image one: #
        # ---------- #

        elli_one = elli_11 + 2 * elli_12 + 3 * elli_13 + 4 * elli_14
        im_four_folds_one = nib.Nifti1Image(elli_one, np.eye(4))
        nib.save(img=im_four_folds_one, filename=jph(pfo_examples, 'fourfolds_one.nii.gz'))

        # ---------- #
        # image two: #
        # ---------- #

        elli_two = elli_21 + 2 * elli_22 + 3 * elli_23 + 4 * elli_24
        im_four_folds_two = nib.Nifti1Image(elli_two, np.eye(4))
        nib.save(img=im_four_folds_two, filename=jph(pfo_examples, 'fourfolds_two.nii.gz'))


if __name__ == '__main__':
    creation_steps = {'Examples folder'   : True,
                      'Punt e mes'        : True,
                      'C'                 : True,
                      'Planetaruim'       : True,
                      'Buckle ellipsoids' : True,
                      'Ellipsoids family' : True,
                      'Cubes in the sky'  : True,
                      'Sandwich'          : True,
                      'Four-folds'        : True}

    path_folder_examples = jph(root_dir, 'data_examples')

    generate_figures(creation_steps, path_folder_examples)
