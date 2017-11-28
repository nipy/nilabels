"""
Generating multimodal template of ellipsoids,
where other modalities are given by:
 - main ellipsoid in space, and main ellipsoid with cheese holes A
 - independent salt and pepper noise resampled in a different space B.
 - sobel filters in space C
 - filtering by some set of functions resampled in space C
Default naming convention.
"""
import os
from os.path import join as jph

import nibabel as nib
import numpy as np
import scipy.ndimage.filters as fil
from scipy import ndimage

from labels_manager.tools.phantoms_generator.shapes_for_phantoms import o_shape, ellipsoid_shape
from labels_manager.tools.detections.get_segmentation import intensity_segmentation
from labels_manager.tools.aux_methods.utils_nib import set_new_data
from labels_manager.tools.defs import root_dir


def stardust_array(omega, max_radius=4, num_stars=40):

    centers = [(np.random.randint(max_radius, omega[0] - max_radius),
                np.random.randint(max_radius, omega[1] - max_radius),
                np.random.randint(max_radius, omega[2] - max_radius)) for _ in range(num_stars)]
    radii = [np.random.choice(list(range(1, max_radius + 1))) for _ in range(num_stars)]

    sky = np.zeros(omega, dtype=np.bool)

    for center, radius in zip(centers, radii):
        for x in xrange(center[0] - radius, center[0] + radius):
            for y in xrange(center[1] - radius, center[1] + radius):
                for z in xrange(center[2] - radius, center[2] + radius):
                    if (center[0] - x) ** 2 + (center[1] - y) ** 2 + (center[2] - z) ** 2 <= radius ** 2:
                        sky[x, y, z] = 1

    return sky


def salt_and_pepper_noise(array, prob=0.01):
    """
    Add salt and pepper noise to image
    prob: Probability of the noise
    """
    salt_pepper_values = {'pepper' : np.max(array),
                          'salt'   : np.min(array)}
    output = np.copy(array)
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            for k in range(array.shape[2]):
                rdn = np.random.uniform(0, 1)
                if rdn < prob:
                    s_or_p = np.random.choice(['salt', 'pepper'])
                    output[i, j, k] = salt_pepper_values[s_or_p]
    return output


def apply_sobel_filter(array):
    dx = fil.sobel(array, 0)  # horizontal derivative
    dy = fil.sobel(array, 1)  # vertical derivative
    dz = fil.sobel(array, 2)
    sob = np.sqrt(dx**2 + dy**2 + dz**2)
    sob *= 10 / np.max(sob)
    return sob


def apply_laplacian_filter(array, alpha=30):
    """
    Laplacian is approximated with difference of Gaussian
    :param array:
    :param alpha:
    :return:
    """
    blurred_f = ndimage.gaussian_filter(array, 3)
    filter_blurred_f = ndimage.gaussian_filter(blurred_f, 1)
    return blurred_f + alpha * (blurred_f - filter_blurred_f)


def apply_filter_invert(array, trim_toll=0.05):

    output = np.zeros_like(array)
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            for k in range(array.shape[2]):
                if array[i, j, k] > np.max(array) * trim_toll:
                    output[i, j, k] = 1./array[i, j, k]

    return output


def create_template_target(pfo_destination, t_type='Template'):
    """
    Phantom Target and Template Creator.
    Same code can create both.
    Template is a template of segmented smoothed 3d ellipsoids.
    Target is a segmented and smoothed sphere.
    Segmentation is performed in the same way.
    :param pfo_destination:
    :param t_type: can be Template or Target, where a target is so for the segmentation propagations.
    :return:
    """
    if t_type == 'Template':
        # parameters
        number_of_charts = 10
        charts_names = ['e{0:02d}'.format(n) for n in range(1, number_of_charts + 1)]
        omega = (100, 100, 100)
        mid_point = np.array([50, 50, 50])
        segmentation_levels = 7
        sigma_smoothing = 6
        foreground = 10

        template_list_suffix_modalities = [['Main', 'MainHoled'],
                                           ['SaltPepper', 'Sobel'],
                                           ['FilterMedian', 'FilterLaplacian', 'FilterInvert']]
        template_list_suffix_masks = ['roi_mask', 'roi_reg_mask']

    elif t_type == 'Target':
        # if not template is a target:
        charts_names = ['t01']
        omega = (100, 100, 100)
        mid_point = np.array([50, 50, 50])
        segmentation_levels = 7
        sigma_smoothing = 6
        foreground = 10

        template_list_suffix_modalities = [['Main', 'MainHoled'],
                                           ['SaltPepper', 'Sobel'],
                                           ['FilterMedian', 'FilterLaplacian', 'FilterInvert']]
        template_list_suffix_masks = ['roi_mask', 'roi_reg_mask']
    else:
        raise IOError
    # create main modality, masks and ground truth segmentation.
    os.system('rm -r {}'.format(pfo_destination))
    os.system('mkdir {}'.format(pfo_destination))
    for ch in charts_names:
        # --- create folder structure
        pfo_ch_dir = jph(pfo_destination, ch)
        os.system('mkdir {}'.format(pfo_ch_dir))
        pfo_ch_mod = jph(pfo_ch_dir, 'mod')
        pfo_ch_masks = jph(pfo_ch_dir, 'masks')
        pfo_ch_segm = jph(pfo_ch_dir, 'segm')
        os.system('mkdir {}'.format(pfo_ch_mod))
        os.system('mkdir {}'.format(pfo_ch_masks))
        os.system('mkdir {}'.format(pfo_ch_segm))
        # --- get the main ellipses
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
        # get the string length
        dist_2_focus = np.linalg.norm(first_focus - second_focus)
        epsilon = 3
        dist = np.random.uniform(low=dist_2_focus + epsilon,
                                 high=dist_2_focus + epsilon + np.linalg.norm(first_focus - np.array([25, 25, 50])))

        # -------------------------- #
        # -- Ellipsoid - space 0 --- #
        # -------------------------- #

        # Ellipsoid - 00
        pfi_mod_00 = jph(pfo_ch_mod, '{0}_{1}.nii.gz'.format(ch, template_list_suffix_modalities[0][0]))
        if t_type == 'Template':
            ellips_data = ellipsoid_shape(omega, first_focus, second_focus, dist,
                                          foreground_intensity=foreground, dtype=np.float64)
        elif t_type == 'Target':
            ellips_data = o_shape(omega=omega, radius=25, foreground_intensity=foreground, dtype=np.float64)
        else:
            raise IOError
        ellips_data = fil.gaussian_filter(ellips_data, sigma=sigma_smoothing)
        nib_ellipsoids = nib.Nifti1Image(ellips_data, affine=np.eye(4))
        nib.save(nib_ellipsoids, filename=pfi_mod_00)

        # Ellipsoid - 01
        pfi_mod_01 = jph(pfo_ch_mod, '{0}_{1}.nii.gz'.format(ch, template_list_suffix_modalities[0][1]))
        stardust = stardust_array(omega)
        cheese_holes = np.logical_not(stardust)
        nib_ellipsoids = nib.Nifti1Image(ellips_data * cheese_holes, affine=np.eye(4))
        nib.save(nib_ellipsoids, filename=pfi_mod_01)

        # Ellipsoid - segmentation space 0
        pfi_mod_00_segm = jph(pfo_ch_segm, '{0}_{1}_approved.nii.gz'.format(ch, template_list_suffix_modalities[0][0]))
        ellips_data_seg = intensity_segmentation(ellips_data, num_levels=segmentation_levels)
        nib_ellipsoids_seg = nib.Nifti1Image(ellips_data_seg, affine=np.eye(4))
        nib.save(nib_ellipsoids_seg, filename=pfi_mod_00_segm)

        # Ellipsoid - mask space 0
        pfi_mod_00_roi_mask = jph(pfo_ch_masks, '{0}_{1}_{2}.nii.gz'.format(
            ch, template_list_suffix_modalities[0][0], template_list_suffix_masks[0]))

        os.system('seg_maths {0} -bin {1}'.format(pfi_mod_00_segm, pfi_mod_00_roi_mask))
        os.system('seg_maths {0} -dil 1 {1}'.format(pfi_mod_00_roi_mask, pfi_mod_00_roi_mask))

        # Ellipsoid - mask space 0 registration
        pfi_mod_00_roi_reg_mask = jph(pfo_ch_masks, '{0}_{1}_{2}.nii.gz'.format(
            ch, template_list_suffix_modalities[0][0], template_list_suffix_masks[1]))

        os.system('seg_maths {0} -bin {1}'.format(pfi_mod_01, pfi_mod_00_roi_reg_mask))
        os.system('seg_maths {0} -ero 0.5 {1}'.format(pfi_mod_00_roi_reg_mask, pfi_mod_00_roi_reg_mask))
        os.system('seg_maths {0} -mul {1} {2}'.format(
            pfi_mod_00_roi_reg_mask, pfi_mod_00_roi_mask, pfi_mod_00_roi_reg_mask))

        # -------------------------- #
        # -- Ellipsoid - space 1 --- #
        # -------------------------- #
        # resample the image in the new space:
        pfi_mod_00_resampled = jph(pfo_destination, 'z_{0}_{1}.nii.gz'.format(
            ch, template_list_suffix_modalities[1][0]))
        pfi_reference_space = jph(pfo_destination, 'z_resampling_space.nii.gz')
        pfi_id = jph(pfo_destination, 'z_id.txt')
        aff = (10. / 8.) * np.eye(4)
        aff[3, 3] = 1
        nib_target = nib.Nifti1Image(np.ones([80, 80, 80]), affine=aff)
        nib.save(nib_target, filename=pfi_reference_space)
        np.savetxt(pfi_id, np.eye(4))
        os.system('reg_resample -ref {0} -flo {1} -trans {2} -res {3}'.format(pfi_reference_space, pfi_mod_00,
                                                                              pfi_id, pfi_mod_00_resampled))

        # add salt and pepper to pfi_mod_00_resampled and save
        pfi_mod_10 = jph(pfo_ch_mod, '{0}_{1}.nii.gz'.format(ch, template_list_suffix_modalities[1][0]))
        im_mod_10_resampled = nib.load(pfi_mod_00_resampled)
        im_mod_10 = set_new_data(im_mod_10_resampled, salt_and_pepper_noise(im_mod_10_resampled.get_data()))
        nib.save(im_mod_10, pfi_mod_10)
        # get sobel filter to pfi_mod_00_resampled and save
        pfi_mod_11 = jph(pfo_ch_mod, '{0}_{1}.nii.gz'.format(ch, template_list_suffix_modalities[1][1]))
        im_mod_10_resampled = nib.load(pfi_mod_00_resampled)
        im_mod_11 = set_new_data(im_mod_10_resampled, apply_sobel_filter(im_mod_10_resampled.get_data()))
        nib.save(im_mod_11, pfi_mod_11)

        # Ellipsoid - mask space 1
        pfi_mod_10_roi_mask = jph(pfo_ch_masks, '{0}_{1}_{2}.nii.gz'.format(
            ch, template_list_suffix_modalities[1][0], template_list_suffix_masks[0]))
        os.system('reg_resample -ref {0} -flo {1} -trans {2} -res {3}'.format(pfi_reference_space, pfi_mod_00_roi_mask,
                                                                              pfi_id, pfi_mod_10_roi_mask))

        os.system('seg_maths {0} -dil 3 {1}'.format(pfi_mod_10_roi_mask, pfi_mod_10_roi_mask))
        os.system('seg_maths {0} -smol 0.5 {1}'.format(pfi_mod_10_roi_mask, pfi_mod_10_roi_mask))

        # Ellipsoid - segmentation space 1
        pfi_mod_10_segm = jph(pfo_ch_segm, '{0}_{1}_approved.nii.gz'.format(ch, template_list_suffix_modalities[1][0]))
        os.system('reg_resample -ref {0} -flo {1} -trans {2} -res {3}'.format(pfi_reference_space, pfi_mod_00_segm,
                                                                              pfi_id, pfi_mod_10_segm))

        # -------------------------- #
        # -- Ellipsoid - space 2 --- #
        # -------------------------- #
        # resample the image in the new space:
        pfi_mod_20_resampled = jph(pfo_destination, 'z_{0}_{1}.nii.gz'.format(
            ch, template_list_suffix_modalities[2][0]))
        pfi_reference_space = jph(pfo_destination, 'z_resampling_space.nii.gz')
        aff = np.eye(4)
        aff[0, 0] = 10. / 8.
        aff[1, 1] = 10. / 6.
        aff[2, 2] = 10. / 6.
        np.savetxt(pfi_id, np.eye(4))
        nib_target = nib.Nifti1Image(np.ones([80, 60, 60]), affine=aff)
        nib.save(nib_target, filename=pfi_reference_space)
        os.system('reg_resample -ref {0} -flo {1} -trans {2} -res {3}'.format(pfi_reference_space, pfi_mod_00,
                                                                              pfi_id, pfi_mod_20_resampled))
        # get median filter
        pfi_mod_20 = jph(pfo_ch_mod, '{0}_{1}.nii.gz'.format(ch, template_list_suffix_modalities[2][0]))
        im_mod_20_resampled = nib.load(pfi_mod_20_resampled)
        im_mod_20 = set_new_data(im_mod_10_resampled, fil.median_filter(im_mod_20_resampled.get_data(), 3))
        nib.save(im_mod_20, pfi_mod_20)

        # get Laplacian sharpening filter
        pfi_mod_21 = jph(pfo_ch_mod, '{0}_{1}.nii.gz'.format(ch, template_list_suffix_modalities[2][1]))
        im_mod_21 = set_new_data(im_mod_10_resampled, apply_laplacian_filter(im_mod_20_resampled.get_data(), 3))
        nib.save(im_mod_21, pfi_mod_21)

        # get inverting filter
        pfi_mod_22 = jph(pfo_ch_mod, '{0}_{1}.nii.gz'.format(ch, template_list_suffix_modalities[2][2]))
        im_mod_22 = set_new_data(im_mod_10_resampled, apply_filter_invert(im_mod_20_resampled.get_data()))
        nib.save(im_mod_22, pfi_mod_22)

        # Ellipsoid - mask space 2
        pfi_mod_20_roi_mask = jph(pfo_ch_masks, '{0}_{1}_{2}.nii.gz'.format(
            ch, template_list_suffix_modalities[2][0], template_list_suffix_masks[0]))
        os.system('reg_resample -ref {0} -flo {1} -trans {2} -res {3}'.format(pfi_reference_space, pfi_mod_00_roi_mask,
                                                                              pfi_id, pfi_mod_20_roi_mask))
        os.system('seg_maths {0} -dil 2 {1}'.format(pfi_mod_20_roi_mask, pfi_mod_20_roi_mask))
        os.system('seg_maths {0} -smol 0.5 {1}'.format(pfi_mod_20_roi_mask, pfi_mod_20_roi_mask))

        # Ellipsoid - segmentation space 2
        pfi_mod_20_segm = jph(pfo_ch_segm, '{0}_{1}_approved.nii.gz'.format(ch, template_list_suffix_modalities[2][0]))
        os.system('reg_resample -ref {0} -flo {1} -trans {2} -res {3}'.format(pfi_reference_space, pfi_mod_00_segm,
                                                                              pfi_id, pfi_mod_20_segm))

    os.system('rm {}/z_*'.format(pfo_destination))


if __name__ == '__main__':

    pfo_examples = jph(root_dir, 'data_examples')
    pfo_template = jph(pfo_examples, 'dummy_template')
    pfo_targets = jph(pfo_examples, 'dummy_target')

    os.system('mkdir -p {}'.format(pfo_examples))
    os.system('mkdir -p {}'.format(pfo_template))
    os.system('mkdir -p {}'.format(pfo_targets))

    # generate template and a target
    create_template_target(pfo_template, t_type='Template')
    create_template_target(pfo_targets, t_type='Target')
