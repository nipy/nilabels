import os
from os.path import join as jph

import nibabel as nib
import numpy as np
import scipy.ndimage.filters as fil

from LABelsToolkit.tools.aux_methods.utils import print_and_run
from LABelsToolkit.tools.phantoms_generator.shapes_for_phantoms import sphere_shape
from LABelsToolkit.tools.visualiser.see_volume import see_array
from LABelsToolkit.tools.phantoms_generator.shapes_for_headlike_phantoms import headlike_phantom


def generate_atlas_at_folder(pfo_where_to_save_atlas, atlas_name='test', randomness_shape=0.3, randomness_noise=0.4):

    assert os.path.exists(pfo_where_to_save_atlas), 'Input folder {} does not exist'.format(pfo_where_to_save_atlas)
    pfo_mod = jph(pfo_where_to_save_atlas, 'mod')
    pfo_segm = jph(pfo_where_to_save_atlas, 'segm')
    pfo_masks = jph(pfo_where_to_save_atlas, 'masks')

    print_and_run('mkdir {}'.format(pfo_mod))
    print_and_run('mkdir {}'.format(pfo_segm))
    print_and_run('mkdir {}'.format(pfo_masks))

    # B) Create modality and segmentation ground truth:
    intensities = (0.9, 0.3, 0.6, 0.8)
    omega = (80, 90, 80)
    mod_gt, segm_gt = headlike_phantom(omega=omega, random_perturbation=randomness_shape, intensities=intensities)

    # B1) get roi mask (from the ground truth):
    roi_mask = segm_gt.astype(np.bool)

    # C) Create other modalities

    # -- invert intensities:
    mod_inv = 1 - mod_gt
    np.place(mod_inv, mod_inv == 1, 0)

    # -- add random noise: (25% of the randomness_noise for the background plus a gaussian filter of 50% size voxel)
    #     The noise granularity is fixed with parameters 10 and 3.
    noise_array = np.random.uniform(-10, 10, size=omega).astype(np.float64)
    noise_array = 0.25 * randomness_noise * fil.gaussian_filter(noise_array, 3)

    mod_gt_noise = mod_gt + noise_array

    noise_array = np.random.uniform(-10, 10, size=omega).astype(np.float64)
    noise_array = 0.25 * randomness_noise * fil.gaussian_filter(noise_array, 3)

    mod_inv_noise = mod_inv + noise_array

    # -- add hypo-intensities artefacts:
    num_artefacts_per_type = int(10 * randomness_noise / 2)
    noise_hypo = np.zeros_like(noise_array).astype(np.int32)
    for j in range(num_artefacts_per_type):
        random_radius = 0.05 * randomness_noise * np.min(omega) *np.random.randn()  # 5% of the min direction
        random_centre = [np.random.uniform(0 + random_radius, j - random_radius) for j in omega]

        noise_hypo = noise_hypo + sphere_shape(omega, random_centre, random_radius, foreground_intensity=1, dtype=np.int32)
    see_array(noise_hypo, block=True)

    noise_hypo = 1 - 1 * (noise_hypo.astype(np.bool))
    # filter the results:
    mod_gt_noise = fil.gaussian_filter(mod_gt_noise * noise_hypo, 0.5 * randomness_noise)
    mod_inv_noise = fil.gaussian_filter(mod_inv_noise * noise_hypo, 0.5 * randomness_noise)

    # D) Get the Registration Mask (based on):
    reg_mask = noise_hypo * roi_mask

    see_array([mod_gt, segm_gt, roi_mask.astype(np.int32), reg_mask.astype(np.int32), mod_gt_noise, mod_inv_noise])

    # E) save all in the data structure
    im_segm_gt  = nib.Nifti1Image(segm_gt, affine=np.eye(4))
    im_mod_gt   = nib.Nifti1Image(mod_gt, affine=np.eye(4))
    im_mod1     = nib.Nifti1Image(mod_gt_noise, affine=np.eye(4))
    im_mod2     = nib.Nifti1Image(mod_inv_noise, affine=np.eye(4))
    im_roi_mask = nib.Nifti1Image(roi_mask.astype(np.int32), affine=np.eye(4))
    im_reg_mask = nib.Nifti1Image(reg_mask.astype(np.int32), affine=np.eye(4))

    nib.save(im_segm_gt, jph(pfo_segm, '{}_segmGT.nii.gz'.format(atlas_name)))
    nib.save(im_mod_gt, jph(pfo_mod, '{}_modGT.nii.gz'.format(atlas_name)))
    nib.save(im_mod1, jph(pfo_mod, '{}_mod1.nii.gz'.format(atlas_name)))
    nib.save(im_mod2, jph(pfo_mod, '{}_mod2.nii.gz'.format(atlas_name)))
    nib.save(im_roi_mask, jph(pfo_masks, '{}_roi_mask.nii.gz'.format(atlas_name)))
    nib.save(im_reg_mask, jph(pfo_masks, '{}_reg_mask.nii.gz'.format(atlas_name)))


def generate_multi_atlas_at_folder(pfo_where_to_create_the_multi_atlas, number_of_subjects=10,
                                   multi_atlas_root_name='sj', randomness_shape=0.3, randomness_noise=0.4):
    """
    Generate a phatom multi atlas of head-like shapes.
    This is based on LABelsToolkit.tools.phantoms_generator.shapes_for_headlike_phantoms.headlike_phantom
    and uses .generate_atlas_at_folder to generate a single element.
    :param pfo_where_to_create_the_multi_atlas: path to file where the multi atlas structure will be saved.
    :param number_of_subjects: [10] how many subjects in the multi atlas
    :param multi_atlas_root_name: root name for the multi atlas
    :param randomness_shape: randomness in the geometry of the backgorund shape. Must be between 0 and 1.
    :param randomness_noise: randomness in the simulated noise signal and artefacts. Must be between 0 and 1.
    :return:
    """
    print_and_run('mkdir -p {}'.format(pfo_where_to_create_the_multi_atlas))

    for sj in range(number_of_subjects):

        # A) Generate folder structure
        sj_name = multi_atlas_root_name + str(sj).zfill(len(str(number_of_subjects)) + 1)

        print('Creating atlas {0} ({1}/{2})'.format(sj_name, sj+1, number_of_subjects))
        generate_atlas_at_folder(jph(pfo_where_to_create_the_multi_atlas, sj_name), atlas_name=sj_name,
                                 randomness_shape=randomness_shape, randomness_noise=randomness_noise)



if __name__ == '__main__':
    omega = (80, 90, 80)
    # noise1 = np.random.choice(range(10), size=omega).astype(np.float64)
    # noise2 = fil.gaussian_filter(noise1, 5)
    # print np.mean(noise2)
    # print np.std(noise2)
    # noise3 = noise2 / (np.mean(noise2) + 2 * np.std(noise2))
    # see_array([noise1, noise2, noise3])

    # noise_array = np.random.uniform(-10, 10, size=omega).astype(np.float64)
    # noise_array = 0.2 * 0.4 * fil.gaussian_filter(noise_array, 3)
    # see_array(noise_array)
    #
    generate_atlas_at_folder('/Users/sebastiano/Desktop/z_test_phantom_atlas', randomness_noise=1)
