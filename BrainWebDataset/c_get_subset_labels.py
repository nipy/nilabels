"""
Create subset data from the BrainWeb database after nifti conversion.

THE SUM of the fuzzy segmentations IS NOT ALWAYS 1 (=255) (thanks BrainWeb...!)
Remove the excedence from the higher value in the stack.
go to Float between zero and 1 and then renormalize.
"""
import os
import numpy as np
import nibabel as nib

import BrainWebDataset.a_params as bw

from LABelsToolkit.main import LABelsToolkit as LaB


def get_selection_dataset(subset, get_other_tissue_probability=True, keep_fuzzy_uint8=False):
    """
    From the dataset converted, it will provide a new dataset with a selection of segmentations.
    It will provide 3 images in the subject folder:
    > T1
    > CRIPS
    > 4d volumes with the FUZZY probability of the required subset and the other tissues in one label if required.
    :param subset: selection of segmentation as a sublist of names_tissues
    :param get_other_tissue_probability: [True] will merge all the other tissue in one label and provide this tissue
    probability as well as the others in the stack.
    :param keep_fuzzy_uint8: [False] if True fuzzy will be between 0 and 255, if False, result will be between 0 and 1.
    :return:
    """

    other_tissue = set(bw.names_tissues) - set(subset + ['bck'])

    old_labels = [0, ] + [bw.names_tissues.index(k) for k in subset] + [bw.names_tissues.index(k) for k in other_tissue]
    new_labels = [0, ] + list(range(1, len(subset) + 1))
    labels_mask = [0, ] + [1, ] * len(subset) + [0, ] * len(other_tissue)

    if get_other_tissue_probability:
        labels_plus_one = np.max(new_labels) + 1
        new_labels = new_labels + [labels_plus_one, ] * len(other_tissue)
    else:
        new_labels = new_labels + [0, ] * len(other_tissue)

    os.system('mkdir {}'.format(bw.pfo_data))

    for sj in bw.subjects_num_list:

        print('Subject {}'.format(sj))

        pfo_sj_nifti = os.path.join(bw.pfo_nifti_in_root, 'BW{}'.format(sj))

        pfo_sj_nifti_new = os.path.join(bw.pfo_data, 'BW{}'.format(sj))

        # os.system('mkdir {}'.format(pfo_sj_nifti_new))

        pfi_nifti_T1    = os.path.join(pfo_sj_nifti, 'BW{}_{}.nii.gz'.format(sj, bw.name_T1.upper()))
        pfi_nifti_CRISP = os.path.join(pfo_sj_nifti, 'BW{}_{}.nii.gz'.format(sj, bw.name_crisp.upper()))

        pfi_nifti_T1_new    = os.path.join(pfo_sj_nifti_new, 'BW{}_{}.nii.gz'.format(sj, bw.name_T1.upper()))
        pfi_nifti_CRISP_new = os.path.join(pfo_sj_nifti_new, 'BW{}_{}.nii.gz'.format(sj, bw.name_crisp.upper()))

        os.system('cp {} {}'.format(pfi_nifti_T1, pfi_nifti_T1_new))
        os.system('cp {} {}'.format(pfi_nifti_CRISP, pfi_nifti_CRISP_new))

        lab = LaB()
        lab.manipulate_labels.relabel(pfi_nifti_CRISP_new, pfi_nifti_CRISP_new, old_labels, new_labels)

        pfi_nifti_MASK = os.path.join(pfo_sj_nifti_new, 'BW{}_MASK.nii.gz'.format(sj))

        os.system('cp {} {}'.format(pfi_nifti_CRISP_new, pfi_nifti_MASK))

        lab.manipulate_labels.relabel(pfi_nifti_MASK, pfi_nifti_MASK, new_labels, labels_mask)

        list_nib_fuzzy_segm = []

        for s in subset:
            pfi_output_nifti_fuzzy = os.path.join(pfo_sj_nifti, 'BW{}_{}.nii.gz'.format(sj, s))
            list_nib_fuzzy_segm.append(nib.load(pfi_output_nifti_fuzzy))

        if get_other_tissue_probability:
            array_other_tissue = np.zeros_like(list_nib_fuzzy_segm[0].get_data(), dtype=np.int32)
            for s_other in other_tissue:
                pfi_output_nifti_fuzzy = os.path.join(pfo_sj_nifti, 'BW{}_{}.nii.gz'.format(sj, s_other))
                array_other_tissue += nib.load(pfi_output_nifti_fuzzy).get_data()

        sum_fuzzy = np.zeros_like(list_nib_fuzzy_segm[0].get_data(), dtype=np.int32)

        for a in list_nib_fuzzy_segm:
            sum_fuzzy += a.get_data()

        if get_other_tissue_probability:
            sum_fuzzy += array_other_tissue

        if keep_fuzzy_uint8:

            if get_other_tissue_probability:
                new_data_no_bkg = np.stack([a.get_data().astype(np.uint16) for a in list_nib_fuzzy_segm] + [array_other_tissue], axis=3)
            else:
                new_data_no_bkg = np.stack([a.get_data().astype(np.uint16) for a in list_nib_fuzzy_segm], axis=3)
            sum_probabilities = np.sum(new_data_no_bkg, axis=3)
            excedence_matrix = np.zeros_like(new_data_no_bkg)
            X, Y, Z = np.where(sum_probabilities > 255)
            for x, y, z in zip(X, Y, Z):
                pos_max = np.argmax(new_data_no_bkg[x, y, z, :])
                excedence_matrix[x, y, z, pos_max] = sum_probabilities[x, y, z] - 255

            new_data_no_bkg = new_data_no_bkg - excedence_matrix

            sum_fuzzy = np.sum(new_data_no_bkg, axis=3)

            assert np.max(sum_fuzzy) <= 255

            bkg_data = 255 - sum_fuzzy

            new_data = np.stack([bkg_data] + [new_data_no_bkg[...,p] for p in range(len(subset))], axis=3).astype(np.uint8)

            new_image = nib.Nifti1Image(new_data, list_nib_fuzzy_segm[0].affine, header=list_nib_fuzzy_segm[0].header)

            pfi_nifti_fuzzy_new = os.path.join(pfo_sj_nifti_new, 'BW{}_FUZZY.nii.gz'.format(sj))
            nib.save(new_image, pfi_nifti_fuzzy_new)

        else:

            excedence_matrix = (sum_fuzzy > 255) * (sum_fuzzy - 255)

            normaliser_single_slice = 255 * np.ones_like(sum_fuzzy) + excedence_matrix
            if get_other_tissue_probability:
                new_data_no_bkg = np.nan_to_num(np.stack([a.get_data().astype(np.float64)/normaliser_single_slice for a in list_nib_fuzzy_segm] + [array_other_tissue.astype(np.float64)/normaliser_single_slice], axis=3))
            else:
                new_data_no_bkg = np.nan_to_num(np.stack([a.get_data().astype(np.float64) / normaliser_single_slice for a in list_nib_fuzzy_segm], axis=3))

            assert np.max(new_data_no_bkg) <= 1.0
            assert np.min(new_data_no_bkg) >= .0

            bkg_data = 1 - np.sum(new_data_no_bkg, axis=3)
            new_data = np.stack([bkg_data] + [new_data_no_bkg[..., p] for p in range(new_data_no_bkg.shape[-1])], axis=3).astype(
                np.float64)

            s = np.sum(new_data, axis=3)
            np.testing.assert_array_almost_equal(s, np.ones_like(s), decimal=10)

            new_image = nib.Nifti1Image(new_data, list_nib_fuzzy_segm[0].affine)

            pfi_nifti_fuzzy_new = os.path.join(pfo_sj_nifti_new, 'BW{}_FUZZY.nii.gz'.format(sj))
            nib.save(new_image, pfi_nifti_fuzzy_new)


if __name__ == '__main__':
    get_selection_dataset(['wm', 'gm', 'csf'], get_other_tissue_probability=True, keep_fuzzy_uint8=False)
