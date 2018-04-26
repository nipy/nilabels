"""
BrainWeb Dataset:
From the downloaded data to the one resampled in the space of the T1.
Date of the dataset download 26-3-2018
---
Based on Numpy, Nibabel, nifti-reg and unix command gunzip.
"""
import numpy as np
import nibabel as nib
import os

import BrainWebDataset.a_params as bw
from LABelsToolkit.tools.aux_methods.utils_rotations import basic_90_rot_ax, axial_90_rotations


def data_to_nifti(pfi_data, pfi_output_nifti, resolution=1.0, shift_intensities=True, swap_ones_with_zeros=False):
    """
    Here the parameters for the conversion (header)
    :param pfi_data:
    :param pfi_output_nifti:
    :param resolution: can be 1 or 0.5
    :param shift_intensities:
    :param swap_ones_with_zeros: vessel probability is 1/255 even outside the skull. This is not consistent with other
     labels. To solve this feature, the flag set the probability to 0/255 where it is 1/255.
    :return:
    """
    print('\nConverting : {} \ninto       : {}'.format(pfi_data, pfi_output_nifti))

    data_type = np.uint8

    if resolution == 1.0:
        shape = (181, 256, 256)
        aff = np.diag([1.0, ] * 3 + [1])
        aff[:-1, 3] = np.array([-127.75, -145.75, -72.25])
    elif resolution == 0.5:
        shape = (362, 434, 362)
        aff = np.diag([0.5, ] * 3 + [1])
        aff[:-1, 3] = np.array([- 90.25, - 126.25, -72.25])
    else:
        raise IOError

    m = np.fromfile(pfi_data, dtype=data_type).reshape(shape)
    m = axial_90_rotations(m, 1, 1)
    m = m[::-1, :, :]
    if shift_intensities:
        m = (m - 128) % 256
    if swap_ones_with_zeros:
        places = m == 1
        assert isinstance(places, np.ndarray)
        if np.any(places):
            np.place(m, places, 0)

    m_nib = nib.Nifti1Image(m, affine=aff)
    m_nib.set_data_dtype(data_type)
    nib.save(m_nib, pfi_output_nifti)


def resample_with_identity(pfi_reference, pfi_floating, pfi_output, pfo_tmp):
    """
    Use nifty-reg to resample the segmentations in the same space
    :param pfi_reference: path to file of the reference
    :param pfi_floating: path to flowting of the reference
    :param pfi_output: path to file output
    :param pfo_tmp: Temporary file
    :return:
    """
    np.savetxt(os.path.join(pfo_tmp, 'id.txt'), np.eye(4))
    cmd = 'reg_resample -ref {} -flo {} -trans {} -res {}'.format(
        pfi_reference, pfi_floating, os.path.join(pfo_tmp, 'id.txt'), pfi_output)
    print(cmd)
    os.system(cmd)


def converter(pfo_raw, pfo_nifti, pfo_tmp):
    """
    Create the folder structure, from raw (where all the 20 subjects x 14 files had been downloaded)
    to the nifti folder where the nifti files are divided into subfolders.
    :param pfo_raw: path to raw folder
    :param pfo_nifti: path to nifti folder
    :param pfo_tmp: path to temporary file
    :return:
    """

    assert os.path.exists(pfo_raw)
    assert os.path.exists(pfo_nifti)

    cmd = 'mkdir {}'.format(pfo_tmp)
    os.system(cmd)

    # -- unzip raw --

    print('Unzipping all, may take a while...')

    cmd = 'cp {0}/subject* {1}'.format(pfo_raw, pfo_tmp)
    os.system(cmd)

    cmd = 'gunzip {0}/subject*'.format(pfo_tmp)
    os.system(cmd)

    # -- Convert and save nifti for each subject --

    for sj in bw.subjects_num_list:

        print('\n\nSubject {}'.format(sj))

        sj_name = 'BW{}'.format(sj)
        pfo_sj_nifti = os.path.join(pfo_nifti, sj_name)

        os.system('mkdir {}'.format(pfo_sj_nifti))

        # -- Convert T1 --

        pfi_raw_T1 = os.path.join(pfo_tmp, 'subject{}_{}_{}.rawb'.format(sj, bw.name_T1, bw.suffix_t1))
        assert os.path.exists(pfi_raw_T1), pfi_raw_T1

        pfi_nifti_T1 = os.path.join(pfo_sj_nifti, 'BW{}_{}.nii.gz'.format(sj, bw.name_T1.upper()))
        data_to_nifti(pfi_raw_T1, pfi_nifti_T1, resolution=1.0, shift_intensities=False)

        # -- Convert crisp --

        pfi_raw_CRISP = os.path.join(pfo_tmp, 'subject{}_{}_{}.rawb'.format(sj, bw.name_crisp, bw.suffix_crisp))
        assert os.path.exists(pfi_raw_CRISP), pfi_raw_CRISP

        pfi_nifti_CRISP_segm_space  = os.path.join(pfo_tmp, 'BW{}_{}.nii.gz'.format(sj, bw.name_crisp.upper()))
        data_to_nifti(pfi_raw_CRISP, pfi_nifti_CRISP_segm_space, resolution=0.5, shift_intensities=False)

        pfi_nifti_CRISP_final = os.path.join(pfo_sj_nifti, 'BW{}_{}.nii.gz'.format(sj, bw.name_crisp.upper()))
        resample_with_identity(pfi_nifti_T1, pfi_nifti_CRISP_segm_space, pfi_nifti_CRISP_final, pfo_tmp)

        # -- Convert segmentations --

        for tt in bw.names_tissues:
            pfi_raw = os.path.join(pfo_tmp, 'subject{}_{}_{}.rawb'.format(sj, tt, bw.suffix_tissues))
            swap_ones_with_zeros = False
            if tt == 'vessels':
                pfi_raw = os.path.join(pfo_tmp, 'subject{}_{}.rawb'.format(sj, tt))
                swap_ones_with_zeros = True

            assert os.path.exists(pfi_raw), pfi_raw

            pfi_output_nifti_segm_space = os.path.join(pfo_tmp, 'BW{}_{}.nii.gz'.format(sj, tt))
            data_to_nifti(pfi_raw, pfi_output_nifti_segm_space, resolution=0.5, shift_intensities=True, swap_ones_with_zeros=swap_ones_with_zeros)

            pfi_output_nifti_final = os.path.join(pfo_sj_nifti, 'BW{}_{}.nii.gz'.format(sj, tt))
            resample_with_identity(pfi_nifti_T1, pfi_output_nifti_segm_space, pfi_output_nifti_final, pfo_tmp)

    # -- Clean --

    cmd = 'rm -r {}'.format(pfo_tmp)
    os.system(cmd)


if __name__ == '__main__':

    # create the conversion, may take some minutes - Paths are stored in the a_params python module
    converter(bw.pfo_raw_in_root, bw.pfo_nifti_in_root, bw.pfo_tmp_in_root)

    # Add the label descriptor for ITK-Snap:
    pfi_label_descriptor = os.path.join(bw.pfo_root, 'labels_descriptor.txt')
    text_file = open(pfi_label_descriptor, "w")
    text_file.write(bw.lab_desc)
    text_file.close()
