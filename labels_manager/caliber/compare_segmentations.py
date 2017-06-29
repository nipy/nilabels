"""
Module based on nifty path to file due to the need of performing registration on the data.
--------
Notes on how to get SVFs from NiftyReg.

SVF are obtained from NiftyReg as follows reg_f3d with command -vel, returning the corresponding cpp grid as the
control point grid we are interested in.
The dense vector field that corresponds to the given gpp grid is then provided with -flow and it
is obtained in 'deformation coordinates' (Eulerian coordinate system).
To have it in displacement coordinate system (Lagrangian coordinate system) for our elaboration we need to
subtract them the identity with python (not with - disp in niftyReg, otherwise it will be exponentiated again).
"""
import numpy as np
import scipy
import nibabel as nib
import os
from os.path import join as jph
import pandas as pd

from labels_manager.tools.descriptions.manipulate_descriptors import LabelsDescriptorManager



def get_dice_score(pfi_binary_image1, pfi_binary_image2):
    """
    Dice score between 2 binary segmentation. If not binary, they are binarized.
    :param pfi_binary_image1: first image
    :param pfi_binary_image2: second image
    :return: dice score between the two images
    """
    im1 = nib.load(pfi_binary_image1)
    im2 = nib.load(pfi_binary_image2)
    assert im1.shape == im2.shape
    card_im1 = np.count_nonzero(im1.get_data())
    card_im2 = np.count_nonzero(im2.get_data())
    card_im1_intersect_im2 = np.count_nonzero(im1.get_data() * im2.get_data())

    return 2 * card_im1_intersect_im2  / float(card_im1 + card_im2)


def get_dispersion(pfi_binary_image1, pfi_binary_image2, pfo_intermediate_files, tag=0):
    pfi_warp_aff = jph(pfo_intermediate_files, 'dispersion_aff_warped_interp0_' + str(tag) +'.nii.gz')
    pfi_transf_aff = jph(pfo_intermediate_files, 'dispersion_aff_transf_interp0_' + str(tag) +'.txt')
    cmd1 = 'reg_aladin -ref {0} -flo {1} -res {2} -aff {3} -interp 0'.format(pfi_binary_image1,
                                                                             pfi_binary_image2,
                                                                             pfi_warp_aff,
                                                                             pfi_transf_aff)
    os.system(cmd1)
    pfi_warp_nrig = jph(pfo_intermediate_files, 'dispersion_aff_warped_interp0_' + str(tag) +'.nii.gz')
    pfi_svf = jph(pfo_intermediate_files, 'dispersion_aff_transf_interp0' + str(tag) +'.nii.gz')
    cmd2 = 'reg_f3d -ref {0} -flo {1} -res {2} -aff {3} -interp 0 -vel'.format(pfi_binary_image1,
                                                                               pfi_warp_aff,
                                                                               pfi_warp_nrig,
                                                                               pfi_svf)
    os.system(cmd2)
    # get the dense SVF - tests to be made
    # cmd3 = 'reg_transform -disp {0} {1}'.format(pfi_svf, )
    svf = nib.load(pfi_svf)
    norms = scipy.linalg.norm(svf.get_data(), axis=4)
    return np.median(norms.flatten())


def get_precision(pfi_binary_image1, pfi_binary_image2, pfo_intermediate_files, tag=0):
    pfi_warp_aff = jph(pfo_intermediate_files, 'dispersion_aff_warped_interp0_' + str(tag) +'.nii.gz')
    pfi_transf_aff = jph(pfo_intermediate_files, 'dispersion_aff_transf_interp0_' + str(tag) +'.txt')
    if not os.path.exists(pfi_transf_aff):
        cmd1 = 'reg_aladin -ref {0} -flo {1} -res {2} -aff {3} -interp 0'.format(pfi_binary_image1,
                                                                                 pfi_binary_image2,
                                                                                 pfi_warp_aff,
                                                                                 pfi_transf_aff)
        os.system(cmd1)
    t = np.loadtxt(pfi_transf_aff)
    return np.abs(np.linalg.det(t))


def get_errors_data_frame(pfi_segm1, pfi_segm2, pfo_intermediate_files, pfi_label_descriptor,
                          pfi_output_table=None, erase_intermediate=False, tag=0):

    im1 = nib.load(pfi_segm1)
    im2 = nib.load(pfi_segm2)
    assert im1.shape == im2.shape
    list_labels_1 = sorted(list(set(im1.get_data().flat)))
    list_labels_2 = sorted(list(set(im2.get_data().flat)))
    assert list_labels_1 == list_labels_2

    ldm = LabelsDescriptorManager(pfi_label_descriptor)
    multi_label = ldm.get_multi_label_dict()

    # create data-frame to fill:
    regions = multi_label.keys()
    s_dice  = pd.Series(np.zeros(len(regions)), index=regions)
    s_dispe = pd.Series(np.zeros(len(regions)), index=regions)
    s_prec  = pd.Series(np.zeros(len(regions)), index=regions)

    # split images in binarised segmentations components.