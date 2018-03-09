import os
from os.path import join as jph
import nibabel as nib

import numpy as np
from scipy.optimize import minimize

from LABelsToolkit.tools.aux_methods.utils import print_and_run


# WARNING: to be tested!


class ICV_estimator(object):
    """
    ICV estimation as in
    Iglesias JE, Ferraris S, Modat M, Gsell W, Deprest J, van der Merwe JL, Vercauteren T: "Template-free estimation of
    intracranial volume: a preterm birth animal model study", MICCAI workshop: Fetal and Infant Image Analysis, 2017.
    Note: not the same code used to produce the paper
    """
    def __init__(self, pfi_list_subjects_to_coregister, pfo_output, S=None, m=None,
                      n=0.001, a=0.001, b=0.1, alpha=0.001, beta=0.1):
        # input subjects
        self.pfi_list_subjects_to_coregister = pfi_list_subjects_to_coregister
        self.pfo_output = pfo_output
        self.num_subjects = len(pfi_list_subjects_to_coregister)
        self.subjects_id = None
        # optimisation function parameters
        self.S = S
        self.m = m
        self.n = n
        self.a = a
        self.b = b
        self.alpha = alpha
        self.beta = beta
        # graph connection is complete by default. Have to edit this directly.
        self.graph_connections = [[i, j] for i in xrange(self.num_subjects) for j in xrange(i+1, self.num_subjects)]
        # folder structure
        self.pfo_warped = jph(self.pfo_output, 'warped')
        self.pfo_transformations = jph(self.pfo_output, 'transformations')
        # run initialisations
        self.__initialise_list_id__()

    def __initialise_list_id__(self):
        self.subjects_id = [os.path.dirname(s).split('.')[0]
                            for s in self.pfi_list_subjects_to_coregister
                            if (s.endswith('.nii') or s.endswith('.nii.gz'))]

    def generate_transformations(self):

        cmd_1 = 'mkdir -p {0} '.format(self.pfo_warped)
        cmd_2 = 'mkdir -p {0} '.format(self.pfo_transformations)

        print_and_run(cmd_1)
        print_and_run(cmd_2)

        for i in [c[0] for c in self.graph_connections]:
            for j in [c[1] for c in self.graph_connections]:
                fname_i_j = self.subjects_id[i] + '_' + self.subjects_id[j]
                fname_j_i = self.subjects_id[j] + '_' + self.subjects_id[i]
                pfi_aff_i_j = jph(self.pfo_transformations, fname_i_j + '.txt')
                pfi_res_i_j = jph(self.pfo_warped, fname_i_j + '.nii.gz')
                pfi_aff_j_i = jph(self.pfo_transformations, fname_j_i + '.txt')
                pfi_res_j_i = jph(self.pfo_warped, fname_j_i + '.nii.gz')

                cmd_reg_i_j = 'reg_aladin -ref {0} -flo {1} -aff {2} -res {3} -speeeeed '.format(
                            self.pfi_list_subjects_to_coregister[i], self.pfi_list_subjects_to_coregister[j],
                            pfi_aff_i_j, pfi_res_i_j)
                cmd_reg_j_i = 'reg_aladin -ref {0} -flo {1} -aff {2} -res {3} -speeeeed '.format(
                            self.pfi_list_subjects_to_coregister[j], self.pfi_list_subjects_to_coregister[i],
                            pfi_aff_j_i, pfi_res_j_i)

                print_and_run(cmd_reg_i_j)
                print_and_run(cmd_reg_j_i)

    def compute_S(self):

        if not os.path.exists(self.pfo_transformations):
            msg = "Folder {} not created. Did you run generate_transformations first?".format(self.pfo_transformations)
            raise IOError(msg)

        S = np.zeros(2, 2)

        for i in xrange(self.num_subjects):
            for j in xrange(i+1, self.num_subjects):

                pfi_aff_i_j = jph(self.pfo_transformations,
                                  self.subjects_id[i] + '_' + self.subjects_id[j] + '.txt')
                pfi_aff_j_i = jph(self.pfo_transformations,
                                  self.subjects_id[j] + '_' + self.subjects_id[i] + '.txt')

                S[i, j] = np.linalg.det(np.loadtxt(pfi_aff_i_j))
                S[j, i] = np.linalg.det(np.loadtxt(pfi_aff_j_i))

        self.S = S

    def compute_m_from_list_masks(self, pfi_list_brain_masks, increase_volume_estimate=0.05):
        # compute m from a list of propagated segmentation of the brain.
        # the icv is the mean volume of some propagated segmentation, whose path is stored in the list
        # pfi_list_brain_mask
        sum_vol = 0.
        for p in pfi_list_brain_masks:
            im_nib = nib.load(p)
            one_voxel_volume = np.round(np.abs(np.prod(np.diag(im_nib.get_affine()))), decimals=6)  # (in mm)
            sum_vol += np.count_nonzero(im_nib.get_data()) * one_voxel_volume

        mean_vol_estimate = (sum_vol / float(len(pfi_list_brain_masks))) * (1 + increase_volume_estimate)
        self.m = mean_vol_estimate

    def icv_estimator(self):

        assert self.S.shape[0] == self.S.shape[1]

        log_estimate_v = self.m * np.ones(self.S.shape[0], dtype=np.float64)

        def cost(v, S, m, n, a, b, alpha, beta):

            sum_abs_log_diff = 0
            for i in xrange(len(v)):
                for j in xrange(i+1, len(v)):
                    sum_abs_log_diff += np.abs(S[i, j] - v[i] + v[j])
            mean_v = np.mean(v)
            N = S.shape[0]

            a1 = alpha + np.linalg.det(S)
            a2 = np.log(beta + sum_abs_log_diff)
            a3 = (2 * a + N) / float(2)
            a4 = np.log(b + 0.5 * np.sum([(v_i + mean_v) ** 2 for v_i in list(v)]) +
                        (N * n * (mean_v - m) ** 2) / (2 * (N + n)))
            return a1 * a2 + a3 * a4

        init_values = np.array([log_estimate_v, self.S, self.m, self.n, self.a, self.b, self.alpha, self.beta])
        log_answer = minimize(cost, init_values, method='trust-ncg', tol=1e-6)

        return np.exp(log_answer)
