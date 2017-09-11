import numpy as np
import pandas as pa
import os
import nibabel as nib
from scipy import ndimage as nd

from labels_manager.tools.image_colors_manipulations.relabeller import keep_only_one_label
from labels_manager.tools.aux_methods.utils_nib import set_new_data
from labels_manager.tools.aux_methods.utils import print_and_run


def lncc_distance(values_patch1, values_patch2):
    """
    Import values below the patches, containing the same number of eolem
    :param values_patch1:
    :param values_patch2:
    :return:
    """
    patches = [values_patch1.flatten(), values_patch2.flatten()]
    np.testing.assert_array_equal(patches[0].shape, patches[1].shape)

    for index_p, p in enumerate(patches):
        den = float(np.linalg.norm(p))
        if den == 0: patches[index_p] = np.zeros_like(p)
        else: patches[index_p] = patches[index_p] / den

    return patches[0].dot(patches[1])


def centroid_array(arr, labels):
    centers_of_mass = [np.array([0, 0, 0])] * len(labels)
    for l_id, l in enumerate(labels):
        coordinates_l = np.where(arr == l)  # returns [X_vector, Y_vector, Z_vector]
        centers_of_mass[l_id] = (1 / float(len(coordinates_l[0]))) * np.array([np.sum(k) for k in coordinates_l])
    return centers_of_mass


def centroid(im, labels, return_mm3=True):
    """
    Centroid (center of mass, baricenter) of a list of labels.
    :param im:
    :param labels: list of labels, e.g. [3] or [2, 3, 45]
    :param return_mm3: if true the answer is in mm if false in voxel indexes.
    :return: list of centroids, one for each label in the input order.
    """
    centers_of_mass = centroid_array(im.get_data(), labels)

    if return_mm3:
        centers_of_mass = [im.affine[:3, :3].dot(cm.astype(np.float64)) for cm in centers_of_mass]
    else:
        centers_of_mass = [np.round(cm).astype(np.uint64) for cm in centers_of_mass]
    return centers_of_mass


def covariance_matrices(im, labels, return_mm3=True, subtract_mean=True):
    """
    Considers the label as a point distribution in the space, and returns the covariance matrix of the points
    distributions.
    :param im: input nibabel image
    :param labels: list of labels input.
    :param return_mm3: if true the answer is in mm if false in voxel indexes.
    :return: covariance matrix of the point distribution of the label
    """
    cov_matrices = [np.zeros([3, 3])] * len(labels)
    for l_id, l in enumerate(labels):
        coords = np.where(im.get_data() == l)  # returns [X_vector, Y_vector, Z_vector]
        if subtract_mean:
            coords = [coords[j] - np.mean(coords[j]) for j in range(3)]
        cov_matrices[l_id] = np.cov(coords)
    if return_mm3:
        cov_matrices = [im.affine[:3, :3].dot(cm.astype(np.float64)) for cm in cov_matrices]

    return cov_matrices

# --- distances

def dice_score(im_segm1, im_segm2, labels_list, labels_names, verbose=0):
    """
    Dice score between paired labels of segmentations.
    :param im_segm1: nibabel image with labels
    :param im_segm2: nibabel image with labels
    :param labels_list:
    :param labels_names:
    :return: dice score of the label label of the two segmentations.
    """
    def dice_score_l(lab):
        place1 = im_segm1.get_data() == lab  # slow but readable, can be refactored later.
        place2 = im_segm2.get_data() == lab
        assert isinstance(place1, np.ndarray)
        assert isinstance(place2, np.ndarray)
        return 2 * np.count_nonzero(place1 * place2) / float(np.count_nonzero(place1) + np.count_nonzero(place2))

    scores = []
    for l in labels_list:
        d = dice_score_l(l)
        scores.append(d)
        if verbose > 0:
            print('Dice scores label {0} : {1} '.format(l, d))

    return pa.Series(scores, index=labels_names)


def dispersion(im_segm1, im_segm2, labels_list, labels_names, return_mm3=True, verbose=0):

    # def dispersion_l(lab, verbose=verbose):
    #     c1 = centroid_array(im_segm1.get_data(), labels=[lab,])[0]
    #     c2 = centroid_array(im_segm2.get_data(), labels=[lab,])[0]
    #     if return_mm3:
    #         c1 = im_segm1.affine[:3, :3].dot(c1.astype(np.float64))
    #         c2 = im_segm1.affine[:3, :3].dot(c2.astype(np.float64))
    #     d = np.sqrt( sum((c1 - c2)**2) )
    #     if verbose > 0:
    #         print('Dispersion, label {0} : {1}'.format(l, d))
    #     return d
    #
    # np.testing.assert_array_almost_equal(im_segm1.affine, im_segm2.affine)
    # return pa.Series(np.array([dispersion_l(l) for l in labels_list]), index=labels_names)

    cs1 = centroid_array(im_segm1.get_data(), labels=labels_list)
    cs2 = centroid_array(im_segm2.get_data(), labels=labels_list)
    if return_mm3:
        cs1 = [im_segm1.affine[:3, :3].dot(c.astype(np.float64)) for c in cs1]
        cs2 = [im_segm1.affine[:3, :3].dot(c.astype(np.float64)) for c in cs2]

    np.testing.assert_array_almost_equal(im_segm1.affine, im_segm2.affine)
    return pa.Series(np.array([np.sqrt( sum((c1 - c2)**2) ) for c1, c2 in zip(cs1, cs2)]), index=labels_names)


def precision(im_segm1, im_segm2, pfo_intermediate_files, labels_list, labels_names, verbose=0):
    # deprecated as too imprecise :-) !
    print_and_run('mkdir -p {}'.format(pfo_intermediate_files))
    precision_per_label = []

    for l in labels_list:
        im_segm_single1 = set_new_data(im_segm1, keep_only_one_label(im_segm1.get_data(), l))
        im_segm_single2 = set_new_data(im_segm2, keep_only_one_label(im_segm2.get_data(), l))

        pfi_im_segm_single1 = os.path.join(pfo_intermediate_files, 'im_segm_single_lab{}_1.nii.gz'.format(l))
        pfi_im_segm_single2 = os.path.join(pfo_intermediate_files, 'im_segm_single_lab{}_2.nii.gz'.format(l))

        nib.save(im_segm_single1, pfi_im_segm_single1)
        nib.save(im_segm_single2, pfi_im_segm_single2)

        pfi_warp_1_2 = os.path.join(pfo_intermediate_files, 'warped_single_lab{}_1_on_2.nii.gz'.format(l))
        pfi_warp_2_1 = os.path.join(pfo_intermediate_files, 'warped_single_lab{}_2_on_1.nii.gz'.format(l))
        pfi_aff_1_2 = os.path.join(pfo_intermediate_files, 'aff_single_lab{}_1_on_2.txt'.format(l))
        pfi_aff_2_1 = os.path.join(pfo_intermediate_files, 'warped_single_lab{}_2_on_1.txt'.format(l))

        cmd1_2 = 'reg_aladin -ref {0} -flo {1} -res {2} -aff {3} -interp 0 -speeeeed'.format(pfi_im_segm_single1,
                                                                                             pfi_im_segm_single2,
                                                                                             pfi_warp_1_2,
                                                                                             pfi_aff_1_2)
        cmd2_1 = 'reg_aladin -ref {0} -flo {1} -res {2} -aff {3} -interp 0 -speeeeed'.format(pfi_im_segm_single2,
                                                                                             pfi_im_segm_single1,
                                                                                             pfi_warp_2_1,
                                                                                             pfi_aff_2_1)
        print_and_run(cmd1_2)
        print_and_run(cmd2_1)

        t_1_2 = np.loadtxt(pfi_aff_1_2)
        t_2_1 = np.loadtxt(pfi_aff_2_1)

        d = 1 / float(np.max([np.abs(np.linalg.det(t_1_2)) ** (-1), np.abs(np.linalg.det(t_2_1)) ** (-1)]))
        precision_per_label.append(d)
        if verbose > 0:
            print('------------------------------------')
            print('Precision, label {0} : {1}'.format(l, d))
            print('------------------------------------')
    pa_series = pa.Series(np.array(precision_per_label), index=labels_names)
    pa_series.to_pickle(os.path.join(pfo_intermediate_files, 'final_precision.pickle'))

    return pa_series


def covariance_distance(im_segm1, im_segm2, labels_list, labels_names, return_mm3=True):
    """
    Considers the label as a point distribution in the space, and returns the covariance matrix of the points
    distributions.
    :return:
    See: Herdin 2005, Correlation matrix distance, a meaningful measure for evaluation of non-stationary MIMO channels
    """
    def cov_dist(m1, m2):
        """
        Covariance distance between matrices m1 and m2, defined as
        d = 1 - (trace(m1 * m2)) / (norm_fro(m1) + norm_fro(m2))
        :param m1: matrix
        :param m2: matrix
        :return: 1 - (np.trace(m1.dot(m2))) / (np.linalg.norm(m1) + np.linalg.norm(m2))
        """
        return 1 - (np.trace(m1.dot(m2)) / (np.linalg.norm(m1, ord='fro') + np.linalg.norm(m2, ord='fro')))

    cvs1 = covariance_matrices(im_segm1, labels=labels_list, return_mm3=return_mm3)
    cvs2 = covariance_matrices(im_segm2, labels=labels_list, return_mm3=return_mm3)

    return pa.Series(np.array([cov_dist(a1, a2) for a1, a2 in zip(cvs1, cvs2)]), index=labels_names)


def hausdorff_distance(im_segm1, im_segm2, labels_list, labels_names, return_mm3=True):
    """
    From 2 segmentations sampled in overlapping grids (with affine in starndard form) it returns the hausdoroff
    distance for each label in the labels list and list names it returns the pandas series with the corresponding
    distances for each label.
    :param im_segm1:
    :param im_segm2:
    :param labels_list:
    :param labels_names:
    :param return_mm3:
    :return:
    """
    def d_H(im1, im2, l):
        arr1 = im1.get_data() == l
        arr2 = im2.get_data() == l
        # df1 = scipy.ndimage.morphology.distance_transform_edt(arr1)
        assert isinstance(arr2, np.ndarray)
        if return_mm3:
            df2 = nd.morphology.distance_transform_edt(arr2, sampling=list(np.diag(im1.array[:3, :3])))
        else:
            df2 = nd.morphology.distance_transform_edt(arr2, sampling=None)
        assert isinstance(df2, np.ndarray)
        d = np.max(df2 * arr1)
        return d

    return pa.Series(
        np.array([np.max([d_H(im_segm1, im_segm2, l), d_H(im_segm2, im_segm1, l)]) for l in labels_list]),
        index=labels_names)


def box_sides_length(im, labels_list, labels_names, return_mm3=True):
    """
    im_segmentation, labels_to_box, labels_to_box_names
    Box surrounding the label in the list labels_to_box.
    A box is an (ordered!) couple of 3d points.
    :param im: sampled on an orthogonal grid.
    :param labels_list:
    :param labels_names:
    :return:
    """
    def get_box_length_single_label(arr, l, scaling_factors):
        coords = np.where(arr == l)  # returns [X_vector, Y_vector, Z_vector]
        dists = [np.abs(np.max(k) - np.min(k)) for k in coords]
        if return_mm3:
            dists = [d * dd for d, dd in zip(coords, scaling_factors)]
        return dists

    return pa.Series(
        np.array([get_box_length_single_label(im.get_data(), l, np.diag(im.affine)) for l in labels_list]),
        index=labels_names)



