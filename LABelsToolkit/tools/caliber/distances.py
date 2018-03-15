import numpy as np
import pandas as pa
import os
import nibabel as nib
from scipy import ndimage as nd

from LABelsToolkit.tools.image_colors_manipulations.relabeller import keep_only_one_label
from LABelsToolkit.tools.aux_methods.utils_nib import set_new_data
from LABelsToolkit.tools.aux_methods.utils import print_and_run
from LABelsToolkit.tools.detections.contours import contour_from_array_at_label_l


# --- Auxiliaries

def centroid_array(arr, labels):
    centers_of_mass = [np.array([0, 0, 0])] * len(labels)
    for l_id, l in enumerate(labels):
        coordinates_l = np.where(arr == l)  # returns [X_vector, Y_vector, Z_vector]
        if len(coordinates_l[0]) == 0:
            centers_of_mass[l_id] = np.nan
        else:
            centers_of_mass[l_id] = (1 / float(len(coordinates_l[0]))) * np.array([np.sum(k) for k in coordinates_l])
    return centers_of_mass


def centroid(im, labels, return_mm3=True):
    """
    Centroid (center of mass, barycenter) of a list of labels.
    :param im:
    :param labels: list of labels, e.g. [3] or [2, 3, 45]
    :param return_mm3: if true the answer is in mm if false in voxel indexes.
    :return: list of centroids, one for each label in the input order.
    """
    centers_of_mass = centroid_array(im.get_data(), labels)
    ans = []
    if return_mm3:
        for cm in centers_of_mass:
            if isinstance(cm, np.ndarray):
                ans += [im.affine[:3, :3].dot(cm.astype(np.float64))]
            else:
                ans += [cm]
    else:
        for cm in centers_of_mass:
            if isinstance(cm, np.ndarray):  # else it is np.nan.
                ans += [np.round(cm).astype(np.uint64)]
            else:
                ans += [cm]
    return ans


def covariance_matrices(im, labels, return_mm3=True):
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
        if np.count_nonzero(coords) > 0:
            cov_matrices[l_id] = np.cov(coords)
        else:
            cov_matrices[l_id] = np.nan * np.ones([3, 3])
    if return_mm3:
        cov_matrices = [im.affine[:3, :3].dot(cm.astype(np.float64)) for cm in cov_matrices]

    return cov_matrices


def covariance_distance_from_matrices(m1, m2, mul_factor=1):
    """
    Covariance distance between matrices m1 and m2, defined as
    d = factor * (1 - (trace(m1 * m2)) / (norm_fro(m1) + norm_fro(m2)))
    :param m1: matrix
    :param m2: matrix
    :param mul_factor: multiplicative factor for the formula, it equals to the maximal value the distance can reach
    :return: mul_factor * (1 - (np.trace(m1.dot(m2))) / (np.linalg.norm(m1) + np.linalg.norm(m2)))
    """
    if np.nan not in m1 and np.nan not in m2:
        return \
            mul_factor * (1 - (np.trace(m1.dot(m2)) / (np.linalg.norm(m1, ord='fro') * np.linalg.norm(m2, ord='fro'))))
    else:
        return np.nan


# --- global distances: (segm, segm) |-> real


def global_dice_score(im_segm1, im_segm2):
    """
    Global dice score as in Munoz-Moreno et al. 2013
    :param im_segm1:
    :param im_segm2:
    :return:
    """
    all_labels1 = set(im_segm1.get_data().astype(np.int).flat) - {0}
    all_labels2 = set(im_segm1.get_data().astype(np.int).flat) - {0}
    sum_intersections = np.sum([np.count_nonzero((im_segm1.get_data() == l) * (im_segm2.get_data() == l))
                         for l in set.union(all_labels1, all_labels2)])
    return 2 * sum_intersections / float(np.count_nonzero(im_segm1.get_data()) + np.count_nonzero(im_segm2.get_data()))


def global_outline_error(im_segm1, im_segm2):
    """
    Volume of the binarised image differences over the average binarised volume of the two images.
    :param im_segm1:
    :param im_segm2:
    :param labels_list:
    :return:
    """
    num_voxels_1, num_voxels_2 = np.count_nonzero(im_segm1.get_data()), np.count_nonzero(im_segm2.get_data())
    num_voxels_diff = np.count_nonzero(im_segm1.get_data() - im_segm2.get_data())
    return num_voxels_diff / (.5 * (num_voxels_1 + num_voxels_2))


# --- Single labels distances (segm, segm, label) |-> real


def dice_score_l(im_segm1, im_segm2, lab):
    place1 = im_segm1.get_data() == lab  # slow but readable, can be refactored later.
    place2 = im_segm2.get_data() == lab
    non_zero_place1 = np.count_nonzero(place1)
    non_zero_place2 = np.count_nonzero(place2)
    if non_zero_place1 + non_zero_place2 == 0:
        return np.nan
    else:
        return 2 * np.count_nonzero(place1 * place2) / float(non_zero_place1 + non_zero_place2)


def d_H(im1, im2, lab, return_mm3):
    """
    Asymmetric component of the Hausdorff distance.
    :param im1: first image
    :param im2: second image
    :param lab: label in the image
    :param return_mm3: final unit of measures of the result.
    :return: max(d(x, contourY)), x: point belonging to the first contour,
                                  contourY: contour of the second segmentation.
    """
    arr1 = im1.get_data() == lab
    arr2 = im2.get_data() == lab
    if np.count_nonzero(arr1) == 0 or np.count_nonzero(arr2) == 0:
        return np.nan
    if return_mm3:
        dt2 = nd.distance_transform_edt(1 - arr2, sampling=list(np.diag(im1.affine[:3, :3])))
    else:
        dt2 = nd.distance_transform_edt(1 - arr2, sampling=None)
    return np.max(dt2 * arr1)


def hausdorff_distance_l(im_segm1, im_segm2, lab, return_mm3):
    return np.max([d_H(im_segm1, im_segm2, lab, return_mm3), d_H(im_segm2, im_segm1, lab, return_mm3)])


def symmetric_contour_distance_l(im1, im2, lab, return_mm3, formula='normalised'):
    """
    Generalised normalised symmetric contour distance.
    On the set {d(x, contourY)) | x in contourX}, several statistics can be computed.
     Mean, median and standard deviation can be useful, as well as a more robust normalisation
     Formula can be
    :param im1:
    :param im2:
    :param lab:
    :param return_mm3:
    :param formula: 'normalised', 'averaged', 'median', 'std'
    'normalised' = (\sum_{x in contourX} d(x, contourY))  + \sum_{y in contourY} d(y, contourX))) / (|contourX| + |contourY|)
    'averaged'   = 0.5 (mean({d(x, contourY)) | x in contourX}) + mean({d(y, contourX)) | y in contourY}))
    'median'     = 0.5 (median({d(x, contourY)) | x in contourX}) + median({d(y, contourX)) | y in contourY}))
    'std'        = 0.5 \sqrt(std({d(x, contourY)) | x in contourX})^2 + std({d(y, contourX)) | y in contourY})^2)
    :return:
    """
    arr1 = im1.get_data() == lab
    arr2 = im2.get_data() == lab

    if np.count_nonzero(arr1) == 0 or np.count_nonzero(arr2) == 0:
        return np.nan

    arr1_contour = contour_from_array_at_label_l(arr1, 1)
    arr2_contour = contour_from_array_at_label_l(arr2, 1)

    if return_mm3:
        dtb1 = nd.distance_transform_edt(1 - arr1_contour, sampling=list(np.diag(im1.affine[:3, :3])))
        dtb2 = nd.distance_transform_edt(1 - arr2_contour, sampling=list(np.diag(im1.affine[:3, :3])))
    else:
        dtb1 = nd.distance_transform_edt(1 - arr1_contour)
        dtb2 = nd.distance_transform_edt(1 - arr2_contour)

    dist_border1_array2 = arr2_contour * dtb1
    dist_border2_array1 = arr1_contour * dtb2

    if formula == 'normalised':
        return (np.sum(dist_border1_array2) + np.sum(dist_border2_array1)) / (np.count_nonzero(arr1_contour) + np.count_nonzero(arr2_contour))
    elif formula == 'averaged':
        return .5 * (np.mean(dist_border1_array2) + np.mean(dist_border2_array1))
    elif formula == 'median':
        return .5 * (np.median(dist_border1_array2) + np.median(dist_border2_array1))
    elif formula == 'std':
        return np.sqrt( .5 * (np.std(dist_border1_array2)**2 + np.std(dist_border2_array1)**2))
    elif formula == 'average_std':
        return .5 * (np.mean(dist_border1_array2) + np.mean(dist_border2_array1)), \
               np.sqrt(.5 * (np.std(dist_border1_array2) ** 2 + np.std(dist_border2_array1) ** 2))
    else:
        raise IOError


# --- distances - (segm, segm) |-> pandas.Series (indexed by labels)

def dice_score(im_segm1, im_segm2, labels_list, labels_names, verbose=1):
    """
    Dice score between paired labels of segmentations.
    :param im_segm1: nibabel image with labels
    :param im_segm2: nibabel image with labels
    :param labels_list:
    :param labels_names:
    :param verbose:
    :return: dice score of the label label of the two segmentations.
    """
    scores = []
    for l in labels_list:
        d = dice_score_l(im_segm1, im_segm2, l)
        scores.append(d)
        if verbose > 0:
            print('    Dice scores label {0} : {1} '.format(l, d))

    return pa.Series(scores, index=labels_names)


def covariance_distance(im_segm1, im_segm2, labels_list, labels_names, return_mm3=True, verbose=1, factor=100):
    """
    Considers the label as a point distribution in the space, and returns the covariance matrix of the points
    distributions.
    :return:
    See: Herdin 2005, Correlation matrix distance, a meaningful measure for evaluation of non-stationary MIMO channels
    """
    cvs1 = covariance_matrices(im_segm1, labels=labels_list, return_mm3=return_mm3)
    cvs2 = covariance_matrices(im_segm2, labels=labels_list, return_mm3=return_mm3)

    cov_dist = []
    for l, a1, a2 in zip(labels_list, cvs1, cvs2):
        d = covariance_distance_from_matrices(a1, a2, mul_factor=factor)
        cov_dist.append(d)
        if verbose > 0:
            print('    Covariance distance label {0} : {1} '.format(l, d))

    return pa.Series(np.array(cov_dist), index=labels_names)


def hausdorff_distance(im_segm1, im_segm2, labels_list, labels_names, return_mm3=True, verbose=1):
    """
    From 2 segmentations sampled in overlapping grids (with affine in starndard form) it returns the hausdoroff
    distance for each label in the labels list and list names it returns the pandas series with the corresponding
    distances for each label.
    :param im_segm1:
    :param im_segm2:
    :param labels_list:
    :param labels_names:
    :param return_mm3:
    :param verbose:
    :return:
    """
    hausd_dist = []
    for l in labels_list:
        d = hausdorff_distance_l(im_segm1, im_segm2, l, return_mm3)
        hausd_dist.append(d)
        if verbose > 0:
            print('    Hausdoroff distance label {0} : {1} '.format(l, d))

    return pa.Series(np.array(hausd_dist), index=labels_names)


def symmetric_contour_distance(im_segm1, im_segm2, labels_list, labels_names, return_mm3=True, verbose=1,
                               formula='normalised'):
    nscd_dist = []
    for l in labels_list:
        d = symmetric_contour_distance_l(im_segm1, im_segm2, l, return_mm3, formula)
        nscd_dist.append(d)
        if verbose > 0:
            print('    {0}-SCD {1} : {2} '.format(formula, l, d))

    return pa.Series(np.array(nscd_dist), index=labels_names)


# --- varaizioni over symmetric contour distance:


def normalised_symmetric_contour_distance(im_segm1, im_segm2, labels_list, labels_names, return_mm3=True, verbose=1):
    return symmetric_contour_distance(im_segm1, im_segm2, labels_list, labels_names,
                                      return_mm3=return_mm3, verbose=verbose, formula='normalised')


def averaged_symmetric_contour_distance(im_segm1, im_segm2, labels_list, labels_names, return_mm3=True, verbose=1):
    return symmetric_contour_distance(im_segm1, im_segm2, labels_list, labels_names,
                                      return_mm3=return_mm3, verbose=verbose, formula='averaged')


def median_symmetric_contour_distance(im_segm1, im_segm2, labels_list, labels_names, return_mm3=True, verbose=1):
    return symmetric_contour_distance(im_segm1, im_segm2, labels_list, labels_names,
                                      return_mm3=return_mm3, verbose=verbose, formula='median')


def std_symmetric_contour_distance(im_segm1, im_segm2, labels_list, labels_names, return_mm3=True, verbose=1):
    return symmetric_contour_distance(im_segm1, im_segm2, labels_list, labels_names,
                                      return_mm3=return_mm3, verbose=verbose, formula='std')


# --- extra:

def box_sides_length(im, labels_list, labels_names, return_mm3=True):
    """
    im_segmentation, labels_to_box, labels_to_box_names
    Box surrounding the label in the list labels_to_box.
    A box is an (ordered!) couple of 3d points.
    :param im: sampled on an orthogonal grid.
    :param labels_list:
    :param labels_names:
    :param return_mm3:
    :return:
    """
    def box_sides_length_l(arr, lab, scaling_factors):
        coordinates = np.where(arr == lab)  # returns [X_vector, Y_vector, Z_vector]
        dists = [np.abs(np.max(k) - np.min(k)) for k in coordinates]
        if return_mm3:
            dists = [d * dd for d, dd in zip(coordinates, scaling_factors)]
        return dists

    return pa.Series(
        np.array([box_sides_length_l(im.get_data(), l, np.diag(im.affine)) for l in labels_list]),
        index=labels_names)


def mahalanobis_distance(im, im_mask=None, trim=False):
    # mono modal, image is vectorised, covariance is the std.
    if im_mask is None:
        mu = np.mean(im.get_data().flatten())
        sigma2 = np.std(im.get_data().flatten())
        return set_new_data(im, np.sqrt((im.get_data() - mu) * sigma2 * (im.get_data() - mu)))
    else:
        np.testing.assert_array_equal(im.affine, im_mask.affine)
        np.testing.assert_array_equal(im.shape, im_mask.shape)
        mu = np.mean(im.get_data().flatten() * im_mask.get_data().flatten())
        print mu
        sigma2 = np.std(im.get_data().flatten() * im_mask.get_data().flatten())
        new_data = np.sqrt((im.get_data() - mu) * sigma2**(-1) * (im.get_data() - mu))
        if trim:
            new_data = new_data * im_mask.get_data().astype(np.bool)
        return set_new_data(im, new_data)




def s_dispersion(im_segm1, im_segm2, labels_list, labels_names, return_mm3=True):
    # definition of s_dispersion is not the conventional definition of precision

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


def s_precision(im_segm1, im_segm2, pfo_intermediate_files, labels_list, labels_names, verbose=0):
    # deprecated as too imprecise :-) ! -  definition of s_precision is not the conventional definition of precision
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