import numpy as np
import pandas as pa
from scipy import ndimage as nd

from nilabels.tools.detections.contours import get_internal_contour_with_erosion_at_label


# --- Auxiliaries

def centroid_array(arr, labels):
    """
    Auxiliary of centroid, for arrays in array coordinates.
    :param arr: numpy array of any dimension > 1 .
    :param labels: list of labels
    :return: list of centre of masses for the selected values in the array.
    If the labels in the labels list is not in the array it returns nan.
    """
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
    :param im: nifti image from nibabel.
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


def covariance_distance_between_matrices(m1, m2, mul_factor=1):
    """
    Covariance distance between matrices m1 and m2, defined as
    d = factor * (1 - (trace(m1 * m2)) / (norm_fro(m1) + norm_fro(m2)))
    :param m1: matrix
    :param m2: matrix
    :param mul_factor: multiplicative factor for the formula, it equals to the maximal value the distance can reach
    :return: mul_factor * (1 - (np.trace(m1.dot(m2))) / (np.linalg.norm(m1) + np.linalg.norm(m2)))
    """
    if np.nan in m1 or np.nan in m2:
        cd = np.nan
    else:
        cd = mul_factor * (1 - (np.trace(m1.dot(m2)) / (np.linalg.norm(m1, ord='fro') * np.linalg.norm(m2, ord='fro'))))
    return cd

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
    :return:
    """
    num_voxels_1, num_voxels_2 = np.count_nonzero(im_segm1.get_data()), np.count_nonzero(im_segm2.get_data())
    num_voxels_diff = np.count_nonzero(im_segm1.get_data() - im_segm2.get_data())
    return num_voxels_diff / (.5 * (num_voxels_1 + num_voxels_2))


# --- Single labels distances (segm, segm, label) |-> real


def dice_score_one_label(im_segm1, im_segm2, lab):
    """
    Dice score for a single label. The input images must have the same grid shape (but can have different affine part).
    :param im_segm1: nibabel image representing a segmentation
    :param im_segm2: as im_segm1
    :param lab: a label.
    :return: dice score distance for the given label. If the label is not present, it returns a nan.
    """
    place1 = im_segm1.get_data() == lab
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
    :param im1: first image (nibabel format)
    :param im2: second image (nibabel format)
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


def hausdorff_distance_one_label(im_segm1, im_segm2, lab, return_mm3):
    return np.max([d_H(im_segm1, im_segm2, lab, return_mm3), d_H(im_segm2, im_segm1, lab, return_mm3)])


def symmetric_contour_distance_one_label(im1, im2, lab, return_mm3, formula='normalised'):
    """
    Generalised normalised symmetric contour distance.
    On the sets {d(x, contourY)) | x in contourX} and {d(y, contourX)) | y in contourY}, several statistics
    can be computed. Mean, median and standard deviation can be useful, as well as a more robust normalisation.
     Formula can be
    :param im1: nibabel image with a segmentation
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

    arr1_contour = get_internal_contour_with_erosion_at_label(arr1, 1)
    arr2_contour = get_internal_contour_with_erosion_at_label(arr2, 1)

    if return_mm3:
        dtb1 = nd.distance_transform_edt(1 - arr1_contour, sampling=list(np.diag(im1.affine[:3, :3])))
        dtb2 = nd.distance_transform_edt(1 - arr2_contour, sampling=list(np.diag(im2.affine[:3, :3])))
    else:
        dtb1 = nd.distance_transform_edt(1 - arr1_contour)
        dtb2 = nd.distance_transform_edt(1 - arr2_contour)

    dist_border1_array2 = arr2_contour * dtb1
    dist_border2_array1 = arr1_contour * dtb2

    dist_border1_array2 = dist_border1_array2[dist_border1_array2 > 0]
    dist_border2_array1 = dist_border2_array1[dist_border2_array1 > 0]

    if formula == 'normalised':
        return (np.sum(dist_border1_array2) + np.sum(dist_border2_array1)) / float(np.count_nonzero(arr1_contour) + np.count_nonzero(arr2_contour))
    elif formula == 'averaged':
        return .5 * (np.mean(dist_border1_array2) + np.mean(dist_border2_array1))
    elif formula == 'median':
        return .5 * (np.median(dist_border1_array2) + np.median(dist_border2_array1))
    elif formula == 'std':
        return np.sqrt(.5 * (np.std(dist_border1_array2)**2 + np.std(dist_border2_array1)**2))
    elif formula == 'average_std':
        return .5 * (np.mean(dist_border1_array2) + np.mean(dist_border2_array1)), \
               np.sqrt(.5 * (np.std(dist_border1_array2) ** 2 + np.std(dist_border2_array1) ** 2))
    else:
        raise IOError('adf')


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
        d = dice_score_one_label(im_segm1, im_segm2, l)
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
        d = covariance_distance_between_matrices(a1, a2, mul_factor=factor)
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
        d = hausdorff_distance_one_label(im_segm1, im_segm2, l, return_mm3)
        hausd_dist.append(d)
        if verbose > 0:
            print('    Hausdoroff distance label {0} : {1} '.format(l, d))

    return pa.Series(np.array(hausd_dist), index=labels_names)


def symmetric_contour_distance(im_segm1, im_segm2, labels_list, labels_names, return_mm3=True, verbose=1,
                               formula='normalised'):
    nscd_dist = []
    for l in labels_list:
        d = symmetric_contour_distance_one_label(im_segm1, im_segm2, l, return_mm3, formula)
        nscd_dist.append(d)
        if verbose > 0:
            print('    {0}-SCD {1} : {2} '.format(formula, l, d))

    return pa.Series(np.array(nscd_dist), index=labels_names)


# --- variants over symmetric contour distance:


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
    Length of the rectangular hull surrounding the labels in the given list.
    The rectangle is parallel to the matrix coordinate system.
    :param im: sampled on an orthogonal grid.
    :param labels_list: list of labels
    :param labels_names: list of labels names that will appear in the Pandas series
    :param return_mm3: if True the answer is provided in the real space coordinates.
    :return: output pandas series. One row for each label.
    """
    def box_sides_length_l(arr, lab, scaling_factors):

        if lab not in arr:
            return np.nan
        coordinates = np.where(arr == lab)  # returns [X_vector, Y_vector, Z_vector]
        if return_mm3:
            coordinates = [d * dd for d, dd in zip(scaling_factors, coordinates)]
        return [np.abs(np.max(coordinates[k]) - np.min(coordinates[k])) for k in range(len(coordinates))]

    boxes_values = [box_sides_length_l(im.get_data(), l, np.diag(im.affine)[:-1]) for l in labels_list]
    return pa.Series(boxes_values, index=labels_names)
