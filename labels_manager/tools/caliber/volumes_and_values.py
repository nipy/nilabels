"""
This module is divided into two parts.
First one -> essential functions, input are nibabel objects, output are reals or arrays.
             The first part refers to the number of voxels.
Second one -> uses the first one, to plot volumes, normalisation outputs values in pandas arrays or dataframes.
"""
import numpy as np
import pandas as pa

from labels_manager.tools.aux_methods.utils_nib import one_voxel_volume


#  ---- First part ----

def get_total_num_nonzero_voxels(im_segm, list_labels_to_exclude=None):
    """
    :param im_segm:
    :param list_labels_to_exclude:
    :return:
    """
    seg = np.copy(im_segm.get_data())
    if list_labels_to_exclude is not None:
        for label_k in list_labels_to_exclude:
            places = seg != label_k
            seg = seg * places
        num_voxels = np.count_nonzero(seg)
    else:
        num_voxels = np.count_nonzero(im_segm.get_data())
    return num_voxels


def get_num_voxels_from_labels_list(im_segm, labels_list):
    """
    :param im_seg: image segmentation
    :param labels_list: integer, list of labels [l1, l2, ..., ln], or list of list of labels if labels needs to be
    considered together.
    e.g. labels_list = [1,2,[3,4]] -> values below label 1, values below label 2, values below label 3 and 4.
    :return: np.arrays with the number of voxels below each input label or list input label.
    """
    num_voxels_per_label = np.zeros(len(labels_list)).astype(np.int64)

    for k, label_k in enumerate(labels_list):
        if isinstance(label_k, int):
            all_places = im_segm.get_data() == label_k
            num_voxels_per_label[k] = np.count_nonzero(np.nan_to_num(all_places))
        elif isinstance(label_k, list):
            all_places = np.zeros_like(im_segm.get_data(), dtype=np.bool)
            for label_k_j in label_k:
                all_places += im_segm.get_data() == label_k_j
            num_voxels_per_label[k] = np.count_nonzero(np.nan_to_num(all_places))
        else:
            raise IOError('Labels list must be like [1,2,[3,4]], where [3, 4] are considered as a single label.')

    return num_voxels_per_label


def get_values_below_labels_list(im_seg, im_anat, labels_list):
    """
    :param im_seg: image segmentation
    :param im_anat: anatomical image, corresponding to the segmentation.
    :param labels_list: integer, list of labels [l1, l2, ..., ln], or list of list of labels if labels needs to be
    considered together.
    e.g. labels_list = [1,2,[3,4]] -> values belows label 1, values below label 2, values below label 3 and 4.
    :return: list of np.arrays. Each containing all the values below the corresponding labels.
    """
    assert im_seg.shape == im_anat.shape

    values_below_each_label = []

    for label_k in labels_list:
        if isinstance(label_k, int):
            coords = np.where(im_seg.get_data() == label_k)
            values_below_each_label.append(im_anat.get_data()[coords].flatten())
        elif isinstance(label_k, list):
            vals = np.array([])
            for label_k_j in label_k:
                coords = np.where(im_seg.get_data() == label_k_j)
                vals = np.concatenate((vals, im_anat.get_data()[coords].flatten()), axis=0)
            values_below_each_label.append(vals)
        else:
            raise IOError('Labels list must be like [1,2,[3,4]], where [3, 4] are considered as a single label.')

    return values_below_each_label

#  ---- Second part ----



def get_volumes_per_label(im_segm, labels, labels_names, tot_volume_prior=None, verbose=0):
    """
    :param im_segm:
    :param labels:
    :param labels_names: index of labels in the final dataframes.
    :param tot_volume_prior:
    :param verbose:
    :return:
    """
    num_voxels = get_total_num_nonzero_voxels(im_segm)
    vol_mm3 = num_voxels * one_voxel_volume(im_segm)
    if tot_volume_prior is None:
        tot_volume_prior = 1
    if tot_volume_prior == 'tot':
        tot_volume_prior = vol_mm3

    if labels_names == 'tot':

        data_frame = pa.DataFrame({'Labels': pa.Series(labels, index=[labels_names]),
                                   'Num voxels': pa.Series(num_voxels, index=[labels_names]),
                                   'Volume': pa.Series(vol_mm3, index=[labels_names]),
                                   'Vol over Tot': pa.Series(vol_mm3 / tot_volume_prior, index=[labels_names])})

    else:
        all_places = np.zeros_like(im_segm.get_data(), dtype=np.bool)
        for label_k in labels:
            if isinstance(label_k, int):
                all_places += im_segm.get_data() == label_k
            else:
                for label_k_j in label_k:
                    all_places += im_segm.get_data() == label_k_j

        flat_volume_voxel = np.nan_to_num( (all_places.astype(np.float64)).flatten() )

        non_zero_voxels = np.count_nonzero(flat_volume_voxel)
        volumes = non_zero_voxels * one_voxel_volume(im_segm)
        if tot_volume_prior is None:
            num_voxels = get_total_num_nonzero_voxels(im_segm)
            tot_volume_prior = num_voxels * one_voxel_volume(im_segm)

        vol_over_tot = volumes / float(tot_volume_prior)

        data_frame = pa.DataFrame({'Labels': pa.Series(labels, index=labels_names),
                                   'Num voxels': pa.Series(non_zero_voxels, index=labels_names),
                                   'Volume': pa.Series(volumes, index=labels_names),
                                   'Vol over Tot': pa.Series(vol_over_tot, index=labels_names)})
    if verbose > 0:
        print(data_frame)

    return data_frame


#  ---- DUMP ----
'''

def se_values_below_labels(im_seg, im_anat, labels_list, labels_names=None):
    """

    :param im_seg: image segmentation
    :param im_anat: image anatomy
    :param labels_list: integer, list of labels [l1, l2, ..., ln], or list of list of labels if labels needs to be
    considered together.
    e.g. labels_list = [1,2,[3,4]] -> values below label 1, values below label 2, values below label 3 and 4.
    :return: list of numpy arrays, each element is the flat array of the values below the labels
    """
    values = []
    for label_k in labels_list:
        # TODO: optimise with np.where after testing
        if isinstance(label_k, int):
            all_places += im_seg.get_data() == label_k
        else:
            all_places = np.zeros_like(im_seg.get_data(), dtype=np.bool)
            for label_k_j in label_k:
                all_places += im_seg.get_data() == label_k_j

        masked_scalar_data = np.nan_to_num(
            (all_places.astype(np.float64) * im_anat.get_data().astype(np.float64)).flatten())
        # remove zero elements from the array:
        non_zero_masked_scalar_data = masked_scalar_data[np.where(masked_scalar_data > 1e-6)]  # 1e-6

        if non_zero_masked_scalar_data.size == 0:  # if not non_zero_masked_scalar_data is an empty array.
            non_zero_masked_scalar_data = 0.

        values.append(non_zero_masked_scalar_data)
    if labels_names is None:
        return values
    else:
        return pa.Series(values, index=labels_names)


def from_values_below_labels_to_mu_std(values_below_labels, labels, labels_names, verbose=0):
    """

    :param values_below_labels: output of values_below_labels
    :param labels:
    :param labels_names:
    :param verbose:
    :return:
    """
    data_frame = pa.DataFrame({'Labels'              : pa.Series(labels, index=labels_names),
                               'Average below label' : pa.Series([np.mean(v) for v in values_below_labels], index=labels_names),
                               'Std below label'     : pa.Series([np.std(v) for v in values_below_labels], index=labels_names)})

    if verbose > 0:
        print(data_frame)

    return data_frame


def get_volumes_per_label(im_segm, labels, labels_names, tot_volume_prior=None, verbose=0):
    """
    :param im_segm:
    :param labels:
    :param labels_names: index of labels in the final dataframes.
    :param tot_volume_prior:
    :param verbose:
    :return:
    """
    num_voxels, vol_mm3 = get_total_volume(im_segm)
    if tot_volume_prior is None:
        tot_volume_prior = 1
    if tot_volume_prior == 'tot':
        tot_volume_prior = vol_mm3

    if labels_names == 'tot':

        data_frame = pa.DataFrame({'Labels': pa.Series(labels, index=[labels_names]),
                                   'Num voxels': pa.Series(num_voxels, index=[labels_names]),
                                   'Volume': pa.Series(vol_mm3, index=[labels_names]),
                                   'Vol over Tot': pa.Series(vol_mm3 / tot_volume_prior, index=[labels_names])})

    else:
        all_places = np.zeros_like(im_segm.get_data(), dtype=np.bool)
        for label_k in labels:
            if isinstance(label_k, int):
                all_places += im_segm.get_data() == label_k
            else:
                for label_k_j in label_k:
                    all_places += im_segm.get_data() == label_k_j

        flat_volume_voxel = np.nan_to_num( (all_places.astype(np.float64)).flatten() )

        non_zero_voxels = np.count_nonzero(flat_volume_voxel)
        volumes = non_zero_voxels * one_voxel_volume(im_segm)
        if tot_volume_prior is None:
            tot_volume_prior = get_total_volume(im_segm, labels_to_exclude=[0])[1]

        vol_over_tot = volumes / float(tot_volume_prior)

        data_frame = pa.DataFrame({'Labels': pa.Series(labels, index=labels_names),
                                   'Num voxels': pa.Series(non_zero_voxels, index=labels_names),
                                   'Volume': pa.Series(volumes, index=labels_names),
                                   'Vol over Tot': pa.Series(vol_over_tot, index=labels_names)})
    if verbose > 0:
        print(data_frame)

    return data_frame


def get_mu_std_below_labels(im_segm, im_anatomical, labels, labels_names, verbose=0):
    """

    :param im_anatomical:
    :param im_segm:
    :param labels: can be an integer, or a list.
     If it is a list, it can contain sublists.
     If labels are in the sublist, volumes will be computed for all the labels in the list.
    e.g. [1,2,[3,4]] -> volume of label 1, volume of label 2, volume of label 3 and 4.
    :param labels_names:
    :param verbose:
    :return:
    """
    values_below_labels = get_values_below_labels(im_segm, im_anatomical, labels)
    df = from_values_below_labels_to_mu_std(values_below_labels, labels, labels_names, verbose=verbose)
    return df
'''
