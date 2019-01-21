"""
This module is divided into two parts.
First one -> essential functions, input are nibabel objects, output are reals or arrays.
             The first part refers to the number of voxels.
Second one -> it uses the first part, to plot volumes, normalisation outputs values in pandas arrays or dataframes.
"""
import numpy as np
import pandas as pa

from nilabels.tools.aux_methods.utils_nib import one_voxel_volume


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
        num_voxels = int(np.count_nonzero(im_segm.get_data()))
    return num_voxels


def get_num_voxels_from_labels_list(im_segm, labels_list):
    """
    :param im_segm: image segmentation
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


def get_values_below_labels_list(im_segm, im_anat, labels_list):
    """
    :param im_segm: image segmentation
    :param im_anat: anatomical image, corresponding to the segmentation.
    :param labels_list: integer, list of labels [l1, l2, ..., ln], or list of list of labels if labels needs to be
    considered together.
    e.g. labels_list = [1,2,[3,4]] -> values belows label 1, values below label 2, values below label 3 and 4.
    :return: list of np.arrays. Each containing all the values below the corresponding labels.
    """
    assert im_segm.shape == im_anat.shape

    values_below_each_label = []

    for label_k in labels_list:
        if isinstance(label_k, int):
            coords = np.where(im_segm.get_data() == label_k)
            values_below_each_label.append(im_anat.get_data()[coords].flatten())
        elif isinstance(label_k, list):
            vals = np.array([])
            for label_k_j in label_k:
                coords = np.where(im_segm.get_data() == label_k_j)
                vals = np.concatenate((vals, im_anat.get_data()[coords].flatten()), axis=0)
            values_below_each_label.append(vals)
        else:
            raise IOError('Labels list must be like [1,2,[3,4]], where [3, 4] are considered as a single label.')

    return values_below_each_label


def get_volumes_per_label(im_segm, labels, labels_names, tot_volume_prior=None, verbose=0):
    """
    Get a separate volume for each label in a data-frame
    :param im_segm: nibabel segmentation
    :param labels: labels you want to measure, or 'all' if you want them all or 'tot' to have the total of the non zero
                   labels.
    :param labels_names: list with the indexes of labels in the final dataframes, corresponding to labels list.
    :param tot_volume_prior: factor the volumes will be divided with.
    :param verbose: > 0 will provide the intermediate stepsp
    :return:
    """
    num_non_zero_voxels = get_total_num_nonzero_voxels(im_segm)
    vol_non_zero_voxels_mm3 = num_non_zero_voxels * one_voxel_volume(im_segm)
    if tot_volume_prior is None:
        tot_volume_prior = vol_non_zero_voxels_mm3
    if labels_names not in [None, 'all', 'tot']:
        if len(labels) != len(labels_names):
            raise IOError('Inconsistent labels - labels_names input.')

    if labels_names == 'all':
        labels_names = ['reg {}'.format(l) for l in labels]

    if labels_names == 'tot':
        labels_names = ['tot']

        non_zero_voxels = np.count_nonzero(im_segm.get_data())
        volumes = non_zero_voxels * one_voxel_volume(im_segm)
        vol_over_tot = volumes / float(tot_volume_prior)

        data_frame = pa.DataFrame({'Num voxels': pa.Series([non_zero_voxels], index=labels_names),
                                   'Volume': pa.Series([volumes], index=labels_names),
                                   'Vol over Tot': pa.Series([vol_over_tot], index=labels_names)})
        return data_frame

    else:
        non_zero_voxels_list = []
        volumes_list = []
        vol_over_tot_list = []

        for label_k in labels:
            all_places = np.zeros_like(im_segm.get_data(), dtype=np.bool)
            if isinstance(label_k, int):
                all_places += im_segm.get_data() == label_k
            else:
                for label_k_j in label_k:
                    all_places += im_segm.get_data() == label_k_j

            flat_volume_voxel = np.nan_to_num((all_places.astype(np.float64)).flatten())

            non_zero_voxels = np.count_nonzero(flat_volume_voxel)
            volumes = non_zero_voxels * one_voxel_volume(im_segm)

            vol_over_tot = volumes / float(tot_volume_prior)

            non_zero_voxels_list.append(non_zero_voxels)
            volumes_list.append(volumes)
            vol_over_tot_list.append(vol_over_tot)

        data_frame = pa.DataFrame({'Label'        : pa.Series(labels,               index=labels_names),
                                   'Num voxels'   : pa.Series(non_zero_voxels_list, index=labels_names),
                                   'Volume'       : pa.Series(volumes_list,         index=labels_names),
                                   'Vol over Tot' : pa.Series(vol_over_tot_list,    index=labels_names)})

        data_frame = data_frame.rename_axis('Region')
        data_frame = data_frame.reset_index()

        if verbose > 0:
            print(data_frame)

        return data_frame
