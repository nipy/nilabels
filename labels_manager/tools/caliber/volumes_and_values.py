import numpy as np
import pandas as pa

from labels_manager.tools.aux_methods.utils_nib import one_voxel_volume


def get_total_volume(im_segm, labels_to_exclude=None):
    """

    :param im_segm:
    :param labels_to_exclude:
    :return:
    """
    seg = np.copy(im_segm.get_data())
    if labels_to_exclude is not None:
        for label_k in labels_to_exclude:
            places = seg != label_k
            seg = seg * places
        num_voxels = np.count_nonzero(seg)
    else:
        num_voxels = np.count_nonzero(im_segm.get_data())
    vol_mm3 = num_voxels * one_voxel_volume(im_segm)
    return num_voxels, vol_mm3


def get_volumes_per_label(im_segm, labels, labels_names, tot_volume_prior=None, verbose=0):
    """
    :param im_segm:
    :param labels:
    :param labels_names: index of labels in the final dataframes.
    :param tot_volume_prior:
    :param verbose:
    :return:
    """
    # Get volumes per regions:
    num_voxels = np.zeros(len(labels), dtype=np.uint64)

    for index_label_k, label_k in enumerate(labels):

        if isinstance(label_k, int):
            places = im_segm.get_data() == label_k
        else:
            places = np.zeros_like(im_segm.get_data(), dtype=np.bool)
            for label_k_j in label_k:
                places += im_segm.get_data() == label_k_j

        num_voxels[index_label_k] = np.count_nonzero(places)

    volume = one_voxel_volume(im_segm) * num_voxels.astype(np.float64)

    # Get tot volume
    if tot_volume_prior is None:
        tot_volume_prior = get_total_volume(im_segm)[0]

    # get volumes over total volume:
    vol_over_tot = volume / float(tot_volume_prior)

    # pandas series:
    se_num_voxel    = pa.Series(num_voxels,   index=labels_names)
    se_volume       = pa.Series(volume,       index=labels_names)
    se_vol_over_tot = pa.Series(vol_over_tot, index=labels_names)
    # final data frame
    data_frame = pa.DataFrame({'Num voxels'   : se_num_voxel,
                               'Volume'       : se_volume,
                               'Vol over Tot' : se_vol_over_tot}, columns=['Num voxels', 'Volume', 'Vol over Tot'])

    if verbose > 0:
        # show the data-frame at console:
        print(data_frame)

    return data_frame


def get_values_below_labels(im_seg, im_anat, labels_list):
    """

    :param im_seg: image segmentation
    :param im_anat: image anatomy
    :param labels_list: integer, list of labels [l1, l2, ..., ln], or list of list of labels if labels needs to be
    considered together.
    e.g. labels_list = [1,2,[3,4]] -> volume of label 1, volume of label 2, volume of label 3 and 4.
    :return: list of numpy arrays, each element is the flat array of the values below the labels
    """
    values = []
    for label_k in labels_list:
        # TODO: test and optimise with np.where
        if isinstance(label_k, int):
            all_places = im_seg.get_data() == label_k
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

    return values

def from_values_below_labels_to_mu_std(values_below_labels, labels, labels_names, verbose=0):
    """
    :param values_below_labels: output of values_below_labels
    :return:
    """
    data_frame = pa.DataFrame({'Labels'              : pa.Series(labels, index=labels_names),
                               'Average below label' : pa.Series(np.mean(values_below_labels), index=labels_names),
                               'Std below label'     : pa.Series(np.std(values_below_labels), index=labels_names)})

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
    if verbose:
        print(df)
    return df
