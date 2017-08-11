import numpy as np
from tabulate import tabulate

from labels_manager.tools.aux_methods.utils import one_voxel_volume


def get_total_volume(im_segm, labels_to_exclude=None, return_mm3=True):
    """

    :param im_segm:
    :param labels_to_exclude:
    :param return_mm3:
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

    if return_mm3:
        mm_3 = num_voxels * one_voxel_volume(im_segm)
        return mm_3
    else:
        return num_voxels


def get_volumes_per_label(im_segm, labels='all', tot_volume_prior=None, verbose=0, return_mm3=True):
    """

    :param im_segm:
    :param labels: Labels can be as well a list of lists. In this case the volume is measured together (Left + right)
    :param tot_volume_prior:
    :param verbose:
    :param return_mm3:
    :return:
    """
    if isinstance(labels, int):
        labels = [labels, ]
    elif isinstance(labels, list):
        pass
    elif labels == 'all':
        labels = list(np.sort(list(set(im_segm.flat))))
    else:
        raise IOError("Input labels must be a list, a list of lists, or an int or the string 'all'.")

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

    if return_mm3:
        volume = one_voxel_volume(im_segm) * num_voxels.astype(np.float64)
    else:
        volume = num_voxels.astype(np.float64)[:]

    # Get tot volume
    if tot_volume_prior is None:
        tot_volume_prior = get_total_volume(im_segm)

    # get volumes over total volume:
    vol_over_tot = volume / float(tot_volume_prior)

    # show a table at console:
    if verbose:
        headers = ['labels', 'Vol', 'Vol/totVol']
        table = [[r, v, v_t] for r, v, v_t in zip(labels, volume, vol_over_tot)]
        print(tabulate(table, headers=headers))

    return num_voxels, volume, vol_over_tot


def get_average_below_labels(im_segm, im_anatomical, labels='all', verbose=0):
    """
    can be an integer, or a list.
     If it is a list, it can contain sublists.
     If labels are in the sublist, volumes will be computed for all the labels in the list.
    e.g. [1,2,[3,4]] -> volume of label 1, volume of label 2, volume of label 3 and 4.
    :param im_anatomical:
    :param im_segm:
    :param labels:
    :param verbose:
    :return:
    """
    if isinstance(labels, int):
        labels = [labels, ]
    elif isinstance(labels, list):
        pass
    elif labels == 'all':
        labels = list(np.sort(list(set(im_segm.flat))))
    else:
        raise IOError("Input labels must be a list, a list of lists, or an int or the string 'all'.")

    # Get volumes per regions:
    values = np.zeros(len(labels), dtype=np.float64)

    for index_label_k, label_k in enumerate(labels):

        if isinstance(label_k, int):
            all_places = im_segm.get_data() == label_k
        else:
            all_places = np.zeros_like(im_segm.get_data(), dtype=np.bool)
            for label_k_j in label_k:
                all_places += im_segm.get_data() == label_k_j

        masked_scalar_data = np.nan_to_num(
            (all_places.astype(np.float64) * im_anatomical.get_data().astype(np.float64)).flatten())
        # remove zero elements from the array:
        non_zero_masked_scalar_data = masked_scalar_data[np.where(masked_scalar_data > 1e-6)]  # 1e-6

        if non_zero_masked_scalar_data.size == 0:  # if not non_zero_masked_scalar_data is an empty array.
            non_zero_masked_scalar_data = 0.

        values[index_label_k] = np.mean(non_zero_masked_scalar_data)

        # mean_voxel = np.mean(non_zero_masked_scalar_data)
        # if self.return_mm3:
        #     values[index_label_k] = ( 1 / self._one_voxel_volume ) * mean_voxel
        # else:
        #     values[index_label_k] = mean_voxel

        if verbose:
            print('Mean below the labels for the given image {0} : {1}'.format(labels[index_label_k],
                                                                               values[index_label_k]))
            if isinstance(non_zero_masked_scalar_data, np.ndarray):
                print 'non zero masked scalar data : ' + str(len(non_zero_masked_scalar_data))
    return values
