"""
Measurements on labels.
"""
import numpy as np
import os
import nibabel as nib
from tabulate import tabulate


class SegmentationAnalyzer(object):

    def __init__(self, pfi_segmentation, pfi_scalar_im, icv_factor=None, return_mm3=True):

        for p in [pfi_segmentation, pfi_scalar_im]:
            if not os.path.exists(p):
                raise IOError('Input data path {} does not exist.'.format(p))

        self.pfi_segmentation = pfi_segmentation
        self.return_mm3 = return_mm3
        self.pfi_scalar_im = pfi_scalar_im
        self.icv_factor = icv_factor
        self.labels_to_exclude = None

        self._segmentation = None
        self._scalar_im = None
        self._one_voxel_volume = None

        self.update()

    def update(self):

        self._segmentation =  nib.load(self.pfi_segmentation)
        self._scalar_im = nib.load(self.pfi_scalar_im)

        np.testing.assert_array_almost_equal(self._scalar_im.get_affine(), self._segmentation.get_affine())
        np.testing.assert_array_almost_equal(self._scalar_im.shape, self._segmentation.shape)

        self._one_voxel_volume = np.round(np.abs(np.prod(np.diag(self._segmentation.get_affine()))), decimals=6)

    def get_total_volume(self):

        if self.labels_to_exclude is not None:

            seg = np.copy(self._segmentation.get_data())
            for index_label_k, label_k in enumerate(self.labels_to_exclude):
                places = self._segmentation.get_data() != label_k
                seg =  seg * places.astype(np.int)

            num_voxels = np.count_nonzero(seg)
        else:
            num_voxels = np.count_nonzero(self._segmentation.get_data())

        if self.return_mm3:
            mm_3 = num_voxels * self._one_voxel_volume
            return mm_3
        else:
            return num_voxels

    def get_volumes_per_label(self, selected_labels, verbose=0):
        """
        :param selected_labels: can be an integer, or a list.
         If it is a list, it can contain sublists.
         If labels are in the sublist, volumes will be computed for all the labels in the list.
        e.g. [1,2,[3,4]] -> volume of label 1, volume of label 2, volume of label 3 and 4.
        :param verbose:
        :return:
        """
        if isinstance(selected_labels, int):
            selected_labels = [selected_labels, ]
        elif isinstance(selected_labels, list):
            pass
        else:
            raise IOError('Input labels must be a list or an int.')

        # get tot volume
        tot_brain_volume = self.get_total_volume()

        # Get volumes per regions:
        voxels = np.zeros(len(selected_labels), dtype=np.uint64)

        for index_label_k, label_k in enumerate(selected_labels):

            if isinstance(label_k, int):
                places = self._segmentation.get_data()  == label_k
            else:
                places = np.zeros_like(self._segmentation.get_data(), dtype=np.bool)
                for label_k_j in label_k:
                    places += self._segmentation.get_data() == label_k_j

            voxels[index_label_k] = np.count_nonzero(places)

        if self.return_mm3:
               vol = self._one_voxel_volume * voxels.astype(np.float64)
        else:
            vol = voxels.astype(np.float64)[:]

        # get volumes over total volume:
        vol_over_tot = vol / float(tot_brain_volume)

        # get volume over ICV estimates
        if self.icv_factor is not None:
            vol_over_icv = vol / float(self.icv_factor)
        else:
            vol_over_icv = np.zeros_like(vol)

        # show a table at console:
        if verbose:
            headers = ['labels', 'Vol', 'Vol/totVol', 'Vol/ICV']
            table = [[r, v, v_t, v_icv] for r, v, v_t, v_icv in zip(selected_labels, vol, vol_over_tot, vol_over_icv)]
            print(tabulate(table, headers=headers))

        return vol, voxels, vol_over_tot, vol_over_icv

    def get_average_below_labels(self, selected_labels, verbose=0):
        """
        :param selected_labels:  can be an integer, or a list.
         If it is a list, it can contain sublists.
         If labels are in the sublist, volumes will be computed for all the labels in the list.
        e.g. [1,2,[3,4]] -> volume of label 1, volume of label 2, volume of label 3 and 4.
        :param verbose:
        :return:
        """
        if isinstance(selected_labels, int):
            selected_labels = [selected_labels, ]
        elif isinstance(selected_labels, list):
            pass
        else:
            raise IOError('Input labels must be a list or an int.')

        # Get volumes per regions:
        values = np.zeros(len(selected_labels), dtype=np.float64)

        for index_label_k, label_k in enumerate(selected_labels):

            if isinstance(label_k, int):
                all_places = self._segmentation.get_data() == label_k
            else:
                all_places = np.zeros_like(self._segmentation.get_data(), dtype=np.bool)
                for label_k_j in label_k:
                    all_places += self._segmentation.get_data() == label_k_j

            masked_scalar_data = (all_places.astype(np.float64) * self._scalar_im.get_data().astype(np.float64)).flatten()
            # remove zero elements from the array:
            # non_zero_masked_scalar_data = [j for j in masked_scalar_data if j > 1e-6]

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
                print('Mean below the labels for the given image {0} : {1}'.format(selected_labels[index_label_k], values[index_label_k]))
                if isinstance(non_zero_masked_scalar_data, np.ndarray):
                    print 'non zero masked scalar data : ' + str(len(non_zero_masked_scalar_data))
        return values
