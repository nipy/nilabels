import os

import nibabel as nib
import numpy as np
import pandas as pa

from nilabels.definitions import definition_label
from nilabels.tools.aux_methods.utils import labels_query
from nilabels.tools.aux_methods.utils_path import connect_path_tail_head
from nilabels.tools.caliber.distances import dice_score, covariance_distance, \
    hausdorff_distance, global_outline_error, global_dice_score, normalised_symmetric_contour_distance
from nilabels.tools.caliber.volumes_and_values import get_values_below_labels_list, get_volumes_per_label


class LabelsMeasure(object):
    """
    Facade of the methods in tools.detections and tools.caliber, where methods are accessed through
    paths to images rather than with data.
    Methods under LabelsManagerDetect are taking in general one or more input
    and return some feature of the segmentations \n {}.
    """.format(definition_label)

    def __init__(self, input_data_folder=None, output_data_folder=None, return_mm3=True, verbose=0):
        self.pfo_in     = input_data_folder
        self.pfo_out    = output_data_folder
        self.return_mm3 = return_mm3
        self.verbose    = verbose

    def volume(self, segmentation_filename, labels=None, tot_volume_prior=None, where_to_save=None):
        """
        :param segmentation_filename: filename of the segmentation S.
        :param labels: list of labels, multi-labels (as sublists, e.g. left right will be considered one label)
        :param tot_volume_prior: as an intra-cranial volume factor.
        :param where_to_save:
        :return:
        """
        pfi_segm = connect_path_tail_head(self.pfo_in, segmentation_filename)
        assert os.path.exists(pfi_segm), pfi_segm
        im_segm = nib.load(pfi_segm)
        labels_list, labels_names = labels_query(labels, im_segm.get_data())
        df_volumes_per_label = get_volumes_per_label(im_segm, labels=labels_list, labels_names=labels_names,
                                                     tot_volume_prior=tot_volume_prior, verbose=self.verbose)
        if self.verbose > 0:
            print(df_volumes_per_label)
        if where_to_save is not None:
            pfi_output_table = connect_path_tail_head(self.pfo_out, where_to_save)
            df_volumes_per_label.to_pickle(pfi_output_table)
        return df_volumes_per_label

    def get_total_volume(self, segmentation_filename):
        return self.volume(segmentation_filename, labels='tot')

    def values_below_labels(self, segmentation_filename, anatomy_filename, labels=None):
        """
        :param segmentation_filename:
        :param anatomy_filename:
        :param labels:
        :return: pandas series with label names and corresponding vectors of labels values
        """
        pfi_anat = connect_path_tail_head(self.pfo_in, anatomy_filename)
        pfi_segm = connect_path_tail_head(self.pfo_in, segmentation_filename)
        assert os.path.exists(pfi_anat)
        assert os.path.exists(pfi_segm)
        im_anat = nib.load(pfi_anat)
        im_segm = nib.load(pfi_segm)

        labels_list, labels_names = labels_query(labels, segmentation_array=im_segm.get_data())
        labels_values = get_values_below_labels_list(im_segm, im_anat, labels_list)
        return pa.Series(labels_values, index=labels_names)

    def dist(self, segm_1_filename, segm_2_filename, labels_list=None, labels_names=None,
             metrics=(dice_score, covariance_distance, hausdorff_distance, normalised_symmetric_contour_distance),
             where_to_save=None):

        pfi_segm1 = connect_path_tail_head(self.pfo_in, segm_1_filename)
        pfi_segm2 = connect_path_tail_head(self.pfo_in, segm_2_filename)
        assert os.path.exists(pfi_segm1), pfi_segm1
        assert os.path.exists(pfi_segm2), pfi_segm2

        if self.verbose > 0:
            print("Distances between segmentations: \n -> {0} \n -> {1} \n...started!".format(pfi_segm1, pfi_segm2))

        im_segm1 = nib.load(pfi_segm1)
        im_segm2 = nib.load(pfi_segm2)
        if labels_list is None:
            labels_list1, labels_names1 = labels_query('all', im_segm1.get_data())
            labels_list2, labels_names2 = labels_query('all', im_segm2.get_data())
            labels_list  = list(set(labels_list1) & set(labels_list2))
            labels_list.sort(key=int)
            labels_names = None

        if labels_names is None:
            labels_names = labels_list

        dict_distances_per_label = {}

        for d in metrics:
            if self.verbose > 0:
                print('{} computation started'.format(d.func_name))
            pa_se = d(im_segm1, im_segm2, labels_list, labels_names, self.return_mm3)  # TODO get as function with variable number of arguments
            dict_distances_per_label.update({d.func_name : pa_se})

        df_distances_per_label = pa.DataFrame(dict_distances_per_label,
                                              columns=dict_distances_per_label.keys())

        # df_distances_per_label.loc['mean'] = df_distances_per_label.mean()
        # df_distances_per_label.loc['std'] = df_distances_per_label.std()

        if self.verbose > 0:
            print(df_distances_per_label)

        if where_to_save is not None:
            pfi_output_table = connect_path_tail_head(self.pfo_out, where_to_save)
            df_distances_per_label.to_pickle(pfi_output_table)

        return df_distances_per_label

    def global_dist(self, segm_1_filename, segm_2_filename, where_to_save=None,
                    global_metrics=(global_outline_error, global_dice_score)):

        pfi_segm1 = connect_path_tail_head(self.pfo_in, segm_1_filename)
        pfi_segm2 = connect_path_tail_head(self.pfo_in, segm_2_filename)

        assert os.path.exists(pfi_segm1), pfi_segm1
        assert os.path.exists(pfi_segm2), pfi_segm2

        if self.verbose > 0:
            print("\nGlobal distances between segmentations: \n -> {0} \n -> {1} "
                  "\nComputations started!".format(pfi_segm1, pfi_segm2))

        im_segm1 = nib.load(pfi_segm1)
        im_segm2 = nib.load(pfi_segm2)

        se_global_distances = pa.Series(np.array([d(im_segm1, im_segm2) for d in global_metrics]),
                                        index=[d.__name__ for d in global_metrics])
        if where_to_save is not None:
            where_to_save = connect_path_tail_head(self.pfo_out, where_to_save)
            se_global_distances.to_pickle(where_to_save)

        return se_global_distances

    def topology(self):
        # TODO: island detections, graph detections, cc detections from detector tools
        print('topology for {} is in the TODO list!'.format(self.__class__))

    def groupwise_global_measures_comparisons(self, list_path_A, list_path_B, pfo_where_to_save,
                                              name_list_path_A=None, name_list_path_B=None,
                                              list_distances=(global_dice_score, global_outline_error),
                                              prefix_output='distances_comparison', save_human_readable=True,
                                              verbose=1):
        list_path_A = [connect_path_tail_head(self.pfo_in, ph) for ph in list_path_A]
        list_path_B = [connect_path_tail_head(self.pfo_in, ph) for ph in list_path_B]

        # input sanity check:
        for pfi_im_A in list_path_A:
            assert os.path.exists(pfi_im_A), pfi_im_A
        for pfi_im_B in list_path_B:
            assert os.path.exists(pfi_im_B), pfi_im_B
        if name_list_path_A is None:
            name_list_path_A = [os.path.basename(n).replace('.nii', '').replace('.gz', '') for n in list_path_A]
        if name_list_path_B is None:
            name_list_path_B = [os.path.basename(n).replace('.nii', '').replace('.gz', '') for n in list_path_B]

        # Initialise one data-frame for each metric/score selected
        dictionary_of_measurements = {}
        for d in list_distances:
            dictionary_of_measurements.update({d.__name__: pa.DataFrame(np.zeros([len(list_path_A), len(list_path_B)]),
                                               index=name_list_path_A, columns=name_list_path_B)})

        # Fill values in each data-frame
        for pfi_im_A, name_A in zip(list_path_A, name_list_path_A):
            for pfi_im_B, name_B in zip(list_path_B, name_list_path_B):

                im_A = nib.load(pfi_im_A)
                im_B = nib.load(pfi_im_B)
                for d in list_distances:
                    d_A_B = d(im_A, im_B)
                    dictionary_of_measurements[d.__name__][name_B][name_A] = d_A_B
                    if verbose > 0:
                        print('{0:<15} {1:<15} \n{2:<15} : {3}'.format(pfi_im_A, pfi_im_B, d.__name__, d_A_B))
                        print(dictionary_of_measurements[d.__name__])

        # prepare output folder
        os.system('mkdir -p {}'.format(pfo_where_to_save))

        for d in list_distances:
            # Save each dataframe independently
            pfi_df_global_dice_score = os.path.join(pfo_where_to_save, '{0}_{1}.pickle'.format(prefix_output, d.__name__))
            dictionary_of_measurements[d.__name__].to_pickle(pfi_df_global_dice_score)
            if save_human_readable:
                pfi_df_global_dice_score_txt = os.path.join(pfo_where_to_save, '{0}_{1}.txt'.format(prefix_output, d.__name__))
                with open(pfi_df_global_dice_score_txt, 'w') as outfile:
                    dictionary_of_measurements[d.__name__].to_string(outfile)
