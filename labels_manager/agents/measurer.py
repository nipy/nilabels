import os
import nibabel as nib
import pandas as pa
import numpy as np

from labels_manager.tools.aux_methods.utils import labels_query
from labels_manager.tools.aux_methods.utils_path import connect_path_tail_head
from labels_manager.tools.caliber.volumes import get_volumes_per_label, get_average_below_labels
from labels_manager.tools.caliber.distances import dice_score, dispersion, precision, covariance_distance, \
    hausdorff_distance
from labels_manager.tools.defs import definition_label


class LabelsManagerMeasure(object):
    """
    Facade of the methods in tools.detections and tools.caliber, where methods are accessed through
    paths to images rather than with data.
    Methods under LabelsManagerDetect are taking in general one or more input
    and return some feature of the segmentations {}.
    """.format(definition_label)

    def __init__(self, input_data_folder=None, output_data_folder=None, return_mm3=True, verbose=0):
        self.pfo_in = input_data_folder
        self.pfo_out = output_data_folder
        self.return_mm3 = return_mm3
        self.verbose = verbose

    def volume(self, segmentation_filename, labels=None, anatomy_filename=None, tot_volume_prior=None,
               where_to_save=None):
        """
        :param segmentation_filename: filename of the segmentation S.
        :param labels: list of labels, multi-labels (as sublists, e.g. left right will be considered one label)
        :param anatomy_filename: filename of the anatomical image A. (A,S) forms a chart.
        :param tot_volume_prior: as an intra-cranial volume factor.
        :param where_to_save:
        :return:
        """
        pfi_segm = connect_path_tail_head(self.pfo_in, segmentation_filename)
        assert os.path.exists(pfi_segm)
        im_segm = nib.load(pfi_segm)

        labels_list, labels_names = labels_query(labels, im_segm.get_data())

        df_volumes_per_label = get_volumes_per_label(im_segm, labels=labels_list, labels_names=labels_names,
                                                     tot_volume_prior=tot_volume_prior, verbose=self.verbose)
        if anatomy_filename is not None:
            pfi_anatomy = connect_path_tail_head(self.pfo_in, anatomy_filename)
            assert os.path.exists(pfi_anatomy)
            im_anatomy = nib.load(pfi_anatomy)
            df_average_below_labels = get_average_below_labels(im_segm, im_anatomy, labels_list, labels_names=labels_names,
                                                               verbose=self.verbose)

            df_volumes_per_label['Average below label'] = df_average_below_labels.values

        if self.verbose > 0:
            print(df_volumes_per_label)

        if where_to_save is not None:
            pfi_output_table = connect_path_tail_head(self.pfo_out, where_to_save)
            assert os.path.exists(pfi_output_table)
            df_volumes_per_label.to_pickle(pfi_output_table)
        return df_volumes_per_label

    def get_total_volume(self, segmentation_filename):

        return self.volume(segmentation_filename, labels='tot')

    def dist(self, segm_1_filename, segm_2_filename, labels=None,
             metrics=(dice_score, dispersion, covariance_distance, hausdorff_distance),
             where_to_save=None):

        pfi_segm1 = connect_path_tail_head(self.pfo_in, segm_1_filename)
        pfi_segm2 = connect_path_tail_head(self.pfo_in, segm_2_filename)

        assert os.path.exists(pfi_segm1), pfi_segm1
        assert os.path.exists(pfi_segm2), pfi_segm2

        if self.verbose > 0:
            print("Distances between segmentations: \n -> {0} \n -> {1} \n...started!".format(pfi_segm1, pfi_segm2))

        im_segm1 = nib.load(pfi_segm1)
        im_segm2 = nib.load(pfi_segm2)

        labels_list1, labels_names1 = labels_query(labels, im_segm1.get_data())
        labels_list2, labels_names2 = labels_query(labels, im_segm2.get_data())

        labels_list  = list(set(labels_list1) & set(labels_list2))
        labels_names = list(set(labels_names1) & set(labels_names2))

        labels_list.sort(key=int)
        labels_names.sort(key=int)

        if self.verbose > 0:
            print("Labels image 1: {}".format(labels_list1))
            print("Labels image 2: {}".format(labels_list2))
            print("Labels intersection {}".format(labels_list))
            disjoint_union = list( (set(labels_names1) | set(labels_names2)) - (set(labels_names1) & set(labels_names2)) )
            print("Labels disjoint union {}".format(disjoint_union))

        dict_distances_per_label = {}

        for d in metrics:
            if self.verbose > 0:
                print('{} computation started'.format(d.func_name))
            pa_se = d(im_segm1, im_segm2, labels_list, labels_names, self.return_mm3)
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

    def topology(self):
        # WIP: island detections, graph detections, cc detections from detector tools
        print('topology for {} is in the TODO list!'.format(self.__class__))
