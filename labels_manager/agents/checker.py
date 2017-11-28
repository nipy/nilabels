import os
import numpy as np
import nibabel as nib

from labels_manager.tools.aux_methods.utils_path import connect_path_tail_head
from labels_manager.tools.aux_methods.utils_nib import set_new_data
from labels_manager.tools.detections.check_imperfections import check_missing_labels


class LabelsManagerChecker(object):
    """
    Facade of the methods in tools, for work with paths to images rather than
    with data. Methods under LabelsManagerFuse access label fusion methods.
    """
    def __init__(self, input_data_folder=None, output_data_folder=None):
        self.pfo_in = input_data_folder
        self.pfo_out = output_data_folder

    def missing_labels(self, pfi_segmentation, pfi_labels_descriptor=None):
        # TODO with check_segmentation
        pass