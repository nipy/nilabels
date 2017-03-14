import os
import nibabel as nib
import numpy as np

from labels_manager.tools.aux_methods.sanity_checks import connect_tail_head_path


class LabelsManagerMeasure(object):

    def __init__(self, input_data_folder=None, output_data_folder=None):
        self.pfo_in = input_data_folder
        self.pfo_out = output_data_folder


    def get_list_labels(self, filename_in):
        pfi_in = connect_tail_head_path(self.pfo_in, filename_in)
        im_seg = nib.load(pfi_in)
        list_labels = np.sort(list(set(im_seg.get_data().flatten())))
        return list_labels
