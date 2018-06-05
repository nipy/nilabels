import nibabel as nib
from scipy import ndimage

from LABelsToolkit.tools.aux_methods.utils_path import connect_path_tail_head
from LABelsToolkit.tools.detections.check_imperfections import check_missing_labels
from LABelsToolkit.tools.descriptions.manipulate_descriptors import LabelsDescriptorManager


class LABelsToolkitChecker(object):
    """
    Facade of the methods in tools, for work with paths to images rather than
    with data. Methods under LabelsManagerFuse access label fusion methods.
    """
    def __init__(self, input_data_folder=None, output_data_folder=None):
        self.pfo_in = input_data_folder
        self.pfo_out = output_data_folder

    def missing_labels(self, pfi_segmentation, pfi_labels_descriptor, pfi_where_to_save_the_log_file=None):
        ldm = LabelsDescriptorManager(pfi_labels_descriptor)
        im_se = nib.load(pfi_segmentation)
        check_missing_labels(im_se, ldm, pfi_where_log=pfi_where_to_save_the_log_file)

    def number_connected_components_per_label(self, input_segmentation, where_to_save_the_log_file=None):
        pfi_segm = connect_path_tail_head(self.pfo_in, input_segmentation)
        im = nib.load(pfi_segm)
        msg = 'Labels check number of connected components for segmentation {} \n\n'.format(pfi_segm)
        for l in sorted(list(set(im.get_data().flat))):
            msg_l = 'Label {} has {} connected components'.format(l, ndimage.label(im.get_data() == l)[1])
            print(msg_l)
            msg += msg_l + '\n'

        if where_to_save_the_log_file is not None:
            f = open(where_to_save_the_log_file, 'w')
            f.write(msg)
            f.close()
