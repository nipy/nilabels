import nibabel as nib
from scipy import ndimage

from nilabels.tools.aux_methods.label_descriptor_manager import LabelsDescriptorManager
from nilabels.tools.aux_methods.utils_nib import compare_two_nib
from nilabels.tools.aux_methods.utils_path import connect_path_tail_head
from nilabels.tools.detections.check_imperfections import check_missing_labels


class LabelsChecker(object):
    """
    Facade of the methods in tools, for work with paths to images rather than
    with data. Methods under LabelsManagerFuse access label fusion methods.
    """
    def __init__(self, input_data_folder=None, output_data_folder=None):
        self.pfo_in = input_data_folder
        self.pfo_out = output_data_folder

    def missing_labels(self, path_input_segmentation, path_input_labels_descriptor, pfi_where_to_save_the_log_file=None):
        pfi_segm = connect_path_tail_head(self.pfo_in, path_input_segmentation)
        pfi_ld   = connect_path_tail_head(self.pfo_in, path_input_labels_descriptor)
        ldm = LabelsDescriptorManager(pfi_ld)
        im_se = nib.load(pfi_segm)
        in_descriptor_not_delineated, delineated_not_in_descriptor = \
            check_missing_labels(im_se, ldm, pfi_where_log=pfi_where_to_save_the_log_file)
        return in_descriptor_not_delineated, delineated_not_in_descriptor

    def number_connected_components_per_label(self, input_segmentation, where_to_save_the_log_file=None):
        pfi_segm = connect_path_tail_head(self.pfo_in, input_segmentation)
        im = nib.load(pfi_segm)
        msg = 'Labels check number of connected components for segmentation {} \n\n'.format(pfi_segm)
        labs_list, cc_list = [], []
        for lab in sorted(list(set(im.get_data().flat))):
            cc = ndimage.label(im.get_data() == lab)[1]
            msg_l = 'Label {} has {} connected components'.format(lab, cc)
            print(msg_l)
            msg += msg_l + '\n'
            labs_list.append(lab)
            cc_list. append(cc)
        if where_to_save_the_log_file is not None:
            f = open(where_to_save_the_log_file, 'w')
            f.write(msg)
            f.close()
        return labs_list, cc_list

    def compare_two_nifti_images(self, path_first_image, path_second_image):
        pfi_first  = connect_path_tail_head(self.pfo_in, path_first_image)
        pfi_second = connect_path_tail_head(self.pfo_in, path_second_image)
        im1 = nib.load(pfi_first)
        im2 = nib.load(pfi_second)
        return compare_two_nib(im1, im2)
