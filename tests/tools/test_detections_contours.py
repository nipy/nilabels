import os
import numpy as np
import nibabel as nib

from nilabels.tools.detections.contours import contour_from_array_at_label
from nilabels.tools.aux_methods.label_descriptor_manager import LabelsDescriptorManager


def test_contour_from_array_at_label():

    im_arr  = np.zeros([10, 10, 10])


    # TODO
    pass


def test_contour_from_segmentation():
    pass


def test_get_xyz_borders_of_a_label():
    pass


def test_get_internal_contour_with_erosion_at_label():
    pass


if __name__ == "__main__":

    test_contour_from_array_at_label()
    test_contour_from_segmentation()
    test_get_xyz_borders_of_a_label()
    test_get_internal_contour_with_erosion_at_label()
