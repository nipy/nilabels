import os
import numpy as np
import nibabel as nib

from nilabels.tools.detections.check_imperfections import check_missing_labels
from nilabels.tools.aux_methods.label_descriptor_manager import LabelsDescriptorManager

from tests.tools.decorators_tools import pfo_tmp_test, \
    write_and_erase_temporary_folder_with_dummy_labels_descriptor


@write_and_erase_temporary_folder_with_dummy_labels_descriptor
def test_check_missing_labels():

    # Instantiate a labels descriptor manager
    pfi_ld = os.path.join(pfo_tmp_test, 'labels_descriptor.txt')
    ldm = LabelsDescriptorManager(pfi_ld, labels_descriptor_convention='itk-snap')

    # Create dummy image
    data = np.zeros([10, 10, 10])
    data[:3, :3, :3] = 1
    data[3:5, 3:5, 3:5] = 12
    data[5:7, 5:7, 5:7] = 3
    data[7:10, 7:10, 7:10] = 7

    im_segm = nib.Nifti1Image(data, affine=np.eye(4))

    # Apply check_missing_labels, then test the output
    pfi_log = os.path.join(pfo_tmp_test, 'check_imperfections_log.txt')
    in_descriptor_not_delineated, delineated_not_in_descriptor = check_missing_labels(im_segm, ldm, pfi_log)

    print(in_descriptor_not_delineated, delineated_not_in_descriptor)

    np.testing.assert_equal(in_descriptor_not_delineated, {8, 2, 4, 5, 6})  # in label descriptor, not in image
    np.testing.assert_equal(delineated_not_in_descriptor, {12})  # in image not in label descriptor
    assert os.path.exists(pfi_log)


if __name__ == "__main__":

    test_check_missing_labels()

