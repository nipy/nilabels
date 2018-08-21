import nibabel as nib


from nilabel.tools.aux_methods.label_descriptor_manager import LabelsDescriptorManager
from nilabel.main import Nilabel as NiL


pfi_segm = '/Users/sebastiano/Desktop/test_segmentation.nii.gz'
pfi_ld = '/Users/sebastiano/Desktop/labels_descriptor.txt'

pfi_output_msg = '/Users/sebastiano/Desktop/output.txt'

ldm = LabelsDescriptorManager(pfi_ld)

im_se = nib.load(pfi_segm)
la = NiL()

# TODO
la.check.missing_labels()
# check_missing_labels(im_se, ldm, pfi_output_msg)
