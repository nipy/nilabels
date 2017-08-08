import numpy as np
import nibabel as nib

from labels_manager.tools.aux_methods.utils import print_and_run, set_new_data


def divide_by_median_below_labels_path(pfi_input, pfi_segmentation, pfi_output, thr_zeros=True):

    im_input = nib.load(pfi_input)
    im_segm = nib.load(pfi_segmentation)

    list_labels = list(np.sort(list(set(im_segm.get_data().flat))))
    mask_data = np.zeros_like(im_segm.get_data(), dtype=np.bool)
    for label_k in list_labels[1:]:  # want to exclude the first one
        mask_data += im_segm.get_data() == label_k

    masked_im_data = np.nan_to_num((mask_data.astype(np.float64) * im_input.get_data().astype(np.float64)).flatten())
    non_zero_masked_im_data = masked_im_data[np.where(masked_im_data > 1e-6)]
    median = np.median(non_zero_masked_im_data)
    assert isinstance(median, float)
    output_im = set_new_data(im_input, (1 / float(median))* im_input.get_data())
    nib.save(output_im, pfi_output)

    if thr_zeros:
        cmd = 'seg_maths {0} -thr 0 {0}'.format(pfi_output)
        print_and_run(cmd)
