
import pandas as pa
import numpy as np

from labels_manager.tools.descriptions.manipulate_descriptors import LabelsDescriptorManager
from labels_manager.tools.aux_methods.utils_nib import one_voxel_volume


def check_missing_labels(im_segm, labels_descriptor, pfi_where_log=None):
    """

    :param im_segm:
    :param labels_descriptor: instance of LabelsDescriptorManager
    :return:
    """


    assert isinstance(labels_descriptor, LabelsDescriptorManager)
    labels_dict = labels_descriptor.get_dict(as_string=False)
    labels_list = labels_dict.keys()
    labels_names = [labels_dict[l][2] for l in labels_list]

    labels_in_the_image = set(im_segm.get_data().flatten())
    intersection = labels_in_the_image & set(labels_list)
    print('Labels in the descriptor not delineated : \n{}'.format(set(labels_list) - intersection))
    print('Labels delineated not in the descriptor : \n{}'.format(labels_in_the_image - intersection))

    num_voxels_per_label = []
    for label_k in labels_list:
        all_places = im_segm.get_data() == label_k
        len_non_zero_places = len(all_places[np.where(all_places > 1e-6)])
        if len_non_zero_places == 0:
            print('Label {0} present in label descriptor and not delineated in the given segmentation'.format(label_k))
        num_voxels_per_label.append(len_non_zero_places)

    se_voxels = pa.Series(num_voxels_per_label, index=labels_names)
    se_volume = pa.Series(one_voxel_volume(im_segm) * np.array(num_voxels_per_label), index=labels_names)

    df = pa.DataFrame({'Num voxels' :se_voxels, 'Volumes' : se_volume})
    # print df


if __name__ == "__main__":
    # TODO move to examples
    import nibabel as nib
    from labels_manager.tools.descriptions.manipulate_descriptors import LabelsDescriptorManager

    pfi_segm = '/Users/sebastiano/Desktop/1305_approved_v2.nii.gz'
    pfi_ld = '/Users/sebastiano/Desktop/labels_descriptor.txt'

    ldm = LabelsDescriptorManager(pfi_ld)

    im_se = nib.load(pfi_segm)
    check_missing_labels(im_se, ldm)

