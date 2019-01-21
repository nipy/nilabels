import numpy as np
import pandas as pa
from skimage import measure

from nilabels.tools.aux_methods.utils_nib import one_voxel_volume
from nilabels.tools.aux_methods.label_descriptor_manager import LabelsDescriptorManager


def check_missing_labels(im_segm, labels_descriptor, pfi_where_log=None):
    """
    :param im_segm: nibabel image of a segmentation.
    :param labels_descriptor: instance of LabelsDescriptorManager
    :param pfi_where_log: path to file where to save a log of the output.
    :return: correspondences between labels in the segmentation and labels in the label descriptors,
    number of voxels, volume and number of connected components per label, all in a log file.
    """
    assert isinstance(labels_descriptor, LabelsDescriptorManager)
    labels_dict = labels_descriptor.get_dict_itk_snap()
    labels_list_from_descriptor = labels_dict.keys()

    labels_in_the_image = set(im_segm.get_data().astype(np.int).flatten())
    intersection = labels_in_the_image & set(labels_list_from_descriptor)

    in_descriptor_not_delineated = set(labels_list_from_descriptor) - intersection
    delineated_not_in_descriptor = labels_in_the_image - intersection

    msg = 'Labels in the descriptor not delineated: \n{}\n'.format(in_descriptor_not_delineated)
    msg += 'Labels delineated not in the descriptor: \n{}'.format(delineated_not_in_descriptor)
    print(msg)

    if pfi_where_log is not None:
        labels_names = [labels_dict[l][2] for l in labels_list_from_descriptor]

        num_voxels_per_label = []
        num_connected_components_per_label = []
        for label_k in labels_list_from_descriptor:
            all_places = im_segm.get_data() == label_k

            cc_l = len(set(list(measure.label(all_places, background=0).flatten()))) - 1
            num_connected_components_per_label.append(cc_l)

            len_non_zero_places = len(all_places[np.where(all_places > 1e-6)])
            if len_non_zero_places == 0:
                msg_l = '\nLabel {0} present in label descriptor and not delineated in the ' \
                        'given segmentation.'.format(label_k)
                msg += msg_l
                print(msg_l)
            num_voxels_per_label.append(len_non_zero_places)

        se_voxels = pa.Series(num_voxels_per_label, index=labels_names)
        se_volume = pa.Series(one_voxel_volume(im_segm) * np.array(num_voxels_per_label), index=labels_names)

        df = pa.DataFrame({'Num voxels' :se_voxels, 'Volumes' : se_volume,
                           'Connected components': num_connected_components_per_label})

        df.index = df.index.map('{:<30}'.format)
        df['Num voxels'] = df['Num voxels'].map('{:<10}'.format)
        df['Volumes'] = df['Volumes'].map('{:<10}'.format)
        df['Connected components'] = df['Connected components'].map('{:<10}'.format)

        f = open(pfi_where_log, "w+")
        df.to_string(f)
        f.close()

        f = open(pfi_where_log, "a+")
        f.write('\n\n' + msg)
        f.close()

        print('Log status saved in {}'.format(pfi_where_log))

    return in_descriptor_not_delineated, delineated_not_in_descriptor
