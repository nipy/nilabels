import pandas as pa
import numpy as np

from skimage import measure

from LABelsToolkit.tools.aux_methods.utils_nib import one_voxel_volume


def check_missing_labels(im_segm, labels_descriptor, pfi_where_log=None):
    """
    :param im_segm: nibabel image of a segmentation.
    :param labels_descriptor: instance of LabelsDescriptorManager
    :return: correspondences between labels in the segmentation and labels in the label descriptors,
    number of voxels, volume and number of connected components per label, all in a log file.
    """
    assert isinstance(labels_descriptor, LabelsDescriptorManager)
    labels_dict = labels_descriptor.get_dict(as_string=False)
    labels_list = labels_dict.keys()

    labels_in_the_image = set(im_segm.get_data().astype(np.int).flatten())
    intersection = labels_in_the_image & set(labels_list)
    msg = 'Labels in the descriptor not delineated: \n{}\n'.format(set(labels_list) - intersection)
    msg += 'Labels delineated not in the descriptor: \n{}'.format(labels_in_the_image - intersection)
    print(msg)

    if pfi_where_log is not None:
        labels_names = [labels_dict[l][2] for l in labels_list]

        num_voxels_per_label = []
        num_connected_components_per_label = []
        for label_k in labels_list:
            all_places = im_segm.get_data() == label_k

            cc_l = len(set(list(measure.label(all_places, background=0).flatten()))) - 1
            num_connected_components_per_label.append(cc_l)

            len_non_zero_places = len(all_places[np.where(all_places > 1e-6)])
            if len_non_zero_places == 0:
                msg_l = '\nLabel {0} present in label descriptor and not delineated in the given segmentation.'.format(label_k)
                msg +=  msg_l
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



if __name__ == "__main__":
    # TODO move to examples:
    import nibabel as nib
    from LABelsToolkit.tools.descriptions.manipulate_descriptors import LabelsDescriptorManager

    pfi_segm = '/Users/sebastiano/Desktop/test_segmentation.nii.gz'
    pfi_ld = '/Users/sebastiano/Desktop/labels_descriptor.txt'

    pfi_output_msg = '/Users/sebastiano/Desktop/output.txt'

    ldm = LabelsDescriptorManager(pfi_ld)

    im_se = nib.load(pfi_segm)
    check_missing_labels(im_se, ldm, pfi_output_msg)
