import numpy as np
import pandas as pa
import os
import nibabel as nib

from labels_manager.tools.manipulations_colors.relabeller import keep_only_one_label
from labels_manager.tools.aux_methods.utils_nib import set_new_data
from labels_manager.tools.aux_methods.utils import print_and_run


def lncc_distance(values_patch1, values_patch2):
    """
    Import values below the patches, containing the same number of eolem
    :param values_patch1:
    :param values_patch2:
    :return:
    """
    patches = [values_patch1.flatten(), values_patch2.flatten()]
    np.testing.assert_array_equal(patches[0].shape, patches[1].shape)

    for index_p, p in enumerate(patches):
        den = float(np.linalg.norm(p))
        if den == 0: patches[index_p] = np.zeros_like(p)
        else: patches[index_p] = patches[index_p] / den

    return patches[0].dot(patches[1])


def centroid_array(arr, labels):
    centers_of_mass = [np.array([0, 0, 0])] * len(labels)
    for l_id, l in enumerate(labels):
        coordinates_l = np.where(arr == l)  # returns [X_vector, Y_vector, Z_vector]
        centers_of_mass[l_id] = (1 / float(len(coordinates_l[0]))) * np.array([np.sum(k) for k in coordinates_l])
    return centers_of_mass


def centroid(im, labels, real_space_coordinates=True):
    """
    Centroid (center of mass, baricenter) of a list of labels.
    :param im:
    :param labels: list of labels, e.g. [3] or [2, 3, 45]
    :param real_space_coordinates: if true the answer is in mm if false in voxel indexes.
    :return: list of centroids, one for each label in the input order.
    """
    centers_of_mass = centroid_array(im.get_data(), labels)

    if real_space_coordinates:
        centers_of_mass = [im.affine[:3, :3].dot(cm.astype(np.float64)) for cm in centers_of_mass]
    else:
        centers_of_mass = [np.round(cm).astype(np.uint64) for cm in centers_of_mass]
    return centers_of_mass


def dice_score(im_segm1, im_segm2, labels_list, labels_names):
    """
    Dice score between paired labels of segmentations.
    :param im_segm1: nibabel image with labels
    :param im_segm2: nibabel image with labels
    :param labels_list:
    :param labels_names:
    :return: dice score of the label label of the two segmentations.
    """
    def dice_score_l(lab):
        place1 = im_segm1.get_data() == lab
        place2 = im_segm2.get_data() == lab
        assert isinstance(place1, np.ndarray)
        assert isinstance(place2, np.ndarray)
        return 2 * np.count_nonzero(place1 * place2) / (np.count_nonzero(place1) + np.count_nonzero(place2))

    scores = np.zeros(len(labels_list), dtype=np.float64)

    for id_l , l in enumerate(labels_list):
        scores[id_l] = dice_score_l(l)

    return pa.Series(scores, index=labels_names)


def dispersion(im_segm1, im_segm2, labels_list=None, labels_names=None, return_mm3=True):

    def dispersion_l(lab):
        place1 = im_segm1.get_data() == lab
        place2 = im_segm2.get_data() == lab
        c1 = centroid_array(place1, labels=[True,])[0]
        c2 = centroid_array(place2, labels=[True,])[0]
        if return_mm3:
            c1 = im_segm1.affine[:3, :3].dot(c1.astype(np.float64))
            c2 = im_segm1.affine[:3, :3].dot(c2.astype(np.float64))
        return np.sqrt((c1 - c2)**2)

    np.testing.assert_array_almost_equal(im_segm1.affine, im_segm2.affine)
    return pa.Series(np.array([dispersion_l(l) for l in labels_list]), index=labels_names)


def precision(im_segm1, im_segm2, pfo_intermediate_files, labels_list, labels_names):

    print_and_run('mkdir -p {}'.format(pfo_intermediate_files))
    precision_per_label = []

    for l in labels_list:
        im_segm_single1 = set_new_data(im_segm1, keep_only_one_label(im_segm1.get_data(), l))
        im_segm_single2 = set_new_data(im_segm2, keep_only_one_label(im_segm2.get_data(), l))

        pfi_im_segm_single1 = os.path.join(pfo_intermediate_files, 'im_segm_single_lab{}_1.nii.gz'.format(l))
        pfi_im_segm_single2 = os.path.join(pfo_intermediate_files, 'im_segm_single_lab{}_2.nii.gz'.format(l))

        nib.save(im_segm_single1, pfi_im_segm_single1)
        nib.save(im_segm_single2, pfi_im_segm_single2)

        pfi_warp_1_2 = os.path.join(pfo_intermediate_files, 'warped_single_lab{}_1_on_2.nii.gz'.format(l))
        pfi_warp_2_1 = os.path.join(pfo_intermediate_files, 'warped_single_lab{}_2_on_1.nii.gz'.format(l))
        pfi_aff_1_2 = os.path.join(pfo_intermediate_files, 'aff_single_lab{}_1_on_2.txt'.format(l))
        pfi_aff_2_1 = os.path.join(pfo_intermediate_files, 'warped_single_lab{}_2_on_1.txt'.format(l))

        cmd1_2 = 'reg_aladin -ref {0} -flo {1} -res {2} -aff {3} -interp 0'.format(pfi_im_segm_single1,
                                                                                   pfi_im_segm_single2,
                                                                                   pfi_warp_1_2,
                                                                                   pfi_aff_1_2)
        cmd2_1 = 'reg_aladin -ref {0} -flo {1} -res {2} -aff {3} -interp 0'.format(pfi_im_segm_single2,
                                                                                   pfi_im_segm_single1,
                                                                                   pfi_warp_2_1,
                                                                                   pfi_aff_2_1)
        print_and_run(cmd1_2)
        print_and_run(cmd2_1)

        t_1_2 = np.loadtxt(pfi_aff_1_2)
        t_2_1 = np.loadtxt(pfi_aff_2_1)

        precision_per_label.append(np.max(np.abs(np.linalg.det(t_1_2)) ** (-1), np.abs(np.linalg.det(t_2_1)) ** (-1)))

    pa_series = pa.Series(np.array(precision_per_label), index=labels_names)
    pa_series.dump_to_pickle(os.path.join(pfo_intermediate_files, 'final_precision.pickle'))

    return pa_series


# TODO below --------

def box_sides(im_segmentation, labels_to_box):
    """
    im_segmentation, labels_to_box, labels_to_box_names
    Box surrounding the label in the list labels_to_box.
    A box is an (ordered!) couple of 3d points.
    :param im_segmentation:
    :param labels_to_box:
    :param labels_to_box_names:
    :return:
    """
    one_label_data = keep_only_one_label(im_segmentation, label_to_keep=labels_to_box)
    ans = []
    # for d in range(len(one_label_data.shape)):
    #     ans.append(np.sum(binarise_a_matrix(np.sum(one_label_data, axis=d), dtype=np.int)))
    # return ans
    pass


def hausdorff_distance():
    # see Abdel Aziz Taha and Allan Hanbury 2015
    # An Efficient Algorithm for Calculating the Exact Hausdorff Distance
    pass
