import os
from os.path import join as jph

import nibabel as nib
import numpy as np

import nilabels.tools.image_colors_manipulations.relabeller as rel
import nilabels.tools.image_shape_manipulations.splitter as sp
from nilabels.agents.agents_controller import AgentsController
from nilabels.definitions import root_dir
from nilabels.tools.aux_methods.utils_nib import set_new_data

if __name__ == '__main__':

    if not os.path.exists(jph(root_dir, 'data_examples')):
        print('Run generate_simple_phantoms.py before this, please.')
        raise IOError

    # Create output folder:
    cmd = 'mkdir -p {}'.format(jph(root_dir, 'data_output'))
    os.system(cmd)

    # Instantiate a manager from the class LabelsManager
    lt = AgentsController(jph(root_dir, 'data_examples'), jph(root_dir, 'data_output'))
    print('Input folder: ' + lt._pfo_in)
    print('Output folder: ' + lt._pfo_out)

    run_example = {'Relabel'                          : False,
                   'Permute'                          : False,
                   'Erase'                            : False,
                   'Assign all others a value'        : False,
                   'Keep one label'                   : False,
                   'Extend slice'                     : False,
                   'Split in 4d'                      : False,
                   'Split in 4d only some'            : False,
                   'Merge in 4d'                      : True,
                   'Axial symmetrisation'             : True,
                   'Symmetrisation with registration' : True}

    open_figures = True

    """
    Example 1:
    Relabel the segmentation punt_seg.nii.gz increasing the label values by 1.
    """
    if run_example['Relabel']:

        # data:
        fin_punt_seg_original = 'punt_seg.nii.gz'
        fin_punt_seg_new      = 'punt_seg_relabelled.nii.gz'

        list_old_labels = [1, 2, 3, 4, 5, 6]
        list_new_labels = [2, 3, 4, 5, 6, 7]

        # Using the manager:
        lt.manipulate_labels.relabel(fin_punt_seg_original, fin_punt_seg_new,
                                     list_old_labels, list_new_labels)

        # Without the managers: loading the data and applying the relabeller
        im_seg = nib.load(jph(root_dir, 'data_examples', fin_punt_seg_original))
        data_seg = im_seg.get_data()
        data_seg_new = rel.relabeller(data_seg, list_old_labels, list_new_labels)

        im_relabelled = set_new_data(im_seg, data_seg_new)

        # Results comparison:
        nib_seg_new = nib.load(jph(root_dir, 'data_output', fin_punt_seg_new))
        nib_seg_new_data = nib_seg_new.get_data()
        np.testing.assert_array_equal(data_seg_new, nib_seg_new_data)

        if open_figures:
            # figure before:
            cmd = 'itksnap -g {0} -s {1}'.format(
                   jph(root_dir, 'data_examples', 'punt.nii.gz'),
                   jph(root_dir, 'data_examples', fin_punt_seg_original))
            os.system(cmd)
            # figure after
            cmd = 'itksnap -g {0} -s {1}'.format(
                   jph(root_dir, 'data_examples', 'punt.nii.gz'),
                   jph(root_dir, 'data_output', fin_punt_seg_new))
            os.system(cmd)

    """
    Example 2:
    Permute labels according to a given permutation
    """
    if run_example['Permute']:

        # data:
        fin_punt_seg_original = 'punt_seg.nii.gz'
        fin_punt_seg_new      = 'punt_seg.nii.gz'
        perm = [[1, 2, 3], [3, 1, 2]]

        # Using the manager:
        lt.manipulate_labels.permute_labels(fin_punt_seg_original, fin_punt_seg_new, perm)

        # without the manager:
        im_seg = nib.load(jph(root_dir, 'data_examples', fin_punt_seg_original))
        data_seg = im_seg.get_data()
        data_seg_new = rel.permute_labels(data_seg, perm)

        # Results comparison:
        nib_seg_new = nib.load(jph(root_dir, 'data_output', fin_punt_seg_new))
        nib_seg_new_data = nib_seg_new.get_data()
        np.testing.assert_array_equal(data_seg_new, nib_seg_new_data)

        if open_figures:
            # figure before:
            cmd = 'itksnap -g {0} -s {1}'.format(
                   jph(root_dir, 'data_examples', 'punt.nii.gz'),
                   jph(root_dir, 'data_examples', fin_punt_seg_original))
            os.system(cmd)
            # figure after
            cmd = 'itksnap -g {0} -s {1}'.format(
                   jph(root_dir, 'data_examples', 'punt.nii.gz'),
                   jph(root_dir, 'data_output', fin_punt_seg_new))
            os.system(cmd)

    """
    Example 3:
    Erase some labels.
    """
    if run_example['Erase']:

        # data:
        fin_punt_seg_original = 'punt_seg.nii.gz'
        fin_punt_seg_new      = 'punt_seg.nii.gz'
        labels_to_erase = [4, 5, 6]

        # using the manager:
        lt.manipulate_labels.erase_labels(fin_punt_seg_original, fin_punt_seg_new, labels_to_erase=labels_to_erase)

        # without the manager:
        im_seg = nib.load(jph(root_dir, 'data_examples', fin_punt_seg_original))
        data_seg = im_seg.get_data()
        data_seg_new = rel.erase_labels(data_seg, labels_to_erase)

        # Results comparison:
        nib_seg_new = nib.load(jph(root_dir, 'data_output', fin_punt_seg_new))
        nib_seg_new_data = nib_seg_new.get_data()
        np.testing.assert_array_equal(data_seg_new, nib_seg_new_data)

        if open_figures:
            # figure before:
            cmd = 'itksnap -g {0} -s {1}'.format(
                   jph(root_dir, 'data_examples', 'punt.nii.gz'),
                   jph(root_dir, 'data_examples', fin_punt_seg_original))
            os.system(cmd)
            # figure after
            cmd = 'itksnap -g {0} -s {1}'.format(
                   jph(root_dir, 'data_examples', 'punt.nii.gz'),
                   jph(root_dir, 'data_output', fin_punt_seg_new))
            os.system(cmd)

    """
    Example 4:
    Assign to the excluded labels the same value.
    """
    if run_example['Assign all others a value']:

        # data:
        fin_punt_seg_original = 'punt_seg.nii.gz'
        fin_punt_seg_new      = 'punt_seg.nii.gz'
        labels_to_keep = [1, 3, 5]
        new_value = 100

        # using the manager:
        lt.manipulate_labels.assign_all_other_labels_the_same_value(fin_punt_seg_original,
                                                                    fin_punt_seg_new, labels_to_keep=labels_to_keep, same_value_label=new_value)

        # without the manager:
        im_seg = nib.load(jph(root_dir, 'data_examples', fin_punt_seg_original))
        data_seg = im_seg.get_data()
        data_seg_new = rel.assign_all_other_labels_the_same_value(data_seg,
                labels_to_keep=labels_to_keep, same_value_label=new_value)

        # Results comparison:
        nib_seg_new = nib.load(jph(root_dir, 'data_output', fin_punt_seg_new))
        nib_seg_new_data = nib_seg_new.get_data()
        np.testing.assert_array_equal(data_seg_new, nib_seg_new_data)

        if open_figures:
            # figure before:
            cmd = 'itksnap -g {0} -s {1}'.format(
                   jph(root_dir, 'data_examples', 'punt.nii.gz'),
                   jph(root_dir, 'data_examples', fin_punt_seg_original))
            os.system(cmd)
            # figure after
            cmd = 'itksnap -g {0} -s {1}'.format(
                   jph(root_dir, 'data_examples', 'punt.nii.gz'),
                   jph(root_dir, 'data_output', fin_punt_seg_new))
            os.system(cmd)

    """
    Example 5:
    Keep only one label.
    """
    if run_example['Keep one label']:
        # data:
        fin_punt_seg_original = 'punt_seg.nii.gz'
        fin_punt_seg_new      = 'punt_seg.nii.gz'
        label_to_keep = 6
        # using the manager:
        lt.manipulate_labels.keep_one_label(fin_punt_seg_original, fin_punt_seg_new,
                                            label_to_keep=label_to_keep)
        # without the manager:
        im_seg = nib.load(jph(root_dir, 'data_examples', fin_punt_seg_original))
        data_seg = im_seg.get_data()
        data_seg_new = rel.keep_only_one_label(data_seg, label_to_keep=label_to_keep)
        # Results comparison:
        nib_seg_new = nib.load(jph(root_dir, 'data_output', fin_punt_seg_new))
        nib_seg_new_data = nib_seg_new.get_data()
        np.testing.assert_array_equal(data_seg_new, nib_seg_new_data)

        if open_figures:
            # figure before:
            cmd = 'itksnap -g {0} -s {1}'.format(
                   jph(root_dir, 'data_examples', 'punt.nii.gz'),
                   jph(root_dir, 'data_examples', fin_punt_seg_original))
            os.system(cmd)
            # figure after
            cmd = 'itksnap -g {0} -s {1}'.format(
                   jph(root_dir, 'data_examples', 'punt.nii.gz'),
                   jph(root_dir, 'data_output', fin_punt_seg_new))
            os.system(cmd)

    """
    Example 6:
    extend slice new dimension (not only for segmentations).
    """
    if run_example['Extend slice']:
        # data:
        fin_punt_original = 'punt.nii.gz'
        fin_punt_new      = 'punt_extended.nii.gz'

        fin_punt_seg_original = 'punt_seg.nii.gz'
        fin_punt_seg_new      = 'punt_seg_extended.nii.gz'

        new_axis = 3
        num_slices = 5
        # using the manager:
        lt.manipulate_intensities.extend_slice_new_dimension(fin_punt_original, fin_punt_new,
                                                             new_axis=new_axis, num_slices=num_slices)
        lt.manipulate_intensities.extend_slice_new_dimension(fin_punt_seg_original, fin_punt_seg_new,
                                                             new_axis=new_axis, num_slices=num_slices)
        # without the manager:
        im_seg = nib.load(jph(root_dir, 'data_examples', fin_punt_seg_original))
        data_seg = im_seg.get_data()
        data_seg_new = np.stack([data_seg, ] * num_slices, axis=new_axis)

        # Results comparison:
        nib_seg_new = nib.load(jph(root_dir, 'data_output', fin_punt_seg_new))
        nib_seg_new_data = nib_seg_new.get_data()
        np.testing.assert_array_equal(data_seg_new, nib_seg_new_data)

        if open_figures:
            # figure before:
            cmd = 'itksnap -g {0} -s {1}'.format(
                   jph(root_dir, 'data_examples', fin_punt_original),
                   jph(root_dir, 'data_examples', fin_punt_seg_original))
            os.system(cmd)
            # figure after - (*) the segmentation can be loaded in itksnap in only one slice.
            cmd = 'itksnap -g {0} -s {1}'.format(
                   jph(root_dir, 'data_output', fin_punt_new),
                   jph(root_dir, 'data_examples', fin_punt_seg_original))  # (*)
            os.system(cmd)

    """
    Example 7:
    Split labels in 4d
    """
    if run_example['Split in 4d']:
        # data:
        fin_seg_original = 'mes_seg.nii.gz'
        fin_seg_new      = 'mes_seg_4d.nii.gz'
        # using the manager
        lt.manipulate_intensities.split_in_4d(fin_seg_original, fin_seg_new)
        # without the manager:
        im_seg = nib.load(jph(root_dir, 'data_examples', fin_seg_original))
        data_seg = im_seg.get_data()
        list_labels = list(np.sort(list(set(data_seg.flat))))
        data_seg_new = sp.split_labels_to_4d(data_seg, list_labels)
        # results comparison
        nib_seg_new = nib.load(jph(root_dir, 'data_output', fin_seg_new))
        nib_seg_new_data = nib_seg_new.get_data()
        np.testing.assert_array_equal(data_seg_new, nib_seg_new_data)

        if open_figures:
            # figure before and after:
            cmd = 'itksnap -g {0}; itksnap -g {1};'.format(
                   jph(root_dir, 'data_examples', fin_seg_original), jph(root_dir, 'data_output', fin_seg_new))
            os.system(cmd)

    """
    Example 8:
    Split labels in 4d, only some labels.
    """

    if run_example['Split in 4d only some']:
        # data:
        fin_punt_seg_original = 'punt_seg.nii.gz'
        fin_punt_seg_new      = 'punt_seg_4d.nii.gz'
        list_labels = [4, 5, 6]
        # using the manager
        lt.manipulate_intensities.split_in_4d(fin_punt_seg_original, fin_punt_seg_new, list_labels=list_labels,
                                              keep_original_values=False)
        # without the manager:
        im_seg = nib.load(jph(root_dir, 'data_examples', fin_punt_seg_original))
        data_seg = im_seg.get_data()
        data_seg_new = sp.split_labels_to_4d(data_seg, list_labels, keep_original_values=False)
        # results comparison
        nib_seg_new = nib.load(jph(root_dir, 'data_output', fin_punt_seg_new))
        nib_seg_new_data = nib_seg_new.get_data()
        np.testing.assert_array_equal(data_seg_new, nib_seg_new_data)

        if open_figures:
            # figure after:
            cmd = 'itksnap -g {0}'.format(
                   jph(root_dir, 'data_output', fin_punt_seg_new))
            os.system(cmd)

    """
    Example 9:
    Merge labels in 4d
    """
    if run_example['Merge in 4d']:
        # data:
        fin_seg_original = 'mes_seg.nii.gz'
        fin_seg_splitted = 'mes_seg_splitted.nii.gz'
        fin_seg_merged = 'mes_seg_merged.nii.gz'
        # split:
        lt.manipulate_intensities.split_in_4d(fin_seg_original, fin_seg_splitted)
        # merge
        lt.set_input_data_folder(jph(root_dir, 'data_output'))

        lt.manipulate_intensities.merge_from_4d(fin_seg_splitted, fin_seg_merged)
        # check you got back the same:
        pfi_seg_before = jph(root_dir, 'data_examples', fin_seg_original)
        pfi_seg_after = jph(root_dir, 'data_output', fin_seg_merged)
        im_before = nib.load(pfi_seg_before)
        im_after  = nib.load(pfi_seg_after)
        np.testing.assert_array_equal(im_before.get_data(), im_after.get_data())

        # set back the input data folder to data_examples
        lt.set_input_data_folder(jph(root_dir, 'data_examples'))

    """
    Example 10:
    Symmetrise axial
    """
    if run_example['Axial symmetrisation']:
        fin_original = 'mes_seg.nii.gz'
        fin_transformed = 'mes_now_punt.nii.gz'
        # transform mes in a punt
        lt.symmetrize.symmetrise_axial(fin_original, fin_transformed, axis='x', plane_intercept=128,
                                       side_to_copy='above', keep_in_data_dimensions=True)

        if open_figures:
            cmd = 'itksnap -g {0}'.format(jph(root_dir, 'data_output', fin_transformed))
            os.system(cmd)

    """
    Example 11:
    Symmetrise with registration
    """
    if run_example['Symmetrisation with registration']:
        fin_anatomy = 'ellipsoids.nii.gz'
        fin_seg = 'ellipsoids_seg_half.nii.gz'
        # transform mes in a punt
        lt.symmetrize.symmetrise_with_registration(fin_anatomy, fin_seg,
                                                   list_labels_input=[1, 2, 3, 4, 5, 6],
                                                   results_folder_path=jph(root_dir, 'data_output'),
                                                   result_img_path=jph(root_dir, 'data_output','ellipsoid_seg_SYM.nii.gz'),
                                                   list_labels_transformed=[7, 8, 9, 10, 11, 12])

        if open_figures:
            cmd = 'itksnap -g {0} -s {1}'.format(jph(root_dir, 'data_examples', fin_anatomy),
                                                 jph(root_dir, 'data_output','ellipsoid_seg_SYM.nii.gz'))
            os.system(cmd)
