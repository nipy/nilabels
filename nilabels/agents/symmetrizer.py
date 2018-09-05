import os

import nibabel as nib
import numpy as np

from nilabels.tools.aux_methods.utils_path import get_pfi_in_pfi_out, connect_path_tail_head
from nilabels.tools.aux_methods.utils_nib import set_new_data
from nilabels.tools.image_colors_manipulations.relabeller import relabeller
from nilabels.tools.aux_methods.utils_rotations import flip_data, symmetrise_data


class SegmentationSymmetrizer(object):
    """
    Facade of the methods in tools. symmetrizer, for work with paths to images rather than
    with data. Methods under LabelsManagerManipulate are taking in general
    one or more input manipulate them according to some rule and save the
    output in the output_data_folder or in the specified paths.
    """

    def __init__(self, input_data_folder=None, output_data_folder=None):
        self.pfo_in = input_data_folder
        self.pfo_out = output_data_folder

    def symmetrise_axial(self, filename_in, filename_out=None, axis='x', plane_intercept=10,
        side_to_copy='below', keep_in_data_dimensions=True):

        pfi_in, pfi_out = get_pfi_in_pfi_out(filename_in, filename_out, self.pfo_in, self.pfo_out)

        im_segm = nib.load(pfi_in)
        data_labels = im_segm.get_data()
        data_symmetrised = symmetrise_data(data_labels,
                                           axis_direction=axis,
                                           plane_intercept=plane_intercept,
                                           side_to_copy=side_to_copy,
                                           keep_in_data_dimensions_boundaries=keep_in_data_dimensions)

        im_symmetrised = set_new_data(im_segm, data_symmetrised)
        nib.save(im_symmetrised, pfi_out)
        print('Symmetrised axis {0}, plane_intercept {1}, image of {2} saved in {3}.'.format(axis, plane_intercept, pfi_in, pfi_out))
        return pfi_out

    def symmetrise_with_registration(self,
                                     filename_anatomy,
                                     filename_segmentation,
                                     list_labels_input,
                                     result_img_path,
                                     results_folder_path=None,
                                     list_labels_transformed=None,
                                     coord='z',
                                     reuse_registration=False):
        """
        Symmetrise a segmentation with registration: it uses NiftyReg.
        The old side is symmetrised in the new side, with new relabelling.
        Method based on paths even if in tools
        :param filename_anatomy: Path to File anatomical image
        :param filename_segmentation: Path to File segmentation of the anatomical image
        :param results_folder_path: Path to FOlder where intermediate results are stored
        :param result_img_path: Path to File symmetrised segmentation
        :param list_labels_input: labels that will be taken into account in the symmetrisation from the old side.
        :param list_labels_transformed: corresponding labels in the same order. If None, labels of the new side
        will be kept with the same numbering in the new side.
        :param reuse_registration: if a registration is already present in the pfo_results, and you only need to change the
        labels value, it will spare you some time when set to True.
        :param coord: coordinate of the registration: in RAS, 'z' will symmetrise Left on Right.
        :return: symmetrised segmentation.
        ---
        NOTE: requires niftyreg.
        """

        pfi_in_anatomy = connect_path_tail_head(self.pfo_in, filename_anatomy)
        pfi_in_segmentation = connect_path_tail_head(self.pfo_in, filename_segmentation)

        if results_folder_path is None:
            if self.pfo_out is not None:
                results_folder_path = self.pfo_out
            else:
                results_folder_path = self.pfo_in
        else:
            results_folder_path = os.path.dirname(pfi_in_segmentation)

        pfi_out_segmentation = connect_path_tail_head(results_folder_path, result_img_path)

        def flip_data_path(input_im_path, output_im_path, axis='x'):
            # wrap flip data, having path for inputs and outputs.
            if not os.path.isfile(input_im_path):
                raise IOError('input image file does not exist.')

            im_labels = nib.load(input_im_path)
            data_labels = im_labels.get_data()
            data_flipped = flip_data(data_labels, axis_direction=axis)

            im_relabelled = set_new_data(im_labels, data_flipped)
            nib.save(im_relabelled, output_im_path)

        # side A is the input, side B is the one where we want to symmetrise.
        # --- Initialisation  --- #

        # check input:
        if not os.path.isfile(pfi_in_anatomy):
            raise IOError('input image file {} does not exist.'.format(pfi_in_anatomy))
        if not os.path.isfile(pfi_in_segmentation):
            raise IOError('input segmentation file {} does not exist.'.format(pfi_in_segmentation))

        # erase labels that are not in the list from image and descriptor

        out_labels_side_A_path = os.path.join(results_folder_path, 'z_labels_side_A.nii.gz')
        labels_im = nib.load(pfi_in_segmentation)
        labels_data = labels_im.get_data()
        labels_to_erase = list(set(labels_data.flat) - set(list_labels_input + [0]))

        # Relabel: from pfi_segmentation to out_labels_side_A_path
        im_pfi_segmentation = nib.load(pfi_in_segmentation)

        segmentation_data_relabelled = relabeller(im_pfi_segmentation.get_data(), list_old_labels=labels_to_erase,
                                                  list_new_labels=[0, ] * len(labels_to_erase))
        nib_labels_side_A_path = set_new_data(im_pfi_segmentation, segmentation_data_relabelled)
        nib.save(nib_labels_side_A_path, out_labels_side_A_path)

        # --- Create side B  --- #

        # flip anatomical image and register it over the non flipped
        out_anatomical_flipped_path = os.path.join(results_folder_path, 'z_anatomical_flipped.nii.gz')
        flip_data_path(pfi_in_anatomy, out_anatomical_flipped_path, axis=coord)

        # flip the labels
        out_labels_flipped_path = os.path.join(results_folder_path, 'z_labels_flipped.nii.gz')
        flip_data_path(out_labels_side_A_path, out_labels_flipped_path, axis=coord)

        # register anatomical flipped over non flipped
        out_anatomical_flipped_warped_path = os.path.join(results_folder_path, 'z_anatomical_flipped_warped.nii.gz')
        out_affine_transf_path = os.path.join(results_folder_path, 'z_affine_transformation.txt')

        if not reuse_registration:
            cmd = 'reg_aladin -ref {0} -flo {1} -aff {2} -res {3}'.format(pfi_in_anatomy,
                                                                          out_anatomical_flipped_path,
                                                                          out_affine_transf_path,
                                                                          out_anatomical_flipped_warped_path)
            print('Registration started!\n')
            os.system(cmd)

            # propagate the registration to the flipped labels
            out_labels_side_B_path = os.path.join(results_folder_path, 'z_labels_side_B.nii.gz')
            cmd = 'reg_resample -ref {0} -flo {1} ' \
                  '-res {2} -trans {3} -inter {4}'.format(out_labels_side_A_path,
                                                          out_labels_flipped_path,
                                                          out_labels_side_B_path,
                                                          out_affine_transf_path,
                                                          0)

            print('Resampling started!\n')
            os.system(cmd)
        else:
            out_labels_side_B_path = os.path.join(results_folder_path, 'z_labels_side_B.nii.gz')

        # update labels of the side B if necessarily
        if list_labels_transformed is not None:
            print('relabelling step!')

            assert len(list_labels_transformed) == len(list_labels_input)

            # relabel from out_labels_side_B_path to out_labels_side_B_path
            im_segmentation_side_B = nib.load(out_labels_side_B_path)

            data_segmentation_side_B_new = relabeller(im_segmentation_side_B.get_data(),
                                                      list_old_labels=list_labels_input,
                                                      list_new_labels=list_labels_transformed)
            nib_segmentation_side_B_new = set_new_data(im_segmentation_side_B, data_segmentation_side_B_new)
            nib.save(nib_segmentation_side_B_new, out_labels_side_B_path)

        # --- Merge side A and side B in a single volume according to a criteria --- #
        # out_labels_side_A_path,  out_labels_side_B_path --> result_path.nii.gz

        nib_side_A = nib.load(out_labels_side_A_path)
        nib_side_B = nib.load(out_labels_side_B_path)

        data_side_A = nib_side_A.get_data()
        data_side_B = nib_side_B.get_data()

        symmetrised_data = np.zeros_like(data_side_A)

        # To manage the intersections of labels between old and new side. Vectorize later...
        dims = data_side_A.shape

        print('Pointwise symmetrisation started!')

        for z in range(dims[0]):
            for x in range(dims[1]):
                for y in range(dims[2]):
                    if (data_side_A[z, x, y] == 0 and data_side_B[z, x, y] != 0) or \
                            (data_side_A[z, x, y] != 0 and data_side_B[z, x, y] == 0):
                        symmetrised_data[z, x, y] = np.max([data_side_A[z, x, y], data_side_B[z, x, y]])
                    elif data_side_A[z, x, y] != 0 and data_side_B[z, x, y] != 0:
                        if data_side_A[z, x, y] == data_side_B[z, x, y]:
                            symmetrised_data[z, x, y] = data_side_A[z, x, y]
                        else:
                            symmetrised_data[z, x, y] = 255  # devil label!

        im_symmetrised = set_new_data(nib_side_A, symmetrised_data)
        nib.save(im_symmetrised, pfi_out_segmentation)

