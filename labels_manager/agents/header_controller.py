import os

import nibabel as nib
import numpy as np

from labels_manager.tools.aux_methods.sanity_checks import get_pfi_in_pfi_out, connect_path_tail_head
from labels_manager.tools.image_shape_manipulations.spatial import modify_image_type, \
    modify_affine_transformation

class LabelsManagerHeaderController(object):
    """
    Facade of the methods in tools. symmetrizer, for work with paths to images rather than
    with data. Methods under LabelsManagerManipulate are taking in general
    one or more input manipulate them according to some rule and save the
    output in the output_data_folder or in the specified paths.
    """

    def __init__(self, input_data_folder=None, output_data_folder=None):
        self.pfo_in = input_data_folder
        self.pfo_out = output_data_folder

    def modify_image_type(self, filename_in, filename_out, new_dtype, update_description=None, verbose=1):

        pfi_in, pfi_out = get_pfi_in_pfi_out(filename_in, filename_out, self.pfo_in, self.pfo_out)

        im = nib.load(pfi_in)
        new_im = modify_image_type(im, new_dtype=new_dtype, update_description=update_description, verbose=verbose)
        nib.save(new_im, pfi_out)

    def modify_affine(self, filename_in, filename_aff, filename_out, q_form=True, s_form=True,
                      multiplication_side='left'):

        pfi_in, pfi_out = get_pfi_in_pfi_out(filename_in, filename_out, self.pfo_in, self.pfo_out)

        if filename_aff.endswith('.txt'):
            aff = np.loadtxt(connect_path_tail_head(self.pfo_in, filename_aff))
        else:
            aff = np.load(connect_path_tail_head(self.pfo_in, filename_aff))

        im = nib.load(pfi_in)
        new_im = modify_affine_transformation(im, aff, q_form=q_form, s_form=s_form,
                                              multiplication_side=multiplication_side)
        nib.save(new_im, pfi_out)

    def small_spatial_rotation(self, filename_in, filename_out, angle, ):

        pfi_in, pfi_out = get_pfi_in_pfi_out(filename_in, filename_out, self.pfo_in, self.pfo_out)

        # TODO: create the small rotation and then apply to the matrix

        im = nib.load(pfi_in)
        # new_im = apply_orientation_matrix()
        # nib.save(new_im, pfi_out)

