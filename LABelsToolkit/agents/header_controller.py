import nibabel as nib
import numpy as np

from LABelsToolkit.tools.aux_methods.utils_rotations import get_small_orthogonal_rotation
from LABelsToolkit.tools.aux_methods.utils_path import get_pfi_in_pfi_out, connect_path_tail_head
from LABelsToolkit.tools.aux_methods.utils_nib import modify_image_type, \
    modify_affine_transformation, replace_translational_part


class LABelsToolkitHeaderController(object):
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

    def modify_affine(self, filename_in, affine_in, filename_out, q_form=True, s_form=True,
                      multiplication_side='left'):

        pfi_in, pfi_out = get_pfi_in_pfi_out(filename_in, filename_out, self.pfo_in, self.pfo_out)

        if isinstance(affine_in, str):

            if affine_in.endswith('.txt'):
                aff = np.loadtxt(connect_path_tail_head(self.pfo_in, affine_in))
            else:
                aff = np.load(connect_path_tail_head(self.pfo_in, affine_in))

        elif isinstance(affine_in, np.ndarray):
            aff = affine_in
        else:
            raise IOError('parameter affine_in can be path to an affine matrix .txt or .npy or the numpy array'
                          'corresponding to the affine transformation.')

        im = nib.load(pfi_in)
        new_im = modify_affine_transformation(im, aff, q_form=q_form, s_form=s_form,
                                              multiplication_side=multiplication_side)
        nib.save(new_im, pfi_out)

    def apply_small_rotation(self, filename_in, filename_out, angle=np.pi/6, principal_axis='pitch',
                             respect_to_centre=True):

        if isinstance(angle, list):
            assert isinstance(principal_axis, list)
            assert len(principal_axis) == len(angle)
            rot = np.identity(4)
            for pa, an in zip(principal_axis, angle):
                aff = get_small_orthogonal_rotation(theta=an, principal_axis=pa)
                rot = rot.dot(aff)
        else:
            rot = get_small_orthogonal_rotation(theta=angle, principal_axis=principal_axis)
        
        pfi_in, pfi_out = get_pfi_in_pfi_out(filename_in, filename_out, self.pfo_in, self.pfo_out)
        im = nib.load(pfi_in)

        if respect_to_centre:
            fov_centre = im.affine.dot(np.array(list(np.array(im.shape[:3]) / float(2)) + [1]))

            transl = np.eye(4)
            transl[:3, 3] = fov_centre[:3]

            transl_inv = np.eye(4)
            transl_inv[:3, 3] = -1 * fov_centre[:3]

            rt = transl.dot(rot.dot(transl_inv))

            new_aff = rt.dot(im.affine)
        else:
            new_aff = im.get_affine()[:]
            new_aff[:3, :3] = rot[:3, :3].dot(new_aff[:3, :3])

        new_im = modify_affine_transformation(im_input=im, new_aff=new_aff, q_form=True, s_form=True,
                                              multiplication_side='replace')

        nib.save(new_im, pfi_out)

    def modify_translational_part(self, filename_in, filename_out, new_translation):
        pfi_in, pfi_out = get_pfi_in_pfi_out(filename_in, filename_out, self.pfo_in, self.pfo_out)
        im = nib.load(pfi_in)

        if isinstance(new_translation, str):

            if new_translation.endswith('.txt'):
                tr = np.loadtxt(connect_path_tail_head(self.pfo_in, new_translation))
            else:
                tr = np.load(connect_path_tail_head(self.pfo_in, new_translation))

        elif isinstance(new_translation, np.ndarray):
            tr = new_translation
        elif isinstance(new_translation, list):
            tr = np.array(new_translation)
        else:
            raise IOError('parameter new_translation can be path to an affine matrix .txt or .npy or the numpy array'
                          'corresponding to the new intended translational part.')

        new_im = replace_translational_part(im, tr)
        nib.save(new_im, pfi_out)
