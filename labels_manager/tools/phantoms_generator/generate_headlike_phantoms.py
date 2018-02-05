import os
from os.path import join as jph

import nibabel as nib
import numpy as np
import scipy.ndimage.filters as fil

from labels_manager.tools.phantoms_generator.shapes_for_phantoms import oval_shape


def headlike_phantom(modality=('basic', 'gabor'), omega=(161, 181, 201), anatomy_randomness=0.2,
                     background_noise=0.1, sharpness=0.1, artifacts=None, name='headlike_phantom'):
    """
    :param modality: can be 'basic', 'gabor'.
    :param omega: grid domain.
    :param anatomy_randomness: in interval [0,1].
    :param background_noise:
    :param sharpness:
    :param artifacts:
    :param name:
    :return:
    """
    for d in omega:
        assert d > 50, 'Omega must be at least (50, 50, 50)'

    # parameters
    skull_thickness = 3
    spacing_skull_brain = 2




    # omega centre
    omega_c = [int(omega[k] / 2) for k in range(3)]

    for i in range(omega[0]):
        for j in range(omega[0]):
            for k in range(omega[0]):
                pass


    pass


if __name__ == '__main__':
    data = headlike_phantom()
    im = nib.Nifti1Image(data, np.eye(4))
    nib.save(im, '/Users/sebastiano/Desktop/tt.nii.gz')