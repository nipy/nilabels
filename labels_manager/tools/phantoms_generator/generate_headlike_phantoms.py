import os
from os.path import join as jph

import nibabel as nib
import numpy as np
import scipy.ndimage.filters as fil

from scipy import ndimage

from labels_manager.tools.phantoms_generator.shapes_for_phantoms import oval_shape


def headlike_phantom(modality=('basic', 'gabor'), omega=(161, 181, 201), anatomy_randomness=0.2,
                     background_noise=0.1, sharpness=0.1, artifacts=None, name='headlike_phantom'):
    """
    :param modality: can be 'basic', 'gabor'.
    :param omega: grid domain.
    :param anatomy_randomness: in interval [0,1].
    :param background_noise:
    :param sharpness:
    :param artifacts: list of artifactor parameters or None. The strings has to be ...
    :param name:
    :return:
    """
    for d in omega:
        assert d > 50, 'Omega must be at least (50, 50, 50)'

    # Parameters
    skull_thickness = 3
    spacing_skull_brain = 2

    # omega centre
    omega_c = [int(omega[k] / 2) for k in range(3)]


    sh = oval_shape(omega, omega_c, foreground_intensity=1, direction='y', eta=2, alpha=(0.18,0.18), dd=None)

    if 'bias field' in artifacts:
        pass

    if 'hyper holes' in artifacts:
        pass

    if 'hyper holes' in artifacts:
        pass


    pass


if __name__ == '__main__':
    data = headlike_phantom()
    im = nib.Nifti1Image(data, np.eye(4))
    nib.save(im, '/Users/sebastiano/Desktop/tt.nii.gz')