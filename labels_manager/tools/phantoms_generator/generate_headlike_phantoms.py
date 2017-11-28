import os
from os.path import join as jph

import nibabel as nib
import numpy as np
import scipy.ndimage.filters as fil



def headlike_phantom(curtosis, randomness):
    """
    All the parameters forming the head-like phantom are reduced to two.
    curtosis and ramdoness.
    :param curtosis:
    :param randomness:
    :return: resulting 3d volume. (measurements are performed outside)
    """
    omega = (161, 181, 201)
    omega_centre = [int(omega[k] / 2) for k in range(3)]

    skull_thickness = 3
    spacing_skull_brain = 2

    pass
