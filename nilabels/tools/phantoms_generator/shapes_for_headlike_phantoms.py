import numpy as np
from scipy import ndimage

from nilabels.tools.phantoms_generator.shapes_for_phantoms import oval_shape, sulci_structure, ellipsoid_shape


def headlike_phantom(omega=(161, 181, 201), intensities=(0.9, 0.3, 0.6, 0.8), random_perturbation=.0):
    """
    :param omega: grid domain.
    :param intensities: list of four with instensities of [skull, wm, gm, csf]
    :param random_perturbation: value between 0 and 1 providing the level of randomness.
    :return:
    """
    print('Creating headlike phantom:')

    for d in omega:
        assert d > 69, 'Omega must be at least (70, 70, 70) to contain a head-like phantom'
    # Parameters
    skull_thickness = 3
    wm_spacing = 2
    csf_spacing = 4

    alpha = (0.18, 0.18)
    dd_gm = 2 * np.sqrt(omega[1])
    dd_sk = 2 * np.sqrt(omega[1] + skull_thickness ** 2)

    if random_perturbation > 0:
        alpha = (0.05 * random_perturbation * np.random.randn() + alpha[0],  0.05 * random_perturbation *
                 np.random.randn() + alpha[1])
        epsilon = 0.01 * random_perturbation * np.random.randn()
        dd_gm = epsilon + 2 * np.sqrt(omega[1])
        dd_sk = epsilon + 2 * np.sqrt(omega[1] + skull_thickness ** 2)

    # omega centre
    omega_c = [int(omega[j] / 2) for j in range(3)]
    print('- generate brain shape')
    sh_gm = oval_shape(omega, omega_c, foreground_intensity=1, alpha=alpha, dd=dd_gm)
    print('- generate skull')
    sh_sk = oval_shape(omega, omega_c, foreground_intensity=1, alpha=alpha, dd=dd_sk)
    print('- generate wm')
    # erode brain to get an initial wm-like structure
    struct = ndimage.morphology.generate_binary_structure(3, 2)
    sh_wm = ndimage.morphology.binary_erosion(sh_gm, structure=struct, iterations=wm_spacing)

    # smoothing and then re-take the smoothed as binary.
    sc = sulci_structure(omega, omega_c, foreground_intensity=1, a_b_c=None, dd=None, alpha=alpha,
                         random_perturbation=0.1 * random_perturbation)
    sh_wm = sh_wm.astype(np.bool) ^ sc.astype(np.bool) * sh_wm.astype(np.bool)

    print('- generate csf')
    f1 = np.array(omega_c) + np.array([0, csf_spacing, 0])
    f21 = np.array(omega_c) + np.array([csf_spacing, - 2 * csf_spacing, 0])
    f22 = np.array(omega_c) + np.array([-csf_spacing, - 2 * csf_spacing, 0])
    d = 1.2 * np.linalg.norm(f1 - f21) * np.random.normal(1, random_perturbation / float(5))
    csf = ellipsoid_shape(omega, f1, f21, d, background_intensity=0, foreground_intensity=1)
    csf += ellipsoid_shape(omega, f1, f22, d, background_intensity=0, foreground_intensity=1)
    csf = csf.astype(np.bool)

    # ground truth segmentation:
    segm = (sh_gm + sh_sk + sh_wm + csf).astype(np.int32)

    # ground truth intensities:
    anatomy = segm.astype(np.float64)
    for i, l in zip(intensities, [1, 2, 3, 4]):
        places = segm == l
        if np.any(places):
            np.place(anatomy, places, i)

    return anatomy, segm
