import numpy as np
from scipy import ndimage


def get_morphological_patch(dimension, shape):
    """
    :param dimension: dimension of the image (NOT the shape).
    :param shape: circle or square.
    :return: morphological patch as ndimage
    """
    if shape == 'circle':
        morpho_patch = ndimage.generate_binary_structure(dimension, 1)
    elif shape == 'square':
        morpho_patch = ndimage.generate_binary_structure(dimension, 3)
    else:
        return IOError

    return morpho_patch


def get_morphological_mask(point, omega, radius=5, shape='circle', morpho_patch=None):

    if morpho_patch is None:
        d = omega.ndim
        morpho_patch = get_morphological_patch(d, shape=shape)

    mask = np.zeros(omega, dtype=np.bool)
    mask.itemset(tuple(point), 1)
    for _ in range(radius):
        mask = ndimage.binary_dilation(mask, structure=morpho_patch).astype(mask.dtype)
    return mask


def get_patch_values(point, target_image, radius=5, shape='circle', morfo_mask=None):
    """
    To obtain the list of the values below a maks.
    :param point:
    :param target_image:
    :param radius:
    :param shape:
    :param morfo_mask: To avoid computing the morphological mask at each iteration if this method is called in a loop, this can be provided as input.
    :return:
    """
    if morfo_mask is None:
        morfo_mask = get_morphological_mask(point, target_image.shape, radius=radius, shape=shape)
    coord = np.nonzero(morfo_mask.flatten())[0]
    return np.take(target_image.flatten(), coord)


def midpoint_circle_algorithm(center=(0, 0, 0), radius=4):
    x, y, z = center
    # TODO generalise the midpoint circle algorithm and use it for get_shell_for_given_radius
    pass


def get_shell_for_given_radius(radius, d=3):
    circle = []
    if d == 3:
        for xi in xrange(-radius, radius + 1):
            for yi in xrange(-radius, radius + 1):
                for zi in xrange(-radius, radius + 1):
                    if (radius - 1) ** 2 < xi ** 2 + yi ** 2 + zi ** 2 <= radius ** 2:
                        circle.append((xi, yi, zi))
    if d == 2:
        for xi in xrange(-radius, radius + 1):
            for yi in xrange(-radius, radius + 1):
                if (radius - 1) ** 2 < xi ** 2 + yi ** 2 <= radius ** 2:
                    circle.append((xi, yi))

    return circle