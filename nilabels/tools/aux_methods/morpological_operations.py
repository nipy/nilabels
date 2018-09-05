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
        raise IOError

    return morpho_patch


def get_morphological_mask(point, omega, radius=5, shape='circle', morpho_patch=None):
    """
    Helper to obtain a morphological mask based on get_morphological_patch
    :param point: centre of the mask
    :param omega: grid dimension of the image domain. E.g. [256, 256, 128].
    :param radius: radius of the mask if morpho_patch not given.
    :param shape: 'circle' shape of the mask if morpho_patch not given.
    :param morpho_patch:To avoid computing the morphological mask at each iteration if this method,
    this mask can be provided as input. This bypasses the input radius and shape.
    """
    if morpho_patch is None:
        d = len(omega)
        morpho_patch = get_morphological_patch(d, shape=shape)

    array_mask = np.zeros(omega, dtype=np.bool)
    array_mask.itemset(tuple(point), 1)
    for _ in range(radius):
        array_mask = ndimage.binary_dilation(array_mask, structure=morpho_patch).astype(array_mask.dtype)
    return array_mask


def get_values_below_patch(point, target_image, radius=5, shape='circle', morpho_mask=None):
    """
    To obtain the list of the values below a maks.
    :param point: central point of the patch
    :param target_image: array image whose values we are interested into.
    :param radius: patch radius if morpho_patch is not given.
    :param shape: shape patch if morpho_patch is not given.
    :param morpho_mask: To avoid computing the morphological mask at each iteration if this method,
    this mask can be provided as input. This bypasses the input radius and shape.
    :return:
    """
    if morpho_mask is None:
        morpho_mask = get_morphological_mask(point, target_image.shape, radius=radius, shape=shape)
    coord = np.nonzero(morpho_mask.flatten())[0]
    return np.take(target_image.flatten(), coord)


def get_circle_shell_for_given_radius(radius, dimension=3):
    """
    :param radius: radius of the circle.
    :param dimension: must be 2 or 3.
    :return: matrix coordinate values for a circle of given input radius and dimension centered at the origin.
    E.G.
    >> get_circle_shell_for_given_radius(3,2)
    [(-3, 0), (-2, -2), (-2, -1), (-2, 1), (-2, 2), (-1, -2), (-1, 2), (0, -3), (0, 3), (1, -2), (1, 2), (2, -2),
    (2, -1), (2, 1), (2, 2), (3, 0)]
    Note: implementation not optimal.
    """
    # Generalise midpoint circle algorithm for future version.
    circle = []
    if dimension == 3:
        for xi in range(-radius, radius + 1):
            for yi in range(-radius, radius + 1):
                for zi in range(-radius, radius + 1):
                    if (radius - 1) ** 2 < xi ** 2 + yi ** 2 + zi ** 2 <= radius ** 2:
                        circle.append((xi, yi, zi))
    elif dimension == 2:
        for xi in range(-radius, radius + 1):
            for yi in range(-radius, radius + 1):
                if (radius - 1) ** 2 < xi ** 2 + yi ** 2 <= radius ** 2:
                    circle.append((xi, yi))
    else:
        raise IOError('Dimensions allowed are 2 or 3.')
    return circle
