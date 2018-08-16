import numpy as np


def from_rgb_to_xyz(v_rgb, rgb_working_space='appleD65'):
    """
    Conversion from RGB to XYZ based on the documentation at the website:
    http://www.brucelindbloom.com/index.html?Equations.html
    :param v_rgb: input  triplet (r,g,b)
    :param rgb_working_space:
    :return: v converted in xyz space
    """

    if rgb_working_space == 'appleD65':
        m = np.array([[0.4497288, 0.3162486, 0.1844926],
                      [0.2446525, 0.6720283, 0.0833192],
                      [0.0251848, 0.1411824, 0.9224628]])

    elif rgb_working_space == 'sRGBD65':
        m = np.array([[0.4124564, 0.3575761, 0.1804375],
                      [0.2126729, 0.7151522, 0.0721750],
                      [0.0193339, 0.1191920, 0.9503041]])

    else:
        msg = 'rgb_working_space must be in {0}. See website {1} for documentation.'.format(
                ['appleD65', 'sRGBD65'],
                'http://www.brucelindbloom.com/index.html?Equations.html')

        raise IOError(msg)

    return m.dot(np.array(v_rgb))


def xyz_to_lab(v_xyz, reference_white=(100.0000, 100.0000, 100.0000)):
    """
    Converting colour triplets from xyz to lab through the reference white.
    :param v_xyz: colour vector in xyz for the given reference white.
    :param reference_white:
    :return: corresponding lab colour according to the website:
        http://www.brucelindbloom.com/index.html?Equations.html
    See as well CIE standard convention.
    """
    epsilon = 216 / float(24389)
    kappa   = 24389 / float(27)

    v_r = [v_xyz[j] / reference_white[j] for j in range(3)]

    f = [v_r[j] ** (1. / 3) if v_r[j] > epsilon else (kappa*v_r[j] + 16) / 116 for j in range(3)]

    return f


def rgb_to_lab(v_rgb, rgb_working_space='appleD65', reference_white=(100.0000, 100.0000, 100.0000)):
    """
    :param v_rgb: RGB triplet (as vector)
    :param rgb_working_space: can be 'appleD65' or 'sRGBD65'.
    :param reference_white: reference white triplet.
    :return:
    """
    return xyz_to_lab(from_rgb_to_xyz(v_rgb, rgb_working_space=rgb_working_space), reference_white=reference_white)


def rgb_to_yuv(v_rgb):
    """
    From rgb triplet to YUV space. Apprximation of the RGB to LAB conversion.
    :param v_rgb: RGB triplet
    :return: corresponding YUV triplet
    """
    m = np.array([[0.299, 0.587, 0.114],
                  [-0.14713, -0.28886, 0.436],
                  [0.615, -0.51499, -0.10001]])

    return m.dot(np.array(v_rgb))


def yuv_to_rgb(v_yuv):
    """
    :param v_yuv: YUV triplet
    :return: corresponding RGB triplet
    """
    m_inv = np.array([[1.00000000e+00, -1.17983844e-05, 1.13983458e+00],
                      [1.00000395e+00, -3.94646053e-01, -5.80594234e-01],
                      [9.99979679e-01, 2.03211194e+00, -1.51129807e-05]])

    return np.round(m_inv.dot(np.array(v_yuv))).astype('uint8')


def yuv_to_lab(v_yuv):
    """
    :param v_yuv: YUV triplet
    :return: corresponding LAB triplet
    """
    return rgb_to_lab(yuv_to_rgb(v_yuv))


def get_range_lab_color_space():
    """
    :return: range of the LAB color space in the form
     [min_l, max_l], [min_a, max_a], [min_b, max_b]
    """
    max_l, max_a, max_b = [-np.inf] * 3
    min_l, min_a, min_b = [np.inf] * 3

    for r in range(255):
        for g in range(255):
            for b in range(255):
                l, a, b = rgb_to_lab([r, g, b])
                max_l, max_a, max_b = max(max_l, l), max(max_a, a), max(max_b, b)
                min_l, min_a, min_b = min(min_l, l), min(min_a, a), min(min_b, b)

    return [min_l, max_l], [min_a, max_a], [min_b, max_b]


def get_range_yuv_color_space():
    """
    :return: range of the YUV color space in the form
    [0, max_y], [0, max_u], [0, max_v]
    """
    max_y = rgb_to_yuv([255, 0, 0])[0]
    max_u = rgb_to_yuv([0, 255, 0])[1]
    max_v = rgb_to_yuv([0, 0, 255])[2]

    return [0, max_y], [0, max_u], [0, max_v]


def color_distance(v1, v2, in_color_space='rgb'):
    """
    Distance between v1 and v2 computed in lab-color space, where the difference is closest to the
    human perception.
    :param v1: first triplet color.
    :param v2: second triplet color.
    :param in_color_space: possible options: 'rgb' 'lab' 'xyz' 'yuv'
    :return: distance as l2 norm in lab space
    """
    color_space_map = {'rgb': rgb_to_lab, 'lab': lambda x: x, 'xyz': xyz_to_lab, 'yuv': yuv_to_lab}

    return np.linalg.norm(
           np.array(color_space_map[in_color_space](v1)) - np.array(color_space_map[in_color_space](v2)))
