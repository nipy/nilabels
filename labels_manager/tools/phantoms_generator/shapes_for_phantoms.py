import numpy as np


# ---------- Simple shapes generators ---------------

def o_shape(omega=(250, 250), radius=50,
            background_intensity=0, foreground_intensity=20, dtype=np.uint8):

    m = background_intensity * np.ones(omega, dtype=dtype)

    if len(omega) == 2:
        c = [omega[j] / 2 for j in range(len(omega))]
        for x in xrange(omega[0]):
            for y in xrange(omega[1]):
                if (x - c[0])**2 + (y - c[1])**2 < radius**2:
                    m[x, y] = foreground_intensity
    elif len(omega) == 3:
        c = [omega[j] / 2 for j in range(len(omega))]
        for x in xrange(omega[0]):
            for y in xrange(omega[1]):
                for z in xrange(omega[2]):
                    if (x - c[0])**2 + (y - c[1])**2 + (z - c[2])**2 < radius**2:
                        m[x, y, z] = foreground_intensity
    return m


def c_shape(omega=(250, 250), internal_radius=40, external_radius=60, opening_height=50,
            background_intensity=0, foreground_intensity=20, dtype=np.uint8, margin=None):

    def get_a_2d_c(omega, internal_radius, external_radius, opening_height, background_intensity,
                   foreground_intensity, dtype):

        m = background_intensity * np.ones(omega[:2], dtype=dtype)

        c = [omega[j] / 2 for j in range(len(omega))]
        # create the crown
        for x in xrange(omega[0]):
            for y in xrange(omega[1]):
                if internal_radius**2 < (x - c[0])**2 + (y - c[1])**2 < external_radius**2:
                    m[x, y] = foreground_intensity

        # open the c
        low_lim = int(omega[0] / 2) - int(opening_height / 2)
        high_lim = int(omega[0] / 2) + int(opening_height / 2)

        for x in xrange(omega[0]):
            for y in xrange(int(omega[1] / 2), omega[1]):
                if low_lim < x < high_lim and m[x, y] == foreground_intensity:
                    m[x, y] = background_intensity

        return m

    c_2d = get_a_2d_c(omega=omega[:2], internal_radius=internal_radius, external_radius=external_radius,
                      opening_height=opening_height, background_intensity=background_intensity,
                      foreground_intensity=foreground_intensity, dtype=dtype)

    if len(omega) == 2:
        return c_2d

    elif len(omega) == 3:
        if margin is None:
            return np.repeat(c_2d, omega[2]).reshape(omega)
        else:
            res = np.zeros(omega, dtype=dtype)
            for z in xrange(margin, omega[2] - 2 * margin):
                res[..., z] = c_2d
            return res


def ellipsoid_shape(omega, focus_1, focus_2, distance, background_intensity=0, foreground_intensity=100,
                    dtype=np.uint8):
    sky = background_intensity * np.ones(omega, dtype=dtype)
    for xi in range(omega[0]):
        for yi in range(omega[1]):
            for zi in range(omega[2]):
                if np.sqrt( (focus_1[0] - xi) ** 2 + (focus_1[1] - yi) ** 2 + (focus_1[2] - zi) ** 2 ) + \
                    np.sqrt( (focus_2[0] - xi) ** 2 + (focus_2[1] - yi) ** 2 + (focus_2[2] - zi) ** 2 ) \
                        <= distance:
                    sky[xi, yi, zi] = foreground_intensity
    return sky


def cube_shape(omega, center, side_length, background_intensity=0, foreground_intensity=100, dtype=np.uint8):
    sky = background_intensity * np.ones(omega, dtype=dtype)
    half_side_length = int(np.ceil(side_length / 2))

    for lx in range(-half_side_length, half_side_length + 1):
        for ly in range(-half_side_length, half_side_length + 1):
            for lz in range(-half_side_length, half_side_length + 1):
                sky[center[0] + lx, center[1] + ly, center[2] + lz] = foreground_intensity
    return sky


# ---------- Head-like experiments ---------------


def oval_shape():
    """
    From the ellipsoid equation in canonic form.
    :return:
    """
    pass