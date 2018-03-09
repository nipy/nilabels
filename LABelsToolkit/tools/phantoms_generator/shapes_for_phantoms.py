import numpy as np
from scipy import ndimage


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


def ellipsoid_shape(omega, focus_1, focus_2, distance, background_intensity=0, foreground_intensity=100, dtype=np.uint8):
    sky = background_intensity * np.ones(omega, dtype=dtype)
    for xi in xrange(omega[0]):
        for yi in xrange(omega[1]):
            for zi in xrange(omega[2]):
                if np.sqrt( (focus_1[0] - xi) ** 2 + (focus_1[1] - yi) ** 2 + (focus_1[2] - zi) ** 2 ) + np.sqrt( (focus_2[0] - xi) ** 2 + (focus_2[1] - yi) ** 2 + (focus_2[2] - zi) ** 2 ) <= distance:
                    sky[xi, yi, zi] = foreground_intensity
    return sky


def cube_shape(omega, center, side_length, background_intensity=0, foreground_intensity=100, dtype=np.uint8):
    sky = background_intensity * np.ones(omega, dtype=dtype)
    half_side_length = int(np.ceil(side_length / 2))

    for lx in xrange(-half_side_length, half_side_length + 1):
        for ly in xrange(-half_side_length, half_side_length + 1):
            for lz in xrange(-half_side_length, half_side_length + 1):
                sky[center[0] + lx, center[1] + ly, center[2] + lz] = foreground_intensity
    return sky

def circle_shape(omega, centre, radius, foreground_intensity=100, dtype=np.uint8):
    sky = np.zeros(omega, dtype=dtype)
    for xi in xrange(omega[0]):
        for yi in xrange(omega[1]):
            for zi in xrange(omega[2]):
                if np.sqrt( (centre[0] - xi) ** 2 + (centre[1] - yi) ** 2 + (centre[2] - zi) ** 2 ) <= radius:
                    sky[xi, yi, zi] = foreground_intensity
    return sky

# ---------- Head-like experiments ---------------


def oval_shape(omega, centre, foreground_intensity=1, alpha=(0.18,0.18), dd=None, a_b_c=None, dtype=np.uint8):
    """
    From the ellipsoid equation in canonic form.
    Pebble-like stone shape with a principal direction. Can represent a biological shape phantom.

    :param omega:
    :param centre:
    :param foreground_intensity:
    :param alpha: between 0.1 and 0.3 maximal range
    :param dd: maximal extension, smaller than 2 * np.sqrt(omega[direction])
    :return:
    """
    sky = np.zeros(omega, dtype=dtype)

    if a_b_c is None:
        a_b_c = [1, 2, 1]
    if dd is None:
        dd = 2 * np.sqrt(omega[1])
    a_b_c = dd * np.array(a_b_c)
    for xi in xrange(omega[0]):
        for yi in xrange(omega[1]):
            for zi in xrange(omega[2]):
                if (np.abs(xi - centre[0]) / float(a_b_c[0])) ** 2 * (1 + alpha[0] * zi) / dd + (np.abs(yi - centre[1]) / float(a_b_c[1])) ** 2 + (np.abs(zi - centre[2]) / float(a_b_c[2])) ** 2 * (1 + alpha[1] * yi) / dd < 1:
                    sky[xi, yi, zi] = foreground_intensity

    return sky


def sulci_structure(omega, centre, foreground_intensity=1, a_b_c=None, dd=None, random_perturbation=0, alpha=(0.18,0.18), dtype=np.uint8):
    sky = np.zeros(omega, dtype=dtype)

    if a_b_c is None:
        a_b_c = [1, 2, 1]
    if dd is None:
        dd = 2 * np.sqrt(omega[1])
    a_b_c = dd * np.array(a_b_c)

    thetas = [j * np.pi / 4 for j in range(0, 8)]
    phis = [j * np.pi / 4 for j in range(1,4)]

    radius_internal_foci = a_b_c[0]
    radius_external_foci = a_b_c[1]

    internal_foci = []
    external_foci = []

    for theta in thetas:
        for phi in phis:
            p = np.array([np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)])
            # deform according to ovoidal shape
            p = p * np.array([1 , 1 + np.sqrt(alpha[0]),  1 + np.sqrt(alpha[1])])
            internal_foci.append(radius_internal_foci * p + np.array(centre))
            external_foci.append(radius_external_foci * p + np.array(centre))

    # add north and south pole:
    internal_foci.append(radius_internal_foci * np.array([0, 0, 1]) + np.array(centre))
    internal_foci.append(radius_internal_foci * np.array([0, 0, -1]) + np.array(centre))

    external_foci.append(radius_external_foci * np.array([0, 0, 1]) + np.array(centre))
    external_foci.append(radius_external_foci * np.array([0, 0, -1]) + np.array(centre))

    # generate ellipses:
    for inte, exte in zip(internal_foci, external_foci):
        d = 1.1 * np.linalg.norm(inte - exte)
        if random_perturbation > 0:
            epsilon_radius = random_perturbation * np.random.randn() * d
            epsilon_direction = np.linalg.norm(inte - exte) * 0.5 * random_perturbation * np.random.randn(3)
            sky += ellipsoid_shape(omega, inte, exte + epsilon_direction, d + epsilon_radius, background_intensity=0, foreground_intensity=foreground_intensity)
        else:
            sky += ellipsoid_shape(omega, inte, exte, d, background_intensity=0, foreground_intensity=foreground_intensity)

    return sky


def artifactor(omega, kind='bias field', strengths=0.5):
    """
    Add random artefact to a domain. Different kinds of artefacts and strengths can be selected.
    :param omega: a domain of a 3d volume image.
    :param kind: 'bias field', 'salt and pepper', 'salt pepper and curry', 'gaussian field'
    :param strengths: Value in the interval [0, 1]. 0 nothing happen. 1 strongest artefact.
    :return: All the artefacts are normalised between 0 and 1. This is not related with strengths.
    """
    if kind == 'bias field':
        pass

    if kind == 'change modality':
        pass

    if kind == 'gaussian noise':
        pass

    if kind == 'black holes':
        pass

    if kind == 'white holes':
        pass

    assert len(omega) == 3






if __name__ == '__main__':
    import nibabel as nib
    sky = oval_shape(omega=(81,101,71), centre=(40,50,35))
    im = nib.Nifti1Image(sky, np.eye(4))
    nib.save(im, '/Users/sebastiano/Desktop/zzz_test.nii.gz')

    pass
