import os
import time
import subprocess


def check_pfi_io(pfi_input, pfi_output):
    if not os.path.exists(pfi_input):
        raise IOError('Input file {} does not exists.'.format(pfi_input))
    if pfi_output is not None:
        if not os.path.exists(os.path.dirname(pfi_output)):
            raise IOError('Output file {} is located in a non-existing folder.'.format(
                    pfi_output))
    return True


def check_path_validity(pfi, interval=1, timeout=100):
    if os.path.exists(pfi):
        if pfi.endswith('.nii.gz'):
            mustend = time.time() + timeout
            while time.time() < mustend:
                try:
                    subprocess.check_output('gunzip -t {}'.format(pfi), shell=True)
                except subprocess.CalledProcessError:
                    print "Caught CalledProcessError"
                else:
                    return True
                time.sleep(interval)
            msg = 'File {0} corrupted after 100 tests. \n'.format(pfi)
            raise IOError(msg)
        else:
            return True
    else:
        msg = '{} does not exist!'.format(pfi)
        raise IOError(msg)


def is_valid_permutation(in_perm):
    """
    A permutation is a list of 2 lists of same size:
    a = [[1,2,3], [2,3,1]]
    means permute 1 with 2, 2 with 3, 3 with 1.
    """

    if not len(in_perm) == 2:
        return False
    if not len(in_perm[0]) == len(in_perm[1]):
        return False
    if not all(isinstance(n, int) for n in in_perm[0]):
        return False
    if not all(isinstance(n, int) for n in in_perm[1]):
        return False
    if not set(in_perm[0]) == set(in_perm[1]):
        return False
    return True
