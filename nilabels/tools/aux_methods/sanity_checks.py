import os
import time
import subprocess
import itertools


def check_pfi_io(pfi_input, pfi_output):
    """
    Check if input and output files exist
    :param pfi_input: path to file input
    :param pfi_output: path to file output. Can be None.
    :return: True if pfi_input exists and if output is provided and exists.
    """
    if not os.path.exists(pfi_input):
        raise IOError('Input file {} does not exists.'.format(pfi_input))
    if pfi_output is not None:
        if not os.path.exists(os.path.dirname(pfi_output)):
            raise IOError('Output file {} is located in a non-existing folder.'.format(
                    pfi_output))
    return True


def check_path_validity(pfi, interval=1, timeout=100):
    """
    Workaround function to cope with delayed operations in the cluster.
    Boringly asking if something exists, until timeout or appearance of the sought file happen.
    :param pfi: path to file to assess
    :param interval: seconds
    :param timeout: number of intervals before timeout.
    :return:
    """
    if os.path.exists(pfi):
        if pfi.endswith('.nii.gz'):
            mustend = time.time() + timeout
            while time.time() < mustend:
                try:
                    subprocess.check_output('gunzip -t {}'.format(pfi), shell=True)
                except subprocess.CalledProcessError:
                    print("Caught CalledProcessError")
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


def is_valid_permutation(in_perm, for_labels=True):
    """
    A permutation in generalised Cauchy notation is a list of 2 lists of same size:
    a = [[1,2,3], [2,3,1]]
    means permute 1 with 2, 2 with 3, 3 with 1.
    :param for_labels: if True the permutation elements must be int.
    :param in_perm: input permutation.
    """
    if not len(in_perm) == 2:
        return False
    if not len(in_perm[0]) == len(in_perm[1]) == len(set(in_perm[0])) == len(set(in_perm[1])):
        return False
    if not (sorted(set(in_perm[0])) > sorted(set(in_perm[1])))-(sorted(set(in_perm[0])) < sorted(set(in_perm[1]))) == 0:
        return False
    if for_labels:
        # as dealing with labels, all the elements must be int
        if not all(isinstance(x, int) for x in list(itertools.chain(*in_perm))):
            return False
    return True
