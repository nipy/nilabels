import os
import time
import subprocess


def connect_tail_head_path(tail, head):
    """
    It expects
      1) to find the path to folder in tail and filename in head.
      2) to find the full path in the head (with tail as sub-path).
      3) to have tail with a base path and head to have an additional path + filename.
    :param tail:
    :param head:
    :return:
    """

    if head.startswith(tail):  # os.path.abspath(head).startswith(os.path.abspath(tail)):  # Case 2
        return head
    else:  # case 1, 3
        return os.path.join(tail, head)


def check_pfi_io(pfi_input, pfi_output):
    if not os.path.exists(pfi_input):
        raise IOError('Input file {} does not exists.'.format(pfi_input))
    if pfi_output is not None:
        if not os.path.exists(os.path.dirname(pfi_output)):
            raise IOError('Output file {} is located in a non-existing folder.'.format(
                    pfi_output))
    return True


def get_pfi_in_pfi_out(filename_in, filename_out, pfo_in, pfo_out):
    """
    :param filename_in:
    :param filename_out:
    :param pfo_in:
    :param pfo_out:
    :return:
    """
    pfi_in = connect_tail_head_path(pfo_in, filename_in)    
    if filename_out is None:
        pfi_out = pfi_in
    else:
        if pfo_out is None:
            pfi_out = connect_tail_head_path(pfo_in, filename_out)
        else:
            pfi_out = connect_tail_head_path(pfo_out, filename_out)

    check_pfi_io(pfi_in, pfi_out)
    return pfi_in, pfi_out


def check_path(pfi, interval=1, timeout=100):
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