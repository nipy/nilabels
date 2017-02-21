"""
A permutation is a list of 2 lists of same size:
a = [[1,2,3], [2,3,1]]
means permute 1 with 2, 2 with 3, 3 with 1.
"""


def is_valid_permutation(in_perm):

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
