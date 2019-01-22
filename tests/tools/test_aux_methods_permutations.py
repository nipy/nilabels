from numpy.testing import assert_array_equal, assert_raises

from nilabels.tools.aux_methods.utils import permutation_from_cauchy_to_disjoints_cycles, \
    permutation_from_disjoint_cycles_to_cauchy


# Test permutations:


def test_from_permutation_to_disjoints_cycles():
    cauchy_perm = [[1, 2, 3, 4, 5], [3, 4, 5, 2, 1]]
    cycles_perm = permutation_from_cauchy_to_disjoints_cycles(cauchy_perm)
    expected_ans = [[1, 3, 5], [2, 4]]
    for c1, c2 in zip(expected_ans, cycles_perm):
        assert_array_equal(c1, c2)


def test_from_disjoint_cycles_to_permutation():
    cycles_perm = [[1, 3, 5], [2, 4]]
    cauchy_perm = permutation_from_disjoint_cycles_to_cauchy(cycles_perm)
    expected_ans = [[1, 2, 3, 4, 5], [3, 4, 5, 2, 1]]
    for c1, c2 in zip(cauchy_perm, expected_ans):
        assert_array_equal(c1, c2)


def test_from_permutation_to_disjoints_cycles_single_cycle():
    cauchy_perm = [[1, 2, 3, 4, 5, 6, 7],
                   [3, 4, 5, 1, 2, 7, 6]]
    cycles_perm = permutation_from_cauchy_to_disjoints_cycles(cauchy_perm)
    expected_ans = [[1, 3, 5, 2, 4], [6, 7]]

    for c1, c2 in zip(expected_ans, cycles_perm):
        assert_array_equal(c1, c2)


def test_from_permutation_to_disjoints_cycles_single_cycle_no_valid_permutation():
    cauchy_perm = [[1, 2, 3, 4, 5, 6, 7],
                   [3, 4, 5, 1, 2, 7]]
    with assert_raises(IOError):
        permutation_from_cauchy_to_disjoints_cycles(cauchy_perm)


def test_from_disjoint_cycles_to_permutation_single_cycle():
    cycles_perm = [[1, 3, 5, 2, 4]]
    cauchy_perm = permutation_from_disjoint_cycles_to_cauchy(cycles_perm)
    expected_ans = [[1, 2, 3, 4, 5], [3, 4, 5, 1, 2]]

    for c1, c2 in zip(cauchy_perm, expected_ans):
        assert_array_equal(c1, c2)


if __name__ == '__main__':
    test_from_permutation_to_disjoints_cycles()
    test_from_permutation_to_disjoints_cycles_single_cycle_no_valid_permutation()
    test_from_disjoint_cycles_to_permutation()
    test_from_permutation_to_disjoints_cycles_single_cycle()
    test_from_disjoint_cycles_to_permutation_single_cycle()
