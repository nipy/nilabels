from os.path import join as jph
from numpy.testing import assert_array_equal

from nilabels.definitions import root_dir
from nilabels.tools.aux_methods.utils_path import connect_path_tail_head, get_pfi_in_pfi_out


# TEST aux_methods.sanity_checks.py - NOTE - this is the core of the manager design '''


def test_connect_tail_head_path():
    # Case 1:
    assert connect_path_tail_head('as/df/gh', 'lm.txt') == 'as/df/gh/lm.txt'
    # Case 2:
    assert connect_path_tail_head('as/df/gh', 'as/df/gh/lm/nb.txt') == 'as/df/gh/lm/nb.txt'
    # Case 3:
    assert connect_path_tail_head('as/df/gh', 'lm/nb.txt') == 'as/df/gh/lm/nb.txt'


def test_get_pfi_in_pfi_out():

    tail_a = jph(root_dir, 'tests')
    tail_b = root_dir
    head_a = 'test_auxiliary_methods.py'
    head_b = 'head_b.txt'

    assert_array_equal(get_pfi_in_pfi_out(head_a, None, tail_a, None), (jph(tail_a, head_a), jph(tail_a, head_a)))
    assert_array_equal(get_pfi_in_pfi_out(head_a, head_b, tail_a, None), (jph(tail_a, head_a), jph(tail_a, head_b)))
    assert_array_equal(get_pfi_in_pfi_out(head_a, head_b, tail_a, tail_b), (jph(tail_a, head_a), jph(tail_b, head_b)))


if __name__ == '__main__':
    test_connect_tail_head_path()
    test_get_pfi_in_pfi_out()
