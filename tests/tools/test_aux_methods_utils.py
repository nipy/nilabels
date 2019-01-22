import os
from os.path import join as jph

import numpy as np
from numpy.testing import assert_array_equal

from nilabels.tools.aux_methods.utils import eliminates_consecutive_duplicates, lift_list, labels_query, print_and_run

from tests.tools.decorators_tools import create_and_erase_temporary_folder, test_dir


# TEST tools.aux_methods.utils.py'''


def test_lift_list_1():
    l_in, l_out = [[0, 1], 2, 3, [4, [5, 6]], 7, [8, [9]]], range(10)
    assert_array_equal(lift_list(l_in), l_out)


def test_lift_list_2():
    l_in, l_out = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], range(10)
    assert_array_equal(lift_list(l_in), l_out)


def test_lift_list_3():
    l_in, l_out = [], []
    assert_array_equal(lift_list(l_in), l_out)


def test_eliminates_consecutive_duplicates():
    l_in, l_out = [0, 0, 0, 1, 1, 2, 3, 4, 5, 5, 5, 6, 7, 8, 9], range(10)
    assert_array_equal(eliminates_consecutive_duplicates(l_in), l_out)


@create_and_erase_temporary_folder
def test_print_and_run_create_file():
    cmd = 'touch {}'.format(jph(test_dir, 'z_tmp_test', 'tmp.txt'))
    output_msg = print_and_run(cmd)
    assert os.path.exists(jph(test_dir, 'z_tmp_test', 'tmp.txt'))
    assert output_msg == 'touch tmp.txt'


@create_and_erase_temporary_folder
def test_print_and_run_create_file_safety_on():
    cmd = 'touch {}'.format(jph(test_dir, 'z_tmp_test', 'tmp.txt'))
    output_msg = print_and_run(cmd, safety_on=True)
    assert not os.path.exists(jph(test_dir, 'z_tmp_test', 'tmp.txt'))
    assert output_msg == 'touch tmp.txt'


@create_and_erase_temporary_folder
def test_print_and_run_create_file_safety_off():
    cmd = 'touch {}'.format(jph(test_dir, 'z_tmp_test', 'tmp.txt'))
    output_msg = print_and_run(cmd, safety_on=False, short_path_output=False)
    assert os.path.exists(jph(test_dir, 'z_tmp_test', 'tmp.txt'))
    assert output_msg == 'touch {}'.format(jph(test_dir, 'z_tmp_test', 'tmp.txt'))


def test_labels_query_int_input():
    lab, lab_names = labels_query(1)
    assert_array_equal(lab, [1])
    assert_array_equal(lab_names, ['1'])


def test_labels_query_list_input1():
    lab, lab_names = labels_query([1, 2, 3])
    assert_array_equal(lab, [1, 2, 3])
    assert_array_equal(lab_names, ['1', '2', '3'])


def test_labels_query_list_input2():
    lab, lab_names = labels_query([1, 2, 3, [4, 5, 6]])
    assert_array_equal(lift_list(lab), lift_list([1, 2, 3, [4, 5, 6]]))
    assert_array_equal(lab_names, ['1', '2', '3', '[4, 5, 6]'])


def test_labels_query_all_or_tot_input():
    v = np.arange(10).reshape(5, 2)
    lab, lab_names = labels_query('all', v, remove_zero=False)
    assert_array_equal(lab, np.arange(10))
    lab, lab_names = labels_query('tot', v, remove_zero=False)
    assert_array_equal(lab, np.arange(10))
    lab, lab_names = labels_query('tot', v, remove_zero=True)
    assert_array_equal(lab, np.arange(10)[1:])


if __name__ == '__main__':
    test_lift_list_1()
    test_lift_list_2()
    test_lift_list_3()
    test_eliminates_consecutive_duplicates()
    test_print_and_run_create_file()
    test_print_and_run_create_file_safety_on()
    test_print_and_run_create_file_safety_off()
    test_labels_query_int_input()
    test_labels_query_list_input1()
    test_labels_query_list_input2()
    test_labels_query_all_or_tot_input()
