import os
from os.path import join as jph
import numpy as np
import nibabel as nib
import collections

# PATH MANAGER


test_dir = os.path.dirname(os.path.abspath(__file__))
pfo_tmp_test = jph(test_dir, 'z_tmp_test')

# AUXILIARIES


def is_a_string_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


# DECORATORS


def create_and_erase_temporary_folder(test_func):
    def wrap(*args, **kwargs):
        # 1) Before: create folder
        os.system('mkdir {}'.format(pfo_tmp_test))
        # 2) Run test
        test_func(*args, **kwargs)
        # 3) After: delete folder and its content
        os.system('rm -r {}'.format(pfo_tmp_test))

    return wrap


def create_and_erase_temporary_folder_with_a_dummy_nifti_image(test_func):
    def wrap(*args, **kwargs):
        # 1) Before: create folder
        os.system('mkdir {}'.format(pfo_tmp_test))
        nib_im = nib.Nifti1Image(np.zeros((30, 30, 30)), affine=np.eye(4))
        nib.save(nib_im, jph(pfo_tmp_test, 'dummy_image.nii.gz'))
        # 2) Run test
        test_func(*args, **kwargs)
        # 3) After: delete folder and its content
        os.system('rm -r {}'.format(pfo_tmp_test))

    return wrap


def write_and_erase_temporary_folder(test_func):
    def wrap(*args, **kwargs):
        # 1) Before: create folder
        os.system('mkdir {}'.format(pfo_tmp_test))
        # 2) Run test
        test_func(*args, **kwargs)
        # 3) After: delete folder and its content
        os.system('rm -r {}'.format(pfo_tmp_test))

    return wrap


def write_and_erase_temporary_folder_with_dummy_labels_descriptor(test_func):
    def wrap(*args, **kwargs):

        # 1) Before: create folder
        os.system('mkdir {}'.format(pfo_tmp_test))
        # 1bis) Then, generate dummy descriptor in the generated folder
        descriptor_dummy = \
            """################################################
# ITK-SnAP Label Description File
# File format:
# IDX   -R-  -G-  -B-  -A--  VIS MSH  LABEL
# Fields:
#    IDX:   Zero-based index
#    -R-:   Red color component (0..255)
#    -G-:   Green color component (0..255)
#    -B-:   Blue color component (0..255)
#    -A-:   Label transparency (0.00 .. 1.00)
#    VIS:   Label visibility (0 or 1)
#    IDX:   Label mesh visibility (0 or 1)
#  LABEL:   Label description
################################################
    0     0    0    0        0  0  0    "background"
    1   255    0    0        1  1  1    "label one (l1)"
    2   204    0    0        1  1  1    "label two (l2)"
    3    51   51  255        1  1  1    "label three"
    4   102  102  255        1  1  1    "label four"
    5     0  204   51        1  1  1    "label five (l5)"
    6    51  255  102        1  1  1    "label six"
    7   255  255    0        1  1  1    "label seven"
    8   255  50    50        1  1  1    "label eight" """
        with open(jph(pfo_tmp_test, 'labels_descriptor.txt'), 'w+') as f:
            f.write(descriptor_dummy)
        # 2) Run test
        test_func(*args, **kwargs)
        # 3) After: delete folder and its content
        os.system('rm -r {}'.format(pfo_tmp_test))

    return wrap


def write_and_erase_temporary_folder_with_left_right_dummy_labels_descriptor(test_func):
    def wrap(*args, **kwargs):
        # 1) Before: create folder
        os.system('mkdir {}'.format(pfo_tmp_test))
        # 1bis) Then, generate summy descriptor left right in the generated folder
        d = collections.OrderedDict()
        d.update({0: [[0, 0, 0],       [0, 0, 0], 'background']})
        d.update({1: [[255, 0, 0],     [1, 1, 1], 'label A Left']})
        d.update({2: [[204, 0, 0],     [1, 1, 1], 'label A Right']})
        d.update({3: [[51, 51, 255],   [1, 1, 1], 'label B Left']})
        d.update({4: [[102, 102, 255], [1, 1, 1], 'label B Right']})
        d.update({5: [[0, 204, 51],    [1, 1, 1], 'label C']})
        d.update({6: [[51, 255, 102],  [1, 1, 1], 'label D']})
        d.update({7: [[255, 255, 0],   [1, 1, 1], 'label E Left']})
        d.update({8: [[255, 50, 50],   [1, 1, 1], 'label E Right']})
        with open(jph(pfo_tmp_test, 'labels_descriptor_RL.txt'), 'w+') as f:
            for j in d.keys():
                line = '{0: >5}{1: >6}{2: >6}{3: >6}{4: >9}{5: >6}{6: >6}    "{7}"\n'.format(
                    j, d[j][0][0], d[j][0][1], d[j][0][2], d[j][1][0], d[j][1][1], d[j][1][2], d[j][2])
                f.write(line)
        # 2) Run test
        test_func(*args, **kwargs)
        # 3) After: delete folder and its content
        os.system('rm -r {}'.format(pfo_tmp_test))

    return wrap


def create_and_erase_temporary_folder_with_a_dummy_b_vectors_list(test_func):
    def wrap(*args, **kwargs):
        # 1) Before: create folder
        os.system('mkdir {}'.format(pfo_tmp_test))
        # noinspection PyTypeChecker
        np.savetxt(jph(pfo_tmp_test, 'b_vects_file.txt'), np.random.randn(10, 3))
        # 2) Run test
        test_func(*args, **kwargs)
        # 3) After: delete folder and its content
        os.system('rm -r {}'.format(pfo_tmp_test))

    return wrap
