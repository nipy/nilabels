import numpy as np
import os
import nibabel as nib

from definitions import root_dir
from numpy.testing import assert_array_almost_equal
import matplotlib.pyplot as plt

from nose.tools import assert_equals, assert_raises, assert_almost_equals
from numpy.testing import assert_array_equal, assert_almost_equal
from labels_managers.tools.labels.relabeller import relabeller


def test_relabeller_path_simple_input_simple_input(visual_assessment=False):
    """
    test on a cubic element if the connected components extractor works
    :param visual_assessment: if the user is required to see a couple
    of images with itk-snap (needs to be installed).
    :return:
    """
    cmd = 'mkdir -p {0}'.format(os.path.join(root_dir, 'output'))
    os.system(cmd)

    dims = [105, 20, 20]

    im_data = np.zeros([dims[1], dims[2], 1])
    intervals = [50, 10, 50, 20, 25]
    labels   = [1, 2, 3, 4, 5]

    for i in range(len(intervals)):
        # build the matrix we need
        single_slice = labels[i] * np.ones([dims[1], dims[2], intervals[i]])
        im_data = np.concatenate((im_data, single_slice), axis=2)

    # Apply relabeller
    im_data_renewed = relabeller(im_data, [1, 2, 3, 4, 5], \
                                          [10, 20, 30, 40, 50])

    im_original = nib.Nifti1Image(im_data, np.eye(4))
    im_renewed = nib.Nifti1Image(im_data_renewed, np.eye(4))

    path_im1 = os.path.join(root_dir, 'output/test_im1.nii.gz')
    path_im2 = os.path.join(root_dir, 'output/test_im2.nii.gz')

    nib.save(im_original, path_im1)
    nib.save(im_renewed, path_im2)

    if visual_assessment:

        os.system('itksnap -g {}'.format(path_im1))
        os.system('itksnap -g {}'.format(path_im2))

        print 'Check if the second image opened is as the first multip. by 10.'

    im1 = nib.load(im_original, path_im1)
    im2 = nib.load(im_original, path_im2)

    im1_data = im1.get_data()
    im2_data = im2.get_data()

    # check with a random sampling if the second image is the
    # first multiplied by 10
    num_samples = 20
    i_v = np.random.choice(range(dims[0]), size=(num_samples, 1))
    j_v = np.random.choice(range(dims[1]), size=(num_samples, 1))
    k_v = np.random.choice(range(dims[2]), size=(num_samples, 1))

    points = np.concatenate((i_v, j_v, k_v), axis=1)

    assert_array_equal(points.shape, [num_samples, 3])

    for m in range(num_samples):
        pt_x, pt_y, pt_z = list(num_samples[m, :])
        assert_array_equal(10*im1_data[pt_x, pt_y, pt_z], \
                           im2_data[pt_x, pt_y, pt_z])


def test_split_labels_1():
    # generate a square image:
    dim_im = [50, 30, 40]

    for x in xrange():
        pass



def test_split_labels_2():
    pass

def test_split_labels_path(visual_assessment=True):
    pass

def test_merge_labels_1():
    pass


def test_merge_labels_2():
    pass

def test_merge_labels_path():
    pass


def test_split_merge_labels_1():
    pass


test_relabeller_path_simple_input_simple_input(False)
test_split_labels_1()
test_split_labels_2()
test_split_labels_path()
test_merge_labels_1()
test_merge_labels_2()
test_merge_labels_path()
test_split_merge_labels_1()
