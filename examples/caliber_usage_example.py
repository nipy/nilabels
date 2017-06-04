import os
from os.path import join as jph

from defs import root_dir
from labels_manager.caliber.segmentation_analyzer import SegmentationAnalyzer as SA


if __name__ == '__main__':

    examples_folder = jph(root_dir, 'images_examples')

    pfi_im = jph(examples_folder, 'cubes_in_space.nii.gz')
    pfi_im_bin = jph(examples_folder, 'cubes_in_space_bin.nii.gz')

    for p in [pfi_im, pfi_im_bin, examples_folder]:
        if not os.path.exists(p):
            raise IOError('Run generate_images_examples.py to create the images examples before this, please.')

    # instantiate the SegmentationAnalyzer from caliber: scalar is the binary.
    sa = SA(pfi_segmentation=pfi_im, pfi_scalar_im=pfi_im_bin)

    # total volume:
    print('The image contains 4 cubes of sides 11, 17, 19 and 9:')
    print('11**3 +  17**3 + 19**3 + 9**3 = {} '.format(11**3 +  17**3 + 19**3 + 9**3))
    print('sa.get_total_volume()         = {} '.format(sa.get_total_volume()))

    # Get volumes per label:
    print('The 4 cubes of sides 11, 17, 19 and 9 are labelled 1, 2, 3 and 4 resp.:')
    print('sa.get_volumes_per_label(1)[0] = {}'.format(sa.get_volumes_per_label(1)[0]))
    print('11**3                          = {}'.format(11 ** 3))
    print('sa.get_volumes_per_label(2)[0] = {}'.format(sa.get_volumes_per_label(2)[0]))
    print('17**3                          = {}'.format(17 ** 3))
    print('sa.get_volumes_per_label(3)[0] = {}'.format(sa.get_volumes_per_label(3)[0]))
    print('19**3                          = {}'.format(19 ** 3))
    print('sa.get_volumes_per_label(4)[0] = {}'.format(sa.get_volumes_per_label(4)[0]))
    print('9**3                          = {}'.format(9 ** 3))
    print('sa.get_volumes_per_label([[1, 3]]) = {}'.format(sa.get_volumes_per_label([[1, 3]])[0]))
    print('11* 3 +  19**3                    = {0}'.format(11**3 + 19**3))
    print('\nTo sum up: \n')
    print('sa.get_volumes_per_label([1, 2, 3, 4, [1, 3]]) = {}\n\n'.format(
        sa.get_volumes_per_label([1, 2, 3, 4, [1, 3]])[0]))

    # instantiate the SegmentationAnalyzer from caliber: scalar is not the bin, but have the same values
    # as the segmentation.
    sa = SA(pfi_segmentation=pfi_im, pfi_scalar_im=pfi_im)
    # Get volumes under each label, given the image weight, corresponding to the label itself:
    print('sa.get_average_below_labels(1) = {}'.format(sa.get_average_below_labels(1)))
    print('11**3 * 1  / 11**3             = {}'.format(11 ** 3 * 1/ float(11**3) ))
    print('sa.get_average_below_labels(2) = {}'.format(sa.get_average_below_labels(2)))
    print('17**3  * 2 / 17 ** 3           = {}'.format(17 ** 3 * 2 / float(17 ** 3)))
    print('sa.get_average_below_labels(3) = {}'.format(sa.get_average_below_labels(3)))
    print('19**3 * 3/ 19 ** 3             = {}'.format(19 ** 3 * 3 / float(19 ** 3)))
    print('sa.get_average_below_labels(4) = {}'.format(sa.get_average_below_labels(4)))
    print('9**3 * 4 / 9 ** 3              = {}'.format(9 ** 3 * 4 / float(9 ** 3)))
    print('sa.get_average_below_labels([1, 3]) = {}'.format(sa.get_average_below_labels([1, 3])))
    print('11**3 * 1  / 11**3,  19**3 * 3/ 19 ** 3   = {0}'.format([11 ** 3 * 1 / float(11 ** 3),
                                                                    19 ** 3 * 3 / float(19 ** 3)]))
    print('sa.get_average_below_labels([[1, 3]])   = {}'.format(sa.get_average_below_labels([[1, 3]])))
    print('( 11**3 * 1 +19**3 * 3 ) / (11**3 + 19 ** 3) = {0}'.format( (11**3 * 1 +19**3 * 3) / float(11**3 + 19 ** 3)))
    print('\n\nTo sum up: \n')
    print('sa.get_average_below_labels([1, 2, 3, 4, [1, 3]]) = {}'.format(
        sa.get_average_below_labels([1, 2, 3, 4, [1, 3]])))
