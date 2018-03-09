import os
from os.path import join as jph

from LABelsToolkit.main import LabelsManager as LM
from LABelsToolkit.tools.defs import root_dir


if __name__ == '__main__':

    examples_folder = jph(root_dir, 'data_examples')

    pfi_im = jph(examples_folder, 'cubes_in_space.nii.gz')
    pfi_im_bin = jph(examples_folder, 'cubes_in_space_bin.nii.gz')

    for p in [pfi_im, pfi_im_bin, examples_folder]:
        if not os.path.exists(p):
            raise IOError('Run lm.tools.benchmarking.generate_images_examples.py to create the images examples before this, please.')

    m = LM()

    # total volume:
    print('The image contains 4 cubes of sides 11, 17, 19 and 9:\n')
    print('11**3 +  17**3 + 19**3 + 9**3 = {} '.format(11**3 +  17**3 + 19**3 + 9**3))
    print('sa.get_total_volume()         = {} '.format(m.measure.get_total_volume(pfi_im)['Volume'].values))

    # Get volumes per label:
    print('The 4 cubes of sides 11, 17, 19 and 9 are labelled 1, 2, 3 and 4 resp.:')
    print('Volume measured label 1        = {}'.format( m.measure.volume(pfi_im, labels=1)['Volume'].values) )
    print('11**3                          = {}'.format(11 ** 3))
    print('Volume measured label 2        = {}'.format( m.measure.volume(pfi_im, labels=2)['Volume'].values) )
    print('17**3                          = {}'.format(17 ** 3))
    print('Volume measured label 3        = {}'.format( m.measure.volume(pfi_im, labels=3)['Volume'].values) )
    print('19**3                          = {}'.format(19 ** 3))
    print('Volume measured label 4        = {}'.format( m.measure.volume(pfi_im, labels=4)['Volume'].values) )
    print('9**3                           = {}'.format(9 ** 3))
    print('Volume measured labels ([1, 3]) = {}'.format( m.measure.volume(pfi_im,
                                                                          labels=[[1, 3]])['Volume'].values) )
    print('11* 3 +  19**3                  = {}'.format(11**3 + 19**3))
    print('\nTo sum up: \n')
    print('Volume measured labels ([1, 2, 3, 4, [1, 3]]) = \n{}\n'.format(
        m.measure.volume(pfi_im, labels=[1,2, 3, 4, [1, 3]])
        ))

    print('------------')

    # Get volumes under each label, given the image weight, corresponding to the label itself:
    print('average below label 1 = {}'.format( m.measure.volume(pfi_im, anatomy_filename=pfi_im, labels=1)['Average below label'].values) )
    print('11**3 * 1  / 11**3    = {}'.format(11 ** 3 * 1/ float(11**3) ))
    print('average below label 2 = {}'.format( m.measure.volume(pfi_im, anatomy_filename=pfi_im, labels=2)['Average below label'].values) )
    print('17**3  * 2 / 17 ** 3  = {}'.format(17 ** 3 * 2 / float(17 ** 3)))
    print('average below label 3 = {}'.format( m.measure.volume(pfi_im, anatomy_filename=pfi_im, labels=3)['Average below label'].values) )
    print('19**3 * 3/ 19 ** 3    = {}'.format(19 ** 3 * 3 / float(19 ** 3)))
    print('average below label 4 = {}'.format( m.measure.volume(pfi_im, anatomy_filename=pfi_im, labels=4)['Average below label'].values) )
    print('9**3 * 4 / 9 ** 3     = {}'.format(9 ** 3 * 4 / float(9 ** 3)))
    print('average below labels [1,3]                = {}'.format(
        m.measure.volume(pfi_im, anatomy_filename=pfi_im, labels=[1,3])['Average below label'].values)
    )
    print('11**3 * 1  / 11**3,  19**3 * 3/ 19 ** 3   = {}'.format([11 ** 3 * 1 / float(11 ** 3),
                                                                    19 ** 3 * 3 / float(19 ** 3)]))
    print('average below labels [[1, 3]]   = {}'.format(
        m.measure.volume(pfi_im, anatomy_filename=pfi_im, labels=[[1,3]])['Average below label'].values)
    )
    print('( 11**3 * 1 +19**3 * 3 ) / (11**3 + 19 ** 3) = {0}'.format( (11**3 * 1 +19**3 * 3) / float(11**3 + 19 ** 3)))
    print('\n\nTo sum up: \n')
    print('average below labels [1, 2, 3, 4, [1, 3]] = \n{}'.format(
        m.measure.volume(pfi_im, anatomy_filename=pfi_im, labels=[1, 2, 3, 4, [1, 3]] ))
    )
