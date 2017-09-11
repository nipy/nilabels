import os
from os.path import join as jph

from os.path import join as jph
from labels_manager.main import LabelsManager as LM

from labels_manager.tools.caliber.distances import dice_score, dispersion, covariance_distance, hausdorff_distance

from labels_manager.tools.defs import root_dir

if __name__ == '__main__':

    examples_folder = jph(root_dir, 'data_examples')

    pfi_im = jph(examples_folder, 'cubes_in_space.nii.gz')
    pfi_im_bin = jph(examples_folder, 'cubes_in_space_bin.nii.gz')

    for p in [pfi_im, pfi_im_bin, examples_folder]:
        if not os.path.exists(p):
            raise IOError(
                'Run lm.tools.benchmarking.generate_images_examples.py to create the images examples before this, please.')

    m = LM()


    # m.measure.dist(pfi_automatic_MV, pfi_manual_1, metrics=(dice_score, dispersion, covariance_distance,
    #                                                         hausdorff_distance),
    #                intermediate_files_folder_name=pfo_intermediate_files, where_to_save=where_to_save)

    # instantiate the SegmentationAnalyzer from caliber: scalar is the binary.

