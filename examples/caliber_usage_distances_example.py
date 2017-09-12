import os
from os.path import join as jph

from labels_manager.main import LabelsManager as LM
from labels_manager.tools.caliber.distances import dice_score, dispersion, covariance_distance, hausdorff_distance
from labels_manager.tools.defs import root_dir

if __name__ == '__main__':

    # Paths to input
    pfo_examples = jph(root_dir, 'data_examples')
    pfi_seg1 = jph(pfo_examples, 'fourfolds_one.nii.gz')
    pfi_seg2 = jph(pfo_examples, 'fourfolds_two.nii.gz')
    where_to_save = None

    # Instantiate a Labels Manager class
    m = LM()
    m.measure.return_mm3 = False

    # get the measure
    d = m.measure.dist(pfi_seg1, pfi_seg2,
                       metrics=(dice_score, dispersion, hausdorff_distance, covariance_distance),
                       where_to_save=where_to_save)

    print d
