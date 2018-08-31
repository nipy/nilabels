import os
from os.path import join as jph

from nilabels.agents.agents_controller import AgentsController as NiL
from nilabels.definitions import root_dir

if __name__ == '__main__':

    # generate output folder for examples:
    cmd = 'mkdir -p {}'.format(jph(root_dir, 'data_output'))
    os.system(cmd)

    # instantiate a label manager:
    lt = NiL(jph(root_dir, 'data_examples'), jph(root_dir, 'data_output'))

    # data:
    fin_punt_seg_original = 'mes_seg.nii.gz'
    fin_punt_seg_new = 'mes_seg_relabelled.nii.gz'

    list_old_labels = [1, 2, 3, 4, 5, 6]
    list_new_labels = [2, 3, 4, 5, 6, 7]

    # Using the manager to relabel the data:
    lt.manipulate_labels.relabel(fin_punt_seg_original, fin_punt_seg_new,
                          list_old_labels, list_new_labels)

    # figure before:
    cmd = 'itksnap -g {0} -s {1}'.format(
        jph(root_dir, 'data_examples', 'mes.nii.gz'),
        jph(root_dir, 'data_examples', fin_punt_seg_original))
    os.system(cmd)
    # figure after
    cmd = 'itksnap -g {0} -s {1}'.format(
        jph(root_dir, 'data_examples', 'mes.nii.gz'),
        jph(root_dir, 'data_output', fin_punt_seg_new))
    os.system(cmd)
