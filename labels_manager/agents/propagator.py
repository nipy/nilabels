from labels_manager.tools.manipulations.propagators import simple_propagator
from labels_manager.tools.aux_methods.sanity_checks import connect_tail_head_path


class LabelsManagerPropagate(object):
    """
    Facade of the methods in tools, for work with paths to images rather than
    with data. Methods under LabelsManagerPropagate are aimed at propagating segmentations.
    """
    def __init__(self, input_data_folder=None, output_data_folder=None):
        self.pfo_in = input_data_folder
        self.pfo_out = output_data_folder

    def simple_propagator(self,
                          ref_in,
                          flo_in,
                          flo_mask_in,
                          flo_on_ref_img_out,
                          flo_on_ref_mask_out,
                          flo_on_ref_trans_out):

        ref_in               = connect_tail_head_path(self.pfo_in, ref_in)
        flo_in               = connect_tail_head_path(self.pfo_in, flo_in)
        flo_mask_in          = connect_tail_head_path(self.pfo_in, flo_mask_in)
        flo_on_ref_img_out   = connect_tail_head_path(self.pfo_in, flo_on_ref_img_out)
        flo_on_ref_mask_out  = connect_tail_head_path(self.pfo_in, flo_on_ref_mask_out)
        flo_on_ref_trans_out = connect_tail_head_path(self.pfo_in, flo_on_ref_trans_out)

        simple_propagator(ref_in, flo_in, flo_mask_in,
                          flo_on_ref_img_out, flo_on_ref_mask_out, flo_on_ref_trans_out,
                          settings_reg='', settings_interp=' -inter 0 ',
                          verbose_on=True, safety_on=False)
