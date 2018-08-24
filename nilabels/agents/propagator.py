import os

from nilabels.tools.aux_methods.utils_path import connect_path_tail_head


class LabelsPropagator(object):
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
                          flo_on_ref_trans_out,
                          verbose_on=True,
                          safety_on=False,
                          settings_reg='',
                          settings_interp=' -inter 0 '):

        ref_in               = connect_path_tail_head(self.pfo_in, ref_in)
        flo_in               = connect_path_tail_head(self.pfo_in, flo_in)
        flo_mask_in          = connect_path_tail_head(self.pfo_in, flo_mask_in)
        flo_on_ref_img_out   = connect_path_tail_head(self.pfo_in, flo_on_ref_img_out)
        flo_on_ref_mask_out  = connect_path_tail_head(self.pfo_in, flo_on_ref_mask_out)
        flo_on_ref_trans_out = connect_path_tail_head(self.pfo_in, flo_on_ref_trans_out)

        last_two = lambda l: '/'.join(l.split('/')[-2:])

        cmd_1 = 'reg_aladin -ref {0} -flo {1} -aff {2} -res {3} {4} ; '.format(ref_in,
                                                                               flo_in,
                                                                               flo_on_ref_trans_out,
                                                                               flo_on_ref_img_out,
                                                                               settings_reg)
        cmd_2 = 'reg_resample -ref {0} -flo {1} -trans {2} -res {3} {4}'.format(ref_in,
                                                                                flo_mask_in,
                                                                                flo_on_ref_trans_out,
                                                                                flo_on_ref_mask_out,
                                                                                settings_interp)
        if verbose_on:
            print('\nRegistration images and propagation of mask.\n')
            print('reg_aladin -ref {0} -flo {1} -aff {2} -res {3} {4} ; '.format(last_two(ref_in),
                                                                                 last_two(flo_in),
                                                                                 last_two(flo_on_ref_trans_out),
                                                                                 last_two(flo_on_ref_img_out),
                                                                                 last_two(settings_reg)))

            print('reg_resample -ref {0} -flo {1} -trans {2} -res {3} {4}'.format(last_two(ref_in),
                                                                                  last_two(flo_mask_in),
                                                                                  last_two(flo_on_ref_trans_out),
                                                                                  last_two(flo_on_ref_mask_out),
                                                                                  last_two(settings_interp)))

        if not safety_on:
            os.system(cmd_1 + cmd_2)

