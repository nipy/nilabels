import os


def simple_propagator(ref_in, flo_in, flo_mask_in,
                      flo_on_ref_img_out, flo_on_ref_mask_out, flo_on_ref_trans_out,
                      settings_reg='', settings_interp=' -inter 0 ',
                      verbose_on=True, safety_on=False):

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
        print '\nRegistration images and propagation of mask.\n'
        print 'reg_aladin -ref {0} -flo {1} -aff {2} -res {3} {4} ; '.format(last_two(ref_in),
                                                                             last_two(flo_in),
                                                                             last_two(flo_on_ref_trans_out),
                                                                             last_two(flo_on_ref_img_out),
                                                                             last_two(settings_reg))

        print 'reg_resample -ref {0} -flo {1} -trans {2} -res {3} {4}'.format(last_two(ref_in),
                                                                             last_two(flo_mask_in),
                                                                             last_two(flo_on_ref_trans_out),
                                                                             last_two(flo_on_ref_mask_out),
                                                                             last_two(settings_interp))

    if not safety_on:
        os.system(cmd_1 + cmd_2)
