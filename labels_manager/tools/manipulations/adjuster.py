import nibabel as nib

from labels_manager.tools.aux_methods.utils_nib import set_new_data


def adjust_nifti_image_type_path(pfi_nifti_input, new_dtype, pfi_nifti_output, update_description=None, verbose=1):
    im_input = nib.load(pfi_nifti_input)
    if update_description is not None:
        if not isinstance(update_description, str):
            raise IOError('update_description must be a string')
        hd = im_input.header
        hd['descrip'] = update_description
        im_input.update_header()
    new_im = set_new_data(im_input, im_input.get_data().astype(new_dtype), new_dtype=new_dtype, remove_nan=True)
    if verbose > 0:
        print('Data type before {}'.format(im_input.get_data_dtype()))
        print('Data type after {}'.format(new_im.get_data_dtype()))
    nib.save(new_im, pfi_nifti_output)


def reproduce_slice_fourth_dimension(nib_image, num_slices=10, repetition_axis=3):

    im_sh = nib_image.shape
    if not (len(im_sh) == 2 or len(im_sh) == 3):
        raise IOError('Methods can be used only for 2 or 3 dim images. No conflicts with existing multi, slices')

    new_data = np.stack([nib_image.get_data(), ] * num_slices, axis=repetition_axis)
    output_im = set_new_data(nib_image, new_data)

    return output_im


def reproduce_slice_fourth_dimension_path(pfi_input_image, pfi_output_image, num_slices=10, repetition_axis=3):
    old_im = nib.load(pfi_input_image)
    new_im = reproduce_slice_fourth_dimension(old_im, num_slices=num_slices, repetition_axis=repetition_axis)
    nib.save(new_im, pfi_output_image)
    print 'New image created and saved in {0}'.format(pfi_output_image)

