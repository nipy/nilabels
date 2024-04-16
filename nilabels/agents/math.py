import nibabel as nib

from nilabels.tools.aux_methods.utils_nib import set_new_data
from nilabels.tools.aux_methods.utils_path import connect_path_tail_head, get_pfi_in_pfi_out


class Math:
    """Facade of no external methods. Simple class for quick algebraic manipulations of images with the same grid"""

    def __init__(self, input_data_folder=None, output_data_folder=None):
        self.pfo_in = input_data_folder
        self.pfo_out = output_data_folder

    def sum(self, path_first_image, path_second_image, path_resulting_image):
        pfi_im1, pfi_im2 = get_pfi_in_pfi_out(path_first_image, path_second_image, self.pfo_in, self.pfo_in)
        pfi_result = connect_path_tail_head(self.pfo_out, path_resulting_image)

        im1 = nib.load(pfi_im1)
        im2 = nib.load(pfi_im2)

        if not im1.shape == im2.shape:
            raise OSError("Input images must have the same dimensions.")

        im_result = set_new_data(im1, new_data=im1.get_fdata() + im2.get_fdata())

        nib.save(im_result, pfi_result)
        print(f"Image sum of {pfi_im1} {pfi_im2} saved under {pfi_result}.")
        return pfi_result

    def sub(self, path_first_image, path_second_image, path_resulting_image):
        pfi_im1, pfi_im2 = get_pfi_in_pfi_out(path_first_image, path_second_image, self.pfo_in, self.pfo_in)
        pfi_result = connect_path_tail_head(self.pfo_out, path_resulting_image)

        im1 = nib.load(pfi_im1)
        im2 = nib.load(pfi_im2)

        if not im1.shape == im2.shape:
            raise OSError("Input images must have the same dimensions.")

        im_result = set_new_data(im1, new_data=im1.get_fdata() - im2.get_fdata())

        nib.save(im_result, pfi_result)
        print(f"Image difference of {pfi_im1} {pfi_im2} saved under {pfi_result}.")
        return pfi_result

    def prod(self, path_first_image, path_second_image, path_resulting_image):
        pfi_im1, pfi_im2 = get_pfi_in_pfi_out(path_first_image, path_second_image, self.pfo_in, self.pfo_in)
        pfi_result = connect_path_tail_head(self.pfo_out, path_resulting_image)

        im1 = nib.load(pfi_im1)
        im2 = nib.load(pfi_im2)

        if not im1.shape == im2.shape:
            raise OSError("Input images must have the same dimensions.")

        im_result = set_new_data(im1, new_data=im1.get_fdata() * im2.get_fdata())

        nib.save(im_result, pfi_result)
        print(f"Image product of {pfi_im1} {pfi_im2} saved under {pfi_result}.")
        return pfi_result

    def scalar_prod(self, scalar, path_image, path_resulting_image):
        pfi_image = connect_path_tail_head(self.pfo_in, path_image)
        pfi_result = connect_path_tail_head(self.pfo_out, path_resulting_image)
        im = nib.load(pfi_image)

        im_result = set_new_data(im, new_data=scalar * im.get_fdata())

        nib.save(im_result, pfi_result)
        print(f"Image {pfi_image} times {scalar} saved under {pfi_result}.")
        return pfi_result
