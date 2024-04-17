import numpy as np

from nilabels.tools.aux_methods.utils_nib import set_new_data


def get_rgb_image_from_segmentation_and_label_descriptor(im_segm, ldm, invert_black_white=False, dtype_output=np.int32):
    """From the labels descriptor and a nibabel segmentation image.
    :param im_segm: nibabel segmentation whose labels corresponds to the input labels descriptor.
    :param ldm: instance of class label descriptor manager.
    :param dtype_output: data type of the output image.
    :param invert_black_white: to swap black with white (improving background visualisation).
    :return: a 4d image, where at each voxel there is the [r, g, b] vector in the fourth dimension.
    """

    if not len(im_segm.shape) == 3:
        raise OSError("input segmentation must be 3D.")

    rgb_image_arr = np.ones(list(im_segm.shape) + [3])

    for l in ldm.dict_label_descriptor:
        pl = im_segm.get_fdata() == l
        rgb_image_arr[pl, :] = ldm.dict_label_descriptor[l][0]

    if invert_black_white:
        pl = im_segm.get_fdata() == 0
        rgb_image_arr[pl, :] = np.array([255, 255, 255])
    return set_new_data(im_segm, rgb_image_arr, new_dtype=dtype_output)
