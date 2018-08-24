import numpy as np

from nilabels.tools.aux_methods.utils_nib import set_new_data


def exploded_segmentation(im_segm, direction, intercepts, offset, dtype=np.int):
    """
    Damien Hirst-like sectioning of an anatomical segmentation.
    :param im_segm: nibabel image segmentation
    :param direction: sectioning direction, can be sagittal, axial or coronal
    (conventional names for images oriented to standard (diagonal affine transformation))
    :param intercepts: list of values of the stack plane in the input segmentation.
    Needs to include the max plane and the min plane
    :param offset: voxel to leave empty between one slice and the other
    :return: nibabel image output as sectioning of the input one.
    """
    if direction.lower() == 'axial':
        block = np.zeros([im_segm.shape[0], im_segm.shape[1], offset]).astype(dtype)
        stack = []
        for j in range(1, len(intercepts)):
            stack += [im_segm.get_data()[:, :, intercepts[j-1]:intercepts[j]].astype(dtype)] + [block]
        return set_new_data(im_segm, np.concatenate(stack, axis=2))

    elif direction.lower() == 'sagittal':
        block = np.zeros([offset, im_segm.shape[1], im_segm.shape[2]]).astype(dtype)
        stack = []
        for j in range(1, len(intercepts)):
            stack += [im_segm.get_data()[intercepts[j - 1]:intercepts[j], :, :].astype(dtype)] + [block]
        return set_new_data(im_segm, np.concatenate(stack, axis=0))

    elif direction.lower() == 'coronal':
        block = np.zeros([im_segm.shape[0], offset, im_segm.shape[2]]).astype(dtype)
        stack = []
        for j in range(1, len(intercepts)):
            stack += [im_segm.get_data()[:, intercepts[j - 1]:intercepts[j], :].astype(dtype)] + [block]
        for st in stack:
            print(st.shape)
        return set_new_data(im_segm, np.concatenate(stack, axis=1))

    else:
        raise IOError
