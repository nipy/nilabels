from nilabels.tools.detections.contours import get_xyz_borders_of_a_label, set_new_data
from nilabels.tools.aux_methods.utils_nib import images_are_overlapping


def crop_with_passepartout(im_input, passepartout_values):
    """
    :param im_input:
    :param passepartout_values: in the form [x_min, -x_max, y_min, -y_max, z_min, -z_max]. where -x_max is the
    thickness of the border from the border.
    :return:
    """
    x_min, x_max, y_min, y_max, z_min, z_max = passepartout_values
    cropped_data = im_input.get_data()[x_min:-x_max, y_min:-y_max, z_min:-z_max]
    return set_new_data(im_input, cropped_data)


def crop_with_passepartout_based_on_label_segmentation(im_input_to_crop, im_segm, margins, label):
    """
    :param im_input_to_crop:
    :param im_segm:
    :param margins: in the form [x,y,z] space in voxel in each direction left as passepartout
                    around the selected label of the segmentation.
    :param label: label around which we want to isolate the input image to crop
    :return:
    ---
    Note: if label is not in the segmentation, return the input image.
    """
    assert images_are_overlapping(im_input_to_crop, im_segm)

    v = get_xyz_borders_of_a_label(im_segm.get_data(), label)

    x_min, x_max = v[0] - margins[0], v[1] + margins[0] + 1
    y_min, y_max = v[2] - margins[1], v[3] + margins[1] + 1
    z_min, z_max = v[4] - margins[2], v[5] + margins[2] + 1
    print([x_min, x_max, y_min, y_max, z_min, z_max])
    return crop_with_passepartout(im_input_to_crop, [x_min, -x_max, y_min, -y_max, z_min, -z_max])
