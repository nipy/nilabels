# Early draft
import numpy as np
import copy
from collections import Counter

from labels_manager.tools.aux_methods.morpological_tools import get_shell_for_given_radius, get_morphological_patch, \
    get_morphological_mask, get_patch_values
from labels_manager.tools.aux_methods.utils import triangular_density_function
from labels_manager.tools.measurements.distances import lncc_distance


def weighting_for_LNCC(point, target_image, stack_warped, morphological_mask):
    """
    We want to assess the lncc between a target and each of the image stack of images
    contained into a stack, on a patch shape.
    :param point: point in an image
    :param target_image: an image
    :param stack_warped: a stack of warped images
    :param morphological_mask: patch-like object
    :return:
    """
    num_timepoints = stack_warped.shape[3]
    patch_target = np.array(get_patch_values(point, target_image, morfo_mask=morphological_mask))
    ordered_measurements = []

    for t in range(num_timepoints):
        patch_t = np.array(get_patch_values(point, stack_warped[..., t], morfo_mask=morphological_mask))
        ordered_measurements.append(lncc_distance(patch_target, patch_t))
    return np.array(ordered_measurements)


def weighting_for_whole_label_values(grayscale_value, covering_labels, intensities_distrib_matrix):
    """

    :param grayscale_value:
    :param covering_labels:
    :param intensities_distrib_matrix: output of selector.get_intensities_statistics_matrix()
    :return:
    """
    assert len(covering_labels) == intensities_distrib_matrix.shape[2]
    ans = []
    for id_l, l in enumerate(covering_labels):
        # all stats, label l, slice of the stack id_l
        quart_low, mu, quart_up = list(intensities_distrib_matrix[:, l, id_l])
        # value of the triangular distribution normalised for the mean of the distribution itself
        val_normalised = triangular_density_function(grayscale_value, quart_low, mu, quart_up) / float(mu)
        ans.append(val_normalised)
    return np.array(ans)


def weighting_for_distance_from_trusted_label(point, stack_weights, stack_segmentations):
    """

    :param point: coordinates of a point in the stack.
    :param stack_weights:
    :param stack_segmentations:
    :return:
    """
    x, y, z = point
    list_candidates = list(stack_segmentations[x, y, z, :])
    # the closest to a stack weight with a 1.0 in the stack_weights, with the same value of the list candidates
    r = 1
    island_reached = False
    label_island = -1
    c = [0, 0, 0]
    while r < 1000 or island_reached:
        coord = get_shell_for_given_radius(r)
        for c in coord:
            if 1.0 in stack_weights[x + c[0], y + c[1], z + c[2], :]:
                index_label_island = list(stack_weights[x + c[0], y + c[1], z + c[2], :]).index(1.0)
                label_island = stack_segmentations[x + c[0], y + c[1], z + c[2], index_label_island]
                if label_island in list_candidates:
                    island_reached = True
        r += 1

    pos_label_at_point = stack_segmentations[x + c[0], y + c[1], z + c[2], :].index(label_island)

    ans = [0.0] * len(list_candidates)
    ans[pos_label_at_point] = 1.0

    return np.array(ans)


def get_values_below_label(image, segmentation, label):
    np.testing.assert_array_equal(image.shape, segmentation.shape)
    below_label_places = segmentation == label
    coord = np.nonzero(below_label_places.flatten())[0]
    return np.take(image.flatten(), coord)


def get_intensities_statistics_matrix(warped, segmentation, percentile=10):
    """
    Given an image and a segmentation
    (or a stack of images and a stack of segmentations)
    provides a matrix M

                   | label 1  | label 2 | label 3 | ...
    ------------------------------------------------------------------------
    inf percentile |
    mean           |
    sup percentile |

    where M[0, k] is the superior quartile of the grayscale values below the label k
          M[1, k] is the mean of the grayscale values below label k
          M[0, k] is the inferior quartile of the grayscale values below the label k

    The size of the matrix is 3 x max label.
    If the segmentation has 2 labels 0 and 255, than the matrix has shape 3 x 255 and it is mostly composed by NAN,
    where the labels are not present in the segmentation.

    :param warped: image
    :param segmentation: segmentation
    :param percentile: percentile epsilon so that inf percentile = 0 + percentile , sup percentile = 100 - percentile
    :return: table M
    """
    np.testing.assert_array_equal(warped.shape, segmentation.shape)
    list_all_labels = list(set(segmentation.astype('uint64').flat) - {0})
    list_all_labels.sort()

    if len(warped.shape) == 4:
        num_stacks = warped.shape[3]
        m = np.empty([3, int(list_all_labels[-1] + 1), num_stacks], dtype=np.float64)
        m.fill(np.NAN)
        for stack_id in range(num_stacks):
            for label in list_all_labels:

                vals = get_values_below_label(warped[..., stack_id], segmentation[..., stack_id], label)

                m[0, label, stack_id] = np.percentile(vals, 0 + percentile)
                m[1, label, stack_id] = np.mean(vals)
                m[2, label, stack_id] = np.percentile(vals, 100 - percentile)
    else:
        m = np.empty([3, len(list_all_labels)], dtype=np.float64)
        m.fill(np.NAN)
        for label in list_all_labels:

            vals = get_values_below_label(warped, segmentation, label)

            m[0, label] = np.percentile(vals, 25)
            m[1, label] = np.mean(vals)
            m[2, label] = np.percentile(vals, 75)

    return m


def phase_1_majority_voting(stack_segmentations, threshold=None):
    # input is a numpy array
    num_timepoints = stack_segmentations.shape[3]
    if threshold is None:
        threshold = int(np.ceil(num_timepoints / float(2)))
    sh = list(stack_segmentations.shape)
    sh[-1] += 1
    stack_weights = np.zeros(sh, dtype=np.float32)
    for x in xrange(stack_segmentations.shape[0]):
        print x
        for y in xrange(stack_segmentations.shape[1]):
            for z in xrange(stack_segmentations.shape[2]):

                c = Counter(list(stack_segmentations[x, y, z, :]))
                # when the number of occurrences of the most common is >= threshold
                if c.most_common(1)[0][1] >= threshold:
                    first_majority_index = list(stack_segmentations[x, y, z, :]).index(c.most_common(1)[0][0])
                    stack_weights[x, y, z, first_majority_index] = 1.0

    return stack_weights


def phase_2_majority_voting(target_image, stack_warped, stack_segmentations, stack_weights):

    weight_methods_map = [0.3, 0.4, 0.3]  # lncc, label background, distance

    morpho_patch = get_morphological_patch(3, shape='circle')

    intensities_distrib_matrix = get_intensities_statistics_matrix(stack_warped, stack_segmentations)

    # input is a numpy array
    output_stack_weights = copy.deepcopy(stack_weights)
    for x in xrange(stack_segmentations.shape[0]):
        print x
        for y in xrange(stack_segmentations.shape[1]):
            for z in xrange(stack_segmentations.shape[2]):

                if 1.0 not in stack_weights[x, y, z, :]:

                    morpho_mask = get_morphological_mask([x, y, z], target_image.shape, morpho_patch=morpho_patch)

                    distances_lncc = weighting_for_LNCC([x, y, z], target_image, stack_warped,
                                                        morphological_mask=morpho_mask)

                    weight_bkg = weighting_for_whole_label_values(target_image[x, y, z],
                                                                      stack_segmentations[x, y, z, :],
                                                                      intensities_distrib_matrix)

                    weight_dist = weighting_for_distance_from_trusted_label([x, y, z], stack_weights,
                                                                            stack_segmentations)

                    # Weighted sum of the values:
                    output_stack_weights[x, y, z, :] = np.array(weight_methods_map) * \
                                                       (distances_lncc + weight_bkg + weight_dist)

    return output_stack_weights


def get_uncertain_values_mask(stack_weights):
    # input are numpy array - as intermediate passage.
    uncertain_mask = np.zeros(stack_weights.shape[:-1], dtype=np.float64)
    for x in xrange(stack_weights.shape[0]):
        for y in xrange(stack_weights.shape[1]):
            for z in xrange(stack_weights.shape[2]):
                if 1.0 not in list(stack_weights[x, y, z, :]):
                    uncertain_mask[x, y, z] = 1
    return uncertain_mask


def from_weights_and_segmentations_get_the_final_segmentation(stack_segmentations, stack_weights, extra_label=254):
    # input are numpy array - prototype, very slow!
    ans = np.zeros_like(stack_segmentations[..., 0])
    for x in xrange(ans.shape[0]):
        for y in xrange(ans.shape[1]):
            for z in xrange(ans.shape[2]):
                label = (list(stack_segmentations[x, y, z, :]) + [extra_label]).index(np.max(stack_weights[x, y, z, :]))
                ans[x, y, z] = label
    return ans


def weighted_sum_label_fusion(stack_seg, stack_warp, target, pfi_step_1_output=None, load_step_1_output=True):
    """
    Input are arrays corresponding to images already aligned in the common space.
    :param target:
    :param stack_seg:
    :param stack_warp:
    :return:
    """
    #
    # print 'starting phase 1'
    # stack_weights1 = phase_1_majority_voting(stack_seg)
    # print 'phase 1 end'
    #
    # debug
    stack_weights1 = np.ones((100,100,100, 10))

    stack_weights1[30,30,30, :] = np.zeros(10)

    print 'starting phase 2'
    stack_weights2 = phase_2_majority_voting(target, stack_warp, stack_seg, stack_weights1)
    print 'phase 2 end'
    final_seg = from_weights_and_segmentations_get_the_final_segmentation(stack_seg, stack_weights2)

    return final_seg
