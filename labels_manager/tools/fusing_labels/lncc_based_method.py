import numpy as np
from collections import Counter

from labels_manager.tools.aux_methods.morpological_tools import get_morphological_patch, get_morphological_mask, get_patch_values
from labels_manager.tools.measurements.distances import lncc_distance

def simple_majority_voting_lncc(stack_segmentations, stack_warped, target_image_data, verbose=0, threshold=None):
    """
    All input are numpy arrays.
    Simple majority voting algorithm based on the LNCC measure on the warped, when the majority is not reached.
    :param stack_segmentations:
    :param stack_warped:
    :param target_image_data:
    :param verbose: if 1 or True, simple verbose, if verbose=2 debugging verbose.
    :return:
    """
    assert not np.isnan(np.min(stack_warped))
    assert not np.isnan(np.min(stack_segmentations))
    ans = np.zeros_like(stack_segmentations[..., 0]).astype(np.int16)
    morpho_patch = get_morphological_patch(3, shape='circle')
    num_segmentations = stack_warped.shape[3]

    if threshold is None:
        threshold = int(num_segmentations / 2)

    for x in xrange(target_image_data.shape[0]): #
        for y in xrange(target_image_data.shape[1]):
            for z in xrange(target_image_data.shape[2]):
                c = Counter(list(stack_segmentations[x, y, z, :]))
                if c.most_common(1)[0][1] >= threshold:  # number of occurrences of the most common is >= threshold
                    ans[x, y, z] = c.most_common(1)[0][0]
                else:
                    morphological_mask = get_morphological_mask([x, y, z], target_image_data.shape, morpho_patch=morpho_patch)
                    patch_target = np.array(get_patch_values([x, y, z], target_image_data, morfo_mask=morphological_mask))

                    ordered_measurements = []

                    for t in xrange(num_segmentations):

                        patch_t = np.array(get_patch_values([x, y, z], stack_warped[..., t], morfo_mask=morphological_mask))

                        ordered_measurements.append(lncc_distance(patch_target, patch_t))
                        if verbose == 2:
                            print ordered_measurements

                    pos_max = ordered_measurements.index(np.max(ordered_measurements))

                    if verbose == 2:
                        print x, y, z
                        print ordered_measurements, pos_max
                    ans[x, y, z] = stack_segmentations[x, y, z, pos_max]

    return ans
