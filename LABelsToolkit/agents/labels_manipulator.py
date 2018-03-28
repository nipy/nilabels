import nibabel as nib

from LABelsToolkit.tools.aux_methods.utils_path import get_pfi_in_pfi_out
from LABelsToolkit.tools.aux_methods.utils_nib import set_new_data

from LABelsToolkit.tools.image_colors_manipulations.relabeller import relabeller, \
    permute_labels, erase_labels, assign_all_other_labels_the_same_value, keep_only_one_label


class LABelsToolkitLabelsManipulate(object):
    """
    Facade of the methods in tools, for work with paths to images rather than
    with data. Methods under LabelsManagerManipulate are taking in general
    one or more input manipulate them according to some rule and save the
    output in the output_data_folder or in the specified paths.
    """
    # TODO add filename for labels descriptors and manipulations of labels descriptors.

    def __init__(self, input_data_folder=None, output_data_folder=None):
        self.pfo_in = input_data_folder
        self.pfo_out = output_data_folder

    def relabel(self, pfi_input, pfi_output=None, list_old_labels=(), list_new_labels=()):
        """
        Masks of :func:`labels_manager.tools.manipulations.relabeller.relabeller` using filename
        """

        pfi_in, pfi_out = get_pfi_in_pfi_out(pfi_input, pfi_output, self.pfo_in,
                                             self.pfo_out)
        im_labels = nib.load(pfi_in)
        data_labels = im_labels.get_data()
        data_relabelled = relabeller(data_labels, list_old_labels=list_old_labels,
                                     list_new_labels=list_new_labels)

        im_relabelled = set_new_data(im_labels, data_relabelled)

        nib.save(im_relabelled, pfi_out)
        print('Relabelled image {0} saved in {1}.'.format(pfi_in, pfi_out))
        return pfi_out

    def permute_labels(self, pfi_input, pfi_output=None, permutation=()):

        pfi_in, pfi_out = get_pfi_in_pfi_out(pfi_input, pfi_output, self.pfo_in, self.pfo_out)

        im_labels = nib.load(pfi_in)
        data_labels = im_labels.get_data()
        data_permuted = permute_labels(data_labels, permutation=permutation)

        im_permuted = set_new_data(im_labels, data_permuted)
        nib.save(im_permuted, pfi_out)
        print('Permuted labels from image {0} saved in {1}.'.format(pfi_in, pfi_out))
        return pfi_out

    def erase_labels(self, pfi_input, pfi_output=None, labels_to_erase=()):

        pfi_in, pfi_out = get_pfi_in_pfi_out(pfi_input, pfi_output, self.pfo_in, self.pfo_out)

        im_labels = nib.load(pfi_in)
        data_labels = im_labels.get_data()
        data_erased = erase_labels(data_labels, labels_to_erase=labels_to_erase)

        im_erased = set_new_data(im_labels, data_erased)
        nib.save(im_erased, pfi_out)
        print('Erased labels from image {0} saved in {1}.'.format(pfi_in, pfi_out))
        return pfi_out

    def assign_all_other_labels_the_same_value(self, pfi_input, pfi_output=None,
        labels_to_keep=(), same_value_label=255):

        pfi_in, pfi_out = get_pfi_in_pfi_out(pfi_input, pfi_output, self.pfo_in, self.pfo_out)

        im_labels = nib.load(pfi_in)
        data_labels = im_labels.get_data()
        data_reassigned = assign_all_other_labels_the_same_value(data_labels,
                                labels_to_keep=labels_to_keep, same_value_label=same_value_label)

        im_reassigned = set_new_data(im_labels, data_reassigned)
        nib.save(im_reassigned, pfi_out)
        print('Reassigned labels from image {0} saved in {1}.'.format(pfi_in, pfi_out))
        return pfi_out

    def keep_one_label(self, pfi_input, pfi_output=None, label_to_keep=1):

        pfi_in, pfi_out = get_pfi_in_pfi_out(pfi_input, pfi_output, self.pfo_in, self.pfo_out)

        im_labels = nib.load(pfi_in)
        data_labels = im_labels.get_data()
        data_one_label = keep_only_one_label(data_labels, label_to_keep)

        im_one_label = set_new_data(im_labels, data_one_label)
        nib.save(im_one_label, pfi_out)
        print('Label {0} kept from image {1} and saved in {2}.'.format(label_to_keep, pfi_in, pfi_out))
        return pfi_out
