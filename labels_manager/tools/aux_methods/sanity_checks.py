import os


def arrange_path(base_path, main_input):
    return os.path.join(base_path, main_input)


def connect_tail_head_path(tail, head):
    """
    It expects
      1) to find the path to folder in tail and filename in head.
      2) to find the full path in the head.
      3) to have tail with a base path and head to have an additional path + filename.
    :param tail:
    :param head:
    :return:
    """

    if os.path.dirname(head) == tail:  # case 2
        return head
    else:  # case 1, 3
        return os.path.join(tail, head)


def check_pfi_io(pfi_input, pfi_output):
    if not os.path.exists(pfi_input):
        IOError('Input file {} does not exists.'.format(pfi_input))
    if pfi_output is not None:
        if not os.path.exists(os.path.dirname(pfi_output)):
            IOError('Output file {} is located in a non-existing folder.'.format(
                    pfi_output))


def get_pfi_in_pfi_out(filename_in, filename_out, pfo_in, pfo_out):
    """
    :param filename_in:
    :param filename_out:
    :param pfo_in:
    :param pfo_out:
    :return:
    """
    pfi_in = connect_tail_head_path(pfo_in, filename_in)    
    if filename_out is None:
        pfi_out = pfi_in
    else:
        if pfo_out is None:
            pfi_out = connect_tail_head_path(pfo_in, filename_out)
        else:
            pfi_out = connect_tail_head_path(pfo_out, filename_out)

    check_pfi_io(pfi_in, pfi_out)
    return pfi_in, pfi_out
