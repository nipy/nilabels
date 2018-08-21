import os


def connect_path_tail_head(tail, head):
    """
    It is expected to find
      1) the path to folder in tail and filename in head.
      2) the full path in the head (with tail as sub-path).
      3) the tail with a base path and head to have an additional path + filename.
    :param tail:
    :param head:
    :return:
    """
    if tail is None or tail == '':
        return head
    if head.startswith(tail):  # Case 2
        return head
    else:  # case 1, 3
        return os.path.join(tail, head)


def get_pfi_in_pfi_out(filename_in, filename_out, pfo_in, pfo_out):
    """
    Core method of every facade to connect working folder with input data files.
    :param filename_in: filename input
    :param filename_out: filename output
    :param pfo_in: path to folder input
    :param pfo_out: path to folder output
    :return: connection of the inupt and the output.
    """
    pfi_in = connect_path_tail_head(pfo_in, filename_in)
    if filename_out is None:
        pfi_out = pfi_in
    else:
        if pfo_out is None:
            pfi_out = connect_path_tail_head(pfo_in, filename_out)
        else:
            pfi_out = connect_path_tail_head(pfo_out, filename_out)

    return pfi_in, pfi_out
