import os
from sanity_checks import check_pfi_io


def connect_path_tail_head(tail, head):
    """
    It expects
      1) to find the path to folder in tail and filename in head.
      2) to find the full path in the head (with tail as sub-path).
      3) to have tail with a base path and head to have an additional path + filename.
    :param tail:
    :param head:
    :return:
    """
    if tail is None or tail == '':
        return head
    if head.startswith(tail):  # os.path.abspath(head).startswith(os.path.abspath(tail)):  # Case 2
        return head
    else:  # case 1, 3
        return os.path.join(tail, head)


def get_pfi_in_pfi_out(filename_in, filename_out, pfo_in, pfo_out):
    """
    Core method of every facade
    :param filename_in:
    :param filename_out:
    :param pfo_in:
    :param pfo_out:
    :return:
    """
    pfi_in = connect_path_tail_head(pfo_in, filename_in)
    if filename_out is None:
        pfi_out = pfi_in
    else:
        if pfo_out is None:
            pfi_out = connect_path_tail_head(pfo_in, filename_out)
        else:
            pfi_out = connect_path_tail_head(pfo_out, filename_out)

    # check_pfi_io(pfi_in, pfi_out)
    return pfi_in, pfi_out
