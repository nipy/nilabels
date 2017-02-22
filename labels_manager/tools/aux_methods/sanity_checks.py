import os


def arrange_path(base_path, main_input):
    return os.path.join(base_path, main_input)


def connect_tail_head_path(tail, head):
    if os.path.basename(head) == head:  # head is only a filename
        return os.path.join(tail, head)
    elif os.path.dirname(head) == tail:  # head includes the tail
        return head
    else:  # head is a relative path to tail (existence not verified here).
        os.path.join(head, tail)


def check_pfi_io(pfi_input, pfi_output):
    if not os.path.exists(pfi_input):
        IOError('Input file {} does not exists.'.format(pfi_input))
    if pfi_output is not None:
        if not os.path.exists(os.path.dirname(pfi_output)):
            IOError('Output file {} is located in an unexisting folder.'.format(
                    pfi_output))


def get_pfi_in_pfi_out(filename_in, filename_out, pfo_in, pfo_out):
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
