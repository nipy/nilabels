from definitions import definition_label


class LabelsManagerDetect(object):
    """
    Facade of the methods in tools.detections, where methods are accessed through
    paths to images rather than with data.
    Methods under LabelsManagerDetect are taking in general one or more input
    and return some feature of the segmentations {}.
    """.format(definition_label)

    def __init__(self, input_data_folder=None, output_data_folder=None):
        self.pro_in = input_data_folder
        self.pfo_out = output_data_folder


    # WIP: island detections, graph detections, cc detections.
