from defs import definition_label
import pandas as pa


class LabelsManagerMeasure(object):
    """
    Facade of the methods in tools.detections and tools.caliber, where methods are accessed through
    paths to images rather than with data.
    Methods under LabelsManagerDetect are taking in general one or more input
    and return some feature of the segmentations {}.
    """.format(definition_label)

    def __init__(self, input_data_folder=None, output_data_folder=None):
        self.pfo_in = input_data_folder
        self.pfo_out = output_data_folder

    def volume(self, pfi_segmentation, labels='tot', pfi_anathomy=None):
        pass

    def d(self, metrics):
        pass
    # WIP: island detections, graph detections, cc detections.
