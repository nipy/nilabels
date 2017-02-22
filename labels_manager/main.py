import os

from labels_manager.helpers.manipulations import LabelsManagerManipulate
from labels_manager.helpers.measures import LabelsManagerMeasure
from labels_manager.helpers.detectors import LabelsManagerDetect


class LabelsManager(object):

    def __init__(self, input_data_folder, output_data_folder=None):

        if not (os.path.isdir(input_data_folder) or input_data_folder is None):
            raise IOError('Selected path must be None or must point to an existing folder.')

        if output_data_folder is None:
            output_data_folder = input_data_folder

        self.input_data_folder = input_data_folder

        self.manipulate = LabelsManagerManipulate(input_data_folder, output_data_folder)
        self.measure = LabelsManagerMeasure(input_data_folder, output_data_folder)
        self.detect = LabelsManagerDetect(input_data_folder, output_data_folder)
