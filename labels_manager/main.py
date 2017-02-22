import os

from labels_manager.agents.manipulator import LabelsManagerManipulate
from labels_manager.agents.measurer import LabelsManagerMeasure
from labels_manager.agents.detector import LabelsManagerDetect


class LabelsManager(object):

    def __init__(self, input_data_folder, output_data_folder=None):
        """
        Main agent-class that access all the tools methods by paths through
        agents detector, manipulator and measurer.
        """

        if not (os.path.isdir(input_data_folder) or input_data_folder is None):
            raise IOError('Selected path must be None or must point to an existing folder.')

        self.pfo_in = input_data_folder

        if output_data_folder is None:
            self.pfo_in = input_data_folder
        else:
            self.pfo_out = output_data_folder

        self.manipulate = LabelsManagerManipulate(self.pfo_in, self.pfo_out)
        self.measure = LabelsManagerMeasure(self.pfo_in, self.pfo_out)
        self.detect = LabelsManagerDetect(self.pfo_in, self.pfo_out)
