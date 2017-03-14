import os

from labels_manager.agents.manipulator import LabelsManagerManipulate
from labels_manager.agents.measurer import LabelsManagerMeasure
from labels_manager.agents.detector import LabelsManagerDetect
from labels_manager.agents.fuser import LabelsManagerFuse
from labels_manager.agents.propagator import LabelsManagerPropagate


class LabelsManager(object):

    def __init__(self, input_data_folder, output_data_folder=None):
        """
        Main agent-class that access all the tools methods by paths through
        agents called detect, manipulate and measure.
        """

        if not (os.path.isdir(input_data_folder) or input_data_folder is None):
            raise IOError('Selected path must be None or must point to an existing folder.')

        self._pfo_in = input_data_folder

        if output_data_folder is None:
            self._pfo_out = input_data_folder
        else:
            self._pfo_out = output_data_folder
        self._set_attribute_agents()

    def set_input_data_folder(self, input_data_folder):
        self._pfo_in = input_data_folder
        self._set_attribute_agents()

    def set_output_data_folder(self, output_data_folder):
        self._pfo_out = output_data_folder
        self._set_attribute_agents()

    def _set_attribute_agents(self):
        self.manipulate = LabelsManagerManipulate(self._pfo_in, self._pfo_out)
        self.measure    = LabelsManagerMeasure(self._pfo_in, self._pfo_out)
        self.detect     = LabelsManagerDetect(self._pfo_in, self._pfo_out)
        self.fuse       = LabelsManagerFuse(self._pfo_in, self._pfo_out)
        self.propagate  = LabelsManagerPropagate(self._pfo_in, self._pfo_out)
