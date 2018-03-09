import os

from LABelsToolkit.agents.labels_manipulator import LabelsManagerLabelsManipulate
from LABelsToolkit.agents.intensities_manipulator import LabelsManagerIntensitiesManipulate
from LABelsToolkit.agents.measurer import LabelsManagerMeasure
from LABelsToolkit.agents.fuser import LabelsManagerFuse
from LABelsToolkit.agents.propagator import LabelsManagerPropagate
from LABelsToolkit.agents.symmetrizer import LabelsManagerSymmetrize
from LABelsToolkit.agents.checker import LabelsManagerChecker
from LABelsToolkit.agents.header_controller import LabelsManagerHeaderController


class LabelsManager(object):

    def __init__(self, input_data_folder=None, output_data_folder=None):
        """
        Main agent-class that access all the tools methods given input paths through agents.
        Each agent has a different semantic task, so that recover the tool to be applied to a path will be easier.
        The nomenclature and functionality of each tool is decoupled from the agents that are using them.
        > The input of a method in tools is typically a nibabel image.
        > The input of a method in the agents it typically a path to a nifti image.
        """
        if input_data_folder is not None:
            if not os.path.isdir(input_data_folder):
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
        self.manipulate_labels      = LabelsManagerLabelsManipulate(self._pfo_in, self._pfo_out)
        self.manipulate_intensities = LabelsManagerIntensitiesManipulate(self._pfo_in, self._pfo_out)
        self.measure                = LabelsManagerMeasure(self._pfo_in, self._pfo_out)
        self.fuse                   = LabelsManagerFuse(self._pfo_in, self._pfo_out)
        self.propagate              = LabelsManagerPropagate(self._pfo_in, self._pfo_out)
        self.symmetrize             = LabelsManagerSymmetrize(self._pfo_in, self._pfo_out)
        self.check                  = LabelsManagerChecker(self._pfo_in, self._pfo_out)
        self.header                 = LabelsManagerHeaderController(self._pfo_in, self._pfo_out)
