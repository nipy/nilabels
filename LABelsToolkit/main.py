import os

from LABelsToolkit.agents.labels_manipulator import LABelsToolkitLabelsManipulate
from LABelsToolkit.agents.shape_manipulator import LABelsToolkitShapeManipulate
from LABelsToolkit.agents.intensities_manipulator import LABelsToolkitIntensitiesManipulate
from LABelsToolkit.agents.measurer import LABelsToolkitMeasure
from LABelsToolkit.agents.fuser import LABelsToolkitFuse
from LABelsToolkit.agents.propagator import LABelsToolkitPropagate
from LABelsToolkit.agents.symmetrizer import LABelsToolkitSymmetrize
from LABelsToolkit.agents.checker import LABelsToolkitChecker
from LABelsToolkit.agents.header_controller import LABelsToolkitHeaderController
from LABelsToolkit.tools.icv.icv_estimator import ICV_estimator

class LABelsToolkit(object):

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
        self.manipulate_labels      = LABelsToolkitLabelsManipulate(self._pfo_in, self._pfo_out)
        self.manipulate_intensities = LABelsToolkitIntensitiesManipulate(self._pfo_in, self._pfo_out)
        self.manipulate_shape       = LABelsToolkitShapeManipulate(self._pfo_in, self._pfo_out)
        self.measure                = LABelsToolkitMeasure(self._pfo_in, self._pfo_out)
        self.fuse                   = LABelsToolkitFuse(self._pfo_in, self._pfo_out)
        self.propagate              = LABelsToolkitPropagate(self._pfo_in, self._pfo_out)
        self.symmetrize             = LABelsToolkitSymmetrize(self._pfo_in, self._pfo_out)
        self.check                  = LABelsToolkitChecker(self._pfo_in, self._pfo_out)
        self.header                 = LABelsToolkitHeaderController(self._pfo_in, self._pfo_out)
        self.icv                    = ICV_estimator
