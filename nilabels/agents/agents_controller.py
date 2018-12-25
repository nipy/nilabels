import os

from nilabels.agents.labels_manipulator import LabelsManipulator
from nilabels.agents.shape_manipulator import ShapeManipulator
from nilabels.agents.intensities_manipulator import IntensitiesManipulator
from nilabels.agents.measurer import LabelsMeasure
from nilabels.agents.fuser import LabelsFuser
from nilabels.agents.symmetrizer import SegmentationSymmetrizer
from nilabels.agents.checker import LabelsChecker
from nilabels.agents.header_controller import HeaderController
from nilabels.agents.segmenter import LabelsSegmenter
from nilabels.agents.math import Math


class AgentsController(object):

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
        self.manipulate_labels      = LabelsManipulator(self._pfo_in, self._pfo_out)
        self.manipulate_intensities = IntensitiesManipulator(self._pfo_in, self._pfo_out)
        self.manipulate_shape       = ShapeManipulator(self._pfo_in, self._pfo_out)
        self.measure                = LabelsMeasure(self._pfo_in, self._pfo_out)
        self.fuse                   = LabelsFuser(self._pfo_in, self._pfo_out)
        self.symmetrize             = SegmentationSymmetrizer(self._pfo_in, self._pfo_out)
        self.check                  = LabelsChecker(self._pfo_in, self._pfo_out)
        self.header                 = HeaderController(self._pfo_in, self._pfo_out)
        self.segment                = LabelsSegmenter(self._pfo_in, self._pfo_out)
        self.math                   = Math(self._pfo_in, self._pfo_out)
