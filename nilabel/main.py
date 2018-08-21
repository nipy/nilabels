import os

from nilabel.agents.labels_manipulator import LabelsManipulate
from nilabel.agents.shape_manipulator import ShapeManipulate
from nilabel.agents.intensities_manipulator import IntensitiesManipulate
from nilabel.agents.measurer import LabelsMeasure
from nilabel.agents.fuser import LabelsFuser
from nilabel.agents.propagator import LabelsPropagate
from nilabel.agents.symmetrizer import SegmentationSymmetrize
from nilabel.agents.checker import LabelsChecker
from nilabel.agents.header_controller import HeaderController
from nilabel.agents.segmenter import LabelsSegmenter
from nilabel.tools.icv.icv_estimator import ICV_estimator


class Nilabel(object):

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
        self.manipulate_labels      = LabelsManipulate(self._pfo_in, self._pfo_out)
        self.manipulate_intensities = IntensitiesManipulate(self._pfo_in, self._pfo_out)
        self.manipulate_shape       = ShapeManipulate(self._pfo_in, self._pfo_out)
        self.measure                = LabelsMeasure(self._pfo_in, self._pfo_out)
        self.fuse                   = LabelsFuser(self._pfo_in, self._pfo_out)
        self.propagate              = LabelsPropagate(self._pfo_in, self._pfo_out)
        self.symmetrize             = SegmentationSymmetrize(self._pfo_in, self._pfo_out)
        self.check                  = LabelsChecker(self._pfo_in, self._pfo_out)
        self.header                 = HeaderController(self._pfo_in, self._pfo_out)
        self.segment                = LabelsSegmenter(self._pfo_in, self._pfo_out)
        self.icv                    = ICV_estimator

