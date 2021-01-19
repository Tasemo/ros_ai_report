from torchaudio.datasets import SPEECHCOMMANDS
from enum import Enum, auto
import utils
import os

class DataSetType(Enum):
    TRAINING = auto()
    VALIDATION = auto()
    TESTING = auto()

# each data point contains the following values: waveform, sample_rate, label, speaker_id, utterance_number
class DataSubSet(SPEECHCOMMANDS):
    def __init__(self, type: DataSetType):
        super().__init__("./", download=True)
        
        def load_list(filename):
            filepath = os.path.join(self._path, filename)
            with open(filepath) as fileobj:
                return [os.path.join(self._path, line.strip()) for line in fileobj]

        if type == DataSetType.VALIDATION:
            self._walker = load_list("validation_list.txt")
        elif type == DataSetType.TESTING:
            self._walker = load_list("testing_list.txt")
        elif type == DataSetType.TRAINING:
            # the training set contains of everything not in the validation or testing set
            excludes = load_list("validation_list.txt") + load_list("testing_list.txt")
            excludes = set(excludes)
            self._walker = [w for w in self._walker if w not in excludes]

    @utils.persistent_memoize
    def get_file_count(self):
        return sum(1 for _ in self)

    @utils.persistent_memoize
    def get_labels(self):
        return sorted(set(datapoint[2] for datapoint in self))

    def get_sample_rate(self):
        # sample rate is guaranteed to be the same for all data points
        return self[0][1]
