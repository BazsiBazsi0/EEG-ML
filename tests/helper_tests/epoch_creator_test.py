import unittest
import mne
import numpy as np
from utils.helpers.epoch_creator_helper import EpochCreatorHelper


class TestEpochCreatorHelper(unittest.TestCase):
    def setUp(self):
        # Creating a mock raw object, with "eeg" channels
        self.raw_conc = mne.io.RawArray(
            np.random.rand(10, 1000), mne.create_info(10, 1000, ch_types="eeg")
        )

        self.events = np.array([[200, 0, 1]])
        self.event_id = {"event": 1}
        self.tmin = 0
        self.tmax = 4
        self.proj = True
        self.picks = None
        self.baseline = None
        self.preload = True

    def test_create_epochs(self):
        epochs = EpochCreatorHelper.create_epochs(
            self.raw_conc,
            self.events,
            self.event_id,
            tmin=self.tmin,
            tmax=self.tmax,
            proj=self.proj,
            picks=self.picks,
            baseline=self.baseline,
            preload=self.preload,
        )

        # Assert that the returned object is an instance of mne.Epochs
        self.assertIsInstance(epochs, mne.Epochs)
