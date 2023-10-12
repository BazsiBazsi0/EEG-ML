import unittest
import mne
import numpy as np
from unittest.mock import Mock
from utils.helpers.channel_picker_helper import ChannelPickerHelper
from utils.helpers.epoch_creator_helper import EpochCreatorHelper


class TestHelpers(unittest.TestCase):
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

    def test_pick_channels(self):
        # Create a mock logger
        logger = Mock()

        # Create a mock epochs object with 3 channels
        info = mne.create_info(ch_names=["ch1", "ch2", "ch3"], sfreq=1000.0)
        epochs = mne.EpochsArray(
            [[[1, 2, 3], [4, 5, 6], [7, 8, 9]]], info
        )  # Modified this line

        # Define channel levels
        channel_levels = [["ch1", "ch2"], ["ch1"]]

        # Test valid channel selection
        ChannelPickerHelper.pick_channels(epochs, channel_levels, 0, logger)
        self.assertEqual(epochs.ch_names, ["ch1", "ch2"], "Channel selection failed.")

        # Test no channel selection
        ChannelPickerHelper.pick_channels(
            epochs, channel_levels, len(channel_levels), logger
        )
        self.assertEqual(
            epochs.ch_names, ["ch1", "ch2"], "Channels should not be changed."
        )

        # Test invalid channel selection level
        with self.assertRaises(ValueError):
            ChannelPickerHelper.pick_channels(
                epochs, channel_levels, len(channel_levels) + 1, logger
            )

        # Test failed channel selection
        with self.assertRaises(ValueError):
            ChannelPickerHelper.pick_channels(epochs, [["ch4"]], 0, logger)

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
