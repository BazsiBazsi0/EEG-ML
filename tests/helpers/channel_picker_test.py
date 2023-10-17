import unittest
import mne
import numpy as np
from unittest.mock import Mock
from utils.helpers.channel_picker_helper import ChannelPickerHelper


class TestChannelPickerHelper(unittest.TestCase):
    def setUp(self):
        # Creating a mock raw object, with "eeg" channels
        self.raw_conc = mne.io.RawArray(
            np.random.rand(10, 1000), mne.create_info(10, 1000, ch_types="eeg")
        )

    def test_pick_channels(self):
        # Create a mock logger
        logger = Mock()

        # Create a mock epochs object with 3 channels
        info = mne.create_info(ch_names=["ch1", "ch2", "ch3"], sfreq=1000.0)
        epochs = mne.EpochsArray(
            [[[1, 2, 3], [4, 5, 6], [7, 8, 9]]], info
        )  # Modified this line

        # Define channel levels
        channel_levels: int = 1

        channel_picks: list = ["ch1", "ch2"]

        # Test valid channel selection
        ChannelPickerHelper.pick_channels(epochs, channel_levels, channel_picks, logger)
        self.assertEqual(epochs.ch_names, ["ch1", "ch2"], "Channel selection failed.")

        # Test no channel selection
        ChannelPickerHelper.pick_channels(epochs, channel_levels, channel_picks, logger)
        self.assertEqual(
            epochs.ch_names, ["ch1", "ch2"], "Channels should not be changed."
        )

        # Test invalid channel selection level
        with self.assertRaises(ValueError):
            ChannelPickerHelper.pick_channels(
                epochs, channel_levels, info.ch_names, logger
            )

        # Test failed channel selection, level higher then 3 should include all
        with self.assertRaises(ValueError):
            ChannelPickerHelper.pick_channels(epochs, 5, [["ch4"]], logger)
