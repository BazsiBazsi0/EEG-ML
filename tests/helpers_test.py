import unittest
import mne
from unittest.mock import Mock
from utils.helpers.channel_picker_helper import ChannelPicker


class TestHelpers(unittest.TestCase):
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
        ChannelPicker.pick_channels(epochs, channel_levels, 0, logger)
        self.assertEqual(epochs.ch_names, ["ch1", "ch2"], "Channel selection failed.")

        # Test no channel selection
        ChannelPicker.pick_channels(epochs, channel_levels, len(channel_levels), logger)
        self.assertEqual(
            epochs.ch_names, ["ch1", "ch2"], "Channels should not be changed."
        )

        # Test invalid channel selection level
        with self.assertRaises(ValueError):
            ChannelPicker.pick_channels(
                epochs, channel_levels, len(channel_levels) + 1, logger
            )

        # Test failed channel selection
        with self.assertRaises(ValueError):
            ChannelPicker.pick_channels(epochs, [["ch4"]], 0, logger)
