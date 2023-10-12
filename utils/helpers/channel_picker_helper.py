class ChannelPicker:
    @staticmethod
    def pick_channels(epochs, channel_levels, ch_pick_level, logger):
        """
        Pick EEG channels based on the specified channel level and update the logger.

        This static method selects EEG channels according to the provided channel level.
        It updates the logger with information about the channel selection.

        Parameters:
            epochs (mne.Epochs): EEG data epochs.
            channel_levels (list of list): List of channel selections.
            ch_pick_level (int): EEG channel selection level.
            logger: The logger object for logging information.

        Raises:
            ValueError: If an invalid `ch_pick_level` is provided.

        Returns:
            mne.epochs.Epochs: Modified 'epochs' object with selected EEG channels.


        """
        logger.info(f"EEG channels before selection: \n{epochs[0].ch_names}")

        if ch_pick_level < len(channel_levels):
            epochs.pick_channels(ch_names=channel_levels[ch_pick_level])
            if len(epochs.ch_names) == len(channel_levels[ch_pick_level]):
                logger.info("Channel selection successful.")
            else:
                raise ValueError("Channel selection failed.")
        elif ch_pick_level == len(channel_levels):
            pass
        else:
            raise ValueError("EEG channel selection level is not defined or invalid.")

        logger.info(f"EEG channels remaining after selection: \n{epochs[0].ch_names}")

        return epochs
