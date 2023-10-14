class ChannelPickerHelper:
    @staticmethod
    def pick_channels(epochs, channel_level, channel_picks, logger):
        """
        Pick EEG channels based on the specified channel level and update the logger.

        This static method selects EEG channels according to the provided channel level.
        It updates the logger with information about the channel selection.

        Parameters:
            epochs (mne.Epochs): EEG data epochs.
            channel_picks (list of list): List of channel selections.
            channel_level (int): EEG channel selection level.
            logger: The logger object for logging information.

        Raises:
            ValueError: If an invalid `channel_level` is provided.

        Returns:
            mne.epochs.Epochs: Modified 'epochs' object with selected EEG channels.


        """
        logger.info(f"EEG channels before selection: \n{epochs[0].ch_names}")

        if channel_level < len(channel_picks):
            epochs.pick_channels(ch_names=channel_picks)
            if len(epochs.ch_names) == len(channel_picks):
                logger.info("Channel selection successful.")
            else:
                raise ValueError("Channel selection failed.")
        elif channel_level == len(channel_picks):
            pass
        else:
            raise ValueError("EEG channel selection level is not defined or invalid.")

        logger.info(f"EEG channels remaining after selection: \n{epochs[0].ch_names}")

        return epochs
