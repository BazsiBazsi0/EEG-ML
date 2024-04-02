import mne


class EpochCreatorHelper:
    @staticmethod
    def create_epochs(
        raw_conc,
        events,
        event_id,
        tmin=0,
        tmax=5,
        proj=True,
        picks=None,
        baseline=None,
        preload=True,
    ):
        """
        Create and return an 'mne.epochs.Epochs' object.

        This static method constructs an 'mne.epochs.Epochs' object from provided data.

        Parameters:
            raw_conc (mne.io.Raw): Concatenated EEG data.
            events (numpy.ndarray): Event matrix.
            event_id (dict): Event descriptions to codes.
            tmin (float, optional): Start of the time interval (seconds).
            tmax (float, optional): End of the time interval (seconds).
            proj (bool, optional): Preprocess with SSP (True/False).
            picks (None or array of int, optional): EEG channels for analysis
            baseline (None or tuple, optional): Baseline correction time interval.
            preload (bool, optional): Preload data for faster access (True/False).

        Returns:
            mne.epochs.Epochs: Segmented EEG data.

        Note:
            This method creates EEG epochs and returns the resulting 'Epochs' object.
        """
        return mne.epochs.Epochs(
            raw_conc,
            events,
            event_id,
            tmin=tmin,
            tmax=tmax,
            proj=proj,
            picks=mne.pick_types(
                raw_conc.info,
                meg=False,
                eeg=True,
                stim=False,
                eog=False,
                exclude="bads",
            ),
            baseline=baseline,
            preload=preload,
        )
