import os
import mne
from typing import List


class LabelHelper:
    @staticmethod
    def init_arguments(data_path, subject):
        # The imaginary runs for indexing purposes
        runs = [4, 6, 8, 10, 12, 14]
        # fists
        task2 = [4, 8, 12]
        # legs
        task4 = [6, 10, 14]
        # The subject naming scheme can be adapted using zero fill, example 'S001'
        sub_name = "S" + str(subject).zfill(3)
        # Generates a path for the folder of the subject
        sub_folder = os.path.join(data_path, sub_name)
        subject_runs = []
        return runs, sub_folder, sub_name, subject_runs, task2, task4
    @staticmethod
    def label_annotations(
        descriptions: List[str], old_labels: List[str], new_labels: List[str]
    ) -> List[str]:
        """
        Label annotations based on a mapping of old labels to new labels.

        Args:
            descriptions (List[str]): List of annotations.
            old_labels (List[str]): List of old labels.
            new_labels (List[str]): List of new labels.

        Returns:
            List[str]: Updated annotations.
        """
        for i, desc in enumerate(descriptions):
            if desc in old_labels:
                descriptions[i] = new_labels[old_labels.index(desc)]
        return descriptions
    @staticmethod
    def label_epochs(
        self, raw_filt: mne.io.Raw, run: int, task2: List[int], task4: List[int]
    ) -> mne.Epochs:
        """
        Label epochs based on task-specific annotations.

        Args:
            raw_filt (mne.io.Raw): Processed raw data.
            run (int): Current run number.
            task2 (List[int]): Run numbers for task2.
            task4 (List[int]): Run numbers for task4.

        Returns:
            mne.Epochs: Labeled epochs.

        Labeled annotations:
            - 'B' indicates baseline.
            - 'L' indicates motor imagination of opening and closing the left fist.
            - 'R' indicates motor imagination of opening and closing the right fist.
            - 'LR' indicates motor imagination of opening and closing both fists.
            - 'F' indicates motor imagination of moving both feet.

        The annotation description of the raw runs have <u2 (2 char unicode)
        dtype: {dtype[str_]:()} <U2.

        Description of the built-in data annotations:
            - The description for each run describes the sequence of
            - 'T0': rest
            - 'T1': motion real/imaginary
                - the left fist (in runs 3, 4, 7, 8, 11, and 12)
                - both fists (in runs 5, 6, 9, 10, 13, and 14)
            - 'T2': motion real/imaginary
                - the right fist (in runs 3, 4, 7, 8, 11, and 12)
                - both feet (in runs 5, 6, 9, 10, 13, and 14)

        If we print out the annotation descriptions,
        we would get 'T0' between all of the 'T1' and 'T2' annotations.
        It is easily recognizable that the meaning of 'T0-1-2'
        descriptions is dependent on the run numbers.
        """
        if run in task2:
            raw_filt.annotations.description = self.label_annotations(
                raw_filt.annotations.description, ["T0", "T1", "T2"], ["B", "L", "R"]
            )
        elif run in task4:
            raw_filt.annotations.description = self.label_annotations(
                raw_filt.annotations.description, ["T0", "T1", "T2"], ["B", "LR", "F"]
            )
        return raw_filt
