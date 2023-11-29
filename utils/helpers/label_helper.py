import os
import mne
from typing import List


class LabelHelper:
    # TODO: This class sucks. Either i need to drop the staticmetods or rewrite them into real helpers
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
