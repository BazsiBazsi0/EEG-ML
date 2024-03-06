import os
import mne
import numpy as np
from unittest import TestCase
from utils.helpers.label_helper import LabelHelper


class TestLabelHelper(TestCase):
    def setUp(self):
        self.helper = LabelHelper()
        self.data_path = "/dataset/files"
        self.subject = 1

    def test_init_arguments(self):
        (
            runs,
            sub_folder,
            sub_name,
            subject_runs,
            task2,
            task4,
        ) = LabelHelper.init_arguments(self.data_path, self.subject)
        self.assertEqual(runs, [4, 6, 8, 10, 12, 14])
        self.assertEqual(sub_folder, os.path.join(self.data_path, "S001"))
        self.assertEqual(sub_name, "S001")
        self.assertEqual(subject_runs, [])
        self.assertEqual(task2, [4, 8, 12])
        self.assertEqual(task4, [6, 10, 14])

    def test_label_annotations(self):
        descriptions = ["T0", "T1", "T2"]
        old_labels = ["T0", "T1", "T2"]
        new_labels = ["B", "L", "R"]
        updated_descriptions = self.helper.label_annotations(
            descriptions, old_labels, new_labels
        )
        self.assertEqual(updated_descriptions, ["B", "L", "R"])