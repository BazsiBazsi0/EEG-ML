import os
import mne
import numpy as np
from typing import List
from unittest import TestCase
from utils.helpers.label_helper import LabelHelper

class TestLabelHelper(TestCase):
    def setUp(self):
        self.helper = LabelHelper()
        self.data_path = "/dataset/files"
        self.subject = 1

    def test_init_arguments(self):
        runs, sub_folder, sub_name, subject_runs, task2, task4 = LabelHelper.init_arguments(self.data_path, self.subject)
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
        updated_descriptions = self.helper.label_annotations(descriptions, old_labels, new_labels)
        self.assertEqual(updated_descriptions, ["B", "L", "R"])

    def test_label_epochs(self):
        # Create a dummy raw object with annotations
        info = mne.create_info(ch_names=10, sfreq=1000.0)
        raw = mne.io.RawArray(np.random.random((10, 1000)), info)
        raw.set_annotations(mne.Annotations(onset=[0.5], duration=[0.1], description=["T0"]))
        
        # Test for task2
        run = 4
        task2 = [4]
        task4 = []
        updated_raw = self.helper.label_epochs(raw.copy(), run, task2, task4)
        self.assertEqual(updated_raw.annotations.description[0], "B")

        # Test for task4
        run = 6
        task2 = []
        task4 = [6]
        updated_raw = self.helper.label_epochs(raw.copy(), run, task2, task4)
        self.assertEqual(updated_raw.annotations.description[0], "B")
