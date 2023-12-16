import os
import unittest
import numpy as np
import numpy.testing as npt


class TestFileComparison(unittest.TestCase):
    """
    A unittest class for testing file differences.
    """

    def test_file_equality(self) -> None:
        """
        Test method to check if the data in the filtered files is equal to the data in the reference files.
        """

        # Get the current working directory
        root_dir: str = os.getcwd()

        # Define the directories for filtered and reference data
        filtered_data_dir: str = os.path.join(root_dir, "dataset", "filtered_data")
        reference_data_dir: str = os.path.join(root_dir, "dataset", "reference_data")

        # Walk through the directory tree of the filtered data directory
        for dirpath, dirnames, filenames in os.walk(filtered_data_dir):
            for filename in filenames:
                # Define the paths for the filtered and reference files
                filtered_file: str = os.path.join(dirpath, filename)
                reference_file: str = filtered_file.replace(
                    "filtered_data", "reference_data"
                )

                # Check if the reference file exists
                if os.path.exists(reference_file):
                    # Load the data from the filtered and reference files
                    filtered_data = np.load(filtered_file)
                    reference_data = np.load(reference_file)

                    try:
                        # Assert that the arrays are equal
                        npt.assert_array_equal(filtered_data, reference_data)
                    except AssertionError as e:
                        # If the arrays are not equal, fail the test and print the error message
                        self.fail(
                            f"Arrays in {filtered_file} and {reference_file} are not the same: {str(e)}"
                        )
                else:
                    # If the reference file does not exist, fail the test and print the error message
                    self.fail(f"Reference file {reference_file} does not exist.")
