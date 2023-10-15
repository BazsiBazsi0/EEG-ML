import unittest
import os
from utils.logging_utils import Logger


class TestLogger(unittest.TestCase):
    def setUp(self):
        self.logger = Logger("test_logger")
        self.log_file = "logs/test_logger.log"

    def tearDown(self):
        if os.path.exists(self.log_file):
            os.remove(self.log_file)

    def test_debug(self):
        self.logger.debug("This is a debug message")
        self.assertLogContains("DEBUG", "This is a debug message")

    def test_info(self):
        self.logger.info("This is an info message")
        self.assertLogContains("INFO", "This is an info message")

    def test_warning(self):
        self.logger.warning("This is a warning message")
        self.assertLogContains("WARNING", "This is a warning message")

    def test_error(self):
        self.logger.error("This is an error message")
        self.assertLogContains("ERROR", "This is an error message")

    def test_critical(self):
        self.logger.critical("This is a critical message")
        self.assertLogContains("CRITICAL", "This is a critical message")

    def assertLogContains(self, level, message):
        with open(self.log_file, "r") as f:
            log_contents = f.read()
            self.assertIn(level, log_contents)
            self.assertIn(message, log_contents)
