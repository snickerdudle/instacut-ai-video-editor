import unittest
from instacut.modules.transcript_processor import TranscriptProcessor, secondsToMinutes


class TestTranscriptProcessor(unittest.TestCase):
    def setUp(self):
        self.transcript = [
            {"start": 0, "duration": 10, "text": "Hello"},
            {"start": 10, "duration": 5, "text": "world"},
            {"start": 15, "duration": 20, "text": "This is a test"},
            {"start": 35, "duration": 15, "text": "Goodbye"},
        ]
        self.processor = TranscriptProcessor(self.transcript)

    def test_convertDictToIntervals(self):
        expected_output = {
            (0, 10): "Hello",
            (10, 15): "world",
            (15, 35): "This is a test",
            (35, 50): "Goodbye",
        }
        self.assertEqual(self.processor.interval_dict, expected_output)

    def test_joinIntervals(self):
        input_intervals = [
            {"start": 0, "duration": 10, "text": "Hello"},
            {"start": 10, "duration": 5, "text": "world"},
            {"start": 15, "duration": 20, "text": "This is a test"},
            {"start": 35, "duration": 15, "text": "Goodbye"},
        ]
        expected_output = {
            "start": 0,
            "duration": 50,
            "text": "Hello world This is a test Goodbye",
        }
        self.assertEqual(self.processor.joinIntervals(input_intervals), expected_output)

    def test_trimIntervalsToSLong(self):
        expected_output = [
            {"start": 0, "duration": 15, "text": "Hello world"},
            {"start": 15, "duration": 35, "text": "This is a test Goodbye"},
        ]
        self.assertEqual(self.processor.trimIntervalsToSLong(20), expected_output)

    def test_seconds_to_minutes(self):
        # Test that 60 seconds is 1 minute.
        self.assertEqual(secondsToMinutes(60), "1:00")
        # Test that 61 seconds is 1 minute and 1 second.
        self.assertEqual(secondsToMinutes(61), "1:01")
        # Test that 120 seconds is 2 minutes.
        self.assertEqual(secondsToMinutes(120), "2:00")
        # Test that 121 seconds is 2 minutes and 1 second.
        self.assertEqual(secondsToMinutes(121), "2:01")
        # Test that 3600 seconds is 1 hour.
        self.assertEqual(secondsToMinutes(3600), "1:00:00")
        # Test that 3601 seconds is 1 hour and 1 second.
        self.assertEqual(secondsToMinutes(3601), "1:00:01")
        # Test that 3660 seconds is 1 hour and 1 minute.
        self.assertEqual(secondsToMinutes(3660), "1:01:00")
        # Test that 3661 seconds is 1 hour, 1 minute, and 1 second.
        self.assertEqual(secondsToMinutes(3661), "1:01:01")
