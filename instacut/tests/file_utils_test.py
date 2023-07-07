from dataclasses import dataclass
import tempfile
from typing import Any
import unittest
from unittest.mock import patch, MagicMock

from pathlib import Path

from instacut.utils.file_utils import FileUtils


class TestUrlAndFile(unittest.TestCase):
    def test_saveTextFile(self):
        # Create a temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir = Path(temp_dir)
            file_utils = FileUtils(output_dir=temp_dir)

            # Definethe filename and text for the dummy file
            filename = "dummy_file.txt"
            text = "This is a dummy file."

            # Call the saveTextFile method
            saved_file_path = file_utils.saveTextFile(
                temp_dir / filename, text
            )

            # Assert that the file was  created and contains the correct text
            self.assertTrue(saved_file_path.exists())
            with open(saved_file_path, "r") as f:
                saved_text = f.read()
            self.assertEqual(saved_text, text)

    def test_isUrl(self):
        # Valid URLs
        self.assertTrue(FileUtils.isUrl("http://www.example.com"))
        self.assertTrue(FileUtils.isUrl("https://www.example.com"))

        self.assertTrue(FileUtils.isUrl("https://example.com/path"))

        # Invalid URLs
        self.assertTrue(FileUtils.isUrl("example.com"))
        self.assertTrue(FileUtils.isUrl("www.example.com"))
        self.assertFalse(FileUtils.isUrl("http:/www.example.com"))
        self.assertFalse(FileUtils.isUrl("http//www.example.com"))
        self.assertFalse(FileUtils.isUrl("http://example"))
        self.assertFalse(FileUtils.isUrl("http://example."))
        self.assertFalse(FileUtils.isUrl("http://256.256.256.256"))
        self.assertFalse(FileUtils.isUrl("http://localhost"))
        self.assertFalse(FileUtils.isUrl("ftp://ftp.example.com"))

    def test_isFile(self):
        # Valid file paths
        self.assertTrue(FileUtils.isFile("path/to/file.txt"))
        self.assertTrue(FileUtils.isFile("file.py"))
        self.assertTrue(FileUtils.isFile("/absolute/path/file.txt"))
        self.assertTrue(FileUtils.isFile("../relative/path/file.txt"))

        # Invalid file paths
        self.assertFalse(FileUtils.isFile("path/to/directory/"))
        self.assertFalse(FileUtils.isFile("file"))
        self.assertFalse(FileUtils.isFile("file."))
        self.assertFalse(FileUtils.isFile("/path/to/file/"))
        self.assertFalse(FileUtils.isFile("file.txt/"))


@dataclass
class Video:
    output_dir: Any


@dataclass
class SamplingPolicy:
    name: str


@dataclass
class Frame:
    def saveToDir(self, path):
        with open(path / "frame.jpg", "w") as f:
            f.write("frame")


class TestFileUtils(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory for testing
        self.test_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        # Cleanup the temporary directory
        self.test_dir.cleanup()

    def test_getVideoOutputDir(self):
        video = Video(output_dir=self.test_dir.name)
        output_dir = FileUtils.getVideoOutputDir(video)
        self.assertEqual(output_dir, Path(self.test_dir.name))

    def test_getVideoFileOutputDir(self):
        video = Video(output_dir=self.test_dir.name)
        expected_output_dir = Path(self.test_dir.name) / "video"
        output_dir = FileUtils.getVideoFileOutputDir(video)
        self.assertEqual(output_dir, expected_output_dir)

    def test_getVideoTranscriptOutputDir(self):
        video = Video(output_dir=self.test_dir.name)
        expected_output_dir = Path(self.test_dir.name) / "transcript"
        output_dir = FileUtils.getVideoTranscriptOutputDir(video)
        self.assertEqual(output_dir, expected_output_dir)

    def test_getVideoSummaryOutputDir(self):
        video = Video(output_dir=self.test_dir.name)
        expected_output_dir = Path(self.test_dir.name) / "summary"
        output_dir = FileUtils.getVideoSummaryOutputDir(video)
        self.assertEqual(output_dir, expected_output_dir)

    def test_getVideoFramesOutputDir(self):
        video = Video(output_dir=self.test_dir.name)
        expected_output_dir = Path(self.test_dir.name) / "frames"
        output_dir = FileUtils.getVideoFramesOutputDir(video)
        self.assertEqual(output_dir, expected_output_dir)

    def test_saveTextFile(self):
        file_path = Path(self.test_dir.name) / "test.txt"
        text = "This is a test"
        saved_path = FileUtils.saveTextFile(file_path, text)
        self.assertEqual(saved_path, file_path)
        with open(file_path, "r") as f:
            self.assertEqual(f.read(), text)

    def test_saveVideoTranscript(self):
        video = Video(output_dir=self.test_dir.name)
        transcript = "This is a transcript"
        saved_path = FileUtils.saveVideoTranscript(video, transcript)
        expected_path = Path(self.test_dir.name) / "transcript/transcript.txt"
        self.assertEqual(saved_path, expected_path)
        with open(saved_path, "r") as f:
            self.assertEqual(f.read(), transcript)

    def test_saveVideoSummary(self):
        video = Video(output_dir=self.test_dir.name)
        summary = "This is a summary"
        saved_path = FileUtils.saveVideoSummary(video, summary)
        expected_path = Path(self.test_dir.name) / "summary/summary.txt"
        self.assertEqual(saved_path, expected_path)
        with open(saved_path, "r") as f:
            self.assertEqual(f.read(), summary)

    def test_saveVideoFrames(self):
        video = Video(output_dir=self.test_dir.name)
        sampling_policy = SamplingPolicy(name="test_policy")
        frames = [Frame()]
        saved_path = FileUtils.saveVideoFrames(video, sampling_policy, frames)
        expected_path = Path(self.test_dir.name) / "frames/test_policy"
        self.assertEqual(saved_path, expected_path)
        frame_path = expected_path / "frame.jpg"
        with open(frame_path, "r") as f:
            self.assertEqual(f.read(), "frame")


if __name__ == "__main__":
    unittest.main()
