import tempfile
import unittest
from pathlib import Path

from file_utils import FileUtils


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


if __name__ == "__main__":
    unittest.main()
