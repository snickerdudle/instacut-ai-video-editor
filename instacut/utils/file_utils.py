import re
from dataclasses import dataclass
from pathlib import Path
from typing import Union

PathObj = Union[str, Path]


@dataclass
class FileUtils:
    output_path: PathObj

    def saveTextFile(self, filename: str, text: str) -> Path:
        full_path = Path(self.output_path) / filename
        if not full_path.parent.exists():
            full_path.parent.mkdir(parents=True, exist_ok=True)
        with open(full_path, "w") as f:
            f.write(text)
        return full_path

    @classmethod
    def loadUrlsFromFile(cls, path: PathObj) -> list:
        """Loads a list of URLs from a file.

        Args:
            path (PathObj): Path to the file.
        Returns:
            list: List of URLs.
        """
        with open(path, "r") as f:
            urls = f.read().strip().split("\n")
        return urls

    @classmethod
    def isUrl(cls, input_string):
        url_pattern = re.compile(
            r"^(https?://)?([a-zA-Z0-9]+(-[a-zA-Z0-9]+)*\.)+[a-zA-Z]{2,}(/.*)?$"
        )
        return re.match(url_pattern, input_string) is not None

    @classmethod
    def isFile(cls, input_string):
        file_pattern = re.compile(r"^.*\.\w+$")
        return re.match(file_pattern, input_string) is not None
