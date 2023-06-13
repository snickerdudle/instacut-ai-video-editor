from dataclasses import dataclass
from pathlib import Path
from typing import Union


PathObj = Union[str, Path]


@dataclass
class FileUtils:
    output_path: PathObj

    def saveTextFile(self, filename: str, text: str) -> None:
        full_path = Path(self.output_path) / filename
        if not full_path.parent.exists():
            full_path.parent.mkdir(parents=True, exist_ok=True)
        with open(full_path, "w") as f:
            f.write(text)
        return full_path
