import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Union

PathObj = Union[str, Path]


@dataclass
class FileUtils:
    output_dir: PathObj

    @classmethod
    def getVideoOutputDir(cls, video: Any) -> Path:
        """Gets the directory for the video.

        Args:
            video (Video): The video.

        Returns:
            Path: The directory for the video.
        """
        return Path(video.output_dir)

    @classmethod
    def getVideoFileOutputDir(self, video: Any) -> Path:
        """Gets the directory for the video.

        Args:
            video (Video): The video.

        Returns:
            Path: The directory for the video.
        """
        # Get the video dir
        video_dir = self.getVideoOutputDir(video)
        return video_dir / "video"

    @classmethod
    def getVideoTranscriptOutputDir(cls, video: Any) -> Path:
        """Gets the directory for the video transcript.

        Args:
            video (Video): The video.

        Returns:
            Path: The directory for the video transcript.
        """
        # Get the video dir
        video_dir = cls.getVideoOutputDir(video)
        return video_dir / "transcript"

    @classmethod
    def getVideoSummaryOutputDir(cls, video: Any) -> Path:
        """Gets the directory for the video transcript.

        Args:
            video (Video): The video.

        Returns:
            Path: The directory for the video summary.
        """
        # Get the video dir
        video_dir = cls.getVideoOutputDir(video)
        return video_dir / "summary"

    @classmethod
    def getVideoFramesOutputDir(cls, video: Any) -> Path:
        """Gets the directory for the video frames.

        Args:
            video (Video): The video.

        Returns:
            Path: The directory for the video frames.
        """
        # Get the video dir
        video_dir = cls.getVideoOutputDir(video)
        return video_dir / "frames"

    @classmethod
    def saveTextFile(cls, full_path: PathObj, text: str) -> Path:
        """Saves a text file.

        Args:
            full_path (PathObj): The full path to the file.
            text (str): The text to save.

        Returns:
            Path: The path to the saved file.
        """
        full_path = Path(full_path)
        full_path.parent.mkdir(parents=True, exist_ok=True)
        with open(full_path, "w") as f:
            f.write(text)
        return full_path

    @classmethod
    def saveVideoTranscript(cls, video: Any, transcript: str) -> Path:
        """Saves the video transcript.

        Args:
            video (Video): The video.
            transcript (str): The transcript.

        Returns:
            Path: The path to the saved transcript.
        """
        # Get the video dir
        transcript_dir = cls.getVideoTranscriptOutputDir(video)
        return cls.saveTextFile(transcript_dir / "transcript.txt", transcript)

    @classmethod
    def saveVideoSummary(cls, video: Any, summary: str) -> Path:
        """Saves the video summary.

        Args:
            video (Video): The video.
            summary (str): The summary.

        Returns:
            Path: The path to the saved summary.
        """
        # Get the video dir
        summary_dir = cls.getVideoSummaryOutputDir(video)
        return cls.saveTextFile(summary_dir / "summary.txt", summary)

    @classmethod
    def saveVideoFrames(
        cls, video: Any, sampling_policy: Any, frames: List[Any]
    ) -> Path:
        """Saves the video frames.

        Args:
            video (Video): The video.
            sampling_policy (SamplingPolicy): The sampling policy.
            frames (list): The frames.

        Returns:
            Path: The path to the saved frames.
        """
        # Get the video dir
        frames_dir = cls.getVideoFramesOutputDir(video)
        frames_dir = frames_dir / sampling_policy.name
        frames_dir.mkdir(parents=True, exist_ok=True)
        for frame in frames:
            frame.saveToDir(frames_dir)
        return frames_dir

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
