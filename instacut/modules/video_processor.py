"""Contains the video download, frame splitting and sampling logic."""

import pdb
import re
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any, List, Optional, Union, get_args

from pytube import YouTube
from tqdm import tqdm
from youtube_transcript_api import YouTubeTranscriptApi

from instacut.utils.file_utils import FileUtils
from instacut.utils.llm_utils import OpenAIChat
from instacut.utils.prompts import Prompt

PathObj = Union[str, Path]
VideoObj = Union["Video", str]


class Frame:
    """Class for a frame."""

    def __init__(
        self,
        image: Any,
        parent: Optional[Any] = None,
        timestamp: Optional[float] = None,
        order: Optional[int] = None,
    ):
        self.image = image
        self.parent = parent
        self.timestamp = timestamp
        self.order = order

    def save(self, path: PathObj) -> None:
        """Save the frame to a file."""
        path = Path(path)
        if not path.parent.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
        self.image.save(path)


@dataclass
class VideoMetadata:
    """Class for the metadata of a video."""

    name: str
    length: float  # in seconds
    url: str
    framerate: Optional[int] = None
    resolution: Optional[int] = None
    thumbnail_url: Optional[str] = None


@dataclass
class Video:
    """Class for a video."""

    metadata: VideoMetadata
    output_path: PathObj
    frames: List[Frame]

    @classmethod
    def from_file(cls, path: PathObj) -> "Video":
        """Create a video from a file."""
        raise NotImplementedError

    @classmethod
    def from_url(
        cls, url: str, output_path: Optional[PathObj] = None
    ) -> "Video":
        """Create a video from a URL."""
        yt = YouTube(url)
        metadata = VideoMetadata(name=yt.title, length=yt.length, url=url)
        return cls(metadata=metadata, output_path=Path(output_path), frames=[])
        raise NotImplementedError

    @cached_property
    def unix_name(self) -> str:
        """Get the name of the video."""
        return re.sub(r"[^a-zA-Z0-9]+", "_", self.metadata.name)

    @classmethod
    def convertAllToVideo(
        cls,
        video: Union[VideoObj, List[VideoObj]],
        output_path: Optional[PathObj] = None,
    ) -> List["Video"]:
        """Make sure that all inputs are Video objects."""
        if not isinstance(video, list):
            return [video]
        return_list = []
        for v in video:
            if isinstance(v, str):
                # The current item is a URL.
                return_list.append(cls.from_url(v, output_path=output_path))
            else:
                # The current item is a Video object.
                return_list.append(v)
        return return_list

    def save(self, path: Optional[PathObj]) -> None:
        """Save the video to a file."""
        raise NotImplementedError

    def sample(
        self, sampling_policy: "SamplingPolicyBaseClass"
    ) -> List[Frame]:
        """Sample the frames of the video."""
        return sampling_policy.sample(self)

    def getTranscript(self, include_time=False) -> str:
        """Get the transcript of the video."""
        url = self.metadata.url
        url = re.sub(r".*/watch\?v=", "", url)
        transcript = YouTubeTranscriptApi.get_transcript(url)
        if include_time:
            output_list = []
            for i in transcript:
                output_list.append(
                    "[{}-{}] {}".format(
                        round(i["start"], 2),
                        round(i["start"] + i["duration"], 2),
                        i["text"],
                    )
                )
        else:
            # Convert the transcript to a string without any of the timestamps.
            output_list = [i["text"] for i in transcript]

        transcript = "\n".join(output_list)
        return transcript


@dataclass
class SamplingPolicyBaseClass:
    """Class for sampling the frames of a video."""

    name: str

    def sample(video: Video) -> List[Frame]:
        """Sample the frames of a video."""
        raise NotImplementedError


@dataclass
class VideoProcessorConfig:
    """Class for the configuration of a video processing job."""

    sampling_policy: Union[SamplingPolicyBaseClass, str]
    output_path: PathObj
    prompt: str = Union[Prompt, str]


class VideoProcessor:
    """Class for processing a video or a set of videos."""

    def __init__(
        self,
        config: VideoProcessorConfig,
    ) -> None:
        self.config = config
        self.file_processor = FileUtils(output_path=self.config.output_path)

    def summarize(
        self, video: Union[VideoObj, List[VideoObj]], **kwargs
    ) -> int:
        """Summarize the video(s).

        Args:
            video (Union[VideoObj, List[VideoObj]]): The video(s) to summarize.

        Returns:
            int: The number of videos summarized.
        """
        if not isinstance(video, list):
            video = [video]
        videos = Video.convertAllToVideo(
            video, output_path=Path(self.config.output_path)
        )
        for video in tqdm(
            videos, desc="Summarizing videos", unit="video", leave=False
        ):
            name = video.unix_name
            # Check if exists already to not burn through API calls.
            if (Path(self.config.output_path) / (name + ".txt")).exists():
                print('File "{}" already exists. Skipping'.format(name))
                continue
            transcript = video.getTranscript()
            chat = OpenAIChat(self.config.prompt, **kwargs)
            summary = chat.processMessage(transcript)
            self.file_processor.saveTextFile(name + ".txt", summary)
