import json
import pickle
import re
from collections import namedtuple
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

PathObj = Union[str, Path]
Question = namedtuple(
    "Question",
    ["id", "version", "topic", "question", "options", "type"],
)


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
    def getVideoFramesOutputDir(
        cls,
        video: Any,
        sampling_policy: Optional[Any] = None,
    ) -> Path:
        """Gets the directory for the video frames.

        Args:
            video (Video): The video.
            sampling_policy (Optional[SamplingPolicyBaseClass], optional): The
                sampling policy. If supplied, returns the path to the frames
                directory for the sampling policy. If not supplied, returns the
                path to the frames directory for the video.

        Returns:
            Path: The directory for the video frames.
        """
        # Get the video dir
        video_dir = cls.getVideoOutputDir(video) / "frames"
        if sampling_policy is not None:
            video_dir = video_dir / sampling_policy.name
        return video_dir

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
        frames_dir = cls.getVideoFramesOutputDir(video, sampling_policy)
        frames_dir.mkdir(parents=True, exist_ok=True)
        for frame in frames:
            frame.saveToDir(frames_dir)
        return frames_dir

    @classmethod
    def saveFrameQuestions(
        cls, full_path: PathObj, questions: List[Tuple[namedtuple, List[str]]]
    ) -> Path:
        """Saves the frame questions as a JSON or pickle file.

        Args:
            full_path (PathObj): The full path to the file. Can be a JSON or
                pickle file.
            questions (List[Tuple[namedtuple, List[str]]]): The questions and
            their answers.

        Returns:
            Path: The path to the saved questions.
        """
        # The questions are a list of (namedtuple, answers) tuples.
        # The namedtuple is a dataclass, so we can't save it as JSON. We need
        # to convert it to a list first.
        questions_serialized = [(list(q), answers) for q, answers in questions]

        full_path = Path(full_path)
        full_path.parent.mkdir(parents=True, exist_ok=True)
        save_as_pickle = full_path.suffix != ".json"
        if save_as_pickle:
            with open(full_path, "wb") as f:
                pickle.dump(questions_serialized, f)
        else:
            with open(full_path, "w") as f:
                json.dump(questions_serialized, f, indent=4)

    @classmethod
    def loadFrameQuestions(
        cls, full_path: PathObj
    ) -> List[Tuple[namedtuple, List[str]]]:
        """Loads the frame questions from a JSON or pickle file.

        Args:
            full_path (PathObj): The full path to the file.
            load_as_pickle (bool, optional): Whether to load as a pickle file.

        Returns:
            List[Tuple[namedtuple, List[str]]]: The questions and their answers.
        """
        full_path = Path(full_path)
        load_as_pickle = full_path.suffix != ".json"
        if load_as_pickle:
            with open(full_path, "rb") as f:
                questions_serialized = pickle.load(f)
        else:
            with open(full_path, "r") as f:
                questions_serialized = json.load(f)

        # The questions are a list of (namedtuple, answers) tuples.
        # The namedtuple is a dataclass, so we can't save it as JSON. We need
        # to convert it to a list first.
        questions = []
        for q, answers in questions_serialized:
            # Convert the list back to a namedtuple
            q = Question(*q)
            questions.append((q, answers))
        return questions

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


class BaseConfig:
    def __init__(self, **kwargs):
        self.subconfigs = kwargs

    def __getitem__(self, key):
        return self.subconfigs[key]

    def __contains__(self, key):
        return key in self.subconfigs

    @classmethod
    def from_file(cls, path: PathObj):
        """Loads a config from a file.

        Args:
            path (PathObj): Path to the file.

        Returns:
            BaseConfig: The config.
        """
        if path is None:
            print("No config file provided. Using default config.")
            return cls()
        if not Path(path).exists():
            raise ValueError(f"Config file {path} does not exist.")

        with open(path, "r") as f:
            config = json.load(f)

        return cls.from_dict(config)

    @classmethod
    def from_dict(cls, config_dict: dict):
        """Loads a config from a dict.

        Args:
            config_dict (dict): The config dict.
        Returns:
            BaseConfig: The config.
        """
        # The different portions of the config are stored in different keys. The
        # key for the config is the class name. Retrieve the key for the config
        # from the class name.
        config_key = cls.__name__
        config = config_dict.get(config_key)

        if config is None:
            raise ValueError(
                f"Config dict does not contain a config for {config_key}."
            )
        return cls(**config)

    @classmethod
    def from_parent_config(
        cls,
        parent_config: "BaseConfig",
        create_empty_as_fallback=True,
        **kwargs,
    ):
        """Creates a config from a parent config.

        Args:
            parent_config (BaseConfig): The parent config.
            create_empty_as_fallback (bool, optional): Whether to create an
                empty config if the parent config does not have a config for
                this class. Defaults to True.
            **kwargs: The config kwargs.

        Returns:
            BaseConfig: The config.
        """
        # Create a new config from the parent config
        config_dict = parent_config.subconfigs
        class_name = cls.__name__
        if (config_dict is None) or (class_name not in config_dict):
            if create_empty_as_fallback:
                # The parent config does not have a config for this class
                config = cls(**kwargs)
                return config
            else:
                raise ValueError(
                    f"Parent config does not have a config for {class_name}."
                )

        config_dict[class_name].update(**kwargs)
        config = cls.from_dict(config_dict)
        return config

    def __repr__(self):
        # Get all the vars to be displayed
        vars = self.__dict__.copy()
        # Remove the subconfigs
        vars.pop("subconfigs", None)
        # Convert to a string
        s = ", ".join([f"{k}={v}" for k, v in vars.items()])
        return f"{self.__class__.__name__}({s})"
