from pathlib import Path
from queue import Queue
from typing import Optional, List, Tuple

from instacut.modules.video_processor import (
    Frame,
    VideoProcessor,
    VideoProcessorConfig,
)
from instacut.utils.file_utils import BaseConfig
from instacut.utils.image.image_utils import ImageUtils, ImageUtilsConfig


class InstacutRunner:
    def __init__(
        self,
        config_path: str,
        in_queue: Optional[Queue] = None,
        out_queue: Optional[Queue] = None,
    ):
        self.config_path = Path(config_path).resolve()
        self.in_queue = in_queue or Queue()
        self.out_queue = out_queue or Queue()

    def run_all(self, input_url: str, config: Optional[dict] = None):
        if config is not None:
            config = BaseConfig.from_dict(config)
        else:
            config = Path(self.config_path).resolve()
            config = BaseConfig.from_file(config)

        # Perform the summarization
        self.run_summarization(input_url, config)

        # Perform the Sampling
        frames = self.run_sampling(input_url, config)

        # Perform the Question Answering
        self.run_qa(input_url, list(zip(*frames))[1], config)

    def run_summarization(
        self, input_url: str, config: Optional[dict] = None
    ) -> None:
        if config is not None:
            config = BaseConfig.from_dict(config)
        else:
            config = Path(self.config_path).resolve()
            config = BaseConfig.from_file(config)

        vp_config = VideoProcessorConfig.from_parent_config(config)
        vp = VideoProcessor(vp_config)

        # Perform the Summarization
        vp.summarize(video=input_url)
        self.out_queue.put(
            (
                input_url,
                "SUMMARY COMPLETE",
            )
        )

    def run_sampling(
        self, input_url: str, config: Optional[dict] = None
    ) -> List[Tuple[Optional[List[Frame]], str]]:
        if config is not None:
            config = BaseConfig.from_dict(config)
        else:
            config = Path(self.config_path).resolve()
            config = BaseConfig.from_file(config)

        vp_config = VideoProcessorConfig.from_parent_config(config)
        vp = VideoProcessor(vp_config)

        # Perform the Sampling. Returns a list of (frames, directory) tuples for
        # each video.
        frames = vp.sampleFrames(video=input_url)
        self.out_queue.put(
            (
                input_url,
                "SAMPLING COMPLETE",
            )
        )

        return frames

    def run_qa(
        self,
        input_url: str,
        frames_dirs: List[Frame],
        config: Optional[dict] = None,
    ) -> None:
        if config is not None:
            config = BaseConfig.from_dict(config)
        else:
            config = Path(self.config_path).resolve()
            config = BaseConfig.from_file(config)

        # Perform the Question Answering for every video
        for frame_dir in frames_dirs:
            iu_config = ImageUtilsConfig.from_parent_config(config)
            iu_config.image_path = frame_dir
            iu = ImageUtils(config=iu_config)
            iu.generateAnswers()
            self.out_queue.put(
                (
                    input_url,
                    "QA COMPLETE",
                )
            )
