import argparse

from instacut.modules.video_processor import (
    VideoProcessor,
    VideoProcessorConfig,
)
from instacut.utils.file_utils import FileUtils
from instacut.utils.prompts import video_summarization_prompt_2 as prompt
from instacut.utils.prompts import (
    video_summarization_prompt_3 as timestamp_prompt,
)


def perform_summarization(cli_input):
    """Perform the summarization."""
    print("Performing summarization...")
    # Create a video processor.
    vp_config = VideoProcessorConfig(
        sampling_policy="e10",
        output_dir=r"./data/",
        prompt=timestamp_prompt if args.use_timestamps else prompt,
    )
    vp = VideoProcessor(vp_config)

    # We're working with links, perform the summarization.
    vp.summarize(
        cli_input,
        model=args.model,
        use_timestamps=args.use_timestamps,
        save_transcript=True,
    )


def sample_frames(cli_input, save_frames=False):
    """Sample frames from the video."""
    print("Sampling frames...")
    # Create a video processor.
    vp_config = VideoProcessorConfig(
        sampling_policy="u8",
        output_dir=r"./data/",
        prompt=timestamp_prompt if args.use_timestamps else prompt,
    )
    vp = VideoProcessor(vp_config)

    # We're working with links, perform the summarization.
    return vp.sampleFrames(
        cli_input,
        save_frames=save_frames,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Argument for the input text or file path.
    parser.add_argument("cli_input", nargs="*", help="Input text or file path")
    # Argument for selecting the model to use.
    parser.add_argument(
        "-m",
        "--model",
        nargs="?",
        help="Model to use",
        default="gpt-3.5-turbo",
    )
    # Argument for including the timestamps in the output.
    parser.add_argument(
        "-t",
        "--use_timestamps",
        action="store_true",
        help="Include timestamps in the output",
    )
    args = parser.parse_args()

    cli_input = args.cli_input
    while not cli_input:
        # No arguments were passed to the script.
        cli_input = input("Enter text or file path: ")
    if not isinstance(cli_input, list):
        cli_input = [cli_input]
    if not FileUtils.isUrl(cli_input[0]):
        # We're working with a file, load the URLs from the file.
        cli_input = FileUtils.loadUrlsFromFile(cli_input[0])

    print(
        f"""---
Running the main function. Inputs:
    Model: {args.model}
    Use timestamps: {args.use_timestamps}
    Input: {', '.join(cli_input)}\n---"""
    )

    # perform_summarization(cli_input=cli_input)
    frames = sample_frames(cli_input=cli_input[0], save_frames=True)
