import argparse

from instacut.modules.video_processor import (
    VideoProcessor,
    VideoProcessorConfig,
)
from instacut.utils.file_utils import FileUtils
from instacut.utils.prompts import video_summarization_prompt_2

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("cli_input", nargs="*", help="Input text or file path")
    parser.add_argument(
        "-m",
        "--model",
        nargs="?",
        help="Model to use",
        default="gpt-3.5-turbo",
    )
    args = parser.parse_args()

    cli_input = args.cli_input

    while not cli_input:
        # No arguments were passed to the script.
        cli_input = input("Enter text or file path: ")

    if not isinstance(cli_input, list):
        cli_input = [cli_input]

    vp_config = VideoProcessorConfig(
        sampling_policy="e10",
        output_path=r"./processed/",
        prompt=video_summarization_prompt_2,
    )
    vp = VideoProcessor(vp_config)

    if FileUtils.isUrl(cli_input[0]):
        # We're working with links, perform the summarization.
        vp.summarize(cli_input, model=args.model)

    else:
        # We're working with a file, load the URLs from the file.
        urls = FileUtils.loadUrlsFromFile(cli_input)
        vp.summarize(urls, model=args.model)
