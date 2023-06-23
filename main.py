from video_processor import VideoProcessor, VideoProcessorConfig
from prompts import video_summarization_prompt
import argparse
from file_utils import FileUtils

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("cli_input", nargs="*", help="Input text or file path")
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
        prompt=video_summarization_prompt,
    )
    vp = VideoProcessor(vp_config)

    if FileUtils.isUrl(cli_input[0]):
        # We're working with links, perform the summarization.
        vp.summarize(cli_input)

    else:
        # We're working with a file, load the URLs from the file.
        urls = FileUtils.loadUrlsFromFile(cli_input)
        vp.summarize(urls)
