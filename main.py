import argparse

from instacut.runner import InstacutRunner

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Argument for the input config.
    parser.add_argument(
        "-c",
        "--config",
        nargs="?",
        help="Config to use. Defaults to 'instacut_config.json'",
        default="instacut_config.json",
    )
    args = parser.parse_args()

    config_file = args.config
    instacut_runner = InstacutRunner(config_file)

    # Run the InstacutRunner
    instacut_runner.run_all(
        input_url="https://www.youtube.com/watch?v=e3RRycSmd5A"
    )
