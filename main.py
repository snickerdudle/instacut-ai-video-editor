import argparse

from src.runner import InstacutRunner

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

    input_url = "https://www.youtube.com/watch?v=e-P5IFTqB98&t=140s"

    # Run the InstacutRunner
    instacut_runner.run_all(input_url=input_url)

    # ID, Version, Topic, Question, Options, Type
    custom_questions = [
        (
            "101",
            "0",
            "Yellow submarines",
            "How many yellow submarines are there in this frame?",
            "",
            "INT",
        ),
        (
            "102",
            "0",
            "Yellow submarines color",
            "If you saw a Yellow Submarine underwater, what color would it be?",
            "",
            "STR",
        ),
    ]

    instacut_runner.run_qa(
        input_url=input_url,
        custom_questions=custom_questions,
    )
