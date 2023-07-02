# Image Utils
import argparse
from dataclasses import dataclass
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from instacut.utils.image.models.blip2models import Blip2WithCustomGeneration
from PIL import Image
from transformers import Blip2Processor
from instacut.utils.file_utils import BaseConfig

from typing import Optional, Union

ENUM_ADDITION = " Please select one of the following options, or answer with NA if none apply: "
BOOL_ADDITION = " Please answer yes or no. "
LIST_STR_ADDITION = " If yes, please list all that apply, comma-separated (item1, item2, item3 ...). Otherwise, answer with NA."


class ImageUtilsConfig(BaseConfig):
    def __init__(
        self,
        # The device to use
        device: Optional[Union[int, str]] = 0,
        # The number of images to simultaneously process in the model
        batch_size: Optional[int] = 2**8,
        # The number of images to simultaneously preprocess for embedding caching
        preprocessing_batch_size: Optional[int] = 2**10,
        # The path to the image directory
        image_path: Optional[str] = None,
        # The path to the image prompts
        image_prompts: Optional[str] = None,
        # The processor to use
        processor: Optional[str] = "Salesforce/blip2-flan-t5-xl",
        # The model to use
        model: Optional[str] = "Salesforce/blip2-flan-t5-xl",
    ):
        if isinstance(device, str):
            device = int(device) if device.isdigit() else device
        self.device = device
        self.batch_size = int(batch_size)
        self.preprocessing_batch_size = int(preprocessing_batch_size)
        self.image_path = image_path
        if image_prompts is None:
            image_prompts = Path(__file__).parent / "image_prompts.txt"
        self.image_prompts = image_prompts
        self.processor = processor
        self.model = model


def main(config: ImageUtilsConfig):
    processor = Blip2Processor.from_pretrained(config.processor)
    device_map = {
        "query_tokens": config.device,
        "vision_model": config.device,
        "language_model": config.device,
        "language_projection": config.device,
        "qformer": config.device,
    }
    model = Blip2WithCustomGeneration.from_pretrained(
        config.model,
        device_map=device_map,
        torch_dtype=torch.float16,
    )

    # Start by assembling the input files
    image_path = Path(config.image_path)
    # Get all the jpg files in that directory using "*.jpg" glob
    files = [_ for _ in sorted(image_path.glob("*.jpg"))]

    batch_size = min(config.batch_size, len(files))
    # Here we need to process the images in preprocessing_batch_size chunks,
    # because otherwise we will run out of memory
    image = torch.tensor(
        np.stack([np.array(Image.open(i)) for i in files[:batch_size]]),
        dtype=torch.uint8,
    ).to(config.device)

    # Now get the prompts and compile them into complete questions
    with open(
        config.image_prompts,
        "r",
    ) as f:
        questions = [i for i in f.read().strip().split("\n") if i]
        questions = [i.split("\t") for i in questions[1:]]

    processed_inputs = []
    for _, _, topic, question, options, q_type in tqdm(
        questions, desc="Processing questions", unit="qs", leave=False
    ):
        # For each question, we need to add the appropriate addition to the
        # question, and then process it with the model
        parsed_question = question
        if q_type == "ENUM":
            parsed_question += ENUM_ADDITION
            parsed_question += options
        elif q_type == "BOOL":
            parsed_question += BOOL_ADDITION
        elif q_type == "LIST_STR":
            parsed_question += LIST_STR_ADDITION

        parsed_question = (
            "Please answer the following question:\n"
            + parsed_question
            + "\nYour answer: "
        )

        inputs = processor(
            images=image,
            text=[parsed_question] * batch_size,
            return_tensors="pt",
        ).to(config.device, torch.float16)

        processed_inputs.append((topic, inputs))

        # Start the model generation, and put the result in the model Queue
        predictions = model.generate(**inputs)
        generated_text = processor.batch_decode(
            predictions, skip_special_tokens=True
        )
        print(f"Q: {topic}, A: {generated_text}")

        # Constant image compute speed just with processor:
        #   1.54 s per 128 images = 83.1 images per second
        # Constant image compute speed with processor and model:
        #   2.91 s per 128 images = 43.9 images per second


if __name__ == "__main__":
    # Free all the cache that can be freed
    torch.cuda.empty_cache()

    # Use argparse to get the device and batch size
    parser = argparse.ArgumentParser()
    # Argument for the location of the config file.
    parser.add_argument(
        "-c",
        "--config",
        required=False,
        help="Config file location",
    )
    args = parser.parse_args()

    main(ImageUtilsConfig.from_file(args.config))
