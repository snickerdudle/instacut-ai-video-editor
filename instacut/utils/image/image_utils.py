# Image Utils
import argparse
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Union

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import Blip2Processor

from instacut.utils.file_utils import BaseConfig
from instacut.utils.image.models.blip2models import Blip2WithCustomGeneration

ENUM_ADDITION = " Please select one of the following options, or answer with NA if none apply: "
BOOL_ADDITION = " Please answer yes or no. "
LIST_STR_ADDITION = " If yes, please list all that apply, comma-separated (item1, item2, item3 ...). Otherwise, answer with NA."

PathObj = Union[str, Path]


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


@torch.no_grad()
def preprocess_images(
    processor: Any,
    model: Any,
    files: List[PathObj],
    preprocessing_batch_size: int,
):
    """
    Preprocesses the images in the given list of files.

    Args:
        processor (Any): The processor to use
        model (Any): The model to use
        files (List[PathObj]): The list of files to preprocess
        preprocessing_batch_size (int): The number of images to simultaneously preprocess

    Returns:
        query_output (torch.FloatTensor): The preprocessed set of query tokens
    """
    print("Preprocessing images...")
    # Here we need to process the images in preprocessing_batch_size chunks,
    # because otherwise we will run out of memory
    output_list = []
    for image_start_idx in tqdm(
        range(0, len(files), preprocessing_batch_size),
        desc="Preprocessing mini-batches",
        unit="mini-batch",
        leave=False,
        ncols=80,
    ):
        # Get the image batch. The image batch is a tensor of shape
        # (preprocessing_batch_size, 3, 224, 224)
        image_batch = torch.tensor(
            np.stack(
                [
                    np.array(Image.open(i))
                    for i in files[
                        image_start_idx : image_start_idx
                        + preprocessing_batch_size
                    ]
                ]
            ),
            dtype=torch.uint8,
        ).to(model.device)
        cur_batch_size = image_batch.shape[0]

        # Preprocess the images
        preprocessed_images = processor(
            images=image_batch,
            return_tensors="pt",
        ).to(model.device, torch.float16)

        # Get the image embeddings from the pixel_values
        image_embeds = model.vision_model(
            pixel_values=preprocessed_images.pixel_values,
            return_dict=True,
        ).last_hidden_state

        # Generate the query_outputs
        image_attention_mask = torch.ones(
            image_embeds.size()[:-1],
            dtype=torch.long,
            device=image_embeds.device,
        )
        query_tokens = model.query_tokens.expand(image_embeds.shape[0], -1, -1)
        query_outputs = model.qformer(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_attention_mask,
            return_dict=True,
        )
        query_output = query_outputs.last_hidden_state

        # Add the query_output to the output_list
        output_list.append(query_output)

    # Concatenate the output_list
    query_output = torch.cat(output_list, dim=0)
    return query_output


@torch.no_grad()
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

    ## INPUT FILES
    # Start by assembling the input files
    image_path = Path(config.image_path)
    # Get all the jpg files in that directory using "*.jpg" glob
    files = [_ for _ in sorted(image_path.glob("*.jpg"))]
    print(f"Found {len(files)} images in {image_path}.")

    ## BATCH SIZING
    # Get the correct batch size for processing the images. Too high will cause
    # memory errors, too low will cause the model to run slowly.
    if config.preprocessing_batch_size > 0:
        preprocessing_batch_size = min(
            config.preprocessing_batch_size, config.batch_size
        )
    else:
        preprocessing_batch_size = config.batch_size
    # Make sure we don't try to process more images than we have
    preprocessing_batch_size = min(preprocessing_batch_size, len(files))
    print(f"Using batch size {preprocessing_batch_size} for preprocessing.")

    ## PROMPTS
    # Now get the prompts and compile them into complete questions
    with open(
        config.image_prompts,
        "r",
    ) as f:
        questions = [i for i in f.read().strip().split("\n") if i]
        questions = [i.split("\t") for i in questions[1:]]

    ## IMAGE PREPROCESSING
    # Get the query outputs for all the images
    query_output = preprocess_images(
        processor,
        model,
        files,
        preprocessing_batch_size,
    )

    results_dict = defaultdict(list)
    for q_id, _, topic, question, options, q_type in tqdm(
        questions,
        desc="Processing questions",
        unit="qs",
        leave=False,
        ncols=80,
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

        # Get the batch size correctly configured
        batch_size = min(config.batch_size, query_output.shape[0])
        for batch_idx in tqdm(
            range(0, query_output.shape[0], batch_size),
            desc=f"Processing batches for q({q_id})",
            unit="batch",
            leave=False,
            ncols=80,
        ):
            cur_batch_size = min(batch_size, query_output.shape[0] - batch_idx)
            # Get the text encoding for the query
            text_encoding = processor(
                text=[parsed_question] * cur_batch_size,
                return_tensors="pt",
            ).to(model.device)

            # Generate the embeddings
            predictions = model.generate(
                query_output=query_output[batch_idx : batch_idx + batch_size],
                input_ids=text_encoding.input_ids,
                attention_mask=text_encoding.attention_mask,
                max_new_tokens=30,
            )
            # The generated text is a list of strings, one for each input frame.
            # Since every iteration asks only a single question (repeated BATCH
            # times), all of these returned strings are answering the same question.
            # We need to aggregate these so they can be dumped to a file later.
            generated_text = processor.batch_decode(
                predictions, skip_special_tokens=True
            )
            results_dict[q_id] += generated_text[:]
            # del text_encoding, predictions, generated_text

    print(len(results_dict))
    # Constant image compute speed just with processor:
    #   1.54 s per 128 images = 83.1 images per second
    # Constant image compute speed with processor and model:
    #   2.91 s per 128 images = 43.9 images per second

    # Constant question compute speed with processor and model (with preprocessing):
    #   2.2 s per 545 images = 247.7 images per second


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
