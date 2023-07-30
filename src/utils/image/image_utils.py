# Image Utils
import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image
from transformers import Blip2Processor

from src.utils.file_utils import BaseConfig, FileUtils, Question
from src.utils.image.models.blip2models import Blip2WithCustomGeneration
from src.utils.io_utils import tqdm

ENUM_ADDITION = " Please select one of the following options, or answer with NA if none apply: "
BOOL_ADDITION = " Please answer yes or no. "
LIST_STR_ADDITION = " If yes, please list all that apply, comma-separated (item1, item2, item3 ...). Otherwise, answer with NA."
INT_ADDITION = " Please answer with a number. "

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


class ImageUtils:
    def __init__(self, config: ImageUtilsConfig):
        self.config = config

        # Initialize the processor and model
        self.processor = Blip2Processor.from_pretrained(config.processor)
        device_map = {
            "query_tokens": config.device,
            "vision_model": config.device,
            "language_model": config.device,
            "language_projection": config.device,
            "qformer": config.device,
        }
        self.model = Blip2WithCustomGeneration.from_pretrained(
            config.model,
            device_map=device_map,
            torch_dtype=torch.float16,
        )

    @torch.no_grad()
    def queryFromInputImage(
        self,
        files: List[PathObj],
        preprocessing_batch_size: int,
    ):
        """
        Preprocesses the images in the given list of files, then generates the query
        tokens for each image.

        Args:
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
            ).to(self.model.device)

            # Preprocess the images
            preprocessed_images = self.processor(
                images=image_batch,
                return_tensors="pt",
            ).to(self.model.device, torch.float16)

            # Get the image embeddings from the pixel_values
            image_embeds = self.model.vision_model(
                pixel_values=preprocessed_images.pixel_values,
                return_dict=True,
            ).last_hidden_state

            # Generate the query_outputs
            image_attention_mask = torch.ones(
                image_embeds.size()[:-1],
                dtype=torch.long,
                device=image_embeds.device,
            )
            query_tokens = self.model.query_tokens.expand(
                image_embeds.shape[0], -1, -1
            )
            query_outputs = self.model.qformer(
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

    def parseQuestions(
        self,
        custom_questions: Optional[List[Tuple[str]]] = None,
        config: Optional[ImageUtilsConfig] = None,
    ) -> List[Question]:
        """
        Parses the questions from the given config file or custom list.

        Args:
            custom_questions (Optional[List[Tuple[str]]], optional): The custom questions to use. Defaults to None.
            config (Optional[ImageUtilsConfig], optional): The config to use. Defaults to None.

        Returns:
            List[Tuple[Question, str]]: The list of questions, along with the parsed
                question string.
        """
        assembled_questions = []
        config = config or self.config

        if custom_questions is not None:
            questions = custom_questions
        else:
            with open(
                config.image_prompts,
                "r",
            ) as f:
                questions = [i for i in f.read().strip().split("\n") if i]
                questions = [i.split("\t") for i in questions[1:]]

        for question_tuple in questions:
            question = Question(*question_tuple)
            # For each question, we need to add the appropriate addition to the
            # question, and then process it with the model
            parsed_question = question.question
            if question.type == "ENUM":
                parsed_question += ENUM_ADDITION
                parsed_question += question.options
            elif question.type == "BOOL":
                parsed_question += BOOL_ADDITION
            elif question.type == "LIST_STR":
                parsed_question += LIST_STR_ADDITION
            elif question.type == "INT":
                parsed_question += INT_ADDITION

            parsed_question = (
                "Please answer the following question:\n"
                + parsed_question
                + "\nYour answer: "
            )
            assembled_questions.append(
                (
                    question,
                    parsed_question,
                )
            )
        return assembled_questions

    def generateAnswers(
        self, custom_questions: Optional[List[Tuple[str]]] = None
    ):
        """
        Generates the answers to the questions for the given image directory or custom
        list of questions.

        Args:
            custom_questions (Optional[List[Tuple[str]]], optional): The custom questions to use. Defaults to None.

        Returns:
            None
        """
        # Check if the question answers already exist
        if (
            Path(self.config.image_path) / "questions" / "questions.json"
        ).exists() and custom_questions is None:
            print(
                "Questions already answered. Skipping question answering step."
            )
            return
        ## INPUT FILES
        # Start by assembling the input files
        image_path = Path(self.config.image_path)
        # Get all the jpg files in that directory using "*.jpg" glob
        files = [_ for _ in sorted(image_path.glob("*.jpg"))]
        print(f"Found {len(files)} images in {image_path}.")

        ## BATCH SIZING
        # Get the correct batch size for processing the images. Too high will cause
        # memory errors, too low will cause the model to run slowly.
        if self.config.preprocessing_batch_size > 0:
            preprocessing_batch_size = min(
                self.config.preprocessing_batch_size, self.config.batch_size
            )
        else:
            preprocessing_batch_size = self.config.batch_size
        # Make sure we don't try to process more images than we have
        preprocessing_batch_size = min(preprocessing_batch_size, len(files))
        print(
            f"Using batch size {preprocessing_batch_size} for preprocessing."
        )

        ## IMAGE PREPROCESSING
        # Get the query outputs for all the images
        query_output = self.queryFromInputImage(
            files,
            preprocessing_batch_size,
        )

        ## QUESTION PROCESSING
        results_dict = defaultdict(list)
        results_list = []
        for question_obj, parsed_question in tqdm(
            self.parseQuestions(custom_questions=custom_questions),
            desc="Processing questions",
            unit="qs",
        ):
            # Get the batch size correctly configured
            batch_size = min(self.config.batch_size, query_output.shape[0])
            for batch_idx in tqdm(
                range(0, query_output.shape[0], batch_size),
                desc=f"Processing batches for q({question_obj.id})",
                unit="batch",
            ):
                cur_batch_size = min(
                    batch_size, query_output.shape[0] - batch_idx
                )
                # Get the text encoding for the query
                text_encoding = self.processor(
                    text=[parsed_question] * cur_batch_size,
                    return_tensors="pt",
                ).to(self.model.device)

                # Generate the embeddings
                predictions = self.model.generate(
                    query_output=query_output[
                        batch_idx : batch_idx + batch_size
                    ],
                    input_ids=text_encoding.input_ids,
                    attention_mask=text_encoding.attention_mask,
                    max_new_tokens=30,
                )
                # The generated text is a list of strings, one for each input frame.
                # Since every iteration asks only a single question (repeated BATCH
                # times), all of these returned strings are answering the same question.
                # We need to aggregate these so they can be dumped to a file later.
                generated_text = self.processor.batch_decode(
                    predictions, skip_special_tokens=True
                )
                results_dict[question_obj.id] += generated_text[:]
            results_list.append((question_obj, results_dict[question_obj.id]))

        # Save the results to a file
        FileUtils.saveFrameQuestions(
            full_path=Path(self.config.image_path)
            / "questions"
            / f"questions{'' if custom_questions is None else '_custom'}.json",
            questions=results_list,
        )


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

    # Initialize the image utils and generate answers
    image_utils = ImageUtils(ImageUtilsConfig.from_file(args.config))
    image_utils.generateAnswers()
