import json
import os
import random
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import torch
from PIL import Image

from instacut.utils.image.image_utils import ImageUtils, ImageUtilsConfig


class ImageUtilsTest(unittest.TestCase):
    @staticmethod
    def create_random_images(
        num_images: int,
        height: int = 720,
        width: int = 1280,
        temp_dir: str = None,
    ):
        temp_dir = temp_dir or tempfile.mkdtemp()

        for i in range(num_images):
            image_array = np.random.randint(
                0, 256, (height, width, 3), dtype=np.uint8
            )
            image = Image.fromarray(image_array)
            image.save(os.path.join(temp_dir, f"image_{i}.jpg"))

        return temp_dir

    @staticmethod
    def create_questions(
        num_questions: int = 3,
        temp_dir: str = None,
    ):
        temp_dir = temp_dir or tempfile.mkdtemp()

        questions = [
            "ID\tVersion\tTopic\tQuestion\tOptions\tType",
        ]
        for i in range(num_questions):
            cur_question = []
            # ID
            cur_question.append(str(i))
            # Version
            cur_question.append("0")
            # Topic
            cur_question.append(str(random.randint(0, 100)))
            # Question
            cur_question.append(
                f"What color do you get if you mix yellow and blue?"
            )
            # Options
            cur_question.append("[red, green, purple, orange]")
            # Type
            cur_question.append("ENUM")

            questions.append("\t".join(cur_question))

        os.makedirs(os.path.join(temp_dir, "questions"), exist_ok=True)
        with open(
            os.path.join(temp_dir, "questions", "questions.tsv"), "w"
        ) as f:
            f.write("\n".join(questions))

        return temp_dir

    @staticmethod
    def create_image_utils(tempdir: str, device="cpu"):
        config = ImageUtilsConfig(
            device=device,
            batch_size=4,
            preprocessing_batch_size=8,
            image_path=tempdir,
            image_prompts=os.path.join(tempdir, "questions", "questions.tsv"),
            processor="Salesforce/blip2-flan-t5-xl",
            model="Salesforce/blip2-flan-t5-xl",
        )
        image_utils = ImageUtils(config)
        return image_utils

    def setUp(self):
        self.temp_dir = self.create_random_images(num_images=10)
        self.create_questions(num_questions=3, temp_dir=self.temp_dir)

    @patch(
        "instacut.utils.image.models.blip2models.Blip2WithCustomGeneration.from_pretrained"
    )
    @patch("transformers.Blip2Processor.from_pretrained")
    def test_model_initialization(self, mock_processor, mock_model):
        # Regenerate the image_utils object with the patched model and processor
        image_utils = self.create_image_utils(self.temp_dir)

        assert isinstance(image_utils.model, MagicMock)
        assert isinstance(image_utils.processor, MagicMock)

        mock_processor.assert_called_with(image_utils.config.processor)
        mock_model.assert_called_with(
            image_utils.config.model,
            device_map=unittest.mock.ANY,
            torch_dtype=torch.float16,
        )

        assert image_utils.config.batch_size == 4
        assert image_utils.config.preprocessing_batch_size == 8
        assert image_utils.config.image_prompts == os.path.join(
            self.temp_dir, "questions", "questions.tsv"
        )
        assert image_utils.config.processor == "Salesforce/blip2-flan-t5-xl"
        assert image_utils.config.model == "Salesforce/blip2-flan-t5-xl"

    def test_ingest_questions(self):
        image_utils = self.create_image_utils(self.temp_dir)

        questions = image_utils.parseQuestions()
        assert len(questions) == 3
        question, parsed_question = questions[1]
        assert question.id == "1"
        assert question.version == "0"
        assert question.topic.isnumeric()
        assert (
            question.question
            == "What color do you get if you mix yellow and blue?"
        )
        assert question.type == "ENUM"

        assert (
            parsed_question
            == "Please answer the following question:\nWhat color do you get if you mix yellow and blue? Please select one of the following options, or answer with NA if none apply: [red, green, purple, orange]\nYour answer: "
        )

    def test_query_from_input_image(self):
        image_utils = self.create_image_utils(
            tempdir=self.temp_dir, device="cuda:0"
        )

        # Get all the files in the tempdir directory, without relying on
        # code above to figure out the number of images.
        image_files = [
            os.path.join(self.temp_dir, f)
            for f in os.listdir(self.temp_dir)
            if os.path.isfile(os.path.join(self.temp_dir, f))
        ]

        query_output = image_utils.queryFromInputImage(
            image_files, image_utils.config.preprocessing_batch_size
        )
        assert query_output.shape[0] == len(image_files)
        assert query_output.shape[1] == 32
        assert query_output.shape[2] == 768

    def test_generate_answers(self):
        image_utils = self.create_image_utils(
            tempdir=self.temp_dir, device="cuda:0"
        )
        image_utils.generateAnswers()

        questions_dir = os.path.join(self.temp_dir, "questions")

        # Make sure that the questions.json file was created
        assert os.path.isfile(os.path.join(questions_dir, "questions.json"))

        # Process the JSON and make sure that the answers are correct
        with open(os.path.join(questions_dir, "questions.json"), "r") as f:
            questions = json.load(f)

        assert len(questions) == 3
        for q in questions:
            answers = q[1]
            assert len(answers) == 10
            for a in answers:
                assert a == "green"
