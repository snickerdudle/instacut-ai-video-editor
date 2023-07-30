import json
import os
import random
import tempfile
import unittest
from typing import List, Tuple
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
    def create_sample_questions(num_questions: int = 3) -> List[Tuple[str]]:
        questions = []
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

            questions.append(tuple(cur_question))

        return questions

    @staticmethod
    def create_and_store_sample_questions(
        num_questions: int = 3,
        temp_dir: str = None,
    ):
        temp_dir = temp_dir or tempfile.mkdtemp()

        questions = [
            "ID\tVersion\tTopic\tQuestion\tOptions\tType",
        ]

        generated_questions = ImageUtilsTest.create_sample_questions(
            num_questions=num_questions
        )
        for q in generated_questions:
            questions.append("\t".join(q))

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
        self.create_and_store_sample_questions(
            num_questions=3, temp_dir=self.temp_dir
        )

    @patch(
        "instacut.utils.image.models.blip2models.Blip2WithCustomGeneration.from_pretrained"
    )
    @patch("transformers.Blip2Processor.from_pretrained")
    def test_model_initialization(self, mock_processor, mock_model):
        # Regenerate the image_utils object with the patched model and processor
        image_utils = self.create_image_utils(self.temp_dir)

        self.assertIsInstance(image_utils.model, MagicMock)
        self.assertIsInstance(image_utils.processor, MagicMock)

        mock_processor.assert_called_with(image_utils.config.processor)
        mock_model.assert_called_with(
            image_utils.config.model,
            device_map=unittest.mock.ANY,
            torch_dtype=torch.float16,
        )

        self.assertEqual(image_utils.config.batch_size, 4)
        self.assertEqual(image_utils.config.preprocessing_batch_size, 8)
        self.assertEqual(
            image_utils.config.image_prompts,
            os.path.join(self.temp_dir, "questions", "questions.tsv"),
        )
        self.assertEqual(
            image_utils.config.processor, "Salesforce/blip2-flan-t5-xl"
        )
        self.assertEqual(
            image_utils.config.model, "Salesforce/blip2-flan-t5-xl"
        )

    def test_parse_questions(self):
        image_utils = self.create_image_utils(self.temp_dir)

        questions = image_utils.parseQuestions()
        self.assertEqual(len(questions), 3)
        question, parsed_question = questions[1]
        self.assertEqual(question.id, "1")
        self.assertEqual(question.version, "0")
        self.assertTrue(question.topic.isnumeric())
        self.assertEqual(
            question.question,
            "What color do you get if you mix yellow and blue?",
        )
        self.assertEqual(question.type, "ENUM")

        self.assertEqual(
            parsed_question,
            "Please answer the following question:\nWhat color do you get if you mix yellow and blue? Please select one of the following options, or answer with NA if none apply: [red, green, purple, orange]\nYour answer: ",
        )

    def test_parse_questions_custom_questions(self):
        image_utils = self.create_image_utils(self.temp_dir)

        questions = ImageUtilsTest.create_sample_questions(num_questions=15)

        questions = image_utils.parseQuestions(custom_questions=questions)
        self.assertEqual(len(questions), 15)
        for question, parsed_question in questions:
            self.assertTrue(question.id.isnumeric())
            self.assertTrue(question.version.isnumeric())
            self.assertTrue(question.topic.isnumeric())
            self.assertEqual(
                question.question,
                "What color do you get if you mix yellow and blue?",
            )
            self.assertEqual(question.type, "ENUM")
            self.assertIn("red", question.options)
            self.assertIn("green", question.options)
            self.assertIn("purple", question.options)
            self.assertIn("orange", question.options)

            self.assertEqual(
                parsed_question,
                "Please answer the following question:\nWhat color do you get if you mix yellow and blue? Please select one of the following options, or answer with NA if none apply: [red, green, purple, orange]\nYour answer: ",
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
        self.assertEqual(query_output.shape[0], len(image_files))
        self.assertEqual(query_output.shape[1], 32)
        self.assertEqual(query_output.shape[2], 768)

    def test_generate_answers(self):
        image_utils = self.create_image_utils(
            tempdir=self.temp_dir, device="cuda:0"
        )
        image_utils.generateAnswers()

        questions_dir = os.path.join(self.temp_dir, "questions")

        # Make sure that the questions.json file was created
        self.assertTrue(
            os.path.isfile(os.path.join(questions_dir, "questions.json"))
        )

        # Process the JSON and make sure that the answers are correct
        with open(os.path.join(questions_dir, "questions.json"), "r") as f:
            questions = json.load(f)

        self.assertEqual(len(questions), 3)
        for q in questions:
            answers = q[1]
            self.assertEqual(len(answers), 10)
            for a in answers:
                self.assertEqual(a, "green")
