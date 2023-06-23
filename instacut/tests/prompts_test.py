import unittest
from typing import List, Optional
from instacut.utils.prompts import Prompt


class TestPrompt(unittest.TestCase):
    def test_assemblePrompt(self):
        prompt_template = "The input text is: {self.input_text}"
        aux_text = ["Auxiliary Text"]
        prompt = Prompt(
            prompt_template=prompt_template,
            aux_text=aux_text,
        )

        assembled_prompt = prompt.assemblePrompt("Hello, world!")

        expected_prompt = "The input text is: Hello, world!"
        self.assertEqual(assembled_prompt, expected_prompt)

    def test_assemblePrompt_aux_text(self):
        prompt_template = "The input text is: {self.aux_text[0]} {self.aux_text[1]} {self.input_text}"
        aux_text = ["Auxiliary Text", "ABC"]
        prompt = Prompt(
            prompt_template=prompt_template,
            aux_text=aux_text,
        )

        assembled_prompt = prompt.assemblePrompt("Hello, world!")

        expected_prompt = "The input text is: Auxiliary Text ABC Hello, world!"
        self.assertEqual(assembled_prompt, expected_prompt)

    def test_assemblePrompt_no_input_text(self):
        prompt_template = "The input text is: {self.input_text}"
        aux_text = ["Auxiliary Text"]
        prompt = Prompt(prompt_template=prompt_template, aux_text=aux_text)

        assembled_prompt = prompt.assemblePrompt()

        expected_prompt = "The input text is: "
        self.assertEqual(assembled_prompt, expected_prompt)

    def test_assemblePrompt_updated_input_text(self):
        prompt_template = "The input text is: {self.input_text}"
        aux_text = ["Auxiliary Text"]
        prompt = Prompt(prompt_template=prompt_template, aux_text=aux_text)

        assembled_prompt = prompt.assemblePrompt(input_text="New input")

        expected_prompt = "The input text is: New input"
        self.assertEqual(assembled_prompt, expected_prompt)


if __name__ == "__main__":
    unittest.main()
