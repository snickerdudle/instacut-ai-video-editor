import json
import os
from collections import UserDict
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


class Prompt:
    def __init__(
        self,
        prompt_template: str,
        aux_text: Optional[List[str]] = None,
        input_text: Optional[str] = "",
        additional_params: Optional[dict] = None,
        prompt_id: Optional[int] = None,
        description: Optional[str] = None,
    ):
        self.prompt_template = prompt_template
        self.aux_text = aux_text or []
        self.input_text = input_text
        self.additional_params = additional_params or {}
        self.prompt_id = prompt_id
        self.description = description

    def assemblePrompt(self, input_text: str = "") -> str:
        self.input_text = input_text if input_text else self.input_text
        return self.prompt_template.format(self=self)

    def has(self, key: str) -> bool:
        # Check if the prompt has a specific key in its additional_params
        return key in self.additional_params

    def set(self, key: str, value: str) -> None:
        # Set a specific key in the additional_params
        self.additional_params[key] = value

    def get(self, key: str) -> str:
        # Get the value of a specific key in the additional_params
        return self.additional_params.get(key)

    def remove(self, key: str) -> None:
        # Remove a specific key from the additional_params
        self.additional_params.pop(key, None)

    def __repr__(self) -> str:
        return f"<Prompt {self.prompt_id} ({self.description})>"


class PromptCollection(UserDict):
    def __init__(self, file_path: Optional[str] = None):
        if file_path is None:
            # Get the location of the current file
            file_path = os.path.realpath(__file__)
            # Set the file path to the prompts.json file
            file_path = Path(file_path).parent / "prompts.json"
        self.file_path = file_path
        self.data: dict = self.loadPrompts()

    def loadPrompts(self) -> dict[int, Prompt]:
        # Load the prompts from the file
        prompts = {}
        with open(self.file_path, "r") as f:
            prompt_definition = json.load(f)
        for config_id in prompt_definition["configs"]:
            if config_id in prompts:
                # Check for duplicate prompt IDs
                raise ValueError(
                    f"Encountered a duplicate Prompt ID: {config_id}"
                )
            config = prompt_definition["configs"][config_id]
            aux_text_id, prompt_template_id, additional_params, description = (
                config["aux_text_id"],
                config["prompt_template_id"],
                config["additional_params"],
                config["description"],
            )
            aux_text = prompt_definition["data"]["aux_text"][aux_text_id]
            prompt_template = prompt_definition["data"]["prompt_template"][
                prompt_template_id
            ]
            prompts[config_id] = Prompt(
                prompt_template=prompt_template,
                aux_text=aux_text,
                additional_params=additional_params,
                prompt_id=config_id,
                description=description,
            )
        return prompts

    def __setitem__(self, key: int, value: Prompt) -> None:
        if not isinstance(value, Prompt):
            raise ValueError(
                f"PromptCollection only accepts values of type Prompt"
            )
        # Set a prompt in the collection
        self.data[key] = value
