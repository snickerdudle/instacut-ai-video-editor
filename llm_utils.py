from dataclasses import dataclass
import os
import openai
from prompts import Prompt
from typing import List, Union, Optional

# Get the API key from the environment
openai.api_key = os.environ.get("OPENAI_API_KEY")


@dataclass
class OpenAIChat:
    """Class for a video."""

    prompt: Union[Prompt, str]
    model: str = "gpt-3.5-turbo"
    temperature: float = 0.9
    messages: Optional[List[dict]] = None
    priming_message: Optional[str] = "You are an intelligent assistant"

    def processMessage(self, message: str) -> str:
        """Process a message.

        Args:
            message (str): The message to process.

        Returns:
            str: The response to the message.
        """
        prompt = self.prompt.assemblePrompt(message)
        messages = self.getMessageHistory()
        messages.append({"role": "user", "content": prompt})
        chat = openai.ChatCompletion.create(
            model=self.model, messages=messages, temperature=self.temperature
        )
        reply = chat.choices[0].message.content
        messages.append({"role": "assistant", "content": reply})
        return reply

    def getMessageHistory(self) -> List[dict]:
        """Get the message history."""
        messages = [
            {
                "role": "system",
                "content": self.priming_message,
            }
        ]
        return messages


@dataclass
class OpenAIContinuousChat(OpenAIChat):
    def getMessageHistory(self) -> List[dict]:
        return self.messages
