# Modified BLIP2 model to work with the new image model
from transformers import Blip2ForConditionalGeneration
from torch import nn
import torch

from typing import Optional


class Blip2WithCustomGeneration(Blip2ForConditionalGeneration):
    @torch.no_grad()
    def generate(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        image_embeds: Optional[torch.FloatTensor] = None,
        query_output: Optional[torch.FloatTensor] = None,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        **generate_kwargs,
    ) -> torch.LongTensor:
        """
        Overrides `generate` function to be able to use the model as a conditional generator.

        Args:
            pixel_values (`torch.FloatTensor` of shape (batch_size, num_channels, height, width), *optional*):
                Input images to be processed. If not supplied, we check for `image_embeds`.
            image_embeds (`torch.FloatTensor` of shape (batch_size, sequence_length, hidden_size), *optional*):
                The sequence used as a prompt for the generation.
            query_output (`torch.FloatTensor` of shape (batch_size, sequence_length, hidden_size), *optional*):
                The pre-processed set of query tokens.
            input_ids (`torch.LongTensor` of shape (batch_size, sequence_length), *optional*):
                The sequence used as a prompt for the generation.
            attention_mask (`torch.LongTensor` of shape (batch_size, sequence_length), *optional*):
                Mask to avoid performing attention on padding token indices

        Returns:
            captions (list): A list of strings of length batch_size * num_captions.
        """
        if hasattr(self, "hf_device_map"):
            # preprocess for `accelerate`
            self._preprocess_accelerate()

        if (
            pixel_values is None
            and image_embeds is None
            and query_output is None
        ):
            raise ValueError(
                "You have to specify either pixel_values or image_embeds or query_output"
            )

        if pixel_values is not None:
            image_embeds = self.vision_model(
                pixel_values, return_dict=True
            ).last_hidden_state

        if query_output is None:
            image_attention_mask = torch.ones(
                image_embeds.size()[:-1],
                dtype=torch.long,
                device=image_embeds.device,
            )

            query_tokens = self.query_tokens.expand(
                image_embeds.shape[0], -1, -1
            )
            query_outputs = self.qformer(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_attention_mask,
                return_dict=True,
            )
            query_output = query_outputs.last_hidden_state

        batch_size = query_output.shape[0]

        language_model_inputs = self.language_projection(query_output)
        language_attention_mask = torch.ones(
            language_model_inputs.size()[:-1],
            dtype=torch.long,
            device=language_model_inputs.device,
        )
        if input_ids is None:
            input_ids = (
                torch.LongTensor([[self.config.text_config.bos_token_id]])
                .repeat(batch_size, 1)
                .to(image_embeds.device)
            )
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        attention_mask = torch.cat(
            [
                language_attention_mask,
                attention_mask.to(language_attention_mask.device),
            ],
            dim=1,
        )

        # concatenate query embeddings with prompt embeddings
        inputs_embeds = self.get_input_embeddings()(input_ids)
        inputs_embeds = torch.cat(
            [
                language_model_inputs,
                inputs_embeds.to(language_model_inputs.device),
            ],
            dim=1,
        )

        outputs = self.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            **generate_kwargs,
        )

        return outputs
