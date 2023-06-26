from dataclasses import dataclass
from typing import List, Optional


class Prompt:
    def __init__(
        self,
        prompt_template: str,
        aux_text: Optional[List[str]] = None,
        input_text: Optional[str] = "",
        additional_params: Optional[dict] = None,
    ):
        self.prompt_template = prompt_template
        self.aux_text = aux_text or []
        self.input_text = input_text
        self.additional_params = additional_params or {}

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


summarization_prompt_1 = (
    "Below is a transcript of a video without any "
    "timestamps. Analyze the content of the video, and create a "
    "detailed section-by-section breakdown of the progression of the video and the "
    "topics discussed. Organize the sections hierarchically, grouping individual "
    "events into larger themes, then into larger ones, and on and on until you've "
    "grouped everything into the overarching video group. Instead of creating long, "
    "comma-separated descriptions, bias towards listing the same elements as sub-"
    "items of the current item. Return the analysis in the form of an indented list, "
    "imitating the formatting of this example:\n"
    "{self.aux_text[0]}\n\n"
    "VIDEO TRANSCRIPT:\n{self.input_text}"
)


# Much better prompt for creating outlines with section numbers. Does not
# include timestamps, just the outline.
summarization_prompt_2 = (
    "Please organize the provided video transcript into a hierarchical outline. "
    "Each line in the transcript corresponds to a specific level in the outline. "
    "The lines at the same indentation level should be grouped together. Make "
    "sure to follow the formatting guidelines below for consistent output:\n\n"
    "{self.aux_text[0]}\n\n"
    "Your task is to generate a hierarchical outline for the provided video "
    "transcript, strictly formatted using the guidelines provided above. Ensure "
    "that the formatting of section titles and numbering is consistent with the "
    "guidelines provided above.\n\n"
    "Example:{self.aux_text[1]}\n\n"
    "Transcript:\n\n{self.input_text}"
)

# Same as the above prompt, but with timestamps included.
summarization_prompt_3 = (
    "Please organize the provided video transcript into a hierarchical outline, "
    "including the relevant video timestamps next to each section. The transcript "
    "is a series of lines of text, each corresponding to a specific point in the "
    "video. The lines are prepended by a timestamp, in increasing order, when "
    "they are spoken in the video. "
    "The outline you are to generate should be a layered summarization of the "
    "entire content of the video, and each line in the outline should correspond "
    "to a range of content in the transcript. In the outline, the lines at "
    "the same indentation level should be grouped together. Make sure to follow "
    "the formatting guidelines below for consistent output:\n\n"
    "GUIDELINES:\n"
    "{self.aux_text[0]}\n\n"
    "TASK:\n"
    "Your task is to generate a hierarchical outline for the provided video "
    "transcript, strictly formatted using the guidelines provided above. Be detailed "
    "in your summarization of the different sections of the video, and ensure "
    "that the formatting of section titles, numbering and video timestamps "
    "is consistent with the guidelines provided above. The timestamps you include "
    "in the outline should indicate the *beginning* time of that section, in other "
    "words the first instance that that section's topic was being discussed in the "
    "transcript. Bias towards going three or even four levels deep in the structure "
    "of the outline, making sure to not leave out important aspects of the video "
    "transcript from the hierarchy.\n\n"
    "EXAMPLE OUTLINE:\n"
    "{self.aux_text[1]}\n\n"
    "NEW VIDEO TRANSCRIPT:\n{self.input_text}\n\n"
    "YOUR OUTLINE:\n"
)

summarization_example_1 = """-Video
--Intro
---Describing the adventure
----The narrator provides an overview of the upcoming adventure, setting the stage for the thrilling journey ahead.
--Team Explores
---Discovering the unknown
----The adventurers exploring mysterious landscapes and encountering breathtaking sights.
---Uncovering hidden treasures
----They stumble upon hidden clues, solve intricate puzzles, and unearth valuable artifacts.
--Difficulties ensue
---Overcoming obstacles
----The team faces treacherous terrains, tricky riddles, and physical tests of strength and skill.
---Facing dangerous creatures
----They encounter fierce creatures guarding the path to their goal.
-----Gorgoyles
-----Mutants
-----Giant spiders
-----Venomous snakes
--Coming out on top
---Epic battle
----The adventurers face a formidable enemy or group of adversaries.
-----Enemy is scared of sunlight
-----Enemy is allergic to silver
---Revealing the ultimate secret
----The adventurers discover the long-guarded secret that holds immense power or knowledge.
-----The secret to eternal life
-----The secret to unlimited wealth
-----The secret to ultimate power"""

summarization_example_2 = [
    """1. Use a dash (-) followed by the number and item name for the main sections.
2. Use two dashes (--) followed by the number and item name for the subsections.
3. Use three dashes (---) followed by the number and item name for the sub-subsections.
4. For each item, include its corresponding number based on the indentation level, for example 2.3.1 being section 2, subsection 3, sub-subsection 1.""",
    """-1 Introduction
--1.1 The illusion of the universe
--1.2 The intergalactic medium as raw materials
--1.3 Quasars as the most powerful objects in existence
--1.4 The power of quasars in molding the structure of the universe
-2 Quasars
--2.1 The discovery of quasars
--2.2 Characteristics of quasars
--2.3 The power source of quasars
---2.3.1 Supermassive black holes
---2.3.2 Accretion disks
---2.3.3 Matter conversion
--2.4 The incredible power of quasars
---2.4.1 Feeding habits
---2.4.2 Energy output greater than stars
--2.5 Jet formation
--2.6 Quasar lifespan
---2.6.1 Effect on galaxies
---2.6.2 Impact on star formation
-3 The Milky Way and Quasars
--3.1 Preserving galactic history
--3.2 The possibility of the Milky Way being a quasar
--3.3 The future of Sagittarius A star
--3.4 The merger of Andromeda
---3.4.1 Double quasars
---3.4.2 Future of the Milky Way
-4 Lessons from Brilliant.org
--4.1 Interactive learning tool
--4.2 Black hole lesson
--4.3 Additional lessons
-5 Conclusion
--5.1 Limited edition pin: Dyson Sphere
--5.2 Future limited edition pins""",
]

summarization_example_3 = [
    """1. Use a dash (-) followed by the item name for the main sections. Include the corresponding transcript timestamp in square brackets at the beginning of the line, like [00:00].
2. Use two dashes (--) followed by the item name for the subsections. Include the corresponding transcript timestamp in square brackets at the beginning of the line, like [02:15].
3. Use three dashes (---) followed by the item name for the sub-subsections. Include the corresponding transcript timestamp in square brackets at the beginning of the line, like [03:04].
4. For each item, include its corresponding number based on the indentation level.""",
    """[00:00] - 1. Introduction
[00:00] -- 1.1 Speaker's Background
[00:00] --- 1.1.1 Education
[03:15] --- 1.1.2 Work Experience
[03:45] -- 1.2 Topic Overview
[05:30] - 2. Main Points
[05:30] -- 2.1 Point 1
[05:30] --- 2.1.1 Supporting Evidence 1
[06:45] --- 2.1.2 Supporting Evidence 2
[07:30] -- 2.2 Point 2
[07:30] --- 2.2.1 Supporting Evidence 1
[08:00] --- 2.2.2 Supporting Evidence 2
[08:00] ---- 2.2.2.1 Sub-sub-point 1
[08:30] ---- 2.2.2.2 Sub-sub-point 2
[09:00] -- 2.3 Point 3
[09:00] --- 2.3.1 Supporting Evidence 1
[09:30] --- 2.3.2 Supporting Evidence 2
[10:20] - 3. Conclusion
[10:20] -- 3.1 Summary of Key Points
[11:30] -- 3.2 Closing Remarks

Note: This example assumes a video duration of around 12 minutes. The timestamps will vary based on the actual video duration and content.""",
]


video_summarization_prompt_1 = Prompt(
    prompt_template=summarization_prompt_1, aux_text=[summarization_example_1]
)

video_summarization_prompt_2 = Prompt(
    prompt_template=summarization_prompt_2, aux_text=summarization_example_2
)

video_summarization_prompt_3 = Prompt(
    prompt_template=summarization_prompt_3,
    aux_text=summarization_example_3,
    additional_params={"include_timestamps": True},
)
