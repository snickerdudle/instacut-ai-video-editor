# Image Utils

# %%
import torch
import numpy as np
import os
from pathlib import Path
from transformers import Blip2ForConditionalGeneration, Blip2Processor
import time
from PIL import Image
import multiprocessing as mp

# Free all the cache that can be freed
torch.cuda.empty_cache()

device = 0
device_map = {
    "query_tokens": device,
    "vision_model": device,
    "language_model": device,
    "language_projection": device,
    "qformer": device,
}

processor = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xl")
# can also try with custom device_map
model = Blip2ForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-flan-t5-xl",
    device_map=device_map,
    torch_dtype=torch.float16,
)


# %%
image_path = Path(
    "/home/askharry/src/video-editing/data/the_most_misunderstood_concept_i/frames/uniformseconds_3.0/"
)
# Get all the jpg files in that directory using "*.jpg" glob
files = [_ for _ in sorted(image_path.glob("*.jpg"))]

# url = "https://huggingface.co/hf-internal-testing/blip-test-image/resolve/main/demo.jpg"
# # Download the image, and use imutils to convert it to a numpy array
# image = Image.open(requests.get(url, stream=True).raw)
# # convert to numpy array
# image = np.array(image)

batch_size = 40
batch_size = min(batch_size if batch_size else int("inf"), len(files))
print(batch_size)


image = torch.tensor(
    np.stack([np.array(Image.open(i)) for i in files[:batch_size]]),
    dtype=torch.uint8,
)
image.to(device)

with open("image_prompts.tsv", "r") as f:
    questions = [i for i in f.read().strip().split("\n") if i]
    questions = [i.split("\t") for i in questions[1:]]

# Questions is now ID, Version, Topic, Question, Options, Type


# %%

ENUM_ADDITION = (
    " Please select one of the following options, or return NA if none apply: "
)
BOOL_ADDITION = (
    " Please answer Yes or No, or return NA if the question does not apply: "
)
LIST_STR_ADDITION = " Please list all that apply, comma separated, or NA."


def model_generate(input_queue, model_queue, model):
    pass


# Create a multiprocessing Queue to store the results of the input processing
input_processing_queue = mp.Queue()
# Now create a Queue for the results of the model generation. If the Queue is
# empty, then the model is still processing the input. If the Queue is not
# empty, then the model has finished processing the input and the result is
# ready to be returned.
model_result_queue = mp.Queue()
# Create a multiprocessing Process to handle the model generation
process = mp.Process(
    target=model_generate,
    args=(input_processing_queue, model_result_queue, model),
)

# Now we need to create a governing loop to handle the input processing and
# model generation. This loop will run until the input Queue is empty and the
# model Queue is empty.
while not input_processing_queue.empty() or not model_result_queue.empty():
    pass

for _, _, topic, question, options, q_type in questions:
    start_time = time.time()

    parsed_question = question
    if q_type == "ENUM":
        parsed_question += ENUM_ADDITION
        parsed_question += options
    elif q_type == "BOOL":
        parsed_question += BOOL_ADDITION
    elif q_type == "LIST_STR":
        parsed_question += LIST_STR_ADDITION

    parsed_question = (
        "Please answer the following question: "
        + parsed_question
        + " Your answer: "
    )

    load_time = time.time()
    inputs = processor(
        images=image,
        text=[parsed_question] * batch_size,
        return_tensors="pt",
    ).to(device, torch.float16)
    load_time = time.time() - load_time

    inference_time = time.time()
    predictions = model.generate(**inputs)
    inference_time = time.time() - inference_time
    text_decode_time = time.time()
    generated_text = processor.batch_decode(
        predictions, skip_special_tokens=True
    )
    text_decode_time = time.time() - text_decode_time

    print(f"Q: {topic}, A: {generated_text}")
    print(
        f"Load time: {load_time:.4f}s, Inference time: {inference_time:.4f}s, Text decode time: {text_decode_time:.4f}s"
    )

# %%
