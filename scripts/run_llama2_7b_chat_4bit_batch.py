"""
Overview

This script uses one-shot prompting with the Llama2-7b chat model (4bit format)
to summarize text documents from the Wikilingua dataset.

Instructions

This assumes you are running on a GPU instance with a Python version >= 3.10
and CUDA installed. Google Colab provides T4 GPU servers free of charge which
should be sufficient.

CUDA Prerequisite Setup (Colab):

You will likely need to configure several settings for correct CUDA setup in Colab
by running the below commands.

    !export CUDA_HOME=/usr/local/cuda-12.2
    # Workaround: https://github.com/pytorch/pytorch/issues/107960
    !ldconfig /usr/lib64-nvidia
    !ldconfig -p | grep libcuda

The ldconfig command output should show libcuda.so, or else issues will occur,
If the ldconfig requires a different directory, check for other nvidia libraries
under /user/. If the notebook server has a different version of cuda home installed,
check for that via `ls /user/local/cuda*` and set that to CUDA_HOME.

Then, install dependencies:

    !pip install transformers accelerate bitsandbytes torch datasets
    !pip install tensorrt --extra-index-url https://pypi.nvidia.com

Troubleshooting: if you see an error like 'NotImplementedError: A UTF-8 locale is required.'
Then run the following code in a notebook cell first:

    import locale
    locale.getpreferredencoding = lambda: "UTF-8"

(From https://github.com/googlecolab/colabtools/issues/3409#issuecomment-1446281277)

The notebook server may need to be restarted at this point.
Finally, you can run the below script.
"""

import os
import random
import time

import torch
from datasets import load_dataset
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16
)

model_name = "meta-llama/Llama-2-7b-chat-hf"

model = AutoModelForCausalLM.from_pretrained(
    model_name, device_map="auto", quantization_config=quantization_config
)
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

device = "cuda" if torch.cuda.is_available() else "cpu"


prompt_template = """
You are helpful AI assistant designed to summarize text documents.
The documents are wiki-like articles ranging in length and diverse in content.
The summaries you produce should be succinct but comprehensive,
capturing the essentials of the document and excluding superfluous details.
Below is an example of your task.

----------------------------------------- EXAMPLE START -----------------------------------------

Document: 'If you have a chair that is the right size to stretch the elastic, that will work perfectly. If you don’t have a chair that’s the right size, you can try using the side of a small table, an empty drawer, or an empty poster frame. If you can, line up the sides with the side of the chair. This will help stretch the elastic evenly. Let your elastic sit, being stretched, for 24 hours. If the desired size still isn’t reached, put the elastic back in the stretched position and leave it for several days. Leave it in a warm place to help the elastic band stretch.', 'You want your iron on and set to the highest setting. Run a face cloth or hand towel under water until it is damp, but not soaked. You can either pin each side of your pants to the ironing board—stretched to the desired length. Or, you can simply slide the pants around the ironing board until they are the proper width. Make sure it completely covers the elastic that you’re trying to stretch out. If needed, use two cloths. With the damp cloth on top of your elastic band and your iron on the highest setting, iron over it. Iron for 10 seconds and then let sit for 10 seconds. Continue doing so for 5-10 minutes. This will help your pants fit because as the elastic heats, it will heightens the breaking weight. This means that it will be able to stretch more before reaching its limit. If it hasn’t’ stretch enough, try flipping your elastic band and repeat the process. Do so until you reach your desired fit.', 'This will make it much easier to work with. Plus, you will be less likely to make an error with your scissors if you can see what you’re doing. Sometimes, elastics are sewn into the seam of clothing. If this is the case, you won’t be able to pull the elastic out of them if you cut the somewhere other than the seam. Find the seam by holding on one side of the seam and stretching the other side. If you feel the elastic shift, you are free to cut wherever you’d like. If you feel it snag at the seam, make sure to cut here. To remove the elastic band from your clothing, make a slit (around ½”). If the elastic is sewn into the seam, you’ll have to cut the seam the size of the elastic. Use scissors to go through your slit and cut the elastic. Cut through the entire elastic without cutting any more holes in your clothes. If you want to still be able to tie the pants snugly, affix a long shoelace or ribbon to one end of the elastic using a safety pin. When you pull out the elastic, pull on the end without the ribbon. This will guide your new tie through the waistband. If you do not want a tie, then just slowly pull the elastic out, being careful not to catch a lose string and bunch your fabric. Once the elastic is out/replaced, your clothes are ready to wear. You can sew the slit closed if you want to, but it’s not a necessary step before you wear your clothes.'

Summary: 'Find a chair. Stretch your elastic clothing over your chair. Let it sit. Turn on the iron and dampen a cloth. Prepare your pants. Place the damp cloth over your elastic. Iron the elastic. Repeat as necessary. Turn the clothing inside out. Find the inside seam. Put a small slit on the inside of you clothing. Cut the elastic. Pull the elastic out.'

------------------------------------------ EXAMPLE END ------------------------------------------

Now, summarize the following document enclosed within triple backticks.

```{document}```

Summary:
"""

wikilingua_dataset = load_dataset("wiki_lingua", "english")
data = wikilingua_dataset["train"]


def get_doc(item):
    return item["article"]["document"]


def has_document(item):
    return bool(get_doc(item))


data = list(filter(has_document, data))
total_num_samples = 100
wikilingua_sample = random.sample(data, total_num_samples)

batch_size = 2


def get_prompt(item):
    return prompt_template.format(document=get_doc(item)[0])


# def process_batch(docs: List[str]) -> Tuple[List[str], float]:
#     inputs = tokenizer.batch_encode_plus(
#         docs, return_tensors="pt", padding=True
#     ).to(device)
#
#     generate_params = {
#         "input_ids": inputs['input_ids'],
#         "max_new_tokens": 200,
#         "bos_token_id": 1,
#         "eos_token_id": 2,
#         "pad_token_id": 32000,
#         "temperature": 0.9,
#         "top_p": 0.6,
#     }
#
#     start_time = time.time()
#     with torch.no_grad():
#         outputs = model.generate(**generate_params)
#     duration = time.time() - start_time
#     outputs = outputs.cuda()
#
#     summaries = tokenizer.batch_decode(outputs, skip_special_tokens=True)
#
#     del outputs
#     del inputs
#     torch.cuda.empty_cache()
#
#     return summaries, duration
#
#
#
# Process the documents in batches
# all_summaries = []
# all_durations = []
#
# # total_generation_samples = len(wikilingua_sample)
# total_generation_samples = 20
# for i in range(0, total_generation_samples, batch_size):
#     batch_docs = [get_prompt(item) for item in wikilingua_sample[i:i + batch_size]]
#     summaries, duration = process_batch(batch_docs)
#     all_summaries.extend(summaries)
#     all_durations.append(duration)
#
# summary = all_summaries[0]
# print(summary)
# avg_inference_time = sum(all_durations) / total_generation_samples
# print(f"Inference time: {avg_inference_time}")
#


prompt0 = get_prompt(wikilingua_sample[0])
prompt1 = get_prompt(wikilingua_sample[1])
batch_docs = [prompt0, prompt1]

truncation_length = 4096
tokenizer.pad_token = tokenizer.eos_token
inputs = tokenizer(
    batch_docs,
    return_tensors="pt",
    add_special_tokens=True,
    padding=True,
    truncation=True,
    max_length=truncation_length,
).to(device)

generate_params = {
    "max_new_tokens": 200,
    "input_ids": inputs["input_ids"],
    "bos_token_id": 1,
    "eos_token_id": 2,
    "pad_token_id": 32000,
    "temperature": 0.9,
    "top_p": 0.6,
}

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TORCH_USE_CUDA_DSA"] = "1"

torch.cuda.empty_cache()


generate_params = {
    "max_new_tokens": 200,
    "input_ids": inputs["input_ids"],
    "bos_token_id": 1,
    "eos_token_id": 2,
    "pad_token_id": 32000,
    "temperature": 0.9,
    "top_p": 0.6,
}

start_time = time.time()
output_ids = model.generate(**generate_params)
# TODO: fix error:
# RuntimeError: CUDA error: device-side assert triggered
# CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.
# For debugging consider passing CUDA_LAUNCH_BLOCKING=1.
# Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.
inference_time = time.time() - start_time
skip_special_tokens = True
# TODO: batch decode
# reply = tokenizer.decode(output_ids[-new_tokens:], skip_special_tokens)
# if len(output_ids) > 0:
#     if tokenizer.convert_ids_to_tokens(int(output_ids[-new_tokens])).startswith("▁"):
#         reply = " " + reply
# print(reply)
print(f"Inference time: {inference_time}")
