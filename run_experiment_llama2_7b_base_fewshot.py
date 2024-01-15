"""
Overview

This script uses few-shot learning to coax the base Llama2-7b model
to summarize text documents from the Wikilingua dataset.

Instructions

This assumes you are running on a GPU instance with a Python version >= 3.10
and CUDA installed. Google Colab provides T4 GPU servers free of charge which
should be sufficient.

Second, this requires a HuggingFace account, signing Microsoft's 
and requesting access to Llama2 from the model hub:
https://huggingface.co/meta-llama/Llama-2-7b-hf
(Llama 2 base model access takes a few hours to a few days for the
Microsoft team to process.)

Finally, you need to create an API token in HuggingFace at:
https://huggingface.co/settings/tokens

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

Next, log into huggingface (assuming an active virtualenv or conda env):

    !pip install huggingface_hub
    !huggingface-cli login

Then, install dependencies:

    !pip install datasets transformers accelerate bitsandbytes torch huggingface_hub

The notebook server may need to be restarted at this point.
Finally, you can run the below script.
"""

import random
import time

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16
)

model_id = "meta-llama/Llama-2-7b-hf"

model = AutoModelForCausalLM.from_pretrained(
    model_id, device_map="auto", quantization_config=quantization_config
)
tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

wikilingua_dataset = load_dataset("wiki_lingua", "english")


def get_doc(item):
    return item["article"]["document"]


def has_document(item):
    return bool(get_doc(item))


data = wikilingua_dataset["train"]
data = list(filter(has_document, data))

wikilingua_sample = random.sample(data, 100)

input_max_tokens = 2048


def truncate_text(text: list, max_tokens: int) -> str:
    all_text = "\n ".join(text)
    words = list(filter(None, all_text.split(" ")))
    max_words = max_tokens if len(words) > max_tokens else len(words)
    return " ".join(words[:max_words])


trun_limit = int(input_max_tokens / 8)

sample_1 = wikilingua_sample[0]
document_1 = truncate_text(sample_1["article"]["document"], trun_limit)
summary_1 = truncate_text(sample_1["article"]["summary"], trun_limit)

sample_2 = wikilingua_sample[1]
document_2 = truncate_text(sample_2["article"]["document"], trun_limit)
summary_2 = truncate_text(sample_2["article"]["summary"], trun_limit)

sample_3 = wikilingua_sample[2]
new_document = truncate_text(sample_3["article"]["document"], trun_limit)

prompt = f"""
[Document 1]: {document_1}
[Summary]: {summary_1}

[Document 2]: {document_2}
[Summary]: {summary_2}

[Your Document]: {new_document}
[Summary]:"""

print(prompt)

input_ids = tokenizer.encode(
    prompt, return_tensors="pt", truncation=True, max_length=input_max_tokens
).to(device)

max_new_tokens = 50
start_time = time.time()
summary_ids = model.generate(
    input_ids,
    max_new_tokens=max_new_tokens,
)
inference_time = time.time() - start_time
del input_ids  # free up gpu memory
summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
del summary_ids  # free up gpu memory
torch.cuda.empty_cache()

print(summary)
print(f"inference_time: {inference_time}")

# This achieves a slow working demo, but the quality of the summary appears poor
# upon manual review of several iterations. This experiment was largely limited by RAM.
# Next, we'll try using a fine-tuned Llama2 model.
# inference_time @ unlimited max generation tokens: 227.33489084243774 secs (long response)
# inference_time @ 50 max generation tokens: 15.427298069000244
# inference_time @ 10 max generation tokens: 10.593532085418701 (useless, abbrev. response)
