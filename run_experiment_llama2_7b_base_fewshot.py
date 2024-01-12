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

from datasets import load_dataset, load_metric
import random
import time
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


model_id = "meta-llama/Llama-2-7b-hf"

tokenizer = AutoTokenizer.from_pretrained(model_id)

device = "cuda:0" if torch.cuda.is_available() else "cpu"

# To allow llama 7b to be loaded successfully on T4 Colab instances, we load
# the model in 8bit with fp16 tensors, and map directly to the gpu:
# https://github.com/facebookresearch/llama/issues/394#issuecomment-1645415450
model = AutoModelForCausalLM.from_pretrained(
    model_id, load_in_8bit=True, device_map="auto", torch_dtype=torch.float16
)

wikilingua_dataset = load_dataset("wiki_lingua", "english")
wikilingua_sample = random.sample(list(wikilingua_dataset["train"]), 100)

# 4096 causes OOM on Colab's T4
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

start_time = time.time()
summary_ids = model.generate(
    input_ids,
    max_length=(input_max_tokens + 1),
    length_penalty=5.0,
    num_beams=2,
    early_stopping=True,
)
inference_time = time.time() - start_time
del input_ids  # free up gpu memory
summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
del summary_ids  # free up gpu memory

print(summary)
print(f"inference_time: {inference_time}")

# This achieves a working demo, but the quality of the summary appears poor
# upon manual review of several iterations. This experiment was largely limited by RAM.
# Next, we'll try using a fine-tuned Llama2 model. Since meta-llama/Llama-2-7b-chat-hf
# and togethercomputer/LLaMA-2-7B-32K consume too much RAM for the Colab's T4 servers,
# we'll try TheBloke/Llama-2-7B-Chat-GGUF.
