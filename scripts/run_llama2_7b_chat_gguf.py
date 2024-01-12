"""
Overview

This script uses one-shot prompting with the Llama2-7b chat model (GGUF format)
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

    !pip install ctransformers[cuda]>=0.2.24 datasets torch accelerate

The notebook server may need to be restarted at this point.
Finally, you can run the below script.
"""

import random
import time

import torch
from ctransformers import AutoModelForCausalLM, AutoConfig
from datasets import load_dataset
from transformers import AutoTokenizer

model_id = "TheBloke/Llama-2-7B-Chat-GGUF"

config = AutoConfig.from_pretrained(model_id)

# Explicitly set the max_seq_len
config.max_seq_len = 2048  # 4096 takes too long w/ Q2_K
config.max_answer_len = 512

# model_file = llama-2-7b-chat.Q2_K.gguf
model_file = "llama-2-7b-chat.Q3_K_M.gguf"
# model_file="llama-2-7b-chat.Q4_K_M.gguf"

device = "cuda" if torch.cuda.is_available() else "cpu"
# Set gpu_layers to the number of layers to offload to GPU. Set to 0 if no GPU acceleration is available on your system.
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    model_file=model_file,
    model_type="llama",
    gpu_layers=100,
    config=config,
    torch_dtype=torch.float16,
    device_map=device,
    hf=True,
)


def get_doc(item):
    return item["article"]["document"]


def has_document(item):
    return bool(get_doc(item))


# Try .generate and decice_map via https://github.com/marella/ctransformers/issues/199
wikilingua_dataset = load_dataset("wiki_lingua", "english")
data = wikilingua_dataset["train"]
data = list(filter(has_document, data))
wikilingua_sample = random.sample(data, 100)

eg_sample = wikilingua_sample[0]
eg_doc = get_doc(eg_sample)
eg_summary = eg_sample["article"]["summary"]

new_sample = wikilingua_sample[1]
new_doc = get_doc(new_sample)

prompt = f"""
You are helpful AI assistant designed to summarize text documents. 
The documents are wiki-like articles ranging in length and diverse in content. 
The summaries you produce should be succinct but comprehensive,
capturing the essentials of the document and excluding superfluous details. 
Below is an example of your task.
----------------------------------------- EXAMPLE -----------------------------------------
[Document]: {eg_doc}

[Summary]: {eg_summary}
----------------------------------------- EXAMPLE -----------------------------------------

Now, summarize the following document. 

[Document]: {new_doc}
"""

print(prompt)

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
input_ids = tokenizer.encode(
    prompt, return_tensors="pt", truncation=True, max_length=config.max_seq_len
).to(device)

start_time = time.time()
# summary_ids = model.generate(input_ids, max_length=(input_max_tokens + 1), length_penalty=5.0, num_beams=2, early_stopping=True)
summary_ids = model.generate(
    input_ids=input_ids,
    max_new_tokens=config.max_seq_len,
    # early_stopping=True,
    repetition_penalty=1.1,
    # no_repeat_ngram_size=2,
    temperature=0.6,
    do_sample=True,
    # top_k=5,
    top_p=0.9,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.pad_token_id,
    use_cache=True,
)
inference_time = time.time() - start_time
del input_ids  # free up gpu memory
summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
del summary_ids  # free up gpu memory

print(summary)
print(f"inference_time: {inference_time}")
