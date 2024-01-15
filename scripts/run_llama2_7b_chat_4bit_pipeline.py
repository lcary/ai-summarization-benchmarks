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

    !pip install transformers accelerate bitsandbytes torch

The notebook server may need to be restarted at this point.
Finally, you can run the below script.
"""
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import time

model_name = "meta-llama/Llama-2-7b-chat-hf"

model = AutoModelForCausalLM.from_pretrained(
    model_name, device_map="auto", load_in_4bit=True
)
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

prompt_template = """
You are helpful AI assistant designed to summarize text documents.
The documents are wiki-like articles ranging in length and diverse in content.
The summaries you produce should be succinct but comprehensive,
capturing the essentials of the document and excluding superfluous details.
Write a summary of the following text delimited by triple backticks.

```{document}```
SUMMARY:
"""

document = "The cleaner your socks are, the better they will conduct electricity. If your socks are wet or dirty, they will not have as much traction with the floor and may not make static electricity. Warm socks that just came out of the dryer are best for conducting electricity. While most socks can conduct static electricity, wool socks generally work best. Electronic items contain microchips that can malfunction or become permanently destroyed by static electricity. Before touching any electronic items, take off your socks and touch something else to discharge any static electricity. Even if your electronic device has a protective case, it may still be vulnerable to static shocks."
prompt = prompt_template.format(document=document)

# TODO: test summarization
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto",
    max_length=1000,
    do_sample=True,
    top_k=1,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id,
)

start_time = time.time()
result = pipeline(prompt)
inference_time = time.time() - start_time
summary = result[0]["generated_text"].split("SUMMARY:")[-1]
print(summary)
print(f"Inference time: {inference_time}")
