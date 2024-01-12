"""
Overview

This script uses prompting with the Mistral-7b chat model (GGUF format)
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

    !pip install ctransformers[cuda]>=0.2.24 datasets

The notebook server may need to be restarted at this point.
Finally, you can run the below script.
"""

# TODO use mistral
# see https://discuss.huggingface.co/t/number-of-tokens-2331-exceeded-maximum-context-length-512-error-even-when-model-supports-8k-context-length/57180

from datasets import load_dataset, load_metric
import random
import time
from ctransformers import AutoModelForCausalLM

model_id = "TheBloke/Llama-2-7B-Chat-GGUF"

# Set gpu_layers to the number of layers to offload to GPU. Set to 0 if no GPU acceleration is available on your system.
llm = AutoModelForCausalLM.from_pretrained(model_id, model_type="llama", gpu_layers=100)

wikilingua_dataset = load_dataset("wiki_lingua", "english")
data = wikilingua_dataset["train"]


def get_doc(item):
    return item["article"]["document"]


def has_document(item):
    return bool(get_doc(item))


data = list(filter(has_document, data))
wikilingua_sample = random.sample(data, 100)

sample_1 = wikilingua_sample[0]
document_1 = get_doc(sample_1)
summary_1 = sample_1["article"]["summary"]

prompt = f"""
You are helpful AI assistant designed to summarize text documents. 
The documents are wiki-like articles ranging in length and diverse in content. 
The summaries you produce should be succinct but comprehensive,
capturing the essentials of the document and excluding superfluous details. 
Summarize the following text: {document_1}
"""

start_time = time.time()
summary = llm(prompt)
inference_time = time.time() - start_time

print(summary)
print(f"inference_time: {inference_time}")
