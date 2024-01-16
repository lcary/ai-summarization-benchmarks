"""
Overview

This script uses `ibm-fms` with the Llama2-7b chat model
to summarize text documents from the Wikilingua dataset.

Instructions

This assumes you are running on a GPU instance with a Python version >= 3.10
and CUDA installed. Google Colab provides T4 GPU servers free of charge which
should be sufficient.

To install dependencies:

    !pip install transformers==4.36.1 accelerate bitsandbytes torch datasets ibm-fms

The notebook server may need to be restarted at this point.

Next, log into huggingface (assuming an active virtualenv or conda env):

    !huggingface-cli login

Finally, you can run the below script.

NOTE: Currently this approach is very slow, taking >20 secs per document.
Something is likely wrong with the configuration (fp16 support? flash attn?)
"""

import torch

from fms.models import llama
from fms.models.hf.llama import modeling_llama_hf
from fms.models.hf.utils import register_fms_models


def get_flash_attn_model_and_tokenizer(model_name: str) -> tuple:
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    return model, tokenizer


model, tokenizer = get_flash_attn_model_and_tokenizer(
    model_name="meta-llama/Llama-2-7b-chat-hf"
)


# compile the model -- in HF, the decoder only
model = llama.convert_hf_llama(model)

# Adapt the FMS implementation back to the HF API, so it can be used in
# the huggingface ecosystem. Under the hood this is still the FMS
# implementation.
model = modeling_llama_hf.HFAdaptedLLaMAForCausalLM.from_fms_model(model)
model.to("cuda")
model = model.to(torch.half)
register_fms_models()

torch._inductor.config.joint_graph_constant_folding = False
model.decoder = torch.compile(model.decoder, dynamic=True)
print("Model device after compiling:")
print(next(model.parameters()).device)

import json
import random
import time
from typing import Tuple, List, Iterator

import torch
from datasets import load_dataset, load_metric
from transformers import AutoModelForCausalLM, AutoTokenizer


def get_document(data: dict) -> list:
    return data["article"]["document"]


def get_summary(data: dict) -> list:
    return data["article"]["summary"]


def has_document(data: dict) -> bool:
    return bool(get_document(data))


def load_sample(num_documents: int = 100) -> List[dict]:
    dataset = load_dataset("wiki_lingua", "english")
    dataset = dataset["train"]
    dataset = list(filter(has_document, dataset))
    return random.sample(dataset, num_documents)


def get_doc_summaries(dataset: List[dict]) -> Iterator[Tuple[str, str]]:
    for data in dataset:
        doc = get_document(data)[0]
        summary = get_summary(data)[0]
        yield doc, summary


def get_flash_attn_model_and_tokenizer(model_name: str) -> tuple:
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    return model, tokenizer


class LLM:
    """Wrapper for running LLM generation on a GPU"""

    def __init__(self):
        self.model = None
        self.tokenizer = None

    def _tokenize(self, text: str, **kwargs) -> torch.Tensor:
        return self.tokenizer.encode(
            text, return_tensors="pt", add_special_tokens=True, **kwargs
        )

    def tokenize(self, text: str, max_length: int) -> Tuple[torch.Tensor, int]:
        """
        Tokenizes the prompt for some document text and loads it into GPU memory.

        Returns a tuple with the tokenized prompt and number of tokens in the input text.
        """
        input_ids_prompt = self._tokenize(
            "\n\n\nSummarize the above text in 1-3 sentences."
        )
        max_doc_length = max_length - len(input_ids_prompt[0]) + 1
        input_ids_doc = self._tokenize(text, max_length=max_doc_length, truncation=True)
        text_tokens = len(input_ids_doc[0])
        input_ids = torch.cat((input_ids_doc, input_ids_prompt[:, 1:]), dim=1)
        input_ids = input_ids.cuda()
        return input_ids, text_tokens

    def generate(
        self, input_ids: torch.Tensor, max_new_tokens: int
    ) -> Tuple[str, float]:
        """
        Generates the LLM response and decodes the output.

        Returns a tuple with the text output and the inference time in seconds.
        """
        start_time = time.time()
        output_ids = self.model.generate(
            inputs=input_ids, max_new_tokens=max_new_tokens
        )[0]
        duration = time.time() - start_time
        new_tokens = len(output_ids) - len(input_ids[0])
        output = self.tokenizer.decode(
            output_ids[-new_tokens:], skip_special_tokens=True
        )
        return output, duration


model_name = "meta-llama/Llama-2-7b-chat-hf"
sample_docs = 10
max_new_tokens = 100
max_input_tokens = 4096
experiment = "torch-compile"

dataset = load_sample(sample_docs + 1)
example1 = dataset[0]

llm = LLM()
llm.model = model
llm.tokenizer = tokenizer

# Workaround: torch._dynamo config is updated to fix the following error:
# BackendCompilerFailed: backend='inductor' raised:
# LoweringException: AssertionError: ndim mismatch <function pow_native at 0x7b20d9110550> () [((s5 + 1)//2)]

import torch._dynamo

torch._dynamo.config.suppress_errors = True

# First pass on a compiled model is slow, so let's warm it up:
for _, (doc, gt_summary) in enumerate(get_doc_summaries([example1])):
    print(f"Warming up compiled model")
    input_ids, num_doc_tokens = llm.tokenize(doc, max_length=max_input_tokens)
    pred_summary, inference_secs = llm.generate(input_ids, max_new_tokens)

dataset = dataset[1:]

start = time.time()
data = []
for index, (doc, gt_summary) in enumerate(get_doc_summaries(dataset)):
    print(f"Generating output for {index}/{sample_docs} documents")
    input_ids, num_doc_tokens = llm.tokenize(doc, max_length=max_input_tokens)
    pred_summary, inference_secs = llm.generate(input_ids, max_new_tokens)
    data.append(
        {
            "doc": doc,
            "gt_summary": gt_summary,
            "pred_summary": pred_summary,
            "inference_secs": inference_secs,
            "num_doc_tokens": num_doc_tokens,
        }
    )

total_time = time.time() - start

durations = [item["inference_secs"] for item in data]
doc_token_counts = [item["num_doc_tokens"] for item in data]
avg_inf_time = sum(durations) / len(durations)
avg_doc_tokens = sum(doc_token_counts) / len(doc_token_counts)

print(f"Total runtime (s): {total_time}")
print(f"Avg. inference time (s): {avg_inf_time}")
print(f"Avg. doc tokens: {avg_doc_tokens}")

results = {
    "total_time": total_time,
    "avg_inf_time": avg_inf_time,
    "avg_doc_tokens": avg_doc_tokens,
}

results_file = "results.json"
with open(results_file, "w") as f:
    json.dump(results, f)

print(f"Wrote results to: {results_file}")

data_file = "data.json"
with open(data_file, "w") as f:
    json.dump(data, f)

print(f"Wrote data to: {data_file}")
