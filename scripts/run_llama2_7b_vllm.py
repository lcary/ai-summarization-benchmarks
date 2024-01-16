"""
This script uses zero-shot prompting with the Llama2-7b chat model
to summarize text documents from the Wikilingua dataset.

Follow setup steps in repo root readme, then run:

    !pip install torch==2.1.2 vllm autoawq transformers typer

Note that this script currently errors out with the following message:

    ValueError: The model's max seq len (4096) is larger than the maximum number of tokens
    that can be stored in KV cache (64). Try increasing `gpu_memory_utilization` or
    decreasing `max_model_len` when initializing the engine.

This occurs whether using the base 7b model or the 7b chat model, for lengths tested
(512, 1024, 2048, 4096).

With max_model_len=128, we see a different error:

    ValueError: No available memory for the cache blocks.
    Try increasing `gpu_memory_utilization` when initializing the engine.

See https://github.com/vllm-project/vllm/issues/2418

The optimal GPU memory and model length settings combination I was able to achieve
for maximum model sequence (prompt + output) generation w/out quantization was with the following:

    llm = LLM(
        model="meta-llama/Llama-2-7b-chat-hf",
        max_model_len=1024,
        gpu_memory_utilization=0.93
    )

Note: this processes the documents at a throughput of 3.091 docs/second but runs with lots of errors like:

    WARNING 01-16 16:05:10 scheduler.py:149] Input prompt (2498 tokens) is too long and exceeds limit of 1024

To workaround such errors, we quantize the model with autoawq:
https://docs.vllm.ai/en/latest/quantization/auto_awq.html

The supported autoawq library is under-optimized, decreasing memory consumption at the expense of latency and quality.
Results for processing 100 documents:
 - Total time (sec): 233.35278129577637
 - Docs/sec: 0.4285357108011036
"""
import json
import os
from pathlib import Path
from typing import Tuple, List

import torch
import typer
from datasets import load_dataset

import time

from vllm import LLM
from vllm import SamplingParams
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

EXPERIMENT_DATASET_FILE = Path("data/100_articles.txt")
EXPERIMENT_DATASET_URLS = set(EXPERIMENT_DATASET_FILE.read_text().split())

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def get_url(item: dict) -> list:
    return item["url"]


def get_doc(item: dict) -> list:
    return item["article"]["document"]


def is_in_experiment_dataset(item: dict) -> bool:
    return get_url(item) in EXPERIMENT_DATASET_URLS


def load_sample(num_docs: int) -> list:
    dataset = load_dataset("wiki_lingua", "english")
    dataset = dataset["train"]
    dataset = list(filter(is_in_experiment_dataset, dataset))
    return dataset[:num_docs]


def tokenize(tokenizer, prompt: str, max_length: int) -> Tuple[List[torch.Tensor], int]:
    """
    Tokenizes the prompt for some document text and loads it into GPU memory.

    Returns a tuple with the tokenized prompt and number of tokens in the input text.
    """
    inputs = tokenizer.encode(
        prompt,
        max_length=max_length,
        truncation=True,
        return_tensors="pt",
        add_special_tokens=True,
    )
    num_tokens = len(inputs[0])
    inputs = inputs.to(device)
    return inputs, num_tokens


def run_pipeline(
    model_name: str, prompts: List[str], temperature: float, max_tokens: int
) -> Tuple[List[str], float]:
    llm = LLM(model=model_name, quantization="AWQ")
    start = time.time()
    sampling_params = SamplingParams(temperature=temperature, max_tokens=max_tokens)
    outputs = llm.generate(prompts, sampling_params)
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}")
        print(f"Generated text: {generated_text!r}")
    total_duration = time.time() - start
    return outputs, total_duration


def run_serial_inference(
    model_name: str,
    prompts: List[str],
    temperature: float,
    max_input_length: int,
    max_tokens: int,
    token_counts: list,
    durations: list,
) -> Tuple[List[str], float]:
    model = AutoAWQForCausalLM.from_quantized(model_name, fuse_layers=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    start = time.time()
    outputs = []
    for prompt in prompts:
        inputs, num_tokens = tokenize(tokenizer, prompt, max_input_length)
        inf_start_time = time.time()
        output_ids = model.generate(
            inputs=inputs, max_new_tokens=max_tokens, temperature=temperature
        )[0]
        individual_duration = time.time() - inf_start_time
        new_tokens = len(output_ids) - len(inputs[0])
        output = tokenizer.decode(output_ids[-new_tokens:], skip_special_tokens=True)
        outputs.append(output)
        durations.append(individual_duration)
        token_counts.append(num_tokens)
        print(f"Generated text: {output!r}")
    total_duration = time.time() - start
    return outputs, total_duration


def main(
    temperature: float = 0.8,
    max_tokens: int = 300,
    max_input_length: int = 4096,
    model_name: str = "TheBloke/Llama-2-7b-Chat-AWQ",
    dynamic_batching: bool = True,
    num_docs: int = 100,
):
    docs = load_sample(num_docs)
    prompt_template = (
        "Document: '{}' .\n\nSummary of the above Document in 1-3 sentences: '"
    )
    prompts = [prompt_template.format(get_doc(item)[0]) for item in docs]

    durations = []
    token_counts = []
    if dynamic_batching:
        outputs, total_duration = run_pipeline(
            model_name, prompts, temperature, max_tokens
        )
    else:
        outputs, total_duration = run_serial_inference(
            model_name,
            prompts,
            temperature,
            max_input_length,
            max_tokens,
            token_counts,
            durations,
        )

    total = len(prompts)
    print(f"Total time (sec): {total_duration}")
    avg_doc_per_sec = total / total_duration
    print(f"Docs/sec: {avg_doc_per_sec}")

    gt_summaries = [item["article"]["summary"][0] for item in docs]

    data = []
    if dynamic_batching:
        for item, output, gt_summary in zip(docs, outputs, gt_summaries):
            data.append(
                {
                    "url": get_url(item),
                    "gt_summary": gt_summary,
                    "pred_summary": output.outputs[0].text,
                }
            )
    else:
        for item, output, gt_summary, duration, num_tokens in zip(
            docs, outputs, gt_summaries, durations, token_counts
        ):
            data.append(
                {
                    "url": get_url(item),
                    "gt_summary": gt_summary,
                    "pred_summary": output,
                    "inference_time": duration,
                    "num_doc_tokens": num_tokens,
                }
            )

    results = {
        "total_time": total_duration,
        "total_docs": total,
        "avg_doc_per_sec": avg_doc_per_sec,
        "avg_sec_per_doc": 1 / avg_doc_per_sec,
    }

    results_file = f"results_vllm.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Wrote results to: {results_file}")

    data_file = f"data_vllm.json"
    with open(data_file, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Wrote data to: {data_file}")


if __name__ == "__main__":
    typer.run(main)
