"""
This script uses zero-shot prompting with the Llama2-7b chat model
to summarize text documents from the Wikilingua dataset.

Follow setup steps in repo root readme, then run:

    pip install torch==2.1.2 vllm autoawq transformers typer

Usage:

    python scripts/run_llama2_7b_vllm.py

Docs: https://docs.vllm.ai/en/latest/quantization/auto_awq.html
"""
import json
import os
import time
from pathlib import Path
from typing import Tuple, List

import torch
import typer
from datasets import load_dataset
from vllm import LLM
from vllm import SamplingParams

EXPERIMENT_ARTICLES_FILE = Path(os.getenv("EXPERIMENT_ARTICLES_FILE", "data/100_articles.txt"))
EXPERIMENT_ARTICLES = set(EXPERIMENT_ARTICLES_FILE.read_text().split())

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def get_url(item: dict) -> list:
    return item["url"]


def get_doc(item: dict) -> list:
    return item["article"]["document"]


def is_in_experiment_dataset(item: dict) -> bool:
    return get_url(item) in EXPERIMENT_ARTICLES


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
    inputs = tokenizer.encode(prompt, max_length=max_length, truncation=True, return_tensors='pt', add_special_tokens=True)
    num_tokens = len(inputs[0])
    inputs = inputs.to(device)
    return inputs, num_tokens

def run_pipeline(model_name: str, prompts: List[str], max_tokens: int) -> Tuple[List[str], float]:
    """
    Runs the text summarization inference pipeline on a set of prompts.
    """
    llm = LLM(model=model_name, quantization="AWQ")
    start = time.time()
    sampling_params = SamplingParams(max_tokens=max_tokens)
    outputs = llm.generate(prompts, sampling_params)
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}")
        print(f"Generated text: {generated_text!r}")
    total_duration = time.time() - start
    return outputs, total_duration


def main(
    max_tokens:int  = 300,
    model_name: str = "TheBloke/Llama-2-7b-Chat-AWQ",
    num_docs: int = 100,
):

    docs = load_sample(num_docs)

    prompt_template = "Document: '{}' .\n\nSummary of the above Document in 1-3 sentences: '"

    prompts = [prompt_template.format(get_doc(item)[0]) for item in docs]

    outputs, total_duration = run_pipeline(model_name, prompts, max_tokens)

    total = len(prompts)
    print(f"Total time (sec): {total_duration}")
    avg_doc_per_sec = total / total_duration
    print(f"Docs/sec: {avg_doc_per_sec}")

    gt_summaries = [item["article"]["summary"][0] for item in docs]

    data = []
    for item, output, gt_summary in zip(docs, outputs, gt_summaries):
        data.append(
            {
                "url": get_url(item),
                "gt_summary": gt_summary,
                "pred_summary": output.outputs[0].text,
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
