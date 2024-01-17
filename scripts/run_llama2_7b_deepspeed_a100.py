"""
Runs llama2b inference on a A100 GPU.

    pip install datasets deepspeed-mii typer

Or install from repo root `requirements.txt`

This script can be run with:

    python scripts/run_llama2_7b_deepspeed_a100.py

This script currently raises a warning upon execution:

    Deadlock detected. Resetting KV cache and recomputing requests. Consider limiting number of
    concurrent requests or decreasing max lengths of prompts/generations.

This results in the generation of garbage responses (see `data_
However, the script completes execution at high throughput. This suggests some optimal configuration
of concurrency settings might improve throughput further and fix the issue.
"""
import json
import os
import time
from pathlib import Path

import mii
import typer
from datasets import load_dataset

EXPERIMENT_ARTICLES_FILE = Path(
    os.getenv("EXPERIMENT_ARTICLES_FILE", "data/100_articles.txt")
)
EXPERIMENT_ARTICLES = set(EXPERIMENT_ARTICLES_FILE.read_text().split())
PROMPT_TEMPLATE = (
    "Document: '{}' .\n\nSummary of the above Document in 1-3 sentences: '"
)


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


def main(
    max_new_tokens: int = 300,
    model_name: str = "meta-llama/Llama-2-7b-chat-hf",
    num_docs: int = 100,
):
    pipe = mii.pipeline(model_name)

    docs = load_sample(num_docs)
    prompts = [PROMPT_TEMPLATE.format(get_doc(item)[0]) for item in docs]

    start = time.time()
    response = pipe(prompts, max_new_tokens=max_new_tokens, max_length=4096)
    total_duration = time.time() - start

    total = len(prompts)
    print(f"Total time (sec): {total_duration}")
    avg_doc_per_sec = total / total_duration
    print(f"Docs/sec: {avg_doc_per_sec}")

    gt_summaries = [item["article"]["summary"][0] for item in docs]

    data = []
    for item, output, gt_summary in zip(docs, response, gt_summaries):
        data.append(
            {
                "url": get_url(item),
                "gt_summary": gt_summary,
                "pred_summary": output.generated_text,
            }
        )

    results = {
        "total_time": total_duration,
        "total_docs": total,
        "avg_doc_per_sec": avg_doc_per_sec,
        "avg_sec_per_doc": 1 / avg_doc_per_sec,
    }

    results_file = f"results_deepspeed.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Wrote results to: {results_file}")

    data_file = f"data_deepspeed.json"
    with open(data_file, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Wrote data to: {data_file}")

    pipe.destroy()


if __name__ == "__main__":
    typer.run(main)
