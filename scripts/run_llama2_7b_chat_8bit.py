"""
Overview

This script uses a one-shot prompt with the Llama2-7b chat model (loaded
in 4bit quantized format for inference optimization)
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
    !python -c 'from datasets import load_metric; load_metric("rouge")' || pip install rouge_score


The notebook server may need to be restarted at this point.

Next, log into huggingface (assuming an active virtualenv or conda env):

    !huggingface-cli login

Finally, you can run the below script.
"""
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


rouge = load_metric("rouge")

def evaluate_summary(docs: List[str], summaries: List[str]) -> float:
    scores = rouge.compute(
        predictions=summaries, references=docs, rouge_types=["rouge1"]
    )["rouge1"].mid
    return scores.fmeasure

class LLM:
    """ Wrapper for running LLM generation on a GPU """
    def __init__(self):
        self.model = None
        self.tokenizer = None

    def load_model_and_tokenizer(self, model_name: str):
        """ Loads the model and tokenizer into memory. """
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map='auto',
            load_in_8bit=True,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    def _tokenize(self, text: str, **kwargs) -> torch.Tensor:
        return self.tokenizer.encode(text, return_tensors="pt", add_special_tokens=True, **kwargs)

    def tokenize(self, text: str, max_length: int) -> Tuple[torch.Tensor, int]:
        """
        Tokenizes the prompt for some document text and loads it into GPU memory.

        Returns a tuple with the tokenized prompt and number of tokens in the input text.
        """
        input_ids_prompt = self._tokenize("\n\n\nSummarize the above text in 1-3 sentences.")
        max_doc_length = max_length - len(input_ids_prompt[0]) + 1
        input_ids_doc = self._tokenize(text, max_length=max_doc_length, truncation=True)
        text_tokens = len(input_ids_doc[0])
        input_ids = torch.cat((input_ids_doc, input_ids_prompt[:, 1:]), dim=1)
        input_ids = input_ids.cuda()
        return input_ids, text_tokens

    def generate(self, input_ids: torch.Tensor, max_new_tokens: int) -> Tuple[str, float]:
        """
        Generates the LLM response and decodes the output.

        Returns a tuple with the text output and the inference time in seconds.
        """
        start_time = time.time()
        output_ids = self.model.generate(inputs=input_ids, max_new_tokens=max_new_tokens)[0]
        duration = time.time() - start_time
        new_tokens = len(output_ids) - len(input_ids[0])
        output = self.tokenizer.decode(output_ids[-new_tokens:], skip_special_tokens=True)
        return output, duration

def main():
    model_name = "meta-llama/Llama-2-7b-chat-hf"
    sample_docs = 100
    max_new_tokens = 100
    max_input_tokens = 4096

    dataset = load_sample(sample_docs)

    llm = LLM()
    llm.load_model_and_tokenizer(model_name)

    start = time.time()
    data = []
    for index, (doc, gt_summary) in enumerate(get_doc_summaries(dataset)):
        print(f"Generating output for {index}/{sample_docs} documents")
        input_ids, num_doc_tokens = llm.tokenize(doc, max_length=max_input_tokens)
        pred_summary, inference_secs = llm.generate(input_ids, max_new_tokens)
        data.append({
            "doc": doc,
            "gt_summary": gt_summary,
            "pred_summary": pred_summary,
            "inference_secs": inference_secs,
            "num_doc_tokens": num_doc_tokens
        })

    total_time = time.time() - start

    docs = [item["doc"] for item in data]
    gt_summaries = [item["gt_summary"] for item in data]
    gt_score = evaluate_summary(docs, gt_summaries)
    pred_summaries = [item["pred_summary"] for item in data]
    pred_score = evaluate_summary(docs, pred_summaries)

    durations = [item["inference_secs"] for item in data]
    doc_token_counts = [item["num_doc_tokens"] for item in data]
    avg_inf_time = sum(durations) / len(durations)
    avg_doc_tokens = sum(doc_token_counts) / len(doc_token_counts)

    print(f"Total runtime (s): {total_time}")
    print(f"Avg. inference time (s): {avg_inf_time}")
    print(f"Avg. doc tokens: {avg_doc_tokens}")
    print(f"Ground truth ROGUE-1 f1 score: {gt_score}")
    print(f"Prediction ROGUE-1 f1 score: {pred_score}")

    results = {
        "total_time": total_time,
        "avg_inf_time": avg_inf_time,
        "avg_doc_tokens": avg_doc_tokens,
        "gt_score": gt_score,
        "pred_score": pred_score
    }

    results_file = "results.json"
    with open(results_file, "w") as f:
        json.dump(results, f)

    print(f"Wrote results to: {results_file}")

    data_file = "data.json"
    with open(data_file, "w") as f:
        json.dump(data, f)

    print(f"Wrote data to: {data_file}")


if __name__ == "__main__":
    main()
