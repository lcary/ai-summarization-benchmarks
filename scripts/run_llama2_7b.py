"""
This script uses the Llama2-7b chat model to summarize text documents from the Wikilingua dataset.

To update this script, change the default kwargs to the `main` method, such as another experiment
name for a valid key from the EXPERIMENTS variable. See the repo root readme for more information.
"""
import json
import math
import time
from pathlib import Path
from typing import Tuple, List, Iterator
from urllib.request import urlretrieve

import torch
import typer
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


EXPERIMENT_DATASET_FILE = Path("data/100_articles.txt")
if not EXPERIMENT_DATASET_FILE.exists():
    EXPERIMENT_DATASET_FILE.parent.mkdir(exist_ok=True)
    DATASET = "https://raw.githubusercontent.com/lcary/ai-summarization-benchmarks/main/data/100_articles.txt"
    urlretrieve(DATASET, EXPERIMENT_DATASET_FILE)
EXPERIMENT_DATASET_URLS = set(EXPERIMENT_DATASET_FILE.read_text().split())


def get_document(data: dict) -> list:
    return data["article"]["document"]


def get_summary(data: dict) -> list:
    return data["article"]["summary"]


def get_url(data: dict) -> list:
    return data["url"]


def is_in_experiment_dataset(data: dict) -> bool:
    return get_url(data) in EXPERIMENT_DATASET_URLS


def load_sample(num_docs: int) -> List[dict]:
    dataset = load_dataset("wiki_lingua", "english")
    dataset = dataset["train"]
    dataset = list(filter(is_in_experiment_dataset, dataset))
    return dataset[:num_docs]


Example = Tuple[str, str, str]


def get_examples(dataset: List[dict]) -> Iterator[Example]:
    for data in dataset:
        doc = get_document(data)[0]
        summary = get_summary(data)[0]
        url = get_url(data)
        yield doc, summary, url


def get_default_model_and_tokenizer(model_name: str) -> tuple:
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    return model, tokenizer


def get_4bit_config() -> BitsAndBytesConfig:
    return BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)


def get_4bit_model_and_tokenizer(model_name: str) -> tuple:
    model = AutoModelForCausalLM.from_pretrained(
        model_name, device_map="auto", quantization_config=get_4bit_config()
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    return model, tokenizer


def get_8bit_model_and_tokenizer(model_name: str) -> tuple:
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        load_in_8bit=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    return model, tokenizer


def get_flash_attn_model_and_tokenizer(model_name: str) -> tuple:
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    return model, tokenizer


def get_flash_4bit_model_and_tokenizer(model_name: str) -> tuple:
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        quantization_config=get_4bit_config(),
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    return model, tokenizer


EXPERIMENTS = {
    "default": get_default_model_and_tokenizer,
    "flash-attn": get_flash_attn_model_and_tokenizer,
    "4-bit": get_4bit_model_and_tokenizer,
    "8-bit": get_8bit_model_and_tokenizer,
    "flash-4bit": get_flash_4bit_model_and_tokenizer,
}


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class LLM:
    """Wrapper for running LLM generation on a GPU"""

    PROMPT = "\n\n\nSummarize the above text in 1-3 sentences."

    def __init__(self):
        self.model = None
        self.tokenizer = None

    def load_model_and_tokenizer(self, model_name: str, experiment: str):
        """Loads the model and tokenizer into memory."""
        model, tokenizer = EXPERIMENTS[experiment](model_name)
        self.model = model
        self.tokenizer = tokenizer

    def _tokenize(self, text: str, **kwargs) -> torch.Tensor:
        return self.tokenizer.encode(
            text, return_tensors="pt", add_special_tokens=True, **kwargs
        )

    def tokenize(self, doc: str, max_length: int) -> Tuple[torch.Tensor, int]:
        """
        Tokenizes the prompt for some document text and loads it into GPU memory.

        Returns a tuple with the tokenized prompt and number of tokens in the input text.
        """
        input_ids_prompt = self._tokenize(self.PROMPT)
        max_doc_length = max_length - len(input_ids_prompt[0]) + 1
        input_ids_doc = self._tokenize(doc, max_length=max_doc_length, truncation=True)
        num_doc_tokens = len(input_ids_doc[0])
        input_ids = torch.cat((input_ids_doc, input_ids_prompt[:, 1:]), dim=1)
        input_ids = input_ids.to(device)
        return input_ids, num_doc_tokens

    def batch_tokenize(
        self, batch_docs: List[str], max_length: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        concatenated_docs = [doc + self.PROMPT for doc in batch_docs]

        inputs = self.tokenizer.batch_encode_plus(
            concatenated_docs,
            max_length=max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

        batch_input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        return batch_input_ids, attention_mask

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
        inference_time = time.time() - start_time
        new_tokens = len(output_ids) - len(input_ids[0])
        output = self.tokenizer.decode(
            output_ids[-new_tokens:], skip_special_tokens=True
        )
        return output, inference_time

    def batch_generate(
        self,
        batch_input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        max_new_tokens: int,
    ) -> Tuple[List[str], float]:
        start_time = time.time()
        batch_output_ids = self.model.generate(
            input_ids=batch_input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
        )
        inference_time = time.time() - start_time

        outputs = []
        for input_ids, output_ids in zip(batch_input_ids, batch_output_ids):
            output = self.tokenizer.decode(
                output_ids[len(input_ids) :], skip_special_tokens=True
            )
            outputs.append(output)

        return outputs, inference_time


def batch_process(
    llm: LLM,
    batch: List[Example],
    max_input_tokens: int,
    max_new_tokens: int,
    batch_id: int,
) -> Iterator[dict]:
    batch_docs = [item[0] for item in batch]
    batch_tokens = []
    for doc in batch_docs:
        _, num_doc_tokens = llm.tokenize(doc, max_length=max_input_tokens)
        batch_tokens.append(num_doc_tokens)
    batch_input_ids, attention_mask = llm.batch_tokenize(
        batch_docs, max_length=max_input_tokens
    )
    batch_outputs, inference_time = llm.batch_generate(
        batch_input_ids, attention_mask, max_new_tokens
    )
    torch.cuda.empty_cache()
    for pred_summary, example, num_doc_tokens in zip(
        batch_outputs, batch, batch_tokens
    ):
        (doc, gt_summary, url) = example
        yield {
            "doc": doc,
            "url": url,
            "gt_summary": gt_summary,
            "pred_summary": pred_summary,
            "batch_inference_time": inference_time,
            "num_doc_tokens": num_doc_tokens,
            "batch_id": batch_id,
        }


def serial_inference(llm, dataset, max_input_tokens, max_new_tokens, num_docs):
    data = []
    for index, (doc, gt_summary, url) in enumerate(get_examples(dataset)):
        print(f"Generating output for {index}/{num_docs} documents")
        input_ids, num_doc_tokens = llm.tokenize(doc, max_length=max_input_tokens)
        pred_summary, inference_time = llm.generate(input_ids, max_new_tokens)
        data.append(
            {
                "doc": doc,
                "url": url,
                "gt_summary": gt_summary,
                "pred_summary": pred_summary,
                "inference_time": inference_time,
                "num_doc_tokens": num_doc_tokens,
            }
        )
    durations = [item["inference_time"] for item in data]
    avg_sec_per_doc = sum(durations) / num_docs
    return data, avg_sec_per_doc


def get_batches(dataset, batch_size, num_docs):
    batches = []
    batch = []
    num_batches = math.ceil(num_docs / batch_size)
    for example in get_examples(dataset):
        batch.append(example)
        if len(batch) == batch_size:
            batches.append(batch)
            batch = []
        else:
            continue
    if batch:
        batches.append(batch)
    return batches, num_batches


def batch_inference(
    llm, dataset, batch_size, max_input_tokens, max_new_tokens, num_docs
):
    llm.tokenizer.pad_token = llm.tokenizer.eos_token
    data = []
    batches, num_batches = get_batches(dataset, batch_size, num_docs)
    for batch_id, batch in enumerate(batches):
        print(
            f"Generating output for batch {batch_id + 1}/{num_batches} of all documents"
        )
        for output in batch_process(
            llm, batch, max_input_tokens, max_new_tokens, batch_id
        ):
            data.append(output)
    duration_pairs = set(
        (item["batch_id"], item["batch_inference_time"]) for item in data
    )
    durations = [item[1] for item in duration_pairs]
    avg_sec_per_doc = sum(durations) / num_docs
    return data, avg_sec_per_doc


def main(
    batch_size: int = 1,  # Batching disabled by default
    experiment: str = "default",
    max_new_tokens: int = 300,
    max_input_tokens: int = 4096,
    model_name: str = "meta-llama/Llama-2-7b-chat-hf",
    num_docs: int = 100,
):
    assert batch_size > 0

    dataset = load_sample(num_docs)

    llm = LLM()
    llm.load_model_and_tokenizer(model_name, experiment)

    start = time.time()
    if batch_size == 1:
        data, avg_sec_per_doc = serial_inference(
            llm, dataset, max_input_tokens, max_new_tokens, num_docs
        )
    else:
        data, avg_sec_per_doc = batch_inference(
            llm, dataset, batch_size, max_input_tokens, max_new_tokens, num_docs
        )

    total_time = time.time() - start

    doc_token_counts = [item["num_doc_tokens"] for item in data]
    avg_doc_tokens = sum(doc_token_counts) / num_docs

    avg_doc_per_sec = 1 / avg_sec_per_doc if avg_sec_per_doc else 0.0

    print(f"Total runtime (sec): {total_time}")
    print(f"Avg. inference time (sec/doc): {avg_sec_per_doc}")
    print(f"Avg. throughput (docs/sec): {avg_doc_per_sec}")
    print(f"Avg. doc tokens: {avg_doc_tokens}")

    results = {
        "total_time": total_time,
        "total_docs": num_docs,
        "avg_doc_per_sec": 1 / avg_sec_per_doc,
        "avg_sec_per_doc": avg_sec_per_doc,
        "avg_doc_tokens": avg_doc_tokens,
    }

    batch_suffix = f"_batchsize-{batch_size}" if batch_size > 1 else ""
    experiment_suffix = f"{experiment}{batch_suffix}"

    results_file = f"results_{experiment_suffix}.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Wrote results to: {results_file}")

    # Remove doc text before saving:
    for item in data:
        del item["doc"]

    data_file = f"data_{experiment_suffix}.json"
    with open(data_file, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Wrote data to: {data_file}")


if __name__ == "__main__":
    typer.run(main)
