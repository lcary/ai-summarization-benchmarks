"""
NOTE: This experiment does not run on Colab's T4 due to OOM issues.
"""

import time
import os
import pandas as pd
import random
from datasets import load_dataset, load_metric
from transformers import (
    pipeline,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    BertTokenizerFast,
)
from langchain.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model = "togethercomputer/LLaMA-2-7B-32K"


def load_pipeline(model_name):
    # https://discuss.huggingface.co/t/is-transformers-using-gpu-by-default/8500/3
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, trust_remote_code=True, torch_dtype=torch.float16
    )

    return model, tokenizer


def summarize(text, model, tokenizer):
    input_context = f"Summarize the following text: {text}"
    input_ids = tokenizer.encode(input_context, return_tensors="pt")
    output = model.generate(input_ids, max_length=128, temperature=0.7)
    output_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return output_text


def evaluate_summary(original, summary):
    rouge = load_metric("rouge")
    scores = rouge.compute(
        predictions=[summary], references=[original], rouge_types=["rouge1"]
    )["rouge1"].mid
    return scores.fmeasure  # F1 score


def run_experiments(model_name, data):
    model, tokenizer = load_pipeline(model_name)
    results = []

    for item in data:
        original_text = item["article"]["document"]

        ground_truth_summary = item["article"]["summary"]

        start_time = time.time()
        generated_summary = summarize(original_text, model, tokenizer)
        inference_time = time.time() - start_time
        f1_score = evaluate_summary(ground_truth_summary, generated_summary)

        result = {
            "model": model_name,
            "f1_score": f1_score,
            "inference_time": inference_time,
            "ground_truth_f1": evaluate_summary(ground_truth_summary, original_text),
        }
        print(result)
        results.append(result)

    return pd.DataFrame(results)


def main():
    wikilingua_dataset = load_dataset("wiki_lingua", "english")
    wikilingua_sample = random.sample(list(wikilingua_dataset["train"]), 100)

    df_results = run_experiments(model, wikilingua_sample)

    os.makedirs("data/", exist_ok=True)
    df_results.to_csv("data/summarization_results.csv", index=False)
    print(df_results)


if __name__ == "__main__":
    main()
