"""
This experiment summarizes text documents with the BART model
for a baseline.
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

models = ["slauw87/bart_summarisation"]


def load_pipeline(model_name):
    summarizer = pipeline("summarization", model=model_name, max_length=100)
    return summarizer


def summarize(text, summarizer):
    return summarizer(text)


def evaluate_summary(original, summary):
    rouge = load_metric("rouge")
    scores = rouge.compute(
        predictions=[summary], references=[original], rouge_types=["rouge1"]
    )["rouge1"].mid
    return scores.fmeasure  # F1 score


def run_experiments(models, data):
    results = []

    for model_name in models:
        summarizer = load_pipeline(model_name)

        for item in data:
            original_text = item["article"]["document"]

            ground_truth_summary = item["article"]["summary"]

            start_time = time.time()
            generated_summary = summarize(original_text, summarizer)
            inference_time = time.time() - start_time
            f1_score = evaluate_summary(ground_truth_summary, generated_summary)

            result = {
                "model": model_name,
                "f1_score": f1_score,
                "inference_time": inference_time,
                "ground_truth_f1": evaluate_summary(
                    ground_truth_summary, original_text
                ),
            }
            print(result)
            results.append(result)

    return pd.DataFrame(results)


def main():
    wikilingua_dataset = load_dataset("wiki_lingua", "english")

    wikilingua_sample = random.sample(list(wikilingua_dataset["train"]), 100)

    df_results = run_experiments(models, wikilingua_sample)

    os.makedirs("data/", exist_ok=True)
    df_results.to_csv("data/summarization_results.csv", index=False)
    print(df_results)


if __name__ == "__main__":
    main()
