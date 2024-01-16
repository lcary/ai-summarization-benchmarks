"""
Adds ROGUE-1 F1 scores for each experiment data file in the `data/` folder.

To run this script, run `pip install evaluate rogue-score`, then:

    python scripts/add_rogue_scores.py

This will create backups of all results files and update rogue scores in-place.
"""
import json
from pathlib import Path
from typing import List, Optional

import evaluate
import typer

print("Loading ROGUE metric from the evaluate library...")
rouge = evaluate.load("rouge")


def calculate_rogue1(ground_truth: List[str], predictions: List[str]) -> float:
    score = rouge.compute(
        predictions=predictions, references=ground_truth, rouge_types=["rouge1"]
    )["rouge1"]
    return float(score)


def evaluate_summaries(data: List[dict]) -> float:
    ground_truth = []
    predictions = []
    for item in data:
        ground_truth.append(item["gt_summary"])
        predictions.append(item["pred_summary"])

    return calculate_rogue1(ground_truth, predictions)


def read_json(path: Path):
    print(f"Reading {path}")
    with path.open() as f:
        return json.load(f)


def update_results(data_file):
    data_dir = data_file.parent
    rogue1 = evaluate_summaries(read_json(data_file))
    result_file = data_file.name.replace("data_", "results_")
    result_path = data_dir / result_file
    assert result_path.exists(), f"Results file {result_path} is missing"
    results = read_json(result_path)
    orig_path = str(result_path)
    backup = orig_path + ".bak"
    result_path.rename(backup)
    print(f"Saved results backup to: {backup}")
    results["pred_score"] = rogue1
    if "gt_score" in results:
        del results["gt_score"]
    with open(orig_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote rogue score to: {orig_path}")


def main(experiment: Optional[str] = None):
    pattern = f"data_{experiment}_*.json" if experiment else "data_*.json"
    data_files = list(Path("data/").glob(pattern))
    print(f"Updating {len(data_files)} results...")
    errors = False
    for data_file in data_files:
        try:
            update_results(data_file)
        except (KeyError, AssertionError) as e:
            print(f"Unable to update results for {data_file} due to an error: {str(e)}")
            errors = True
    if errors:
        raise RuntimeError("An unexpected error occurred. See above.")
    print("All results have been successfully updated.")


if __name__ == "__main__":
    typer.run(main)
