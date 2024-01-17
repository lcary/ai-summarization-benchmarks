"""
This will update to add avg_doc_per_sec to legacy results files.
Usage:

    python scripts/update_old_results.py

"""
import json
from pathlib import Path
from typing import Optional

import typer


def read_json(path: Path):
    print(f"Reading {path}")
    with path.open() as f:
        return json.load(f)


def update_results(result_path):
    results = read_json(result_path)
    orig_path = str(result_path)
    if "avg_inf_time" in results and "avg_doc_per_sec" not in results:
        results["avg_doc_per_sec"] = 1 / results["avg_inf_time"]
        backup = orig_path + ".bak"
        print(f"Saved results backup to: {backup}")
        result_path.rename(backup)
        with open(orig_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Wrote throughput to: {orig_path}")


def main(experiment: Optional[str] = None):
    pattern = f"result_{experiment}_*.json" if experiment else "results_*.json"
    files = list(Path("data/").glob(pattern))
    print(f"Updating {len(files)} results...")
    errors = False
    for filepath in files:
        try:
            update_results(filepath)
        except (KeyError, AssertionError) as e:
            print(f"Unable to update results for {filepath} due to an error: {str(e)}")
            errors = True
    if errors:
        raise RuntimeError("An unexpected error occurred. See above.")
    print("All results have been successfully updated.")


if __name__ == "__main__":
    typer.run(main)
