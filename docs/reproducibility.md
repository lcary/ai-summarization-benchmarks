# Reproducibility

See the below steps for reproducing the results.

## Setup

### Step 1. GPU Setup

This assumes you are running on a GPU instance with a Python version >= 3.10
and CUDA installed. Google Colab provides T4 GPU servers free of charge which
should be sufficient for some of the quantization experiments (e.g. 4-bit). That said,
since these experiments can be pretty memory-intensive, and Colab tends to shut down
free accounts on occasion, I recommend purchasing a Colab Pro subscription, which
also gives you a few hours to test on the A100 GPU server.

### Step 2. HuggingFace Account & Model Access

Second, this requires a HuggingFace account, signing Microsoft's terms & conditions,
and requesting access to Llama2 from the HuggingFace model hub:
https://huggingface.co/meta-llama/Llama-2-7b-chat-hf.
The access request takes a few hours to a few days for the
Microsoft team to process.

You will need to create an API token in HuggingFace at:
https://huggingface.co/settings/tokens

### Step 3. install dependencies in Colab:

```
pip install -r requirements.in
```

The pinned transformers and torch versions are required for some of the optimizations tested.
The notebook server may need to be restarted at this point.

### Step 4: Server-side HuggingFace Login

```
huggingface-cli login
```

This will prompt you for a HuggingFace access token which you created from your account.
After entering that in, finally, you can run the experiment script.

## Usage

The primary script to run without vLLM optimizations is `scripts/run_llama2_7b.py`.
To use this script, which evaluates the Llama2-7b Chat model on the WikiLingua dataset, run:

```
pip install transformers==4.36.1 torch==2.1.2 accelerate bitsandbytes datasets typer flash-attn
python scripts/run_llama2_7b.py
```

The script to run with the best results is `scripts/run_llama2_7b_vllm.py`.
For vLLM experiments, the steps are similar, with a different set of dependencies:
```
pip install torch==2.1.2 vllm autoawq transformers typer
python scripts/run_llama2_7b_vllm.py
```

Use the `--help` flag with each script for available options (e.g. model settings).

The full list of requirements on the A100 at time of recording of best results
can be found in the `a100-requirements.txt` file, which can be `pip` installed.

Each script was tested with the following versions:

 - Python 3.10.12
 - CUDA 12.2

## Analysis

See the analysis notebook at [`notebooks/Llama2 7b Benchmarks.ipynb`](`../notebooks/Llama2 7b Benchmarks.ipynb`)
which can be opened using Jupyter Notebook. The `notebooks/` folder has a set of requirements in `requirements.in`
for installing all Python library dependencies with `pip`, which is required for notebook cell execution.

Adding ROGUE-1 scores to the results is a necessary step to creating the analysis. This can be accomplished by
running the `python scripts/add_rogue_scores.py` script, which will update results in-place.
