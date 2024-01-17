# llama2-7b-summarization-benchmarks

Benchmarks testing the [Llama2-7b chat model](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf)
on random text summarization inference tasks from the [Wikilingua](https://huggingface.co/datasets/wiki_lingua)
dataset.

## Experiments

These benchmark experiments tested the time it took for Llama-7b to generate summaries on Wikilingua articles.
These tables are ordered by the number of documents processed per second by the model.

### Constraints

The below experiments were conducted with the following constraints:

 - Task: Text Summarization
 - Number of GPUs: 1
 - 300 maximum generation tokens
 - No input truncation

### Scenarios

The experiments tested multiple optimization scenarios using the Llama2-7b model:

 - `4-bit`: [4-bit Quantization](https://huggingface.co/blog/4bit-transformers-bitsandbytes)
 - `8-bit`: [8-bit Quantization](https://huggingface.co/blog/hf-bitsandbytes-integration)
 - `default`: [Default model loader](https://huggingface.co/docs/transformers/main/model_doc/llama2) (no optimizations)
 - `flash-attn`: [Flash Attention](https://huggingface.co/docs/transformers/perf_infer_gpu_one#flashattention-2)
 - `vllm-awq`: [vLLM with AutoAWQ quantization](https://docs.vllm.ai/en/latest/quantization/auto_awq.html)

### Hardware

For the GPU requirement, these experiments were run on the following instance types:

 - A100 GPU Server (40GB GPU RAM)
 - T4 GPU Server (15GB GPU RAM)

All scripts were run on [Google Colab](https://colab.research.google.com/drive/1H9rehbj9naQ-req4P0xRxFch3PxIklSZ#scrollTo=PFHXYM-S1QqG).

### Dataset

 - 100 randomly sampled documents from Wikilingua's `english` corpus (same set for all experiments)
 - Average document length for the sample: 515.4 tokens

For the set of sampled article URLs in this experiment dataset, see `data/100_articles.txt`.

### Results

See the report notebook at [`notebooks/BenchmarksReport.ipynb`](`./notebooks/BenchmarksReport.ipynb`)

The following experiments were run on A100 and T4 for the above set of documents in the Wikilingua dataset.

| Experiment ID   | Docs/Second | Instance Type | Experiment Script                          | ROGUE-1 F1 |
|-----------------|-------------|---------------|--------------------------------------------|------------|
| 25 `vllm-awq`   | 4.273       | A100 GPU      | `run_llama2_7b_vllm.py`                    | 0.234      |
| 21 `vllm-awq`   | 0.423       | T4 GPU        | `run_llama2_7b_vllm.py`                    | 0.233      |
| 16 `flash-attn` | 0.339       | A100 GPU      | `run_llama2_7b.py --experiment=flash-attn` | 0.282      |
| 17 `default`    | 0.317       | A100 GPU      | `run_llama2_7b.py --experiment=default`    | 0.280      |
| 22 `4-bit`      | 0.183       | T4 GPU        | `run_llama2_7b.py --experiment=4-bit`      | 0.251      |
| 23 `8-bit`      | 0.055       | T4 GPU        | `run_llama2_7b.py --experiment=8-bit`      | 0.275      |

 - **Summary**: `vLLM` optimization had the fastest throughput at >10x faster than any other strategy
   for the A100 GPU. The strategy was less profound on the lower-memory T4 GPU, but still resulted
   in a 2x speedup over other T4 experiments, and allowed the T4 to beat the other A100 experiments.
 - **Output & Results Data**: To see the data (output, duration) collected during inference, 
   find the JSON files with a given Experiment ID in its filename under the `data/` directory.
 - **Quantization**: quantization helped a lot with avoiding out of memory errors on the
   T4 instance type, which was less of a concern on the larger-memory A100.
 - **Memory Usage**: During the experiments, it was typical for T4 GPU memory
   usage to be around 10-15GB, whereas A100 GPU memory usage was around 30-40GB.
 - Many of the experiment types could not be run due to out of memory and compatibility issues. 
   For more details on failed experiments, see the Experiment Notes below.
 - Output quality was gauged by ROGUE-1 F1 scores, since quantization can affect how well
   the model performs on a given task.

## Additional Tests

See additional test results not listed above in `docs/other-tests.md`.

## Next Steps

 - [x] **All Combinations**: Test primary experiment scenarios on all GPU types. Requires purchasing additional GPU credits for the A100 experiments.
 - [x] **Same Datasets**: The experiments were originally run on different random samples of the data. It would be good to standardize the experiments across different instances using the same sample, via a fixed set of articles.
 - [ ] **Larger Experiments**: The experiments were run on small samples of the data. It would be good to scale up the experiments to 1000 records each. This requires renting additional GPUs.
 - [ ] **Mistral**: Running experiments on the Mistral LLM family and compare inference time / ROGUE scores.
 - [ ] **Orca**: Running experiments on the Orca LLM family and compare inference time / ROGUE scores.
 - [ ] **Knowledge Distillation**: One of the most promising avenues to reducing the memory requirements of LLMs is to leverage knowledge distillation (KD), such as is done for the [Orca series](https://arxiv.org/abs/2306.02707). Note that KD is 2-3 orders of magnitude more expensive than simply renting a larger GPU instance, requiring significant GPU runtime for producing the training dataset from the teacher model, and dozens to hundreds of hours on many GPUs for training the student model.

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
!huggingface-cli login
```

This will prompt you for a HuggingFace access token which you created from your account.
After entering that in, finally, you can run the experiment script.

## Usage

The primary script to run with the best results is `scripts/run_llama2_7b.py`.
To use this script, which evaluates the Llama2-7b Chat model on the WikiLingua dataset,
follow the instructions (e.g. `pip install` steps) in the script docstring, then run:

```
pip install transformers==4.36.1 torch==2.1.2 accelerate bitsandbytes datasets typer flash-attn
python scripts/run_llama2_7b.py
```

For vLLM experiments, the steps are similar, although there are different dependencies
to install listed at the top of the file:
```
pip install torch==2.1.2 vllm autoawq transformers typer
python scripts/run_llama2_7b_vllm.py
```

Use the `--help` flag with each script for available options (e.g. model settings).

Each script was tested with the following versions:

 - Python 3.10.12
 - CUDA 12.2

## Troubleshooting

### `UTF-8` locale issues

If you see an error in Colab like `NotImplementedError: A UTF-8 locale is required. Got ANSI_X3.4-1968`
you can fix this by entering the following into a cell and executing the block:
```
import locale
locale.getpreferredencoding = lambda: "UTF-8"
```
From https://github.com/googlecolab/colabtools/issues/3409#issuecomment-1446281277

### `libcuda` issues

If you see issues with torch or HF complaining about missing libraries, it's likely
an issue with the CUDA setup and the `ldconfig` configuration. Example error:
```
AssertionError: libcuda.so cannot found!
```

You will likely need to configure several settings for correct CUDA setup in Colab
by running the below commands:

```
!export CUDA_HOME=/usr/local/cuda-12.2
# Workaround: https://github.com/pytorch/pytorch/issues/107960
!ldconfig /usr/lib64-nvidia
!ldconfig -p | grep libcuda
```

The ldconfig command output should show libcuda.so, or else issues will occur,
If the ldconfig requires a different directory, check for other nvidia libraries
under /user/. If the notebook server has a different version of cuda home installed,
check for that via `ls /user/local/cuda*` and set that to CUDA_HOME. After that,
restart the session on the GPU server.
