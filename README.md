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
 - `flash-4bit`: Combined Flash Attention w/ 4-bit quantization
 - `flash-attn`: [Flash Attention](https://huggingface.co/docs/transformers/perf_infer_gpu_one#flashattention-2)
 - `vllm-awq`: [vLLM with AutoAWQ quantization](https://docs.vllm.ai/en/latest/quantization/auto_awq.html)

### Hardware

For the GPU requirement, these experiments were run on the following instance types:

 - A100 GPU Server (40GB GPU RAM)
 - T4 GPU Servers (15GB GPU RAM)

All scripts were run on [Google Colab](https://colab.research.google.com/drive/1H9rehbj9naQ-req4P0xRxFch3PxIklSZ#scrollTo=PFHXYM-S1QqG).

### Dataset

 - 100 randomly sampled documents from Wikilingua's `english` corpus (same set for all experiments)
 - Average document length for the sample: 515.4 tokens

For the set of sampled article URLs in this experiment dataset, see `data/100_articles.txt`.

### Results

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
   for the A100 GPU. The strategy was less profound on the lower-memory T4 GPU, and was followed by
   several other strategies.
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

### Other Benchmarks

#### A100 Batching Tests

For the two GPU instance types tested, batching prompts did not impact results nearly as much
as installing and running optimization frameworks like `vllm` (which does some automatic batching).
Batching resulted in OOM errors all sizes on the T4 and later
and at size 4 or greater for the A100 for more than 10 docs (see Failed Experiments).
Note that batching was only possible when using Flash Attention on the A100
after a size 4 documents per batch even for 10 documents, otherwise OOMs occurred.
Note: Some of the below batching tests below were on different sets of documents. Since the A100
experiments are not all on the same dataset, performance needs to be interpreted with a grain of salt. 

| Experiment ID   | Batch Size | Docs/Second | Total Documents | Avg. tokens/doc | ROGUE-1 F1 |
|-----------------|------------|-------------|-----------------|-----------------|------------|
| 15 `flash-attn` | 10         | 0.861       | 10              | 561.3           | 0.234      |
| 14 `flash-attn` | 5          | 0.628       | 10              | 561.3           | 0.247      |
| 13 `flash-attn` | 4          | 0.575       | 10              | 561.3           | 0.238      |
| 12 `flash-attn` | 3          | 0.438       | 10              | 561.3           | 0.243      |
| 10 `default`    | 2          | 0.380       | 10              | 561.3           | 0.289      |
| 11 `flash-attn` | 2          | 0.344       | 10              | 561.3           | 0.229      |
| 16 `flash-attn` | 1          | 0.339       | 100             | 515.4           | 0.282      |
| 17 `default`    | 1          | 0.317       | 100             | 515.4           | 0.280      |

#### A100 Quantization Tests

Several quantization tests actually had a negative effect on A100 inference speed.
Note: Some of the below quantization tests were on different sets of documents.

| Experiment ID  | Docs/Second | Total Documents | Avg. tokens/doc | ROGUE-1 F1 |
|----------------|-------------|-----------------|-----------------|------------|
| 1 `default`    | 0.317       | 100             | 515.4           | 0.280      |
| 1 `flash-4bit` | 0.184       | 10              | 561.3           | 0.266      |
| 2 `4-bit`      | 0.180       | 100             | 451.7           | 0.250      |
| 1 `8-bit`      | 0.054       | 10              | 607.2           | 0.234      |

### Failed Experiments

Table of different errors from various experiments:

| Experiment               | GPU Type | Batch Size | Error                | Documents |
|--------------------------|----------|------------|----------------------|-----------|
| `default`                | `T4`     | 1          | `OutOfMemoryError`   | 10        |
| `flash-attn`             | `T4`     | 1          | `RuntimeError`       | 10        |
| `default`                | `T4`     | 2          | `OutOfMemoryError`   | 10        |
| `vllm` (no quantization) | `T4`     | dynamic    | >90% prompts skipped | 100       |
| `default`                | `A100`   | 4          | `OutOfMemoryError`   | 10        |
| `flash-attn`             | `A100`   | 10         | `OutOfMemoryError`   | 100       |

The only way to successfully run experiments with the T4 instance in all cases was quantization.
Other failed experiments that were unsuccessful on either the T4, the A100 or both include:

 - [DeepSpeed Optimizations](https://github.com/microsoft/DeepSpeed/tree/master/blogs/deepspeed-fastgen)
   (script: `run_llama2_7b_deepspeed_t4.py`, `run_llama2_7b_deepspeed_a100.py`)
 - [DeepSpeed ZeRO-Inference](https://github.com/microsoft/DeepSpeedExamples/tree/master/inference/huggingface/zero_inference)
   (script: `run_llama2_7b_deepspeed_zero.py`)
 - [IBM Foundation Model Stack](https://github.com/foundation-model-stack/foundation-model-stack)
   (script: `run_llama2_7b_chat_ibm_fms.py`)
 - [ONNX Inference](https://github.com/microsoft/Llama-2-Onnx)
   (scripts: `run_llama2_7b_onnx.py`, `run_llama2_7b_onnx_pipeline.py`)
 - Few-shot inference with the Llama2-7b base model (script: `run_llama2_7b_base_fewshot.py`)
 - HuggingFace pipelining, GGUF model inference, and more...

In most cases, the optimizations were still too slow or the T4 ran out of memory due to lack of quantization.
More details can be found in the docstrings at the top of the above scripts.

#### Flash Attention Incompatibility

Note that Flash Attention experiments were only run on the A100 GPU since it's incompatible with the T4 machine,
due to the T4 having an unsupported, older GPU type
(error: `RuntimeError: FlashAttention only supports Ampere GPUs or newer.`).
Also, note that the `4-bit` quantization experiments use 16-bit floating points for the computational type
via `bnb_4bit_compute_dtype`, while for `8-bit` quantization, the computational type is set to the default.

#### vLLM Memory Issues

Meanwhile, `vllm` experiments eat up all the machine's GPU memory when optimizing KV cache and
during CUDA graph creation, resulting in out of memory limits unless we decrease the maximum model
sequence length (prompt + output), which results in the majority of documents being skipped. This
is fixed by AWQ quantization at the expense of throughput (>5x slower). More info on this issue can
be found in the `scripts/run_llama_7b_vllm.py` script docstring.

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
