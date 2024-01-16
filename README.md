# llama2-7b-summarization-benchmarks

Benchmarks testing the [Llama2-7b chat model](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf)
on random text summarization inference tasks from the [Wikilingua](https://huggingface.co/datasets/wiki_lingua)
dataset.

## Results

### Experiments

All experiments were run on single-GPU machines in Google Colab (either the T4 or A100).
Experiments were run for multiple optimization scenarios using the Llama2-7b model:

 - `default`: [Default model loader](https://huggingface.co/docs/transformers/main/model_doc/llama2) (no optimizations)
 - `4-bit`: [4-bit Quantization](https://huggingface.co/blog/4bit-transformers-bitsandbytes)
 - `8-bit`: [8-bit Quantization](https://huggingface.co/blog/hf-bitsandbytes-integration)
 - `flash-attn`: [Flash Attention](https://huggingface.co/docs/transformers/perf_infer_gpu_one#flashattention-2)
 - `flash-4bit`: Combined Flash Attention w/ 4-bit quantization
 - `vllm`: [`vLLM` optimizations (KV cache, CUDA graph)](https://docs.vllm.ai/en/latest/index.html)
 - `vllm-awq`: [vLLM with `autoawq` quantizations](https://docs.vllm.ai/en/latest/quantization/auto_awq.html)
 
See the `scripts/run_llama2_7b.py` script for the code used to load the model in most scenarios.
See the `scripts/run_llama_7b_vllm.py` script for VLLM optimizations.
All other scripts are under `scripts/`, such as those documented in the "Experiment Notes" below.

### Test Results

The following tables shows the time it took for Llama-7b to generate summaries on different datasets.
These tables are ordered by the number of documents processed per second by the model.
Output quality was gauged by ROGUE-1 F1 scores, since quantization can affect how well
the model performs on a given task.

#### Constraints

The below experiments were conducted with the following settings:

 - 300 maximum output tokens
 - 4096 maximum input (prompt) tokens

#### T4 Experiments

The following experiments were run on a T4 machine for a set of Wikilingua articles:

| Scenario     | Batch Size | Processed docs/sec | Experiment # | Script                  | ROGUE-1 F1 |
|--------------|------------|--------------------|--------------|-------------------------|------------|
| `vllm-awq`   | dynamic    | 0.423              | 21           | `run_llama2_7b_vllm.py` | 0.233      |
| `4-bit`      | 1          | 0.183              | 22           | `run_llama2_7b.py`      | 0.251      |
| `vllm-awq`   | 1          | 0.150              | 24           | `run_llama2_7b_vllm.py` | 0.279      |
| `8-bit`      | 1          | 0.055              | 23           | `run_llama2_7b.py`      | 0.275      |

Dataset details:
 - 100 randomly sampled documents from Wikilingua's `english` corpus (same set for all experiments)
 - Average document length: 515.4 tokens
 - For the set of article URLs in this experiment dataset, see `data/100_articles.txt`.

To see the data (output, duration) collected during inference, see the `data/` directory.

Many of the experiment types could not be run on the T4
due to out of memory issues and compatibility issues. 
For more details on failed T4 experiments, see the `Experiment Notes` below.

#### A100 Experiments

The following A100 experiments were run on different sets of documents in the Wikilingua dataset.
_Note: I ran out of GPU credits before being able to run all A100 experiments on the same set of 100 articles.
That is why some of the A100 experiments have different document counts and token averages. Since the A100
experiments are not all on the same dataset, performance needs to be interpreted with a grain of salt._ 

| Scenario     | Batch Size | Processed docs/sec | Total Documents | Avg. tokens/doc | Experiment # | ROGUE-1 F1 |
|--------------|------------|--------------------|-----------------|-----------------|--------------|------------|
| `flash-attn` | 10         | 0.861              | 10              | 561.3           | 15           | 0.234      |
| `flash-attn` | 5          | 0.628              | 10              | 561.3           | 14           | 0.247      |
| `flash-attn` | 4          | 0.575              | 10              | 561.3           | 13           | 0.238      |
| `flash-attn` | 3          | 0.438              | 10              | 561.3           | 12           | 0.243      |
| `default`    | 2          | 0.380              | 10              | 561.3           | 10           | 0.289      |
| `flash-attn` | 2          | 0.344              | 10              | 561.3           | 11           | 0.229      |
| `flash-attn` | 1          | 0.339              | 100             | 515.4           | 16           | 0.282      |
| `default`    | 1          | 0.317              | 100             | 515.4           | 17           | 0.280      |
| `flash-4bit` | 1          | 0.184              | 10              | 561.3           | 18           | 0.266      |
| `4-bit`      | 1          | 0.180              | 100             | 451.7           | 2            | 0.250      |
| `8-bit`      | 1          | 0.054              | 10              | 607.2           | 15           | 0.234      |

### Experiment Notes

 - **Quantization**: In general, quantization had more impact on increasing inference speed on the lower-RAM T4 instance type
   (15GB GPU RAM) compared to the A100 (40GB GPU RAM). This confirms how quantization is a
   great fit for educational use or running summarization tasks on edge devices.
 - **Memory Usage**: During the experiments, it was typical for T4 GPU memory
   usage to be around 10-15GB, whereas A100 GPU memory usage was around 30-40GB. 
 - **Batching**: Additionally, batching had a profound effect on inference speed, 
   but also resulted in OOM errors, at size 2 or greater on the T4 and later
   and at size 4 or greater for the A100.
 - **Flash Attention**: Note that batching was only possible when using Flash Attention on the A100
   after a size 4 documents per batch, otherwise OOMs occurred.

#### Failed Experiments

Table of different errors from various experiments:

| Scenario     | GPU Type | Batch Size | Error                | Documents |
|--------------|----------|------------|----------------------|-----------|
| `default`    | `T4`     | 1          | `OutOfMemoryError`   | 10        |
| `flash-attn` | `T4`     | 1          | `RuntimeError`       | 10        |
| `default`    | `T4`     | 2          | `OutOfMemoryError`   | 10        |
| `vllm`       | `T4`     | dynamic    | >90% prompts skipped | 100       |
| `default`    | `A100`   | 4          | `OutOfMemoryError`   | 10        |
| `flash-attn` | `A100`   | 10         | `OutOfMemoryError`   | 100       |

The `default` experiment running out of memory on the T4 instance occurred after an hour of runtime, during
which only two documents were processed successfully. The following can be seen in the logs of this experiment,
showing how the model couldn't even fit within the GPU memory and overflowed to the CPU:
```
WARNING:root:Some parameters are on the meta device device because they were offloaded to the disk and cpu.
```

The only way to successfully run experiments with the T4 instance in all cases was quantization.

Other failed experiments that were unsuccessful on either the T4, the A100 or both include:

 - [DeepSpeed Optimizations](https://github.com/microsoft/DeepSpeed/tree/master/blogs/deepspeed-fastgen)
   (script: `run_llama2_7b_deepspeed.py`)
 - [DeepSpeed ZeRO-Inference](https://github.com/microsoft/DeepSpeedExamples/tree/master/inference/huggingface/zero_inference)
   (script: `run_llama2_7b_deepspeed_zero.py`)
 - [IBM Foundation Model Stack](https://github.com/foundation-model-stack/foundation-model-stack)
   (script: `run_llama2_7b_chat_ibm_fms.py`)
 - [ONNX Inference](https://github.com/microsoft/Llama-2-Onnx)
   (scripts: `run_llama2_7b_onnx.py`, `run_llama2_7b_onnx_pipeline.py`)
 - Few-shot inference with the Llama2-7b base model (script: `run_llama2_7b_base_fewshot.py`)
 - HuggingFace pipelining, GGUF model inference, and more...

In most cases, the optimizations were still too slow or the T4 ran out of memory due to lack of quantization.

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

## Next Steps

 - [ ] **All Combinations**: Test all experiments on all GPU types. Requires purchasing additional GPU credits for the A100 experiments.
 - [ ] **Same Datasets**: The experiments were originally run on different random samples of the data. It would be good to standardize the experiments across different instances using the same sample, via a fixed set of articles.
 - [ ] **Larger Experiments**: The experiments were run on small samples of the data. It would be good to scale up the experiments to 1000 records each. This requires renting additional GPUs.
 - [ ] **T4 Batching Stability**: Testing with batching multiple documents per `generate` call. When testing this approach on the T4 instances, I frequently encountered OOM and CUDA device errors which blocked further experiments. Likely a larger machine is needed or more time spent configuring resource limitations and truncating prompt text to avoid OOMs is required.
 - [ ] **A100 Batching Stability**: A100 experiments experienced out of memory errors when batching on larger sets of documents. Additional memory cleanup is likely required to stabilize memory usage.
 - [ ] **Mistral**: Running experiments on the Mistral LLM family and compare inference time / ROGUE scores.
 - [ ] **Knowledge Distillation**: One of the most promising avenues to reducing the memory requirements of LLMs is to leverage knowledge distillation (KD), such as is done for the [Orca series](https://arxiv.org/abs/2306.02707). Note that KD is 2-3 orders of magnitude more expensive than simply renting a larger GPU instance, requiring significant GPU runtime for producing the training dataset from the teacher model, and dozens to hundreds of hours on many GPUs for training the student model.
