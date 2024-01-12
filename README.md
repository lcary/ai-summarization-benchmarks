# llama2-7b-summarization-benchmarks

Benchmarks testing the Llama2-7b chat model on text summarization inference tasks.

## References

 - Model: https://huggingface.co/meta-llama/Llama-2-7b-chat-hf
 - Dataset: https://huggingface.co/datasets/wiki_lingua
 - Quantization: https://huggingface.co/blog/4bit-transformers-bitsandbytes

## Results

### Experiments

All experiments were run on a single-GPU machine. Experiments include:

 - Default model loader (T4 GPU)
 - 4-bit Quantization (T4 GPU, A100 GPU)
 - 8-bit Quantization (T4 GPU, A100 GPU)

### Test Results

The following table shows the time it took for Llama-7b to generate summaries,
where the throughput is in the number of tokens generated per second by the model.

| Experiment | GPU Instance | Inference Speed | Number of Documents |
|---|---|---|---|
| `default` | `T4` | `OUT OF MEMORY` | 2 |
| `default` | `A100` | 109.73 tokens/sec | 10 |
| `4-bit` | `T4` | 54.23 tokens/sec | 100 |
| `4-bit` | `A100` | 81.20 tokens/sec | 100 |
| `8-bit` | `T4` | 30.08 tokens/sec | 100 |
| `8-bit` | `A100` | 32.77 tokens/sec | 10 |

In general, quantization had more impact on increasing inference speed on the lower-RAM T4 instance type
(15GB GPU RAM) compared to the A100 (40GB GPU RAM). This confirms how quantization is a great fit for educational
use or running summarization tasks on edge devices.

The `default` experiment running out of memory on the T4 instance occurred after an hour of runtime, during
which only two documents were procesed successfully. The following can be seen in the logs of this experiment,
showing how the model couldn't even fit within the GPU memory and overflowed to the CPU:
```
WARNING:root:Some parameters are on the meta device device because they were offloaded to the disk and cpu.
```

Also, note that the `4-bit` quantization experiments use 16-bit floating points for the computational type
via `bnb_4bit_compute_dtype`, while for `8-bit` quantization, the computational type is set to the default.

## Setup

### Step 1. GPU Setup

This assumes you are running on a GPU instance with a Python version >= 3.10
and CUDA installed. Google Colab provides T4 GPU servers free of charge which
should be sufficient for some of the quantization experiments (e.g. 4-bit). That said,
since these experiments can be pretty memory-intensive, and Colab tends to shut down
free accounts on occasion, I recoomend purchasing a Colab Pro subscription, which
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
!pip install transformers accelerate bitsandbytes torch datasets
!python -c 'from datasets import load_metric; load_metric("rouge")' || pip install rouge_score
```

The notebook server may need to be restarted at this point.

### Step 4: Server-side HuggingFace Login

```
!huggingface-cli login
```

This will prompt you for a HuggingFace access token which you created from your account.
After entering that in, finally, you can run the experiment script.

## Usage

The primary script to run with the best results is `run_llama2_7b.py`.
To use this script, which evaluates the Llama2-7b Chat model on the WikiLingua dataset,
follow the instructions (e.g. `pip install` steps) in the script docstring, then run:

```
python run_llama2_7b.py
```

The script was tested with the following versions:

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

### `ldconfig` issues

If you see issues with torch or HF complaining about missing libraries, it's likely
an issue with the CUDA setup and the `ldconfig` configuration.
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

 - [ ] **Dataset Balancing**: The experiments were run on different random samples of the data. It would be good to standardize the experiments across different instances using the same sample, via a random seed.
 - [ ] **Mistral**: Testing on the Mistral LLM family.
 - [ ] **Batching**: Testing with batching multiple documents per `generate` call. When testing this approach on the T4 instances (see `scripts/run_llama2_7b_chat_4bit_batch.py`) I frequently encountered OOM and CUDA device errors which blocked further experiments. Likely a larger machine is needed or more time spent configuring resource limitations and truncating prompt text to avoid OOMs is required.
 - [ ] **Knowledge Distillation**: One of the most promising avenues to reducing the memory requirements of LLMs is to leverage knowledge distillation (KD), such as is done for the [Orca series](https://arxiv.org/abs/2306.02707). Note that KD is 2-3 orders of magnitude more expensive than simply renting a larger GPU instance, requiring significant GPU runtime for producing the training dataset from the teacher model, and dozens to hundreds of hours on many GPUs for training the student model).
