# Other Tests

This contains additional tests and experiments that were run but were unstable or slow.

### Batching Tests

Batching tests were mostly unstable or unsuccessful, and the throughput metrics below
did not apply to other experiments with larger sets of documents and the same batch sizes. 
Some of the below batching tests below were on different sets of documents. Since the A100
tests below are not all on the same dataset, throughput needs to be interpreted with a grain of salt.

| Experiment | Optimization | Batch Size | Docs/Second | Total Documents | Avg. tokens/doc | ROGUE-1 F1 |
|------------|--------------|------------|-------------|-----------------|-----------------|------------|
| 15         | `flash-attn` | 10         | 0.861       | 10              | 561.3           | 0.234      |
| 14         | `flash-attn` | 5          | 0.628       | 10              | 561.3           | 0.247      |
| 13         | `flash-attn` | 4          | 0.575       | 10              | 561.3           | 0.238      |
| 12         | `flash-attn` | 3          | 0.438       | 10              | 561.3           | 0.243      |
| 10         | `none`       | 2          | 0.380       | 10              | 561.3           | 0.289      |
| 11         | `flash-attn` | 2          | 0.344       | 10              | 561.3           | 0.229      |
| 16         | `flash-attn` | 1          | 0.339       | 100             | 515.4           | 0.282      |
| 17         | `none`       | 1          | 0.317       | 100             | 515.4           | 0.280      |

Additional notes:

 - Batching resulted in OOM errors for all batch sizes on the T4 GPU 
 - Batching resulted in OOMs at batch size 10 or greater for the A100
   whenever the dataset had more than 10 docs (see Failed Experiments).
 - Note that batching was only possible when using Flash Attention on the A100
   after a size 4 documents per batch even for 10 documents, otherwise OOMs occurred. 

### A100 Quantization Tests

The below table shows the result of testing different quantization techniques on the A100.

 - `flash-4bit`: This test combined Flash Attention w/ 4-bit quantization
 - Note: Some of the below quantization tests were on different sets of documents.

| Experiment | Optimization | Docs/Second | Total Documents | Avg. tokens/doc | ROGUE-1 F1 |
|------------|--------------|-------------|-----------------|-----------------|------------|
| 17         | `none`       | 0.317       | 100             | 515.4           | 0.280      |
| 27         | `4-bit`      | 0.206       | 100             | 515.4           | 0.254      |
| 18         | `flash-4bit` | 0.184       | 10              | 561.3           | 0.266      |
| 29         | `8-bit`      | 0.059       | 100             | 515.4           | 0.267      |

The above quantization techniques actually had a negative effect on A100 inference speed.

## Failed Tests

Table of different errors from various experiments:

| Optimization             | GPU Type | Batch Size | Error              | Documents |
|--------------------------|----------|------------|--------------------|-----------|
| `none`                   | `T4`     | 1          | `OutOfMemoryError` | 10        |
| `flash-attn`             | `T4`     | 1          | `RuntimeError`     | 10        |
| `none`                   | `T4`     | 2          | `OutOfMemoryError` | 10        |
| `vllm` (no quantization) | `T4`     | dynamic    | Prompts skipped    | 100       |
| `default`                | `A100`   | 4          | `OutOfMemoryError` | 10        |
| `flash-attn`             | `A100`   | 10         | `OutOfMemoryError` | 100       |

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

In most cases, the optimizations were too slow or the server ran out of memory.
In other cases, the experimental constraints were violated.
More details can be found in the docstrings at the top of the above scripts.

### Flash Attention Compatibility

Flash Attention experiments were only run on the A100 GPU since it's incompatible with the T4 machine,
due to the T4 having an unsupported, older GPU type
(error: `RuntimeError: FlashAttention only supports Ampere GPUs or newer.`).
Also, note that the `4-bit` quantization experiments use 16-bit floating points for the computational type
via `bnb_4bit_compute_dtype`, while for `8-bit` quantization, the computational type is set to the default.
Flash Attention is also incompatible on V100 GPU servers.

### vLLM Memory and Compatibility Issues

The `vllm` experiments run without AutoAWQ quantization consumes the entirety of the T4 machine's
GPU memory when optimizing KV cache and during CUDA graph creation, resulting in out of memory issues unless we decrease the maximum model
sequence length (prompt and output combined) is decreased, which results in the majority of documents being skipped.
This is fixed by AWQ quantization at the expense of throughput. More info on this issue can
be found in the `scripts/run_llama_7b_vllm.py` script docstring.

Note vllm with AWQ is incompatible on V100 GPU servers:

```
ValueError: The quantization method awq is not supported for the current GPU. Minimum capability: 75. Current capability: 70.
```
