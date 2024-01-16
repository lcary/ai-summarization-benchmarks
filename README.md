# llama2-7b-summarization-benchmarks

Benchmarks testing the [Llama2-7b chat model](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf)
on text summarization inference tasks using the [Wikilingua](https://huggingface.co/datasets/wiki_lingua)
dataset.

## Experiments

These experiments tested inference time for text summarization with Llama-7b on Wikilingua articles.

### Constraints

The below experiments were conducted with the following constraints:

 - Task: Text Summarization
 - Number of GPUs: 1
 - 300 maximum generation tokens
 - No input document truncation

### Optimizations

The experiments tested multiple optimization scenarios using the Llama2-7b model:

 - `4-bit`: [4-bit Quantization](https://huggingface.co/blog/4bit-transformers-bitsandbytes)
 - `8-bit`: [8-bit Quantization](https://huggingface.co/blog/hf-bitsandbytes-integration)
 - `none`: [Default model loader](https://huggingface.co/docs/transformers/main/model_doc/llama2) (no optimizations)
 - `flash-attn`: [Flash Attention](https://huggingface.co/docs/transformers/perf_infer_gpu_one#flashattention-2)
 - `vllm-awq`: [vLLM with AutoAWQ quantization](https://docs.vllm.ai/en/latest/quantization/auto_awq.html) (**Best Results**)

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

See the report at [`notebooks/Llama2 7b Benchmarks.pdf`](./notebooks/Llama2%207b%20Benchmarks.pdf)
or the report notebook at [`notebooks/Llama2 7b Benchmarks.ipynb`](./notebooks/Llama2%207b%20Benchmarks.ipynb).

The following experiments were run on A100 and T4 for the above set of documents in the Wikilingua dataset.
The below table is ordered by the number of documents processed per second by the model:

| Experiment ID | Optimization | Docs/Second | Instance Type | ROGUE-1 F1 | Experiment Script                          |
|---------------|--------------|-------------|---------------|------------|--------------------------------------------|
| 25            | `vllm-awq`   | **4.273**   | `A100`        | 0.234      | `run_llama2_7b_vllm.py`                    |
| 28            | `vllm-awq`   | **0.540**   | `T4`          | 0.254      | `run_llama2_7b_vllm.py`                    |
| 16            | `flash-attn` | 0.339       | `A100`        | 0.282      | `run_llama2_7b.py --experiment=flash-attn` |
| 17            | `none`       | 0.317       | `A100`        | 0.280      | `run_llama2_7b.py --experiment=default`    |
| 27            | `4-bit`      | 0.206       | `A100`        | 0.264      | `run_llama2_7b.py --experiment=4-bit`      |
| 22            | `4-bit`      | 0.183       | `T4`          | 0.251      | `run_llama2_7b.py --experiment=4-bit`      |
| 29            | `8-bit`      | 0.059       | `A100`        | 0.267      | `run_llama2_7b.py --experiment=8-bit`      |
| 23            | `8-bit`      | 0.055       | `T4`          | 0.275      | `run_llama2_7b.py --experiment=8-bit`      |

 - **Summary**: `vLLM` optimization had the fastest throughput at 10X faster than any other strategy
   for the A100 GPU. The strategy was less profound on the lower-memory T4 GPU, but still resulted
   in a 2X speedup over other T4 experiments, and allowing T4 inference throughput to exceed that of A100 (without vLLM).
 - **Output & Results Data**: To see the data (output, duration) collected during inference, 
   find the JSON files with a given Experiment ID in its filename under the `data/` directory.
 - **Quantization**: quantization helped considerably with memory limitations on the
   T4 instance type, which was less of a concern on the larger-memory A100.
 - **Memory Usage**: During the experiments, it was typical for T4 GPU memory
   usage to be around 10-15GB, whereas A100 GPU memory usage was around 30-40GB.
 - **Failed Experiments**: Many of the experiment types could not be run due to out of memory and compatibility issues. 
   For more details on failed experiments, see the additional tests document.
 - **Model Quality**: Output quality was gauged by ROGUE-1 F1 scores, since quantization can affect how well
   the model performs on a given task.

## Additional Tests

See additional test results not listed above in [`docs/other-tests.md`](./docs/other-tests.md).

## Next Steps

 - [x] **All Combinations**: Test primary experiment scenarios on all GPU types. Requires purchasing additional GPU credits for the A100 experiments.
 - [x] **Same Datasets**: The experiments were originally run on different random samples of the data. It would be good to standardize the experiments across different instances using the same sample, via a fixed set of articles.
 - [ ] **Larger Experiments**: The experiments were run on small samples of the data. It would be good to scale up the experiments to 1000 records each. This requires renting additional GPUs.
 - [ ] **Mistral**: Running experiments on the Mistral LLM family and compare inference time / ROGUE scores.
 - [ ] **Orca**: Running experiments on the Orca LLM family and compare inference time / ROGUE scores.
 - [ ] **Knowledge Distillation**: One of the most promising avenues to reducing the memory requirements of LLMs is to leverage knowledge distillation (KD), such as is done for the [Orca series](https://arxiv.org/abs/2306.02707). Note that KD is 2-3 orders of magnitude more expensive than simply renting a larger GPU instance, requiring significant GPU runtime for producing the training dataset from the teacher model, and dozens to hundreds of hours on many GPUs for training the student model.

## Reproducibility

See [`docs/reproducibility.md`](./docs/reproducibility.md) for reproducing the results and script usage instructions.

## Troubleshooting

See [`docs/troubleshooting.md`](./docs/troubleshooting.md) for troubleshooting steps.
