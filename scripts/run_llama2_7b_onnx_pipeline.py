"""
Overview

This script uses zero-shot learning to with the base Llama2-7b model
to summarize text documents from the Wikilingua dataset.

Instructions

This assumes you are running on a GPU instance with a Python version >= 3.10
and CUDA installed. Google Colab provides T4 GPU servers free of charge which
should be sufficient.

Second, this requires a HuggingFace account, signing Microsoft's
and requesting access to Llama2 from the model hub:
https://huggingface.co/meta-llama/Llama-2-7b-hf
(Llama 2 base model access takes a few hours to a few days for the
Microsoft team to process.)

Finally, you need to create an API token in HuggingFace at:
https://huggingface.co/settings/tokens

Then, install dependencies (note CUDA 12 requires a pre-release version of onnx runtime
only available at a special index URL according to the GitHub issue here:
https://github.com/microsoft/onnxruntime/issues/13932#issuecomment-1870251784):

    !pip install datasets transformers==4.36.1 accelerate bitsandbytes optimum onnx
    !pip install --index-url="https://pkgs.dev.azure.com/onnxruntime/onnxruntime/_packaging/onnxruntime-cuda-12/pypi/simple/" onnxruntime-gpu==1.17.0

Next, log into huggingface, so we can download models using the above API token:

    !huggingface-cli login

The notebook server may need to be restarted at this point.
This script is working with the following versions on a T4 GPU w/ 12GB System RAM and 15GB GPU RAM:

 - Python 3.10.12
 - CUDA 12.2
"""
import time

from optimum.pipelines import pipeline
from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForCausalLM

model_name = (
    "nenkoru/llama-7b-onnx-merged-fp16"  # TODO: this raises a legacy warning, fix?
)

model = ORTModelForCausalLM.from_pretrained(model_name, device_map="auto")

tokenizer_model_name = (
    "meta-llama/Llama-2-7b-hf"  # TODO: compatible with above onnx model?
)

tokenizer = AutoTokenizer.from_pretrained(tokenizer_model_name)

prompt_template = """
Document: {document}

Summary:
"""

document = "The cleaner your socks are, the better they will conduct electricity. If your socks are wet or dirty, they will not have as much traction with the floor and may not make static electricity. Warm socks that just came out of the dryer are best for conducting electricity. While most socks can conduct static electricity, wool socks generally work best. Electronic items contain microchips that can malfunction or become permanently destroyed by static electricity. Before touching any electronic items, take off your socks and touch something else to discharge any static electricity. Even if your electronic device has a protective case, it may still be vulnerable to static shocks."

prompt = prompt_template.format(document=document)

# task = "summarization"  # NOTE: Not supported for ORTModelForCausalLM
task = "text-generation"
summarizer = pipeline(
    task, model=model, tokenizer=tokenizer, device_map="auto", accelerator="ort"
)

start_time = time.time()
result = summarizer(prompt)
inference_time = time.time() - start_time
summary = result[0]["generated_text"].split("Summary:")[-1]
print(summary)
print(f"Inference time: {inference_time}")
