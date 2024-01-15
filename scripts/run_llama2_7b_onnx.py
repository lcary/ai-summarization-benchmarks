"""
Overview

This script uses few-shot learning to coax the base Llama2-7b model
to summarize text documents from the Wikilingua dataset.

Instructions

This assumes you are running on a GPU instance with a Python version >= 3.10
and CUDA installed. Google Colab provides T4 GPU servers free of charge which
should be sufficient.

Second, this requires a GitHub account with
access to Microsoft's Llama2 ONNX repository:
https://github.com/microsoft/Llama-2-Onnx/
(Repo access takes a few hours to a few days for the
Microsoft team to process.)

Submodule Setup:

You will need a personal-access token to clone Microsoft's submodule once
they approve the access request.

 1. Go to https://github.com/settings/tokens and create a new "Personal access tokens (classic)"
    with "repo:*" permissions.
 2. Run the following code from your Colab notebook, where <PAT> contains your generated token:

    !git clone https://github.com/microsoft/Llama-2-Onnx.git
    %cd Llama-2-Onnx
    !git submodule init 7B_FT_float16
    !git config submodule.7B_FT_float16.url https://<PAT>@github.com/microsoft/Llama-2-Onnx-7-FT-16
    !git submodule update

 3. Wait for the download to complete. This will take several minutes to complete.

Troubleshooting 'Host key verification failed.' errors:

To fix error, you need to allow SSH clones for the Colab machine:

    !ssh-keygen
    !apt-get install jq
    !curl --silent https://api.github.com/meta \
      | jq --raw-output '"github.com "+.ssh_keys[]' >> ~/.ssh/known_hosts

Note that it's very important to include the actual GitHub API URL in this request
and that it's susceptible to MITM attacks if you aren't careful.

CUDA Prerequisite Setup (Colab):

You will likely need to configure several settings for correct CUDA setup in Colab
by running the below commands.

    !export CUDA_HOME=/usr/local/cuda-12.2
    # Workaround: https://github.com/pytorch/pytorch/issues/107960
    !ldconfig /usr/lib64-nvidia
    !ldconfig -p | grep libcuda

The ldconfig command output should show libcuda.so, or else issues will occur,
If the ldconfig requires a different directory, check for other nvidia libraries
under /user/. If the notebook server has a different version of cuda home installed,
check for that via `ls /user/local/cuda*` and set that to CUDA_HOME.

Then, install dependencies (note CUDA 12 requires a pre-release version of onnx runtime
only available at a special index URL according to the GitHub issue here:
https://github.com/microsoft/onnxruntime/issues/13932#issuecomment-1870251784):

    !pip install torch numpy sentencepiece coloredlogs
    !pip install --index-url="https://pkgs.dev.azure.com/onnxruntime/onnxruntime/_packaging/onnxruntime-cuda-12/pypi/simple/" onnxruntime-gpu==1.17.0

The notebook server may need to be restarted at this point.
This script is working with the following versions on a T4 GPU w/ 12GB System RAM and 15GB GPU RAM:

 - Python 3.10.12
 - CUDA 12.2
 - onnxruntime-gpu=1.17.0
 - torch==2.1.0+cu121
 - numpy==1.23.5
 - sentencepiece==0.1.99
 - coloredlogs==15.0.1

Verify the setup with:

    !python MinimumExample/Example_ONNX_LlamaV2.py --onnx_file 7B_FT_float16/ONNX/LlamaV2_7B_FT_float16.onnx --embedding_file 7B_FT_float16/embeddings.pth --tokenizer_path tokenizer.model --prompt "What is the lightest element?"

Finally, you can run the below script.

For prompt = "What is the lightest element?"
 - @max_gen_len=256, inference_time: 37.35784649848938 seconds
 - @max_gen_len=50, inference_time: 5.425885438919067
 - @max_gen_len=10, inference_time: 1.177002191543579
 - @max_gen_len=5, inference_time: 0.6269898414611816 (useless, abbrev. response)

For summarization task (article on 'socks')
 - @max_gen_len=50, inference_time: 28.697363138198853 (first attempt)
For summarization task (article on 'shampoo')
 - @max_gen_len=50, inference_time: 55.52840971946716 (first attempt)
 - @max_gen_len=50, inference time: 37.90727734565735 (second attempt)
 - @max_gen_len=50, inference time: 45.60644197463989 (third attempt)
 - @max_gen_len=1, inference time: 8.578016757965088
"""
import os
import random
import time
from typing import List, Tuple

import numpy as np
import onnxruntime
import torch
from sentencepiece import SentencePieceProcessor
from datasets import load_dataset

# Session / model settings:
onnx_file = "7B_FT_float16/ONNX/LlamaV2_7B_FT_float16.onnx"
embedding_file = "7B_FT_float16/embeddings.pth"
tokenizer_path = "tokenizer.model"

class Tokenizer:
    def __init__(self, model_path: str):
        # reload tokenizer
        assert os.path.isfile(model_path), model_path
        self.sp_model = SentencePieceProcessor(model_file=model_path)

        # BOS / EOS token IDs
        self.n_words: int = self.sp_model.vocab_size()
        self.bos_id: int = self.sp_model.bos_id()
        self.eos_id: int = self.sp_model.eos_id()
        self.pad_id: int = self.sp_model.pad_id()

        assert self.sp_model.vocab_size() == self.sp_model.get_piece_size()

    def encode(self, s: str, bos: bool, eos: bool) -> List[int]:
        assert type(s) is str
        t = self.sp_model.encode(s)
        if bos:
            t = [self.bos_id] + t
        if eos:
            t = t + [self.eos_id]
        return t

    def decode(self, t: List[int]) -> str:
        return self.sp_model.decode(t)


# Create the ONNX session
options = onnxruntime.SessionOptions()
llm_session = onnxruntime.InferenceSession(
    onnx_file,
    sess_options=options,
    providers=[
        "DmlExecutionProvider",
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    ],
)


def infer(llm_session: onnxruntime.InferenceSession, prompt: str, max_gen_len: int) -> Tuple[str, float]:
    # get the data type used by the model
    data_type_str = llm_session.get_inputs()[0].type
    if data_type_str == "tensor(float16)":
        data_type = np.float16
    elif data_type_str == "tensor(float32)" or data_type_str == "tensor(float)":
        data_type = np.float32
    else:
        raise Exception(f"Unknown data type {data_type_str}")

    # Get the relevant shapes so we can create the inputs
    for inputs_meta in llm_session._inputs_meta:
        if inputs_meta.name == "x":
            x_shape = inputs_meta.shape
        elif inputs_meta.name == "attn_mask":
            attn_mask_shape = inputs_meta.shape
        elif inputs_meta.name == "k_cache":
            k_cache_shape = inputs_meta.shape

    hidden_size = x_shape[2]
    max_seq_len = attn_mask_shape[1]
    n_layers = k_cache_shape[1]
    n_heads = k_cache_shape[3]

    start_tokenizer = time.time()
    # Initialize the tokenizer and produce the initial tokens.
    tokenizer = Tokenizer(model_path=tokenizer_path)
    tokens = tokenizer.encode(prompt, bos=True, eos=False)
    tokenizer_time = time.time() - start_tokenizer
    print(f"Tokenization time: {tokenizer_time}")

    # create the embedding layer.
    embedding_layer = torch.nn.Embedding(tokenizer.n_words, hidden_size)
    embedding_layer.load_state_dict(torch.load(embedding_file))
    embedding_layer.eval()

    # Create the embeddings of the initial prompt.
    x = embedding_layer(torch.tensor(tokens)).detach().cpu().numpy()
    x = np.expand_dims(x, axis=0).astype(data_type)

    # Create the attention mask.
    attn_mask = -10000.0 * torch.triu(
        torch.ones(attn_mask_shape), diagonal=1
    ).cpu().detach().numpy().astype(data_type)

    # Create the K and V caches.
    head_dim = int(hidden_size / n_heads)
    k_cache = np.zeros([1, n_layers, max_seq_len, n_heads, head_dim], dtype=data_type)
    v_cache = np.zeros([1, n_layers, max_seq_len, n_heads, head_dim], dtype=data_type)

    # Iteratively generate tokens.
    pos = np.array(0)
    output_tokens = []
    total_inference_time = 0.0
    for idx in range(max_gen_len):
        start_time = time.time()
        results = llm_session.run(
            None,
            {
                "x": x,
                "attn_mask": attn_mask,
                "k_cache": k_cache[:, :, :pos],
                "v_cache": v_cache[:, :, :pos],
                "pos": pos.astype(np.int64),
            },
        )
        inference_time = time.time() - start_time
        total_inference_time += inference_time

        logits, k_out, v_out = results[:3]

        # Decide the next token using your preferred sampling strategy.
        next_token = np.argmax(logits, axis=-1).astype(np.int64)
        output_tokens.extend(next_token)

        # Stop if/when we get an ENDOFTEXT token before reaching maximum sequence length
        if next_token == tokenizer.eos_id:
            break

        # Update the cache
        seq_len = x.shape[1]
        k_cache[:, :, pos : pos + seq_len] = k_out
        v_cache[:, :, pos : pos + seq_len] = v_out

        # Update pos and x ready for the next round.
        pos = np.array(int(pos) + seq_len, dtype=np.int64)
        x = embedding_layer(torch.tensor(next_token)).unsqueeze(0)
        x = x.cpu().detach().numpy().astype(data_type)

    output_str = tokenizer.decode(torch.tensor(output_tokens).tolist())
    return output_str, total_inference_time


def get_doc(item):
    return item["article"]["document"]


def has_document(item):
    return bool(get_doc(item))


wikilingua_dataset = load_dataset("wiki_lingua", "english")
data = wikilingua_dataset["train"]
data = list(filter(has_document, data))
wikilingua_sample = random.sample(data, 100)

new_sample = wikilingua_sample[0]
doc = get_doc(new_sample)[0]

prompt_template = """
You are helpful AI assistant designed to summarize text documents. 
The documents are wiki-like articles ranging in length and diverse in content. 
The summaries you produce should be succinct but comprehensive,
capturing the essentials of the document and excluding superfluous details. 
Below is an example of your task.
----------------------------------------- EXAMPLE -----------------------------------------
[Document]: Once you finish washing your bear, get rid of as much water as you can without handling the bear too harshly, so that it dries quicker and more thoroughly. Squeeze water from its limbs, torsos, and head, but be careful to keep their original shape. Do not wring or twist them as you would with a bath towel. Then use a towel to softly pat any remaining moisture from their fur. For best results, let your toy dry on its own. Either set it on top of a drying rack, away from direct sunlight, and let it sit overnight, or set it in a drying bag and hang that from a laundry line, as long as the laundry line is shaded. Do not hang the bear itself from any laundry line or rack, since this may damage it. Setting up a fan to blow directly on the bear will help it to dry out quicker. For quicker results, use a laundry dryer or hair dryer to speed things up. However, there is considerable risk in ruining your bear this way, so be extra careful. If you use a laundry dryer, stick with the air cycle. Check the bear every few minutes to make sure no damage has occurred. If you use a hair dryer, be mindful of the heat. Set it to its coolest setting and hold the dryer at least a foot away from the bear as you dry it. If your bear was too delicate for a machine-wash, consider it to be too delicate for a machine-dry, as well. Once the bear has dried out, use a clean comb or brush to freshen up its fur. Judge the feel of it as you do so. Although the fur’s quality won’t ever be the same once you start washing it, be on the lookout for any areas that feel crunchy. This may be a sign of soap that wasn’t rinsed out, so if your bear feels crunchy all over, either rinse it out and dry it all over again, or be sure to use less soap in the future.

[Summary]: Gently remove excess water. Allow your bear to air-dry. Use a dryer. Brush the bear.
----------------------------------------- EXAMPLE -----------------------------------------

Now, summarize the following document. 

[Document]: {doc}
"""

prompt = prompt_template.format(doc=doc)
print(prompt)

output, inference_time = infer(llm_session, prompt, 50)
print(output)
print(f"inference time: {inference_time}")
