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

 3. This will take several minutes to complete.

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

    !pip install torch numpy sentencepiece
    !!pip install --index-url="https://pkgs.dev.azure.com/onnxruntime/onnxruntime/_packaging/onnxruntime-cuda-12/pypi/simple/" onnxruntime-gpu==1.17.0

The notebook server may need to be restarted at this point.

Verify the setup with:

    !python MinimumExample/Example_ONNX_LlamaV2.py --onnx_file 7B_FT_float16/ONNX/LlamaV2_7B_FT_float16.onnx --embedding_file 7B_FT_float16/embeddings.pth --tokenizer_path tokenizer.model --prompt "What is the lightest element?"

Finally, you can run the below script.
"""
import os
import time
from typing import List, Tuple

import numpy as np
import onnxruntime
import torch
from sentencepiece import SentencePieceProcessor


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


def run_onnx_llamav2(
    prompt: str,
    onnx_file: str,
    embedding_file: str,
    tokenizer_path: str,
    max_gen_len: int = 256,
) -> Tuple[str, float]:
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

    # Initialize the tokenizer and produce the initial tokens.
    tokenizer = Tokenizer(model_path=tokenizer_path)
    tokens = tokenizer.encode(prompt, bos=True, eos=False)

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


onnx_file = "7B_FT_float16/ONNX/LlamaV2_7B_FT_float16.onnx"
embedding_file = "7B_FT_float16/embeddings.pth"
tokenizer_path = "tokenizer.model"
prompt = "What" "is" "the" "lightest" "element?"
max_gen_len = 50
response, inference_time = run_onnx_llamav2(
    prompt,
    onnx_file,
    embedding_file,
    tokenizer_path,
    max_gen_len,
)

print(response)
print(f"inference_time: {inference_time}")
# inference_time: 37.35784649848938 seconds at max_gen_len = 256
# inference_time: 5.425885438919067 at 50 tokens
# inference_time: 1.177002191543579 at 10 tokens
# inference_time: 0.6269898414611816 at 5 tokens  (useless, abbrev. response)
