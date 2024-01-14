"""
Overview

This script uses one-shot prompting with the Llama2-7b chat model (4bit format)
to summarize text documents from the Wikilingua dataset.

Instructions

This assumes you are running on a GPU instance with a Python version >= 3.10
and CUDA installed. Google Colab provides T4 GPU servers free of charge which
should be sufficient.

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

Then, install dependencies:

    !pip install transformers accelerate bitsandbytes torch

The notebook server may need to be restarted at this point.
Finally, you can run the below script.
"""
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList
import torch
import time

model_name = "meta-llama/Llama-2-7b-chat-hf"

model = AutoModelForCausalLM.from_pretrained(
    model_name, device_map="auto", load_in_4bit=True
)
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

prompt_template = """
You are helpful AI assistant designed to summarize text documents.
The documents are wiki-like articles ranging in length and diverse in content.
The summaries you produce should be succinct but comprehensive,
capturing the essentials of the document and excluding superfluous details.
Below is an example of your task.

----------------------------------------- EXAMPLE START -----------------------------------------

Document: 'If you have a chair that is the right size to stretch the elastic, that will work perfectly. If you don’t have a chair that’s the right size, you can try using the side of a small table, an empty drawer, or an empty poster frame. If you can, line up the sides with the side of the chair. This will help stretch the elastic evenly. Let your elastic sit, being stretched, for 24 hours. If the desired size still isn’t reached, put the elastic back in the stretched position and leave it for several days. Leave it in a warm place to help the elastic band stretch.', 'You want your iron on and set to the highest setting. Run a face cloth or hand towel under water until it is damp, but not soaked. You can either pin each side of your pants to the ironing board—stretched to the desired length. Or, you can simply slide the pants around the ironing board until they are the proper width. Make sure it completely covers the elastic that you’re trying to stretch out. If needed, use two cloths. With the damp cloth on top of your elastic band and your iron on the highest setting, iron over it. Iron for 10 seconds and then let sit for 10 seconds. Continue doing so for 5-10 minutes. This will help your pants fit because as the elastic heats, it will heightens the breaking weight. This means that it will be able to stretch more before reaching its limit. If it hasn’t’ stretch enough, try flipping your elastic band and repeat the process. Do so until you reach your desired fit.', 'This will make it much easier to work with. Plus, you will be less likely to make an error with your scissors if you can see what you’re doing. Sometimes, elastics are sewn into the seam of clothing. If this is the case, you won’t be able to pull the elastic out of them if you cut the somewhere other than the seam. Find the seam by holding on one side of the seam and stretching the other side. If you feel the elastic shift, you are free to cut wherever you’d like. If you feel it snag at the seam, make sure to cut here. To remove the elastic band from your clothing, make a slit (around ½”). If the elastic is sewn into the seam, you’ll have to cut the seam the size of the elastic. Use scissors to go through your slit and cut the elastic. Cut through the entire elastic without cutting any more holes in your clothes. If you want to still be able to tie the pants snugly, affix a long shoelace or ribbon to one end of the elastic using a safety pin. When you pull out the elastic, pull on the end without the ribbon. This will guide your new tie through the waistband. If you do not want a tie, then just slowly pull the elastic out, being careful not to catch a lose string and bunch your fabric. Once the elastic is out/replaced, your clothes are ready to wear. You can sew the slit closed if you want to, but it’s not a necessary step before you wear your clothes.'

Summary: 'Find a chair. Stretch your elastic clothing over your chair. Let it sit. Turn on the iron and dampen a cloth. Prepare your pants. Place the damp cloth over your elastic. Iron the elastic. Repeat as necessary. Turn the clothing inside out. Find the inside seam. Put a small slit on the inside of you clothing. Cut the elastic. Pull the elastic out.'

------------------------------------------ EXAMPLE END ------------------------------------------

Now, summarize the following document enclosed within triple backticks.

```{document}```

Summary:
"""

document = "The cleaner your socks are, the better they will conduct electricity. If your socks are wet or dirty, they will not have as much traction with the floor and may not make static electricity. Warm socks that just came out of the dryer are best for conducting electricity. While most socks can conduct static electricity, wool socks generally work best. Electronic items contain microchips that can malfunction or become permanently destroyed by static electricity. Before touching any electronic items, take off your socks and touch something else to discharge any static electricity. Even if your electronic device has a protective case, it may still be vulnerable to static shocks."
prompt = prompt_template.format(document=document)
truncation_length = 4096
input_ids = tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=True).to(
    device
)

if truncation_length is not None:
    input_ids = input_ids[:, -truncation_length:]


class _StopEverythingStoppingCriteria(transformers.StoppingCriteria):
    def __init__(self):
        transformers.StoppingCriteria.__init__(self)

    def __call__(self, input_ids: torch.LongTensor, _scores: torch.FloatTensor) -> bool:
        return False


generate_params = {
    "max_new_tokens": 200,
    "inputs": input_ids,
    "bos_token_id": 1,
    "eos_token_id": 2,
    "pad_token_id": 32000,
    "temperature": 0.9,
    "top_p": 0.6,
    "stopping_criteria": transformers.StoppingCriteriaList(),
    "logits_processor": LogitsProcessorList([]),
}

generate_params["stopping_criteria"].append(_StopEverythingStoppingCriteria())

start_time = time.time()
output_ids = model.generate(**generate_params)[0]
inference_time = time.time() - start_time
output_ids = output_ids.cuda()
new_tokens = len(output_ids) - len(input_ids[0])
skip_special_tokens = True
reply = tokenizer.decode(output_ids[-new_tokens:], skip_special_tokens)
if len(output_ids) > 0:
    if tokenizer.convert_ids_to_tokens(int(output_ids[-new_tokens])).startswith("▁"):
        reply = " " + reply
print(reply)
print(f"Inference time: {inference_time}")
