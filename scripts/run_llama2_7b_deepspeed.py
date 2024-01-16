"""
Required setup:

    !apt-get install libaio-dev
    !pip install git+https://github.com/tjruwase/transformers@kvcache-offload-cpu
    !pip install deepspeed-mii

Currently fails with CPU and GPU OOM errors.
Note: current implementation is incompatible with fp16 models.
"""

import mii

DEPLOYMENT_NAME = "test-llama2-deepspeed"

model_name: str = "meta-llama/Llama-2-7b-chat-hf"

# Workaround for RuntimeError due to old CUDA capabilities version (<8)
# is to use the legacy deepspeed library (mii.deploy) with zero inference enabled
# https://github.com/microsoft/DeepSpeed-MII/issues/273#issuecomment-1813151824
# RuntimeError: Unable to load ragged_device_ops op due to no compute capabilities remaining after filtering

# Config settings: https://github.com/microsoft/DeepSpeed-MII/blob/95d1e1c8890a016f2b5788414754abbbfd4540ae/mii/legacy/config.py#L25
mii.deploy(
    deployment_name=DEPLOYMENT_NAME,
    deployment_type=mii.constants.DeploymentType.NON_PERSISTENT,
    task="text-generation",
    model=model_name,
)

generator = mii.mii_query_handle(DEPLOYMENT_NAME)
result = generator.query(
    {"query": ["DeepSpeed is", "Seattle is"]}, do_sample=True, max_new_tokens=30
)

# mii.terminate(DEPLOYMENT_NAME)

print(result)
