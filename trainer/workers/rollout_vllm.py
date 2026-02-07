import os

import ray
import torch
from ray.util.placement_group import placement_group
from typing import Any
from loguru import logger
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from transformers import AutoModelForCausalLM, PreTrainedModel, Trainer
from vllm import LLM, SamplingParams

from trainer.workers.network_utils import get_ip, get_open_port
from trainer.workers.rlhf_utils import stateless_init_process_group


class InferenceLLM(LLM):
    def __init__(self, *args: Any, **kwargs: Any):
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        super().__init__(*args, **kwargs)


model_name = "Qwen/Qwen2.5-1.5B-Instruct"
train_model = AutoModelForCausalLM.from_pretrained(model_name)
train_model.to("cuda:1")
logger.info(f"train_model: {train_model}")
# Set CUDA_VISIBLE_DEVICES before ray.init() so Ray only sees and manages GPUs 1 and 2
# Ray will then map these to devices 0 and 1 in its worker processes
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
os.environ["RAY_RUNTIME_ENV_SKIP_INSTALL"] = "1"
ray.init(
    runtime_env={
        "excludes": [".venv/", ".git/", "*.pyc", "__pycache__/"],
        "env_vars": {
            "VIRTUAL_ENV": "",  # Unset VIRTUAL_ENV for Ray workers
            "PYTHON_EXECUTABLE": "/home/recoverx/astarag/trainer-rl/.venv/bin/python3",
        },
    }
)
pg_inference = placement_group([{"GPU": 1, "CPU": 0}] * 2)
ray.get(pg_inference.ready())

scheduling_inference = PlacementGroupSchedulingStrategy(
    placement_group=pg_inference,
    placement_group_capture_child_tasks=True,
    placement_group_bundle_index=0,
)

llm = ray.remote(
    num_cpus=0,
    num_gpus=0,
    scheduling_strategy=scheduling_inference,
)(InferenceLLM).remote(
    model=model_name,
    enforce_eager=True,
    worker_extension_cls="trainer.workers.rlhf_utils.WorkerExtension",
    tensor_parallel_size=2,
    distributed_executor_backend="ray",
)

prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]

sampling_params = SamplingParams(temperature=0)
outputs = ray.get(llm.generate.remote(prompts, sampling_params))

print("-" * 50)
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}\nGenerated text: {generated_text!r}")
    print("-" * 50)

# Set up the communication channel between the training process and the
# inference engine.
master_address = get_ip()
master_port = get_open_port()

model_update_group = stateless_init_process_group(
    master_address, master_port, 0, 3, torch.device("cuda:1")
)

handle = llm.collective_rpc.remote(
    "init_weight_update_group", args=(master_address, master_port, 1, 3)
)


ray.get(handle)

# Simulate a training step by zeroing out all model weights.
# In a real RLHF training loop the weights would be updated using the gradient
# from an RL objective such as PPO on a reward model.
for name, p in train_model.named_parameters():
    p.data.zero_()

# Synchronize the updated weights to the inference engine.
for name, p in train_model.named_parameters():
    dtype_name = str(p.dtype).split(".")[-1]
    handle = llm.collective_rpc.remote(
        "update_weight", args=(name, dtype_name, p.shape)
    )
    model_update_group.broadcast(p, src=0, stream=torch.cuda.current_stream())
    ray.get(handle)

# Verify that the inference weights have been updated.
assert all(ray.get(llm.collective_rpc.remote("check_weights_changed")))

# Generate text with the updated model. The output is expected to be nonsense
# because the weights are zero.
outputs_updated = ray.get(llm.generate.remote(prompts, sampling_params))
print("-" * 50)
for output in outputs_updated:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}\nGenerated text: {generated_text!r}")
    print("-" * 50)
