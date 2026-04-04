# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Demonstrates reinforcement learning from human feedback (RLHF) using vLLM
via HTTP API, with IPC-based weight syncing APIs.

Unlike rlhf_nccl.py which uses NCCL and can use separate GPUs, this script
uses CUDA IPC which requires the training model and vLLM server to be on the
same GPU. Memory must be carefully managed to fit both models.

Unlike rlhf.py which creates a vLLM instance programmatically, this script
assumes you have already started a vLLM server using `vllm serve`. It uses:
- OpenAI-compatible API for inference requests
- HTTP endpoints for weight transfer control plane
- CUDA IPC for actual weight data transfer

Prerequisites:
    Start a vLLM server with weight transfer enabled and reduced GPU memory
    utilization to leave room for the training model:

    $ VLLM_SERVER_DEV_MODE=1 VLLM_ALLOW_INSECURE_SERIALIZATION=1 \
        vllm serve facebook/opt-125m --enforce-eager \
        --weight-transfer-config '{"backend": "ipc"}' \
        --load-format dummy \
        --gpu-memory-utilization 0.5

    Then run this script:

    $ python rlhf_http_ipc.py

The example performs the following steps:

* Load the training model on GPU 0 (same GPU as the vLLM server).
* Generate text using the vLLM server via OpenAI-compatible API. The output
  is expected to be nonsense because the server is initialized with dummy weights.
* Initialize weight transfer via HTTP endpoint (no-op for IPC).
* Broadcast the real weights from the training model to the vLLM server
  using CUDA IPC handles.
* Generate text again to show normal output after the weight update.
"""

import os
from typing import cast

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

from vllm.distributed.weight_transfer.ipc_engine import (
    IPCTrainerSendWeightsArgs,
    IPCWeightTransferEngine,
)

from trainer.rollouts.rollout_base import HttpRollout
from trainer.rollouts.rollout_utils import (
    BASE_URL,
    init_weight_transfer_engine,
    pause_generation,
    resume_generation,
)


# Enable insecure serialization for IPC handle serialization
os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"


class IPCHttpRollout(HttpRollout):
    def __init__(self, model_name: str = "facebook/opt-125m") -> None:
        super().__init__(model_name)
        init_weight_transfer_engine(BASE_URL, backend="ipc")

    def sync_weights(self, trainer_model: nn.Module) -> None:
        pause_generation(BASE_URL)
        print("Broadcasting weights via CUDA IPC (HTTP)...")
        trainer_args = IPCTrainerSendWeightsArgs(mode="http", url=BASE_URL)
        IPCWeightTransferEngine.trainer_send_weights(
            iterator=trainer_model.named_parameters(),
            trainer_args=trainer_args,
        )
        resume_generation(BASE_URL)


MODEL_NAME = "facebook/opt-125m"


def main():
    # IPC requires the training model to be on the same GPU as the vLLM server.
    # Align `device` with the GPU where `vllm serve` is bound.
    rollout = IPCHttpRollout(MODEL_NAME)
    device = "cuda:2"
    torch.accelerator.set_device_index(device)

    train_model = cast(
        nn.Module,
        AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.bfloat16),
    )
    train_model.to(torch.device(device))
    train_model.eval()

    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]

    print("-" * 50)
    print("Generating text BEFORE weight update (expect nonsense):")
    print("-" * 50)
    outputs = rollout.generate(prompts)
    for prompt, choice_texts in zip(prompts, outputs):
        print(f"Prompt: {prompt!r}\nGenerated texts: {choice_texts!r}")
        print("-" * 50)

    rollout.sync_weights(train_model)

    print("-" * 50)
    print("Generating text AFTER weight update:")
    print("-" * 50)
    outputs_updated = rollout.generate(prompts)
    for prompt, choice_texts in zip(prompts, outputs_updated):
        print(f"Prompt: {prompt!r}\nGenerated texts: {choice_texts!r}")
        print("-" * 50)


if __name__ == "__main__":
    main()
