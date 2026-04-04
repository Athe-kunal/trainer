# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Demonstrates reinforcement learning from human feedback (RLHF) using vLLM
via HTTP API, with native weight syncing APIs.

Unlike rlhf.py which creates a vLLM instance programmatically, this script
assumes you have already started a vLLM server using `vllm serve`. It uses:
- OpenAI-compatible API for inference requests
- HTTP endpoints for weight transfer control plane
- NCCL for actual weight data transfer

Prerequisites:
    Start a vLLM server with weight transfer enabled:

    $ VLLM_SERVER_DEV_MODE=1 vllm serve facebook/opt-125m \
        --enforce-eager \
        --weight-transfer-config '{"backend": "nccl"}' \
        --load-format dummy

    Then run this script:

    $ python rlhf_http.py

The example performs the following steps:

* Load the training model on GPU 0.
* Generate text using the vLLM server via OpenAI-compatible API. The output
  is expected to be nonsense because the server is initialized with dummy weights.
* Initialize weight transfer via HTTP endpoint.
* Broadcast the real weights from the training model to the vLLM server
  using NCCL.
* Generate text again to show normal output after the weight update.
"""

import threading
from typing import cast

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

from vllm.distributed.weight_transfer.nccl_engine import (
    NCCLTrainerSendWeightsArgs,
    NCCLWeightTransferEngine,
)
from vllm.utils.network_utils import get_ip, get_open_port

from trainer.rollouts.rollout_base import HttpRollout
from trainer.rollouts.rollout_utils import (
    BASE_URL,
    get_world_size,
    init_weight_transfer_engine,
    pause_generation,
    resume_generation,
    update_weights,
)

MODEL_NAME = "facebook/opt-125m"


class NCCLHttpRollout(HttpRollout):
    """HTTP rollout with NCCL weight sync; establishes the NCCL group at init."""

    def __init__(self, model_name: str = "facebook/opt-125m") -> None:
        super().__init__(model_name)
        self.inference_world_size = get_world_size(BASE_URL)
        world_size = self.inference_world_size + 1
        master_address = get_ip()
        master_port = get_open_port()
        rank_offset = 1
        print(
            f"{master_address=} {master_port=} {world_size=} "
            f"{self.inference_world_size=} {rank_offset=}"
        )

        init_thread = threading.Thread(
            target=init_weight_transfer_engine,
            kwargs={
                "base_url": BASE_URL,
                "master_address": master_address,
                "master_port": master_port,
                "rank_offset": rank_offset,
                "world_size": world_size,
                "backend": "nccl",
            },
        )
        init_thread.start()

        self._model_update_group = NCCLWeightTransferEngine.trainer_init(
            dict(
                master_address=master_address,
                master_port=master_port,
                world_size=world_size,
            ),
        )
        init_thread.join()

    def sync_weights(self, trainer_model: nn.Module) -> None:
        pause_generation(BASE_URL)

        names: list[str] = []
        dtype_names: list[str] = []
        shapes: list[list[int]] = []
        for name, param in trainer_model.named_parameters():
            names.append(name)
            dtype_names.append(str(param.dtype).split(".")[-1])
            shapes.append(list(param.shape))

        update_thread = threading.Thread(
            target=update_weights,
            args=(BASE_URL, names, dtype_names, shapes, True),
        )
        update_thread.start()

        print("Broadcasting weights via NCCL...")
        trainer_args = NCCLTrainerSendWeightsArgs(
            group=self._model_update_group,
            packed=True,
        )
        NCCLWeightTransferEngine.trainer_send_weights(
            iterator=trainer_model.named_parameters(),
            trainer_args=trainer_args,
        )
        update_thread.join()

        resume_generation(BASE_URL)


def main():
    rollout = NCCLHttpRollout(MODEL_NAME)
    device = f"cuda:{rollout.inference_world_size}"
    torch.accelerator.set_device_index(device)

    print(f"Loading training model: {MODEL_NAME}")
    train_model = cast(
        nn.Module,
        AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.bfloat16),
    )
    train_model.to(torch.device(device))

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
