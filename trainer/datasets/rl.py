from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
import asyncio
import json
import os
from typing import Any, Callable, Dict, List, Tuple

from loguru import logger
from omegaconf import DictConfig, OmegaConf
import torch
from transformers import AutoTokenizer

from trainer.datasets import base
from trainer.datasets.base import BaseDataset, get_tensor_dict
from trainer.utils.communication import async_request


@dataclass
class Sample:

    class Status(Enum):

        RUNNING = "running"
        ABORTED = "aborted"
        DONE = "done"

    # for initialization
    sample: Dict[str, Any] = field(default_factory=dict)

    # for environment interaction
    state_text: str = ""
    action_text: str = ""

    # for training
    state_dict: Dict[str, List[int | float]] = field(default_factory=dict)
    state_dicts: List[Dict[str, List[int | float]]] = field(default_factory=list)

    # for logging
    turn: int = 0
    metrics: Dict[str, List[float | int | bool]] = field(
        default_factory=lambda: defaultdict(list)
    )

    # for partial rollout
    status: Status = Status.RUNNING
    previous_action_text: str = ""
    previous_response_length: int = 0

    def to_json(self) -> Dict[str, Any]:

        data = self.__dict__
        data["metrics"] = dict(self.metrics)
        data["status"] = self.status.value
        return data


def initialize_state_dict(
    tokenizer: AutoTokenizer, state_text: str
) -> Dict[str, List[int | float]]:

    state = tokenizer.encode(state_text, add_special_tokens=False)
    return {
        "states": state,
        "actions": len(state) * [0],
        "action_mask": len(state) * [0],
        "logps": len(state) * [0.0],
        "rewards": len(state) * [0.0],
    }


def add_llm_response(sample: Sample, response: Dict[str, Any]) -> None:

    # `previous_action_text` is non-empty if aborted before
    sample.action_text = sample.previous_action_text + response["text"]

    # encode(decode(tokens)) may not be identical to tokens. Therefore,
    # token-in-token-out is necessary to guanartee that tokens fed into
    # training and inference engines are identical
    # https://github.com/OpenRLHF/OpenRLHF/pull/1094
    # https://github.com/THUDM/slime/pull/117
    meta_info = response["meta_info"]
    if (
        "output_token_logprobs" in meta_info
        and len(meta_info["output_token_logprobs"][0]) == 3
    ):  # TODO: is this condition correct?
        logp, action, _ = map(list, zip(*meta_info["output_token_logprobs"]))
        sample.state_dict["states"].extend(action)
        sample.state_dict["actions"].extend(action)
        sample.state_dict["action_mask"].extend(len(action) * [1])
        sample.state_dict["logps"].extend(logp)
        sample.state_dict["rewards"].extend(len(action) * [0.0])
        # actual rewards will be overwritten in `add_env_response`

    finish_reason = meta_info["finish_reason"]["type"]
    logger.info(f"{finish_reason=}")
    if finish_reason == "abort":
        # User may mask action tokens to avoid off-policy training
        sample.status = Sample.Status.ABORTED
        sample.previous_action_text = sample.action_text
        sample.previous_response_length += meta_info["completion_tokens"]
        logger.info(
            f"{sample.status=}, {sample.previous_response_length=}, "
            f"{sample.previous_action_text=}"
        )
        return

    sample.turn += 1
    sample.metrics["response_length"].append(
        sample.previous_response_length + meta_info["completion_tokens"]
    )
    sample.metrics["length_clip_ratio"].append(finish_reason == "length")

    # reset if not aborted
    sample.previous_action_text = ""
    sample.previous_response_length = 0


def add_env_response(
    tokenizer: AutoTokenizer, sample: Sample, response: Dict[str, Any]
) -> None:

    sample.state_dict["rewards"][-1] = response["reward"]

    if response["done"]:

        sample.status = Sample.Status.DONE
        sample.state_dicts.append(sample.state_dict)
        sample.metrics["turns"].append(sample.turn)
        sample.metrics["rewards"].append(
            sum([state_dict["rewards"][-1] for state_dict in sample.state_dicts])
        )
        return

    if response["next_state"].startswith(sample.state_text + sample.action_text):
        state_dict_delta = initialize_state_dict(
            tokenizer,
            response["next_state"][len(sample.state_text + sample.action_text) :],
        )
        for key, value in state_dict_delta.items():
            sample.state_dict[key].extend(value)
    else:
        # If the previous state is not a prefix of the next state, the trajectory will
        # contain multiple sequences
        sample.state_dicts.append(sample.state_dict)
        sample.state_dict = initialize_state_dict(tokenizer, response["next_state"])
    sample.state_text = response["next_state"]


def _get_initial_state_text(config: DictConfig, tokenizer: AutoTokenizer, sample: Sample) -> str:
    if config.apply_chat_template:
        return tokenizer.apply_chat_template(
            sample.sample[config.messages_key],
            add_generation_prompt=True,
            tokenize=False,
        )
    return sample.sample[config.prompt_key]


def _prepare_sample_for_generation(
    config: DictConfig,
    tokenizer: AutoTokenizer,
    sample: Sample,
) -> bool:
    if sample.status == Sample.Status.DONE:
        logger.info(f"{sample.status=}")
        return False
    if sample.status == Sample.Status.ABORTED:
        sample.status = Sample.Status.RUNNING
        logger.info(f"{sample.status=}")
        return True

    sample.state_text = _get_initial_state_text(config, tokenizer, sample)
    sample.state_dict = initialize_state_dict(tokenizer, sample.state_text)
    logger.info(f"{sample.status=}, {sample.state_text=}")
    return True


def _build_generate_request(
    sampling_params: Dict[str, Any],
    sample: Sample,
) -> Dict[str, Any]:
    max_new_tokens = sampling_params["max_new_tokens"] - sample.previous_response_length
    request_payload = {
        "input_ids": sample.state_dict["states"],
        "sampling_params": {
            **sampling_params,
            "max_new_tokens": max_new_tokens,
            "no_stop_trim": True,
        },
        "return_logprob": True,
    }
    logger.info(f"{max_new_tokens=}")
    return request_payload


async def _generate_one_turn(
    router_url: str,
    request_payload: Dict[str, Any],
) -> Dict[str, Any]:
    return await async_request(
        router_url,
        "generate",
        json=request_payload,
    )


async def base_generate(
    config: DictConfig,
    tokenizer: AutoTokenizer,
    router_url: str,
    sample: Sample,
    env_step_fn: Callable,
) -> None:
    """
    A typical generate function where user only needs to provide the `env_step`
    function. User may provide their own `generate` function for advanced use.
    """
    sampling_params = OmegaConf.to_container(config.sampling_params)
    logger.info(f"{router_url=}, {sample.status=}")

    should_continue = _prepare_sample_for_generation(config, tokenizer, sample)
    if not should_continue:
        return

    while True:
        request_payload = _build_generate_request(sampling_params, sample)
        llm_response = await _generate_one_turn(router_url, request_payload)
        add_llm_response(sample, llm_response)
        if sample.status == Sample.Status.ABORTED:
            return

        env_response = await env_step_fn(sample)
        add_env_response(tokenizer, sample, env_response)
        if sample.status == Sample.Status.DONE:
            return


class SampleGroup:

    def __init__(
        self, config: DictConfig, tokenizer: AutoTokenizer, sample: Dict[str, Any]
    ):

        self.config = config
        self.tokenizer = tokenizer
        self.samples = [
            Sample(sample=deepcopy(sample)) for _ in range(config.responses_per_prompt)
        ]

    async def generate(self, router_url: str, generate_fn: Callable) -> "SampleGroup":
        """
        This function packs the generation tasks of samples within a group into a single task so that they will return togather.
        """
        await asyncio.gather(
            *(
                generate_fn(self.config, self.tokenizer, router_url, sample)
                for sample in self.samples
            )
        )
        return self

    def print(self):

        sample = self.samples[0]
        print("\n")
        print(sample.state_text + sample.action_text)
        print("[Reward]", sample.metrics["rewards"][0])

    def save(self, step):

        data = [sample.to_json() for sample in self.samples]
        os.makedirs(self.config.save_dir, exist_ok=True)
        with open(f"{self.config.save_dir}/step{step}.jsonl", "a") as f:
            f.write(json.dumps(data) + "\n")

    def to_all_tensor_dicts_and_metrics(
        self,
    ) -> Tuple[
        List[List[Dict[str, torch.Tensor]]], Dict[str, List[float | int | bool]]
    ]:

        all_tensor_dicts, metrics = [], defaultdict(list)
        for sample in self.samples:
            tensor_dicts = []
            for state_dict in sample.state_dicts:
                tensor_dict = get_tensor_dict(
                    state_dict["states"],
                    state_dict["actions"],
                    state_dict["action_mask"],
                )
                tensor_dict["llm_logps"] = torch.FloatTensor(state_dict["logps"][1:])
                tensor_dict["rewards"] = torch.FloatTensor(state_dict["rewards"][1:])
                tensor_dicts.append(tensor_dict)
            all_tensor_dicts.append(tensor_dicts)
            for key, value in sample.metrics.items():
                metrics[key].extend(value)
        return all_tensor_dicts, metrics


class RLDataset(BaseDataset):

    def __getitem__(self, idx: int) -> SampleGroup:

        sample = self.dataset[idx]
        return SampleGroup(self.dataset_config, self.tokenizer, sample)

    def collate_fn(self, batch: Tuple[SampleGroup]) -> SampleGroup:
        return batch[0]

    def convert_to_messages(
        self, sample: Dict[str, Any]
    ) -> List[base.Message] | tuple[List[base.Message], List[base.Message]]: ...
