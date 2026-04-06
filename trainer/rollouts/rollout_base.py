import abc
import asyncio
import importlib.util
from collections import defaultdict
from types import ModuleType
from typing import Any, NamedTuple, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
from loguru import logger
from omegaconf import DictConfig
from openai import AsyncOpenAI, OpenAI
from transformers import AutoTokenizer

from trainer.datasets import RLDataset
from trainer.datasets.base import get_dataloader, pack_tensor_dicts
from trainer.datasets.rl import SampleGroup
from trainer.rollouts.rollout_utils import BASE_URL, agenerate_completions, generate_completions
from trainer.utils.logging import gather_and_log, progress_bar, time_logger


class _RolloutResult(NamedTuple):
    tensor_dict: Optional[dict[str, torch.Tensor]]
    cu_seqs: Optional[torch.Tensor]


class HttpRollout(abc.ABC):
    def __init__(
        self, config: Optional[DictConfig] = None, model_name: str = "facebook/opt-125m"
    ) -> None:
        self.client = OpenAI(
            base_url=f"{BASE_URL}/v1",
            api_key="EMPTY",
        )
        self.aclient = AsyncOpenAI(
            base_url=f"{BASE_URL}/v1",
            api_key="EMPTY",
        )
        self.config = config
        self.model_name = model_name
        self.train_dataloader = None
        self.test_dataloader = None
        self.env_module = None
        if self.config is not None and dist.get_rank() == 0:
            self._initialize_rank_zero_state()

    @abc.abstractmethod
    def sync_weights(self, trainer_model: nn.Module) -> None: ...

    def generate(
        self,
        prompts: list[str],
        max_tokens: int = 32,
        temperature: float = 0,
        n: int = 1,
    ) -> list[list[str]]:
        return generate_completions(
            client=self.client,
            model=self.model_name,
            prompts=prompts,
            max_tokens=max_tokens,
            temperature=temperature,
            n=n,
        )

    def agenerate(
        self,
        prompts: list[str],
        max_tokens: int = 32,
        temperature: float = 0,
        n: int = 1,
    ) -> list[list[str]]:
        return agenerate_completions(
            aclient=self.aclient,
            model=self.model_name,
            prompts=prompts,
            max_tokens=max_tokens,
            temperature=temperature,
            n=n,
        )

    def _initialize_rank_zero_state(self) -> None:
        tokenizer = AutoTokenizer.from_pretrained(
            self.config.server_args.model_path,
            trust_remote_code=True,
        )
        self.train_dataloader, self.test_dataloader = get_dataloader(
            RLDataset,
            self.config,
            tokenizer,
            1,
        )
        logger.info(
            f"{self.model_name=}, "
            f"{len(self.train_dataloader)=}, {len(self.test_dataloader)=}"
        )
        self.env_module = self._load_env_module(self.config.env_path)

    def _load_env_module(self, env_path: Optional[str]) -> ModuleType:
        if not env_path:
            raise ValueError(
                "rollout.env_path must be set for GRPO rollouts. "
                f"Received {env_path=}"
            )
        module_spec = importlib.util.spec_from_file_location("trainer_env_module", env_path)
        if module_spec is None or module_spec.loader is None:
            raise RuntimeError(f"Failed to create module spec for {env_path=}")
        module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)
        if not hasattr(module, "generate"):
            raise AttributeError(f"Environment module must expose a `generate` function. {env_path=}")
        return module

    async def _run_generation(
        self,
        train: bool,
        step: int,
    ) -> _RolloutResult:
        rollout_config = self.config.train if train else self.config.test
        dataloader = self.train_dataloader if train else self.test_dataloader
        groups_to_complete = rollout_config.prompts_per_rollout or len(dataloader)
        logger.info(f"{train=}, {step=}, {groups_to_complete=}")

        first_iter = True
        filtered_groups = 0
        all_tensor_dicts: list[list[dict[str, torch.Tensor]]] = []
        fallback_tensor_dicts: list[list[dict[str, torch.Tensor]]] = []
        metrics: dict[str, list[float | int | bool]] = defaultdict(list)

        tbar = progress_bar(total=groups_to_complete, desc="Rollout")
        sample_groups = dataloader(groups_to_complete)
        for sample_group in sample_groups:
            assert self.env_module is not None
            await sample_group.generate(BASE_URL, self.env_module.generate)
            completed_group = sample_group
            if first_iter:
                completed_group.print()
                first_iter = False
            await asyncio.to_thread(completed_group.save, step)
            tensor_dicts, metrics_delta = completed_group.to_all_tensor_dicts_and_metrics()
            for key, values in metrics_delta.items():
                metrics[key].extend(values)
            if self._should_filter_group(train, rollout_config, metrics_delta):
                filtered_groups += 1
                fallback_tensor_dicts.extend(tensor_dicts)
                continue
            all_tensor_dicts.extend(tensor_dicts)
            tbar.update(1)
        tbar.close()

        if train and not all_tensor_dicts and fallback_tensor_dicts:
            logger.warning(
                "All groups were filtered out. Reusing filtered groups to keep training stable."
            )
            all_tensor_dicts = fallback_tensor_dicts

        if metrics:
            completed_groups = max(1, groups_to_complete)
            metrics["dynamic_filtering_ratio"].append(filtered_groups / completed_groups)
            suffix = "train" if train else "test"
            gather_and_log(
                {f"{k}/{suffix}": v for k, v in metrics.items()},
                step,
            )

        if not train:
            return _RolloutResult(None, None)
        return self._build_rollout_result(all_tensor_dicts)

    def _should_filter_group(
        self,
        train: bool,
        rollout_config: DictConfig,
        metrics_delta: dict[str, list[float | int | bool]],
    ) -> bool:
        if not train or not rollout_config.dynamic_filtering:
            return False
        rewards = metrics_delta.get("rewards")
        if rewards is None or len(rewards) <= 1:
            return False
        rewards_tensor = torch.as_tensor(rewards, dtype=torch.float32)
        return bool(rewards_tensor.std() == 0)

    def _build_rollout_result(
        self,
        all_tensor_dicts: list[list[dict[str, torch.Tensor]]],
    ) -> _RolloutResult:
        flattened_tensor_dicts: list[dict[str, torch.Tensor]] = [
            tensor_dict for tensor_dicts in all_tensor_dicts for tensor_dict in tensor_dicts
        ]
        if not flattened_tensor_dicts:
            logger.warning("No rollout samples left after filtering; returning empty tensors.")
            empty_cu_seqs = torch.zeros(1, dtype=torch.long)
            return _RolloutResult({}, empty_cu_seqs)
        tensor_dict = pack_tensor_dicts(flattened_tensor_dicts)
        seqs = torch.LongTensor([len(tensor_dicts) for tensor_dicts in all_tensor_dicts])
        cu_seqs = torch.cumsum(torch.cat((torch.LongTensor([0]), seqs)), dim=0)
        return _RolloutResult(tensor_dict, cu_seqs)

    @time_logger("rollout")
    async def __call__(
        self, train: bool, step: int
    ) -> tuple[Optional[dict[str, torch.Tensor]], Optional[torch.Tensor]]:
        if self.config is None:
            raise RuntimeError("Rollout config is required before calling __call__.")
        if dist.get_rank() != 0:
            return None, None
        rollout_result = await self._run_generation(train, step)
        return rollout_result.tensor_dict, rollout_result.cu_seqs
