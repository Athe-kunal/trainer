from typing import Callable, Optional, Union, Dict, List, Tuple, Generator, Sequence
from omegaconf import OmegaConf, DictConfig
import os
import asyncio
import importlib
import functools
import multiprocessing
from collections import defaultdict
import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor, Replicate
from transformers import AutoTokenizer
from sglang.srt.server_args import ServerArgs
from sglang.srt.entrypoints.http_server_engine import launch_server_process
from sglang.srt.utils import MultiprocessingSerializer
from sglang_router.launch_router import RouterArgs, launch_router
from trainer.datasets import get_dataloader, pack_tensor_dicts, RLDataset, SampleGroup
from trainer.utils.communication import (
    get_host,
    get_available_port,
    get_gloo_group,
    broadcast_object,
    gather_and_concat_list,
    sync_request,
    async_request,
)
from trainer.utils.logging import progress_bar, time_logger, gather_and_log

from sglang.srt.patch_torch import monkey_patch_torch_reductions

from trainer.flattened_tensor import FlattenedTensorBucket

PROCESSES: list[multiprocessing.Process] = []


def shutdown_processes_when_exit(func: Callable):

    @functools.wraps(func)
    async def wrapper(*args, **kwargs):

        try:
            return await func(*args, **kwargs)
        finally:
            for process in PROCESSES:
                process.terminate()
                process.join(timeout=3)
                if process.is_alive():
                    process.kill()

    return wrapper


class Rollout:
    def __init__(self, config: DictConfig):
        self.config = config
        self._prepare_device_mesh()
        self._preare_environment_variables()

        if dist.get_rank() == 0:
            self.tokenizer = AutoTokenizer.from_pretrained(
                config.server_args.model_path, trust_remote_code=True
            )
            self.train_dataloader, self.test_dataloader = get_dataloader(
                RLDataset, config, self.tokenizer, 1
            )

            self._prepare_environment()
            self.sample_buffer: List[SampleGroup] = []

            self._launch_router_process()

        dist.barrier(group=get_gloo_group())

        if self.device_mesh["tp"].get_local_rank() == 0:
            self._launch_server_process()

    def _prepare_device_mesh(self):

        world_size = dist.get_world_size()
        tp_size = self.config.server_args.tp_size
        assert (
            world_size % tp_size == 0
        ), f"World_size {world_size} must be divisible by tp_size {tp_size}."

        self.device_mesh = dist.device_mesh.init_device_mesh(
            "cuda",
            mesh_dim_names=("dp", "tp"),
            mesh_shape=(world_size // tp_size, tp_size),
        )

    def _prepare_environment_variables(self):

        if "TORCHELASTIC_USE_AGENT_STORE" in os.environ.keys():
            del os.environ["TORCHELASTIC_USE_AGENT_STORE"]
        cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if cuda_visible_devices:
            cuda_visible_devices = cuda_visible_devices.split(",")
            cuda_visible_device = cuda_visible_devices[int(os.environ["LOCAL_RANK"])]
        else:
            cuda_visible_device = os.environ["LOCAL_RANK"]
        cuda_visible_devices = self.device_mesh["tp"].size() * [None]
        dist.all_gather_object(
            cuda_visible_devices,
            cuda_visible_device,
            self.device_mesh["tp"].get_group(),
        )
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(cuda_visible_devices)
        monkey_patch_torch_reductions()

    def _launch_server_process(self):

        server_args = OmegaConf.to_container(self.config.server_args)
        server_args = ServerArgs(
            enable_memory_saver=True,
            host=get_host(),
            port=get_available_port(),
            log_level="error",
            **server_args,
        )
        server_process = launch_server_process(server_args)
        PROCESSES.append(server_process)

        self.worker_url = server_args.url()

        router_url = server_args.url()

        router_url = broadcast_object(
            self.router_url if dist.get_rank() == 0 else None,
            process_group=self.device_mesh["dp"].get_group(),
            group_src=0,
        )
        sync_request(router_url, f"add_worker?url={self.worker_url}")
        self.worker_urls = gather_and_concat_list(
            [self.worker_url], self.device_mesh["dp"].get_group()
        )

    def _launch_router_process(self):

        router_args = RouterArgs(
            host=get_host(), port=get_available_port(), log_level="error"
        )
        self.router_url
