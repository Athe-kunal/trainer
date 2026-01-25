from typing import Optional, Union, Dict, List, Tuple, Generator, Sequence
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
from vllm import AsyncEngineArgs, AsyncLLMEngine
from vllm.distributed.parallel_state import init_distributed_environment
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

PROCESSES = []


def shutdown_processes_when_exit(func):

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
        self._prepare_environment_variables()

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
            self._launch_vllm_engine()

    def _prepare_device_mesh(self):

        world_size = dist.get_world_size()
        tp_size = self.config.server_args.tp_size
        assert (
            world_size % tp_size == 0
        ), f"World_size {world_size} must be divisible by tp_size {tp_size}."

        self.device_mesh = dist.device_mesh.init_device_mesh(
            "cpu",
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

    def _launch_vllm_engine(self):

        # Convert server args to vLLM engine args
        server_args = OmegaConf.to_container(self.config.server_args)

        engine_args = AsyncEngineArgs(
            model=server_args.get("model_path"),
            tensor_parallel_size=server_args.get("tp_size", 1),
            pipeline_parallel_size=server_args.get("pp_size", 1),
            trust_remote_code=True,
            disable_log_requests=True,
            max_model_len=server_args.get("max_model_len"),
            gpu_memory_utilization=server_args.get("gpu_memory_utilization", 0.9),
            dtype=server_args.get("dtype", "auto"),
            seed=server_args.get("seed", 0),
            enable_prefix_caching=server_args.get("enable_prefix_caching", False),
            **{
                k: v
                for k, v in server_args.items()
                if k
                not in [
                    "model_path",
                    "tp_size",
                    "pp_size",
                    "max_model_len",
                    "gpu_memory_utilization",
                    "dtype",
                    "seed",
                    "enable_prefix_caching",
                ]
            },
        )

        # Initialize vLLM async engine
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)

        # Store worker URL for compatibility
        self.worker_url = f"vllm://localhost:{get_available_port()}"

        router_url = broadcast_object(
            self.router_url if dist.get_rank() == 0 else None,
            process_group=self.device_mesh["dp"].get_group(),
            group_src=0,
        )

        # In vLLM, we don't need to register with router in the same way
        # This would need to be adapted based on your router implementation
        self.worker_urls = gather_and_concat_list(
            [self.worker_url], self.device_mesh["dp"].get_group()
        )

    def _launch_router_process(self):

        # Router launch remains the same as it's custom code
        from sglang_router.launch_router import RouterArgs, launch_router

        router_args = RouterArgs(
            host=get_host(), port=get_available_port(), log_level="error"
        )
        self.router_url = f"http://{router_args.host}:{router_args.port}"

        router_process = multiprocessing.Process(
            target=launch_router, args=(router_args,)
        )
        router_process.start()
        PROCESSES.append(router_process)
        sync_request(self.router_url, "health", "GET", 10)

    def _prepare_environment(self):

        spec = importlib.util.spec_from_file_location(
            "custom_module", self.config.env_path
        )
        self.env = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.env)

    @time_logger("rollout")
    async def __call__(
        self, train: bool, step: int
    ) -> Optional[Tuple[Optional[Dict[str, torch.Tensor]], Optional[torch.Tensor]]]:

        def _schedule_tasks(sample_groups: Sequence[SampleGroup]):

            for sample_group in sample_groups:
                pendings.add(
                    asyncio.create_task(
                        sample_group.generate(self.router_url, self.env.generate)
                    )
                )

        if dist.get_rank() == 0:

            config = self.config.train if train else self.config.test
            dataloader = self.train_dataloader if train else self.test_dataloader
            groups_to_complete = config.prompts_per_rollout or len(dataloader)

            tbar = progress_bar(total=groups_to_complete, desc="Rollout")

            pendings, first_iter = set(), True
            filtered_groups, completed_groups = 0, 0
            all_tensor_dicts: List[List[Dict[str, torch.Tensor]]] = []
            metrics: Dict[str, List[Union[float, int, bool]]] = defaultdict(list)

            if train and config.partial_rollout:
                _schedule_tasks(self.sample_buffer)

            while completed_groups < groups_to_complete:

                if first_iter or (train and config.partial_rollout):
                    sample_groups = dataloader(groups_to_complete - len(pendings))
                    _schedule_tasks(sample_groups)

                done, pendings = await asyncio.wait(
                    pendings, return_when=asyncio.FIRST_COMPLETED
                )

                for task in done:
                    if completed_groups < groups_to_complete:
                        tbar.update()
                    completed_groups += 1
                    sample_group = task.result()
                    if first_iter:
                        sample_group.print()
                        first_iter = False
                    await asyncio.to_thread(sample_group.save, step)
                    all_tensor_dicts_delta, metrics_delta = (
                        sample_group.to_all_tensor_dicts_and_metrics()
                    )
                    for k, v in metrics_delta.items():
                        metrics[k].extend(v)
                    if (
                        train
                        and config.dynamic_filtering
                        and len(metrics_delta["rewards"]) > 1
                        and torch.tensor(metrics_delta["rewards"]).std() == 0
                    ):
                        filtered_groups += 1
                        continue
                    all_tensor_dicts.extend(all_tensor_dicts_delta)

            if train and config.partial_rollout:
                # vLLM doesn't have pause_generation, would need custom implementation
                # await async_request(self.worker_urls, "pause_generation")
                done, _ = await asyncio.wait(pendings)
                self.sample_buffer = [task.result() for task in done]
                # await async_request(self.worker_urls, "continue_generation")

            metrics["dynamic_filtering_ratio"].append(
                filtered_groups / completed_groups
            )
            suffix = "train" if train else "test"
            metrics = {f"{k}/{suffix}": v for k, v in metrics.items()}
            gather_and_log(metrics, step)

        # Use GLOO group to avoid affecting vLLM engine
        await asyncio.to_thread(dist.barrier, group=get_gloo_group())

        if not train:
            return

        if self.device_mesh["tp"].get_local_rank() == 0:
            # vLLM handles memory differently, may need custom implementation
            # In vLLM, you would typically free KV cache blocks
            pass

        if dist.get_rank() != 0:
            return None, None

        tensor_dicts: List[Dict[str, torch.Tensor]] = [
            td for tds in all_tensor_dicts for td in tds
        ]
        tensor_dict: Dict[str, torch.Tensor] = pack_tensor_dicts(tensor_dicts)
        seqs = torch.LongTensor(
            [len(tensor_dicts) for tensor_dicts in all_tensor_dicts]
        )
        cu_seqs = torch.cumsum(torch.cat((torch.LongTensor([0]), seqs)), dim=0)

        return tensor_dict, cu_seqs

    @torch.no_grad()
    def update(
        self, named_tensor_generator: Generator[Tuple[str, torch.Tensor], None, None]
    ):

        torch.cuda.empty_cache()
        dist.barrier(group=get_gloo_group())

        if self.device_mesh["tp"].get_local_rank() == 0:
            # vLLM weight update would need custom implementation
            # vLLM doesn't have the same memory occupation API
            pass

        def _update_vllm_weights(
            dtype_to_named_tensors: Dict[torch.dtype, List[Tuple[str, torch.Tensor]]],
        ):

            torch.cuda.synchronize()

            # Collect all tensors for this bucket
            all_named_tensors = []
            for named_tensors in dtype_to_named_tensors.values():
                all_named_tensors.extend(named_tensors)

            gathered_named_tensors = (
                [None for _ in range(self.device_mesh["tp"].size())]
                if self.device_mesh["tp"].get_local_rank() == 0
                else None
            )
            dist.gather_object(
                all_named_tensors,
                gathered_named_tensors,
                group_dst=0,
                group=self.device_mesh["tp"].get_group(),
            )

            if self.device_mesh["tp"].get_local_rank() == 0:
                # vLLM weight loading - would need custom implementation
                # vLLM's model runner has load_model() but not dynamic update like SGLang
                # You might need to:
                # 1. Use vLLM's model.load_weights() method
                # 2. Or access model parameters directly: self.engine.engine.model_executor.driver_worker.model_runner.model

                # Example approach (needs adaptation):
                model = (
                    self.engine.engine.model_executor.driver_worker.model_runner.model
                )
                state_dict = {
                    name: tensor for name, tensor in gathered_named_tensors[0]
                }
                model.load_state_dict(state_dict, strict=False)

        dtype_to_named_tensors = defaultdict(list)
        bucket_size = 0
        for name, tensor in named_tensor_generator:
            param_size = tensor.numel() * tensor.element_size()

            if bucket_size > 0 and bucket_size + param_size > (
                self.config.bucket_size << 20
            ):

                _update_vllm_weights(dtype_to_named_tensors)
                dtype_to_named_tensors = defaultdict(list)
                bucket_size = 0

            tensor = tensor.to(torch.cuda.current_device(), non_blocking=True).detach()
            if isinstance(tensor, DTensor):
                # async version of `tensor.full_tensor()`
                tensor = tensor.redistribute(
                    placements=[Replicate()] * tensor.device_mesh.ndim, async_op=True
                ).to_local()

            dtype_to_named_tensors[tensor.dtype].append((name, tensor))
            bucket_size += param_size

        _update_vllm_weights(dtype_to_named_tensors)

        if self.device_mesh["tp"].get_local_rank() == 0:
            # Resume operations if needed
            pass
        dist.barrier(group=get_gloo_group())
