from typing import Any, Dict, List, Optional, Tuple, Union
from collections import defaultdict
from omegaconf import DictConfig
import os
import asyncio
import importlib
import functools
import torch
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.distributed as dist
from torch.distributed.tensor import DTensor, Replicate
import ray
from ray.util.placement_group import placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from vllm import LLM, SamplingParams
from vllm.outputs import RequestOutput

from trainer.utils.communication import (
    get_host,
    get_available_port,
    get_gloo_group,
    broadcast_object,
    gather_and_concat_list,
    sync_request,
    async_request,
)
from trainer.datasets import get_dataloader, pack_tensor_dicts, RLDataset, SampleGroup
from trainer.utils.logging import progress_bar, time_logger, gather_and_log

from trainer.workers.network_utils import get_ip, get_open_port
from trainer.workers.rlhf_utils import stateless_init_process_group

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


class InferenceLLM(LLM):
    """Configure the vLLM worker for Ray placement group execution."""

    def __init__(self, *args, **kwargs):
        # Remove the top-level CUDA_VISIBLE_DEVICES variable set by Ray
        # so that vLLM can manage its own device placement within the worker.
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        super().__init__(*args, **kwargs)


class Rollout:
    def __init__(self, config: DictConfig) -> None:

        self.config = config
        assert (
            self.config.inference_gpu_ids
        ), f"For Rollout, please provide `inference_gpu_ids`"
        assert self.config.train_gpu_ids, f"Please provide `train_gpu_ids`"

        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.server_args.model_path, trust_remote_code=True
        )

        self.llm = self._setup_ray()
        if dist.get_rank() == 0:
            self._prepare_environment()
            self.train_dataloader, self.test_dataloader = get_dataloader(
                RLDataset, config, self.tokenizer, 1
            )
            self.sample_buffer: List[SampleGroup] = []
        self.model_update_group = self._setup_weight_sync()

    def _setup_ray(self) -> Any:

        num_gpus = len(self.config.inference_gpu_ids.split(","))
        os.environ["CUDA_VISIBLE_DEVICES"] = self.config.inference_gpu_ids
        ray.init()
        pg_inference = placement_group([{"GPU": 1, "CPU": 0}] * num_gpus)
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
            model=self.config.server_args.model_path,
            enforce_eager=True,
            worker_extension_cls="trainer.workers.rlhf_utils.WorkerExtension",
            tensor_parallel_size=self.config.server_args.tp_size,
            distributed_executor_backend="ray",
        )
        return llm

    def _prepare_environment(self):

        spec = importlib.util.spec_from_file_location(
            "custom_module", self.config.env_path
        )
        self.env = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.env)

    def _setup_weight_sync(self) -> Any:

        master_address = get_ip()
        master_port = get_open_port()

        # Calculate world size: 1 (rank-0 trainer) + num_vllm_workers
        # num_vllm_workers is based on tensor_parallel_size, not number of GPUs
        num_vllm_workers = self.config.server_args.tp_size
        world_size = 1 + num_vllm_workers

        # Initialize weight sync group on vLLM workers
        # rank_offset=1 means vLLM workers start at rank 1 (trainer is rank 0)
        handle = self.llm.collective_rpc.remote(
            "init_weight_update_group",
            args=(master_address, master_port, 1, world_size),
        )
        # Only rank-0 creates the weight sync group
        # model_update_group = None
        # if dist.get_rank() == 0:
        train_device_id = int(self.config.train_gpu_ids.split(",")[0])
        logger.info(f"{train_device_id=} | {world_size=}")
        model_update_group = stateless_init_process_group(
            master_address=master_address,
            master_port=master_port,
            rank=0,  # rank-0 is rank 0 in the weight sync group
            world_size=world_size,
            # device=torch.device(f"cuda:{train_device_id}"),
            device=torch.device("cuda:0"),
        )

        ray.get(handle)
        return model_update_group

    async def _generate_with_vllm(self, sample_group: SampleGroup) -> SampleGroup:
        """
        Adapted generate function for vLLM+Ray that mimics the SGLang behavior.
        This wraps the vLLM generation to work with the environment interaction loop.
        """
        # Use the environment's generate function with vLLM backend
        sampling_params = SamplingParams(**self.config.sampling_params)

        async def vllm_generate_fn(config, tokenizer, llm, sample):
            """Generate function adapted for vLLM that works with the env step function"""
            from trainer.datasets.rl import (
                initialize_state_dict,
                add_env_response,
                Sample,
            )

            # Initialize state
            if sample.status == Sample.Status.RUNNING:
                if config.apply_chat_template:
                    sample.state_text = tokenizer.apply_chat_template(
                        sample.sample[config.messages_key],
                        add_generation_prompt=True,
                        tokenize=False,
                    )
                else:
                    sample.state_text = sample.sample[config.prompt_key]
                sample.state_dict = initialize_state_dict(tokenizer, sample.state_text)
            elif sample.status == Sample.Status.ABORTED:
                sample.status = Sample.Status.RUNNING
            elif sample.status == Sample.Status.DONE:
                return

            # Environment interaction loop
            while True:
                # Generate with vLLM
                prompt = tokenizer.decode(sample.state_dict["states"])
                vllm_sampling_params = SamplingParams(
                    **{
                        **sampling_params.__dict__,
                        "max_tokens": sampling_params.max_tokens
                        - sample.previous_response_length,
                        "logprobs": 1,
                    }
                )

                outputs = await asyncio.to_thread(
                    ray.get, llm.generate.remote([prompt], vllm_sampling_params)
                )

                # Process vLLM output into response format
                output = outputs[0]
                response = {
                    "text": output.outputs[0].text,
                    "meta_info": {
                        "output_token_logprobs": (
                            [
                                (token.logprob, token.token_id, token.decoded_token)
                                for token in (output.outputs[0].logprobs or [])
                            ]
                            if output.outputs[0].logprobs
                            else []
                        ),
                        "completion_tokens": len(output.outputs[0].token_ids),
                        "finish_reason": {
                            "type": output.outputs[0].finish_reason or "stop"
                        },
                    },
                }

                # Add LLM response
                from trainer.datasets.rl import add_llm_response

                add_llm_response(sample, response)
                if sample.status == Sample.Status.ABORTED:
                    return

                # Environment step
                env_response = await self.env.step(sample)
                add_env_response(tokenizer, sample, env_response)
                if sample.status == Sample.Status.DONE:
                    return

        # Run generation for all samples in the group
        await asyncio.gather(
            *(
                vllm_generate_fn(self.config, self.tokenizer, self.llm, sample)
                for sample in sample_group.samples
            )
        )
        return sample_group

    @time_logger("rollout")
    async def __call__(
        self, train: bool, step: int
    ) -> Optional[Tuple[Optional[Dict[str, torch.Tensor]], Optional[torch.Tensor]]]:

        def _schedule_tasks(sample_groups: List[SampleGroup]):
            for sample_group in sample_groups:
                pendings.add(
                    asyncio.create_task(self._generate_with_vllm(sample_group))
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
                done, _ = await asyncio.wait(pendings)
                self.sample_buffer = [task.result() for task in done]

            metrics["dynamic_filtering_ratio"].append(
                filtered_groups / completed_groups if completed_groups > 0 else 0
            )
            suffix = "train" if train else "test"
            metrics = {f"{k}/{suffix}": v for k, v in metrics.items()}
            gather_and_log(metrics, step)

        # Use GLOO group to avoid affecting vLLM operations
        await asyncio.to_thread(dist.barrier, group=get_gloo_group())

        if not train:
            return

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

    def update(self, train_model: AutoModelForCausalLM):
        for name, p in train_model.named_parameters():
            dtype_name = str(p.dtype).split(".")[-1]
            handle = self.llm.collective_rpc.remote(
                "update_weight", args=(name, dtype_name, p.shape)
            )
            self.model_update_group.broadcast(
                p, src=0, stream=torch.cuda.current_stream()
            )
            ray.get(handle)


if __name__ == "__main__":
    """
    Test script for the Rollout class with vLLM+Ray backend.

    This script:
    1. Creates a temporary environment module with dummy step() and get_sampling_params() functions
    2. Creates a temporary dataset with sample prompts
    3. Initializes the Rollout class with minimal configuration
    4. Runs both test and train rollouts
    5. Tests weight synchronization between training and inference models

    To run: python trainer/workers/rollout.py

    Requirements:
    - Single GPU available (or modify 'inference_gpu_ids' and 'train_gpu_ids')
    - vLLM and Ray installed
    - Model 'Qwen/Qwen2.5-1.5B-Instruct' accessible (or change to another model)
    """
    import tempfile
    import json
    from omegaconf import OmegaConf

    # Create a simple test environment file
    env_code = """
import asyncio
from vllm import SamplingParams

async def step(sample):
    '''Dummy environment step function that returns a reward and done status'''
    # Simple reward based on response length
    reward = 1.0 if len(sample.action_text) > 10 else 0.5
    return {
        "reward": reward,
        "done": True,  # End after one turn for simplicity
        "next_state": sample.state_text + sample.action_text
    }

def get_sampling_params():
    '''Return vLLM sampling parameters'''
    return SamplingParams(
        temperature=0.7,
        max_tokens=50,
        top_p=0.9,
    )
"""

    # Write environment to a temporary file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(env_code)
        env_path = f.name

    print(f"Created test environment at: {env_path}")

    # Create dummy dataset
    dummy_data = [
        {"prompt": "What is the capital of France?"},
        {"prompt": "Explain quantum computing in simple terms."},
        {"prompt": "Write a haiku about coding."},
    ]

    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        for item in dummy_data:
            f.write(json.dumps(item) + "\n")
        dataset_path = f.name

    print(f"Created test dataset at: {dataset_path}")

    # Create minimal config for testing
    config = OmegaConf.create(
        {
            "inference_gpu_ids": "0",  # Use GPU 0 for inference
            "train_gpu_ids": "0",  # Use GPU 0 for training
            "server_args": {
                "model_path": "Qwen/Qwen2.5-1.5B-Instruct",  # Small model for testing
                "tp_size": 1,
                "dtype": "auto",
            },
            "env_path": env_path,
            "train": {
                "path": dataset_path,
                "prompt_key": "prompt",
                "messages_key": "messages",
                "apply_chat_template": False,
                "train_on_what": [],
                "prompts_per_rollout": 2,  # Just 2 prompts for testing
                "responses_per_prompt": 1,
                "sampling_params": {
                    "temperature": 0.7,
                    "max_new_tokens": 50,
                    "stop": None,
                },
                "dynamic_filtering": False,
                "partial_rollout": False,
                "save_dir": "/tmp/test_rollout/train",
            },
            "test": {
                "path": dataset_path,
                "prompt_key": "prompt",
                "messages_key": "messages",
                "apply_chat_template": False,
                "train_on_what": [],
                "prompts_per_rollout": 1,
                "responses_per_prompt": 1,
                "sampling_params": {
                    "temperature": 0.0,
                    "max_new_tokens": 50,
                    "stop": None,
                },
                "save_dir": "/tmp/test_rollout/test",
            },
            "bucket_size": 512,
        }
    )

    print("=" * 60)
    print("Testing Rollout with vLLM+Ray")
    print("=" * 60)

    try:
        # # Initialize distributed training (single process for testing)
        if not dist.is_initialized():
            dist.init_process_group(
                backend="gloo",
                init_method="tcp://localhost:29500",
                world_size=1,
                rank=0,
            )

        print("\n[1/4] Initializing Rollout...")
        rollout = Rollout(config)
        print("✓ Rollout initialized successfully")

        print("\n[2/4] Running test rollout...")
        import asyncio

        tensor_dict, cu_seqs = asyncio.run(rollout(train=False, step=0))

        if tensor_dict is not None:
            print("✓ Test rollout completed successfully")
            print(f"   - Generated {len(cu_seqs) - 1} sample groups")
            print(f"   - Total sequences: {cu_seqs[-1].item()}")
            print(f"   - Tensor dict keys: {list(tensor_dict.keys())}")
            for k, v in tensor_dict.items():
                print(f"     {k}: shape {v.shape}")

        print("\n[3/4] Running train rollout...")
        tensor_dict, cu_seqs = asyncio.run(rollout(train=True, step=0))

        if tensor_dict is not None:
            print("✓ Train rollout completed successfully")
            print(f"   - Generated {len(cu_seqs) - 1} sample groups")
            print(f"   - Total sequences: {cu_seqs[-1].item()}")
            print(f"   - cu_seqs: {cu_seqs}")

        print("\n[4/4] Testing weight update...")
        from transformers import AutoModelForCausalLM

        test_model = AutoModelForCausalLM.from_pretrained(
            config.server_args.model_path,
            torch_dtype=torch.float16,
        ).to("cuda:0")

        # Zero out weights for testing
        for p in test_model.parameters():
            p.data.zero_()

        rollout.update(test_model)
        print("✓ Weight update completed successfully")

        print("\n" + "=" * 60)
        print("All tests passed! ✓")
        print("=" * 60)

    except Exception as e:
        print(f"\n✗ Error during testing: {e}")
        import traceback

        traceback.print_exc()
    finally:
        # Cleanup
        import os

        if os.path.exists(env_path):
            os.remove(env_path)
            print(f"\nCleaned up test environment file: {env_path}")
        if os.path.exists(dataset_path):
            os.remove(dataset_path)
            print(f"Cleaned up test dataset file: {dataset_path}")
