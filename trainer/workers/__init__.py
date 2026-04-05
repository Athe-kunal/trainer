from loguru import logger
from omegaconf import DictConfig

from .base import BaseWorker

# Alias for backward compatibility
Worker = BaseWorker


def initialize_actor(config: DictConfig, train: bool):

    from hydra.core.hydra_config import HydraConfig

    hydra_config = HydraConfig.get()
    backend = hydra_config.runtime.choices.get("actor" if train else "ref_actor")
    if backend == "fsdp":
        from .fsdp.actor import FSDPActor

        return FSDPActor(config, train)
    else:
        raise NotImplementedError


def initialize_critic(config: DictConfig):

    from hydra.core.hydra_config import HydraConfig

    hydra_config = HydraConfig.get()
    backend = hydra_config.runtime.choices.get("critic")
    if backend == "fsdp":
        from .fsdp.critic import FSDPCritic

        return FSDPCritic(config)
    else:
        raise NotImplementedError


def initialize_rollout(config: DictConfig):

    backend = config.backend
    model_name = config.server_args.model_path
    logger.info(f"{backend=}, {model_name=}")

    if backend == "ipc":
        from trainer.rollouts.rollout_ipc import IPCHttpRollout

        return IPCHttpRollout(model_name)
    elif backend == "nccl":
        from trainer.rollouts.rollout_nccl import NCCLHttpRollout

        return NCCLHttpRollout(model_name)
    else:
        raise NotImplementedError(
            f"Unsupported rollout {backend=}. Expected one of: ipc, nccl"
        )
