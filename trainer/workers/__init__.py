from .base import BaseWorker
from omegaconf import DictConfig

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

    from .rollout import Rollout

    return Rollout(config)
