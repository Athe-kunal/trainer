"""
Dataclass models for trainer configuration.

This module provides type-safe configuration classes that replace DictConfig
throughout the trainer codebase. All configuration is validated at runtime
using Python dataclasses.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Optional
import yaml  # type: ignore[import-untyped]


# =============================================================================
# Optimizer and Scheduler Configurations
# =============================================================================


@dataclass
class OptimizerConfig:
    """Configuration for AdamW optimizer."""

    lr: float = 1e-5
    betas: tuple[float, float] = (0.9, 0.999)
    weight_decay: float = 0.01
    eps: float = 1e-8

    def __post_init__(self):
        # Convert list to tuple if necessary (from YAML loading)
        if isinstance(self.betas, list):
            self.betas = tuple(self.betas)


@dataclass
class SchedulerConfig:
    """Configuration for learning rate scheduler."""

    name: str = "cosine"
    warmup_ratio: float = 0.1


# =============================================================================
# KL and Entropy Configurations
# =============================================================================


@dataclass
class KLConfig:
    """Configuration for KL divergence penalty."""

    coef: float = 0.0
    type: Literal["reward", "advantage", "loss"] = "reward"
    reward_estimator: Literal["k1", "k2", "k3"] = "k1"
    loss_estimator: Literal["k1", "k2", "k3"] = "k1"


@dataclass
class EntropyConfig:
    """Configuration for entropy bonus."""

    coef: float = 0.0


# =============================================================================
# Worker Configurations
# =============================================================================


@dataclass
class BaseWorkerConfig:
    """Base configuration for all workers."""

    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    dtype: Literal["bfloat16", "float16", "float32"] = "bfloat16"
    tp_size: int = 1
    ddp_size: int = 1
    cp_size: int = 1
    enable_gradient_checkpointing: bool = True
    offload_model: bool = False
    offload_optimizer: bool = False
    max_length_per_device: int = 4096
    max_inference_length_per_device: int = 8192


@dataclass
class ActorConfig(BaseWorkerConfig):
    """Configuration for actor model."""

    use_liger_kernel: bool = False
    update_per_rollout: int = 1
    max_grad_norm: float = 1.0
    avg_level: Literal["token", "sequence"] = "token"
    temperature: float = 1.0
    freeze_steps: int = 0

    # PPO specific
    clip: float = 0.2
    tis_coef: float = 0.0

    # DPO/SimPO specific
    beta: float = 0.1
    gamma: float = 0.5

    # ORPO specific
    lambda_orpo: float = 0.1
    eps: float = 1e-6

    # For PPO advantage estimator reference
    adv_estimator: Literal["gae", "reinforce"] = "reinforce"

    # Nested configs
    entropy: EntropyConfig = field(default_factory=EntropyConfig)
    kl: KLConfig = field(default_factory=KLConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)

    def __post_init__(self):
        if isinstance(self.entropy, dict):
            self.entropy = EntropyConfig(**self.entropy)
        if isinstance(self.kl, dict):
            self.kl = KLConfig(**self.kl)
        if isinstance(self.optimizer, dict):
            self.optimizer = OptimizerConfig(**self.optimizer)
        if isinstance(self.scheduler, dict):
            self.scheduler = SchedulerConfig(**self.scheduler)


@dataclass
class RefActorConfig(BaseWorkerConfig):
    """Configuration for reference actor (frozen model for KL computation)."""

    use_liger_kernel: bool = False
    enable_gradient_checkpointing: bool = False
    offload_model: bool = True


@dataclass
class CriticConfig(BaseWorkerConfig):
    """Configuration for critic model."""

    update_per_rollout: int = 1
    max_grad_norm: float = 1.0
    avg_level: Literal["token", "sequence"] = "token"
    clip: float = 0.2

    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)

    def __post_init__(self):
        if isinstance(self.optimizer, dict):
            self.optimizer = OptimizerConfig(**self.optimizer)
        if isinstance(self.scheduler, dict):
            self.scheduler = SchedulerConfig(**self.scheduler)


# =============================================================================
# Advantage Estimation Configuration
# =============================================================================


@dataclass
class AdvantageConfig:
    """Configuration for advantage estimation."""

    estimator: Literal["gae", "reinforce"] = "reinforce"

    # GAE specific
    gamma: float = 0.99
    lamda: float = 0.95

    # REINFORCE specific
    responses_per_prompt: int = 4
    global_norm: bool = False
    norm_var: bool = True


# =============================================================================
# Dataset Configurations
# =============================================================================


@dataclass
class DatasetSplitConfig:
    """Configuration for a dataset split (train/test)."""

    path: Optional[str] = None
    batch_size: int = 8
    max_length: int = 2048


@dataclass
class DataConfig:
    """Configuration for dataset loading."""

    train: DatasetSplitConfig = field(default_factory=DatasetSplitConfig)
    test: DatasetSplitConfig = field(default_factory=DatasetSplitConfig)
    test_ratio: float = 0.1

    def __post_init__(self):
        if isinstance(self.train, dict):
            self.train = DatasetSplitConfig(**self.train)
        if isinstance(self.test, dict):
            self.test = DatasetSplitConfig(**self.test)


# =============================================================================
# Rollout Configurations
# =============================================================================


@dataclass
class SGLangServerConfig:
    """Configuration for SGLang inference server."""

    model_path: str = "Qwen/Qwen2.5-0.5B-Instruct"
    tp_size: int = 1
    trust_remote_code: bool = True
    dtype: str = "bfloat16"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for SGLang ServerArgs."""
        return {
            "model_path": self.model_path,
            "tp_size": self.tp_size,
            "trust_remote_code": self.trust_remote_code,
            "dtype": self.dtype,
        }


@dataclass
class RolloutTrainConfig:
    """Configuration for training rollouts."""

    path: str = "data/train.jsonl"
    batch_size: int = 32
    prompts_per_rollout: Optional[int] = 64
    partial_rollout: bool = False
    dynamic_filtering: bool = False
    max_length: int = 2048


@dataclass
class RolloutTestConfig:
    """Configuration for test/evaluation rollouts."""

    path: Optional[str] = None
    prompts_per_rollout: Optional[int] = None
    max_length: int = 2048


@dataclass
class RolloutConfig:
    """Configuration for rollout generation."""

    env_path: str = "./envs/math_env.py"
    bucket_size: int = 128

    server_args: SGLangServerConfig = field(default_factory=SGLangServerConfig)
    train: RolloutTrainConfig = field(default_factory=RolloutTrainConfig)
    test: RolloutTestConfig = field(default_factory=RolloutTestConfig)
    test_ratio: float = 0.1

    def __post_init__(self):
        if isinstance(self.server_args, dict):
            self.server_args = SGLangServerConfig(**self.server_args)
        if isinstance(self.train, dict):
            self.train = RolloutTrainConfig(**self.train)
        if isinstance(self.test, dict):
            self.test = RolloutTestConfig(**self.test)


# =============================================================================
# Trainer Configuration
# =============================================================================


@dataclass
class TrainerConfig:
    """Configuration for the trainer."""

    # Checkpoint settings
    load_ckpt_from: Optional[str] = None
    save_dir: str = "./checkpoints"
    save_freq: Optional[int] = None

    # Training settings
    total_steps: int = 1000
    n_epochs: int = 3
    eval_only: bool = False
    test_freq: Optional[int] = None

    # Logging
    use_wandb: bool = True
    project: str = "trainer"
    experiment_name: str = "experiment"


# =============================================================================
# Root Configuration
# =============================================================================


@dataclass
class Config:
    """Root configuration containing all training parameters."""

    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    actor: ActorConfig = field(default_factory=ActorConfig)
    ref_actor: Optional[RefActorConfig] = None
    critic: Optional[CriticConfig] = None
    adv: AdvantageConfig = field(default_factory=AdvantageConfig)
    rollout: Optional[RolloutConfig] = None
    data: Optional[DataConfig] = None

    def __post_init__(self):
        if isinstance(self.trainer, dict):
            self.trainer = TrainerConfig(**self.trainer)
        if isinstance(self.actor, dict):
            self.actor = ActorConfig(**self.actor)
        if isinstance(self.ref_actor, dict):
            self.ref_actor = RefActorConfig(**self.ref_actor)
        if isinstance(self.critic, dict):
            self.critic = CriticConfig(**self.critic)
        if isinstance(self.adv, dict):
            self.adv = AdvantageConfig(**self.adv)
        if isinstance(self.rollout, dict):
            self.rollout = RolloutConfig(**self.rollout)
        if isinstance(self.data, dict):
            self.data = DataConfig(**self.data)

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        """Load configuration from a YAML file."""
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls(**data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Config":
        """Create configuration from a dictionary."""
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to a dictionary."""
        from dataclasses import asdict

        return asdict(self)


# =============================================================================
# PPO-specific Configuration
# =============================================================================


@dataclass
class PPOConfig(Config):
    """Configuration specifically for PPO training."""

    ref_actor: RefActorConfig = field(default_factory=RefActorConfig)
    rollout: RolloutConfig = field(default_factory=RolloutConfig)

    def __post_init__(self):
        super().__post_init__()
        if isinstance(self.ref_actor, dict):
            self.ref_actor = RefActorConfig(**self.ref_actor)
        if isinstance(self.rollout, dict):
            self.rollout = RolloutConfig(**self.rollout)


@dataclass
class PPOWithCriticConfig(PPOConfig):
    """Configuration for PPO with GAE (requires critic)."""

    critic: CriticConfig = field(default_factory=CriticConfig)

    def __post_init__(self):
        super().__post_init__()
        if isinstance(self.critic, dict):
            self.critic = CriticConfig(**self.critic)


# =============================================================================
# DPO/ORPO/SimPO-specific Configuration
# =============================================================================


@dataclass
class DPOConfig(Config):
    """Configuration for DPO training."""

    data: DataConfig = field(default_factory=DataConfig)
    ref_actor: RefActorConfig = field(default_factory=RefActorConfig)

    def __post_init__(self):
        super().__post_init__()
        if isinstance(self.data, dict):
            self.data = DataConfig(**self.data)
        if isinstance(self.ref_actor, dict):
            self.ref_actor = RefActorConfig(**self.ref_actor)


@dataclass
class ORPOConfig(Config):
    """Configuration for ORPO training."""

    data: DataConfig = field(default_factory=DataConfig)

    def __post_init__(self):
        super().__post_init__()
        if isinstance(self.data, dict):
            self.data = DataConfig(**self.data)


@dataclass
class SimPOConfig(Config):
    """Configuration for SimPO training."""

    data: DataConfig = field(default_factory=DataConfig)

    def __post_init__(self):
        super().__post_init__()
        if isinstance(self.data, dict):
            self.data = DataConfig(**self.data)


# =============================================================================
# SFT-specific Configuration
# =============================================================================


@dataclass
class SFTConfig(Config):
    """Configuration for supervised fine-tuning."""

    data: DataConfig = field(default_factory=DataConfig)

    def __post_init__(self):
        super().__post_init__()
        if isinstance(self.data, dict):
            self.data = DataConfig(**self.data)


# =============================================================================
# Utility Functions
# =============================================================================


def load_config(path: str) -> Config:
    """Load configuration from a YAML file."""
    return Config.from_yaml(path)


def create_ppo_config(**kwargs) -> PPOConfig:
    """Create a PPO configuration with the given overrides."""
    return PPOConfig(**kwargs)


def create_dpo_config(**kwargs) -> DPOConfig:
    """Create a DPO configuration with the given overrides."""
    return DPOConfig(**kwargs)


def create_sft_config(**kwargs) -> SFTConfig:
    """Create an SFT configuration with the given overrides."""
    return SFTConfig(**kwargs)
