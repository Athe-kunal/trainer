import hydra
from omegaconf import DictConfig
import asyncio
import torch.distributed as dist
from loguru import logger
from tqdm import trange
from trainer.trainer.base import Trainer
from trainer.trainer.utils import init_debugpy_if_enabled
from trainer.workers import initialize_actor, initialize_critic, initialize_rollout
from trainer.utils.communication import initialize_global_process_group, with_session
from trainer.utils.algorithms import compute_advantages


def resolve_rollout_backend(rollout_config: DictConfig) -> str:
    """Maps rollout topology to vLLM weight-transfer backend."""
    topology_to_backend = {
        "colocate": "ipc",
        "disaggregated": "nccl",
    }
    topology = rollout_config.topology
    if topology not in topology_to_backend:
        raise ValueError(
            "rollout.topology must be one of: colocate, disaggregated. "
            f"Received {topology=}"
        )
    backend = topology_to_backend[topology]
    rollout_config.backend = backend
    logger.info(f"{topology=}, {backend=}")
    return backend


class GRPOTrainer(Trainer):

    def __init__(self, config: DictConfig):
        super().__init__(config)

        if not config.trainer.eval_only:

            self.actor = initialize_actor(config.actor, True)
            self.actor.prepare_scheduler(self.config.trainer.total_steps)
            if config.actor.kl.coef > 0:
                self.ref_actor = initialize_actor(config.ref_actor, False)
            if config.adv.estimator == "gae":
                self.critic = initialize_critic(config.critic)
                self.critic.prepare_scheduler(self.config.trainer.total_steps)

        resolve_rollout_backend(self.config.rollout)
        self.rollout = initialize_rollout(self.config.rollout)

    @with_session
    async def train(self):

        if self.config.trainer.eval_only:
            await self.rollout(False, 0)
            return

        initial = self.load_ckpt(
            (self.actor, self.critic)
            if self.config.adv.estimator == "gae"
            else (self.actor,)
        )
        for step in trange(
            1,
            self.config.trainer.total_steps + 1,
            disable=(dist.get_rank() != 0),
            initial=initial,
        ):

            tensor_dict, cu_seqs = await self.rollout(True, step)

            if self.config.actor.kl.coef > 0:
                tensor_dict = self.ref_actor.compute_logps(tensor_dict, step)
            if self.config.adv.estimator == "gae":
                tensor_dict = self.critic.compute_values(tensor_dict, step)
            if (
                self.config.actor.kl.coef > 0
                or self.config.actor.update_per_rollout > 1
            ):
                tensor_dict = self.actor.compute_logps(tensor_dict, step)

            if dist.get_rank() == 0:
                compute_advantages(self.config, tensor_dict, cu_seqs, step)

            self.actor.ppo_update(tensor_dict, step)
            if self.config.adv.estimator == "gae":
                self.critic.ppo_update(tensor_dict, step)
            self.save_ckpt(
                (
                    (self.actor, self.critic)
                    if self.config.adv.estimator == "gae"
                    else (self.actor,)
                ),
                step,
            )

            self.actor.update_rollout(self.rollout, step)
            if (
                self.config.trainer.test_freq is not None
                and step % self.config.trainer.test_freq == 0
            ):
                await self.rollout(False, step)

        self.save_model(
            (self.actor, self.critic)
            if self.config.adv.estimator == "gae"
            else (self.actor,)
        )

    @property
    def train_dataloader(self):
        return self.rollout.train_dataloader


@hydra.main(config_path="config", config_name="grpo", version_base=None)
def main(config: DictConfig):
    init_debugpy_if_enabled()
    initialize_global_process_group(create_gloo_group=True, timeout_second=3000)

    trainer = GRPOTrainer(config)
    asyncio.run(trainer.train())

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
