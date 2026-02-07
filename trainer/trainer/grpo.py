import hydra
from omegaconf import DictConfig
import asyncio
import torch.distributed as dist
from tqdm import trange
from trainer.trainer.base import Trainer
from trainer.workers import initialize_actor, initialize_critic, initialize_rollout
from trainer.workers.rollout import shutdown_processes_when_exit
from trainer.utils.communication import initialize_global_process_group, with_session
from trainer.utils.algorithms import compute_advantages


class PPOTrainer(Trainer):

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

        self.rollout = initialize_rollout(self.config.rollout)

    @shutdown_processes_when_exit
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


@hydra.main(config_path="config", config_name="ppo", version_base=None)
def main(config: DictConfig):
    import os
    
    # Multi-rank debugpy support
    if os.environ.get("ENABLE_DEBUGPY", "0") == "1":
        import debugpy
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        debug_port = 5678 + local_rank  # Each rank gets its own port
        
        debugpy.listen(("0.0.0.0", debug_port))
        print(f"[Rank {local_rank}] Waiting for debugger on port {debug_port}...")
        debugpy.wait_for_client()
        print(f"[Rank {local_rank}] Debugger attached! Starting training...")

    initialize_global_process_group(create_gloo_group=True, timeout_second=3000)

    trainer = PPOTrainer(config)
    asyncio.run(trainer.train())

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
