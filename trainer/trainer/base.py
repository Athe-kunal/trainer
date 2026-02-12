import pathlib
from typing import Dict, Any, Sequence
from omegaconf import OmegaConf, DictConfig
import glob
from loguru import logger
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import wandb
from trainer.workers.base import BaseWorker


class Trainer:

    def __init__(self, config: DictConfig):

        OmegaConf.resolve(config)
        self.load_dir = config.trainer.load_ckpt_from
        if self.load_dir == "latest":
            load_dirs = glob.glob(f"{config.trainer.save_dir}/step*")
            self.load_dir = (
                max(load_dirs, key=lambda dir: int(dir.split("/step")[-1]))
                if load_dirs
                else None
            )
        if self.load_dir is not None:
            load_path = pathlib.Path(self.load_dir)
            if hasattr(config, "actor"):
                config.actor.model_name = str(load_path / "actor" / "model")
            if hasattr(config, "critic"):
                config.critic.model_name = str(load_path / "critic" / "model")
            if hasattr(config, "rollout"):
                config.rollout.server_args.model_path = str(
                    load_path / "actor" / "model"
                )
            self.load_dir = pathlib.Path(self.load_dir)
        self.config = config

        if dist.get_rank() == 0:
            logger.info(f"Config: {OmegaConf.to_yaml(config)}")
            if config.trainer.use_wandb:
                wandb.init(
                    project=config.trainer.project,
                    name=config.trainer.experiment_name,
                    config=OmegaConf.to_container(config),
                )
            else:
                wandb.log = lambda *args, **kwargs: None

    def _get_ckpt(self, step: int) -> Dict[str, Any]:

        ckpt = {"step": step}
        if dist.get_rank() == 0:
            ckpt["dataloader"] = self.train_dataloader.state_dict()
        return ckpt

    def load_ckpt(self, workers: Sequence[BaseWorker]) -> int:

        if self.load_dir is None:
            return 0
        for worker in workers:
            worker_name = "actor" if "Actor" in worker.__class__.__name__ else "critic"
            worker.load_ckpt(self.load_dir / worker_name / "optimizer_scheduler")

        ckpt = self._get_ckpt(0)
        dcp.load(ckpt, checkpoint_id=self.load_dir / "trainer")
        if dist.get_rank() == 0:
            self.train_dataloader.load_state_dict(ckpt["dataloader"])
        return ckpt["step"]

    def save_ckpt(self, workers: Sequence[BaseWorker], step: int):

        if (
            self.config.trainer.save_freq is None
            or step % self.config.trainer.save_freq != 0
        ):
            return

        save_dir = self.config.trainer.save_dir / f"step{step}"
        for worker in workers:
            worker_name = "actor" if "Actor" in worker.__class__.__name__ else "critic"
            worker.save_ckpt(save_dir / worker_name)

        dcp.save(self._get_ckpt(step), checkpoint_id=save_dir / "trainer")

    def save_model(self, workers: Sequence[BaseWorker]):

        save_dir = self.config.trainer.save_dir
        if self.config.trainer.save_freq is not None:
            save_dir += "latest"

        for worker in workers:
            worker_name = "actor" if "Actor" in worker.__class__.__name__ else "critic"
            worker.save_model(save_dir / worker_name)
