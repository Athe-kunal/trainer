import hydra
from omegaconf import DictConfig
import torch.distributed as dist
from tqdm import tqdm
from trainer.trainer.base import Trainer
from trainer.trainer.utils import init_debugpy_if_enabled
from trainer.datasets import KTODataset, get_dataloader
from trainer.workers import initialize_actor
from trainer.utils.communication import initialize_global_process_group


class KTOTrainer(Trainer):

    def __init__(self, config: DictConfig):
        super().__init__(config)

        self.actor = initialize_actor(config.actor, True)
        self.ref_actor = initialize_actor(config.ref_actor, False)
        self.train_dataloader, self.test_dataloader = get_dataloader(
            KTODataset, config.data, self.actor.tokenizer
        )
        self.actor.prepare_scheduler(
            self.config.trainer.n_epochs * len(self.train_dataloader)
        )

    def train(self):

        step = self.load_ckpt((self.actor,))
        for epoch in range(
            step // len(self.train_dataloader), self.config.trainer.n_epochs
        ):
            for tensor_dict in tqdm(
                self.train_dataloader,
                desc=f"Epoch {epoch + 1}",
                disable=(dist.get_rank() != 0),
                initial=step % len(self.train_dataloader),
            ):
                step += 1
                labels = tensor_dict.pop("labels")
                tensor_dict = self.ref_actor.compute_logps(tensor_dict, step, True)
                tensor_dict["labels"] = labels
                self.actor.kto_step(tensor_dict, True, step)
                self.save_ckpt((self.actor,), step)

            for tensor_dict in self.test_dataloader:
                labels = tensor_dict.pop("labels")
                tensor_dict = self.ref_actor.compute_logps(tensor_dict, step, True)
                tensor_dict["labels"] = labels
                self.actor.kto_step(tensor_dict, False, step)

        self.save_model((self.actor,))


@hydra.main(config_path="config", config_name="kto", version_base=None)
def main(config: DictConfig):
    init_debugpy_if_enabled()
    initialize_global_process_group()

    trainer = KTOTrainer(config)
    trainer.train()

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
