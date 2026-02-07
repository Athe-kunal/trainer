import hydra
from omegaconf import DictConfig
import torch.distributed as dist
from tqdm import tqdm
from trainer.trainer.base import Trainer
from trainer.datasets import SFTDataset, get_dataloader
from trainer.workers import initialize_actor
from trainer.utils.communication import initialize_global_process_group


class SFTTrainer(Trainer):

    def __init__(self, config: DictConfig):
        super().__init__(config)

        self.actor = initialize_actor(config.actor, True)
        self.train_dataloader, self.test_dataloader = get_dataloader(
            SFTDataset, config.data, self.actor.tokenizer
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
                self.actor.sft_step(tensor_dict, True, step)
                self.save_ckpt((self.actor,), step)

            for tensor_dict in self.test_dataloader:
                self.actor.sft_step(tensor_dict, False, step)

        self.save_model((self.actor,))


@hydra.main(config_path="config", config_name="sft", version_base=None)
def main(config: DictConfig):
    import os

    # Optional debugpy support - only debugs rank 0 on port 5678
    if os.environ.get("ENABLE_DEBUGPY", "0") == "1":
        import debugpy
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        
        if local_rank == 0:
            # Only rank 0 will wait for debugger
            debugpy.listen(("0.0.0.0", 5678))
            print("[Rank 0] Waiting for debugger on port 5678...")
            debugpy.wait_for_client()
            print("[Rank 0] Debugger attached! Starting training...")

    initialize_global_process_group()

    trainer = SFTTrainer(config)
    trainer.train()

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
