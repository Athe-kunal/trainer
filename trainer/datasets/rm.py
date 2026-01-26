from typing import Tuple, Dict
import torch
from trainer.datasets.dpo import DPODataset


class RMDataset(DPODataset):

    # def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor]]:

    #     sample = self.dataset[idx]
    #     chosen_messages, rejected_messages = self.convert_to_messages(sample)
    #     if self.dataset_config.apply_chat_template:
    #         chosen = self._tokenize_messages(sample[self.config.chosen_key], rm=True)
    #         rejected = self._tokenize_messages(
    #             sample[self.config.rejected_key], rm=True
    #         )
    #         assert len(chosen) == len(rejected) == 1
    #         chosen, rejected = chosen[0], rejected[0]
    #     else:
    #         chosen = self._tokenize_prompt_response(
    #             sample[self.config.prompt_key], sample[self.config.chosen_key], rm=True
    #         )
    #         rejected = self._tokenize_prompt_response(
    #             sample[self.config.prompt_key],
    #             sample[self.config.rejected_key],
    #             rm=True,
    #         )
    #     return chosen, rejected

    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor]]:

        sample = self.dataset[idx]
        chosen_messages, rejected_messages = self.convert_to_messages(sample)
        chosen = self._tokenize_messages(chosen_messages, rm=True)
        rejected = self._tokenize_messages(rejected_messages, rm=True)
        assert len(chosen) == len(rejected) == 1
        chosen, rejected = chosen[0], rejected[0]
        return chosen, rejected
