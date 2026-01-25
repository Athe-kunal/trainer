from typing import Tuple, Dict, List
import torch
from trainer.datasets.base import BaseDataset, pack_tensor_dicts


class SFTDataset(BaseDataset):

    def __getitem__(self, idx: int) -> List[Dict[str, torch.Tensor]]:

        sample = self.dataset[idx]
        if self.config.apply_chat_template:
            if not self.config.messages_key:
                if self.config.system_prompt:
                    messages = [
                        {
                            "role": "system",
                            "content": self.config.system_prompt,
                        },
                    ]
                messages.extend(
                    [
                        {
                            "role": "user",
                            "content": sample[self.config.prompt_key],
                        },
                        {
                            "role": "assistant",
                            "content": sample[self.config.response_key],
                        },
                    ]
                )
            else:
                messages = sample[self.config.messages_key]
            tensor_dicts = self._tokenize_messages(messages)
        else:
            tensor_dicts = [
                self._tokenize_prompt_response(
                    sample[self.config.prompt_key], sample[self.config.response_key]
                )
            ]
        return tensor_dicts

    def collate_fn(
        self, all_tensor_dicts: Tuple[List[Dict[str, torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:

        tensor_dicts = [td for tds in all_tensor_dicts for td in tds]
        return pack_tensor_dicts(tensor_dicts)
