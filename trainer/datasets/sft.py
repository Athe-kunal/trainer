from typing import Any, Tuple, Dict, List
import torch
from trainer.datasets.base import BaseDataset, pack_tensor_dicts, Message


class SFTDataset(BaseDataset):

    def __getitem__(self, idx: int) -> List[Dict[str, torch.Tensor]]:

        sample = self.dataset[idx]
        messages = self.convert_to_messages(sample)
        # if not self.dataset_config.messages_key:
        #     tensor_dict = self._tokenize_prompt_response(messages, rm=False)
        #     tensor_dicts = [tensor_dict]  # Wrap single dict in a list
        # else:
        tensor_dicts = self._tokenize_messages(messages, rm=False)
        return tensor_dicts

    def collate_fn(
        self, all_tensor_dicts: Tuple[List[Dict[str, torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        tensor_dicts = [td for tds in all_tensor_dicts for td in tds]
        return pack_tensor_dicts(tensor_dicts)

    def convert_to_messages(self, sample: Dict[str, Any]) -> List[Message]:
        if not self.dataset_config.apply_chat_template:
            messages = [
                Message(
                    role="user",
                    content=sample[self.dataset_config.prompt_key],
                    train=self._determine_to_train("user"),
                ),
                Message(
                    role="assistant",
                    content=sample[self.dataset_config.response_key],
                    train=self._determine_to_train("assistant"),
                ),
            ]
            if self.dataset_config.system_prompt:
                messages.insert(
                    0,
                    Message(
                        role="system",
                        content=self.dataset_config.system_prompt,
                        train=self._determine_to_train("system"),
                    ),
                )
            return messages
        else:
            messages = sample[self.dataset_config.messages_key]
            return [
                Message(
                    role=m["role"],
                    content=m["content"],
                    train=self._determine_to_train(m["role"]),
                )
                for m in messages
            ]
