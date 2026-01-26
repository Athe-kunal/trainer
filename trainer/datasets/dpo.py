from typing import Any, List, Tuple, Dict
import torch
from trainer.datasets.base import BaseDataset, pack_tensor_dicts, Message


class DPODataset(BaseDataset):

    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor]]:

        sample = self.dataset[idx]
        chosen_messages, rejected_messages = self.convert_to_messages(sample)
        chosen = self._tokenize_messages(chosen_messages)
        rejected = self._tokenize_messages(rejected_messages)
        assert len(chosen) == len(rejected) == 1
        chosen, rejected = chosen[0], rejected[0]
        return chosen, rejected

    def collate_fn(
        self, all_tensor_dicts: Tuple[Tuple[Dict[str, torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:

        tensor_dicts: List[Dict[str, torch.Tensor]] = [
            td for tds in all_tensor_dicts for td in tds
        ]
        return pack_tensor_dicts(tensor_dicts)

    def convert_to_messages(
        self, sample: Dict[str, Any]
    ) -> Tuple[List[Message], List[Message]]:
        if not self.dataset_config.apply_chat_template:
            messages: List[Message] = []
            if self.dataset_config.system_prompt:
                messages.append(
                    Message(
                        role="system",
                        content=self.dataset_config.system_prompt,
                        train=self._determine_to_train("system"),
                    )
                )
            messages.append(
                Message(
                    role="user",
                    content=sample[self.dataset_config.prompt_key],
                    train=self._determine_to_train("user"),
                ),
            )
            chosen = messages + [
                Message(
                    role="assistant",
                    content=sample[self.dataset_config.chosen_key],
                    train=self._determine_to_train("assistant"),
                ),
            ]
            rejected = messages + [
                Message(
                    role="assistant",
                    content=sample[self.dataset_config.rejected_key],
                    train=self._determine_to_train("assistant"),
                ),
            ]
            return chosen, rejected
        else:
            chosen = [
                Message(
                    role=m["role"],
                    content=m["content"],
                    train=self._determine_to_train(m["role"]),
                )
                for m in sample[self.dataset_config.chosen_messages_key]
            ]
            rejected = [
                Message(
                    role=m["role"],
                    content=m["content"],
                    train=self._determine_to_train(m["role"]),
                )
                for m in sample[self.dataset_config.rejected_messages_key]
            ]
            return chosen, rejected
