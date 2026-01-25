from typing import Literal, Tuple, Dict, List, TypedDict
import torch
from trainer.datasets.base import BaseDataset, pack_tensor_dicts

MessagesType: Literal[str] = ["system", "user", "assistant"]


class Message(TypedDict):
    role: MessagesType
    content: str
    train: bool


class SFTDataset(BaseDataset):

    def __getitem__(self, idx: int) -> List[Dict[str, torch.Tensor]]:

        sample = self.dataset[idx]
        if not self.config.apply_chat_template:
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
