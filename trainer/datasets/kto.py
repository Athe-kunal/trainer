from typing import Dict, List, Tuple
import torch

from trainer.datasets.dpo import DPODataset
from trainer.datasets.base import pack_tensor_dicts


class KTODataset(DPODataset):

    def __getitem__(
        self, idx: int
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        chosen, rejected = super().__getitem__(idx)
        chosen["label"] = torch.tensor(1)
        rejected["label"] = torch.tensor(0)
        return chosen, rejected

    def collate_fn(
        self, all_tensor_dicts: Tuple[Tuple[Dict[str, torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        tensor_dicts: List[Dict[str, torch.Tensor]] = [
            td for tds in all_tensor_dicts for td in tds
        ]
        labels = torch.stack([td["label"] for td in tensor_dicts])
        seq_dicts = [
            {k: v for k, v in td.items() if k != "label"} for td in tensor_dicts
        ]
        batch = pack_tensor_dicts(seq_dicts)
        batch["labels"] = labels
        return batch
