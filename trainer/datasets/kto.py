from typing import Dict, List
import torch

from trainer.datasets.sft import SFTDataset


class KTODataset(SFTDataset):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        label_column = self.dataset_config.get("label_column")
        if not label_column:
            raise ValueError(
                "KTODataset requires `label_column` in the dataset config."
            )
        if label_column not in self.dataset.column_names:
            raise ValueError(
                f"KTODataset label_column `{label_column}` not found in dataset columns: "
                f"{self.dataset.column_names}"
            )

    def __getitem__(self, idx: int) -> List[Dict[str, torch.Tensor]]:
        tensor_dicts = super().__getitem__(idx)

        label = self.dataset[idx][self.dataset_config.label_column]
        if label not in (0, 1):
            raise ValueError(
                f"KTODataset expects labels in `{self.dataset_config.label_column}` "
                f"to be 0 or 1, got {label!r}."
            )

        label_tensor = torch.LongTensor([int(label)])
        for tensor_dict in tensor_dicts:
            tensor_dict["label"] = label_tensor
        return tensor_dicts
