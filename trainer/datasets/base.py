import abc
from typing import (
    List,
    Optional,
    Dict,
    Sequence,
    Any,
    Tuple,
    NamedTuple,
    Union,
    Literal,
)
from omegaconf import DictConfig
import os
import datasets
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import AutoTokenizer

from loguru import logger


class Message(NamedTuple):
    role: str
    content: str
    train: bool


def get_tensor_dict(
    states: List[int],
    actions: List[int],
    action_mask: List[int],
    max_length: Optional[int] = None,
    rm: bool = False,
) -> Dict[str, torch.Tensor]:

    if not rm:
        # Causal language model training requires shifting the states and actions by one token
        states = states[:-1]
        actions = actions[1:]
        action_mask = action_mask[1:]

    if max_length is not None:
        states = states[:max_length]
        actions = actions[:max_length]
        action_mask = action_mask[:max_length]

    tensor_dict = {
        "states": torch.LongTensor(states),
        # eos_mask will be later helpful to batch together multiple steps and actions
        # it pads different lengths of sequences to the same length
        "eos_mask": torch.LongTensor((len(states) - 1) * [0] + [1]),
        # position_ids are used to identify the position of the tokens in the sequence
        # It is helpful in context parallelism.
        "position_ids": torch.arange(len(states)),
    }
    if rm:
        # only the EOS token gets action mask as 1
        tensor_dict["action_mask"] = torch.LongTensor(
            (len(states) - 1) * [0] + [1]
        )  # rewards of non-terminal tokens are zeros
    else:
        tensor_dict["actions"] = torch.LongTensor(actions)
        tensor_dict["action_mask"] = torch.LongTensor(action_mask)
    return tensor_dict


def pack_tensor_dicts(
    tensor_dicts: Sequence[Dict[str, torch.Tensor]],
) -> Dict[str, torch.Tensor]:
    return {
        k: pad_sequence([td[k] for td in tensor_dicts], True)
        for k in tensor_dicts[0].keys()
    }


class BaseDataset(Dataset, abc.ABC):

    def __init__(
        self,
        dataset_config: DictConfig,
        tokenizer: AutoTokenizer,
        dataset: datasets.Dataset,
    ):

        self.dataset_config = dataset_config
        self.tokenizer = tokenizer
        self.dataset = dataset

    @abc.abstractmethod
    def convert_to_messages(
        self, sample: Dict[str, Any]
    ) -> List[Message] | tuple[List[Message], List[Message]]:
        raise NotImplementedError("Subclasses must implement this method")

    def _tokenize_prompt_response(
        self, messages: List[Message], rm: bool = False
    ) -> Dict[str, torch.Tensor]:

        states: List[int] = []
        actions: List[int] = []
        action_mask: List[Literal[0, 1]] = []
        for idx, msg in enumerate(messages):
            content = msg.content
            if idx == len(messages) - 1:
                content += self.tokenizer.eos_token
            token_ids = self.tokenizer.encode(content, add_special_tokens=False)
            token_ids_len = len(token_ids)
            states.extend(token_ids)
            if msg.train:
                actions.extend(token_ids)
                action_mask.extend(token_ids_len * [1])
            else:
                actions.extend(token_ids_len * [0])
                action_mask.extend(token_ids_len * [0])
        return get_tensor_dict(
            states, actions, action_mask, self.dataset_config.max_length, rm
        )

    def _tokenize_messages(
        self, messages: List[Message], rm: bool = False
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Convert a multi-turn chat into one or more training sequences.

        - `states`: all tokens (context + assistant), shifted later by get_tensor_dict()
        - `actions`: assistant tokens as targets (0 elsewhere), shifted later
        - `action_mask`: 1 where assistant tokens are targets, else 0

        We include a non-train message only if the *next* message is train=True,
        because it is part of the prompt conditioning the assistant response.

        IMPORTANT:
        action_mask[t] == 1 marks positions where the NEXT token is assistant
        states[t] is the previous token, not the assistant token itself
        """

        prev_text: str = ""
        states: List[int] = []
        actions: List[int] = []
        action_mask: List[bool] = []
        tensor_dicts: List[Dict[str, torch.Tensor]] = []

        def to_hf_messages(msgs: List[Message]) -> List[Dict[str, str]]:
            # HF chat templates expect {"role": ..., "content": ...}
            return [{"role": m.role, "content": m.content} for m in msgs]

        for turn in range(len(messages)):
            is_this_turn_train: bool = messages[turn].train
            is_next_turn_train: bool = (
                turn + 1 < len(messages) and messages[turn + 1].train
            )

            # Skip turns that are neither targets themselves nor part of the prompt
            # for an upcoming target turn. For example, if the current is system prompt
            if not is_this_turn_train and not is_next_turn_train:
                continue

            text: str = self.tokenizer.apply_chat_template(
                to_hf_messages(messages[: turn + 1]),
                add_generation_prompt=is_next_turn_train,
                tokenize=False,
            )

            if text.startswith(prev_text):
                delta_text_token_ids = self.tokenizer.encode(
                    text[len(prev_text) :], add_special_tokens=False
                )
                # Tokenize only the delta string to keep token sequence stable.
                delta_text_token_ids_len = len(delta_text_token_ids)
                states.extend(delta_text_token_ids)
                actions.extend(
                    delta_text_token_ids
                    if is_this_turn_train
                    else delta_text_token_ids_len * [0]
                )
                action_mask.extend(delta_text_token_ids_len * [is_this_turn_train])

            else:
                # Prefix broke (template rendering changed). We only allow a reset
                # right before an assistant/train turn (i.e., we are setting up a new prompt).
                assert (
                    is_next_turn_train
                ), "Template prefix broke at an unexpected point (not right before a train turn)."

                tensor_dicts.append(
                    get_tensor_dict(
                        states, actions, action_mask, self.dataset_config.max_length, rm
                    )
                )

                states = self.tokenizer.encode(text, add_special_tokens=False)
                actions = [0] * len(states)
                action_mask = [False] * len(states)

            prev_text = text

        # Finalize last chunk
        tensor_dicts.append(
            get_tensor_dict(
                states, actions, action_mask, self.dataset_config.max_length, rm
            )
        )
        return tensor_dicts

    def __len__(self):
        return len(self.dataset)

    def _determine_to_train(self, key: str) -> bool:
        return True if key in self.dataset_config.train_on_what else False


class StatefulCycleDataLoader(StatefulDataLoader):

    def __call__(self, batch_size: int) -> List[Dict[str, Any]]:
        """
        Fetch a variable number of data.
        """

        if not hasattr(self, "iterator"):
            self.iterator = iter(self)

        data_list = []
        for _ in range(batch_size):
            try:
                data = next(self.iterator)
            except StopIteration:
                self.iterator = iter(self)
                data = next(self.iterator)
            data_list.append(data)
        return data_list


def get_dataloader(
    dataset_cls: BaseDataset,
    dataset_config: DictConfig,
    tokenizer: AutoTokenizer,
    batch_size: int = None,
) -> Tuple[StatefulDataLoader, StatefulDataLoader]:

    def _load_dataset(
        path: Union[str, List[str]], kwargs: dict[str, Any] | None = None
    ):
        kwargs = kwargs or {}

        def _load_single(name: str):
            ext = os.path.splitext(name)[-1].strip(".")
            is_data_file = ext in ["json", "jsonl", "csv", "parquet", "arrow"]
            if is_data_file and os.path.exists(name):
                if ext == "jsonl":
                    ext = "json"
                return datasets.load_dataset(ext, data_files=name, **kwargs)
            logger.info(f"Loading dataset from {name} with kwargs {kwargs}")
            return datasets.load_dataset(name, **kwargs)

        if isinstance(path, list):
            if not path:
                raise ValueError("Dataset path list must not be empty.")
            return datasets.concatenate_datasets([_load_single(item) for item in path])

        return _load_single(path)

    def _get_dataloader(dataset: BaseDataset, batch_size: int):
        return StatefulCycleDataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            collate_fn=dataset.collate_fn,
        )

    train_dataset = _load_dataset(
        dataset_config.train.path,
        kwargs=dataset_config.train.get("kwargs"),
    )
    if dataset_config.test.path:
        test_dataset = _load_dataset(
            dataset_config.test.path,
            kwargs=dataset_config.test.get("kwargs"),
        )
    else:
        total_size = len(train_dataset)
        indices = np.arange(total_size)
        np.random.seed(42)
        np.random.shuffle(indices)
        split_point = int(dataset_config.test_ratio * total_size)
        train_indices, test_indices = indices[split_point:], indices[:split_point]
        test_dataset = train_dataset.select(test_indices)
        train_dataset = train_dataset.select(train_indices)

    train_dataset = dataset_cls(dataset_config.train, tokenizer, train_dataset)
    test_dataset = dataset_cls(dataset_config.test, tokenizer, test_dataset)

    train_dataloader = _get_dataloader(
        train_dataset, batch_size or dataset_config.train.batch_size
    )
    test_dataloader = _get_dataloader(test_dataset, batch_size or len(test_dataset))
    logger.info(
        f"Loaded {len(train_dataloader)} train samples and {len(test_dataloader)} test samples"
    )
    return train_dataloader, test_dataloader
