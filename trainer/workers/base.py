from typing import Any
from transformers import AutoTokenizer


class BaseWorker:
    def __init__(self, config: dict[str, Any], train: bool) -> None:
        self.config = config
        self.train = train
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_name, trust_remote_code=True
        )
