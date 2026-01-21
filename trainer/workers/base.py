from typing import Any, Union
from transformers import AutoTokenizer
from trainer.datamodels import ActorConfig, CriticConfig, RefActorConfig, BaseWorkerConfig


class BaseWorker:
    def __init__(
        self,
        config: Union[ActorConfig, CriticConfig, RefActorConfig, BaseWorkerConfig, dict[str, Any]],
        train: bool,
    ) -> None:
        self.config = config
        self.train = train
        # Support both dataclass (with attribute access) and dict config
        if hasattr(config, 'model_name'):
            model_name = config.model_name
        else:
            model_name = config.get('model_name')  # type: ignore[union-attr]
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )
