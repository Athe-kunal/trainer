import abc

import torch.nn as nn
from openai import OpenAI, AsyncOpenAI

from trainer.rollouts.rollout_utils import (
    BASE_URL,
    generate_completions,
    agenerate_completions,
)


class HttpRollout(abc.ABC):
    def __init__(self, model_name: str = "facebook/opt-125m") -> None:
        self.client = OpenAI(
            base_url=f"{BASE_URL}/v1",
            api_key="EMPTY",
        )
        self.aclient = AsyncOpenAI(
            base_url=f"{BASE_URL}/v1",
            api_key="EMPTY",
        )
        self.model_name = model_name

    @abc.abstractmethod
    def sync_weights(self, trainer_model: nn.Module) -> None: ...

    def generate(
        self,
        prompts: list[str],
        max_tokens: int = 32,
        temperature: float = 0,
        n: int = 1,
    ) -> list[list[str]]:
        return generate_completions(
            client=self.client,
            model=self.model_name,
            prompts=prompts,
            max_tokens=max_tokens,
            temperature=temperature,
            n=n,
        )

    def agenerate(
        self,
        prompts: list[str],
        max_tokens: int = 32,
        temperature: float = 0,
        n: int = 1,
    ) -> list[list[str]]:
        return agenerate_completions(
            aclient=self.aclient,
            model=self.model_name,
            prompts=prompts,
            max_tokens=max_tokens,
            temperature=temperature,
            n=n,
        )
