from __future__ import annotations

import abc
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Generic,
    NamedTuple,
    SupportsFloat,
    TypeVar,
)
from dataclasses import dataclass


ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")
RenderPromptState = TypeVar("RenderPromptState")


@dataclass
class Prompt:
    prompt: list[dict[str, str]] | str
    data_source: Any
    ability: Any
    reward_model: Any
    extra_info: dict[str, Any]


PromptCallable = Callable[[dict[str, Any]], Prompt]


class EnvStepReturn(NamedTuple):
    obs: ObsType
    reward: SupportsFloat
    terminated: bool
    truncated: bool
    info: dict[str, Any]
    done: bool


class BaseEnv(abc.ABC, Generic[ObsType, ActType]):

    @abc.abstractmethod
    def step(self, action: ActType, meta_info: Any | None) -> EnvStepReturn:
        raise NotImplementedError

    @abc.abstractmethod
    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[ObsType, dict[str, Any]]:
        raise NotImplementedError

    def render(self) -> RenderPromptState | list[RenderPromptState] | None:
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError


class BaseDataset(abc.ABC):
    @abc.abstractmethod
    def prepare_dataset(self, ds_name: str) -> list[Prompt]:
        raise NotImplementedError
