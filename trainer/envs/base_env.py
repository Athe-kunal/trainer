from __future__ import annotations

import abc
from typing import TYPE_CHECKING, Any, Generic, SupportsFloat, TypeVar


ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")
RenderPromptState = TypeVar("RenderPromptState")


class BaseEnv(abc.ABC, Generic[ObsType, ActType]):
    def step(
        self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        raise NotImplementedError

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[ObsType, dict[str, Any]]:
        raise NotImplementedError

    def render(self) -> RenderPromptState | list[RenderPromptState] | None:
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError
