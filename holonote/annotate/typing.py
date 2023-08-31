from __future__ import annotations

from typing import Callable, Literal, TypedDict


class SpecItem(TypedDict):
    type: Callable
    region: Literal["range"] | Literal["single"] | Literal["multi"]


SpecDict = dict[str, SpecItem]
