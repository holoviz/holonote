from __future__ import annotations

from typing import Callable, Literal, TypedDict


class SpecItem(TypedDict):
    """Contains the type and region specification for a region.

    The type is a callable that can be used to convert the data to the
    correct type. This could be `np.datetime64` or `float` for example.

    The region specification is either "range", "single", or "multi".
    """
    type: Callable
    region: Literal["range"] | Literal["single"] | Literal["multi"]


SpecDict = dict[str, SpecItem]
