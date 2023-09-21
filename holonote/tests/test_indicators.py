from __future__ import annotations

import numpy as np
import pandas as pd

from holonote.annotate.annotator import Indicator


def test_range2d_id_matches() -> None:
    value = np.arange(8).reshape(2, 4)
    region_df = pd.DataFrame({"value": list(value), "_id": ["A", "B"]})
    field_df = pd.DataFrame(["B", "A"], index=["B", "A"], columns=["description"])

    # id and description should match
    output = Indicator.ranges_2d(region_df, field_df).data
    expected = pd.DataFrame(
        {
            "x0": {0: 0, 1: 4},
            "y0": {0: 2, 1: 6},
            "x1": {0: 1, 1: 5},
            "y1": {0: 3, 1: 7},
            "description": {0: "A", 1: "B"},
            "id": {0: "A", 1: "B"},
        }
    )
    pd.testing.assert_frame_equal(output, expected)


def test_range1d_id_matches() -> None:
    value = np.arange(4).reshape(2, 2)
    region_df = pd.DataFrame({"value": list(value), "_id": ["A", "B"]})
    field_df = pd.DataFrame(["B", "A"], index=["B", "A"], columns=["description"])

    # id and description should match
    output = Indicator.ranges_1d(
        region_df, field_df, extra_params={"rect_min": -2, "rect_max": -2}
    ).data
    expected = pd.DataFrame(
        {
            "x0": {0: 0, 1: 2},
            "y0": {0: -2, 1: -2},
            "x1": {0: 1, 1: 3},
            "y1": {0: -2, 1: -2},
            "description": {0: "A", 1: "B"},
            "id": {0: "A", 1: "B"},
        }
    )
    pd.testing.assert_frame_equal(output, expected)
