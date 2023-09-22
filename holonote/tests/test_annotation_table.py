from __future__ import annotations

import numpy as np
import pandas as pd

from holonote.annotate import AnnotationTable


def test_table_single_kdim() -> None:
    spec = {"TIME": {"type": np.datetime64, "region": "range"}}
    table = AnnotationTable()
    table.load(primary_key_name="id", fields=["test_description"])
    assert len(table._region_df) == 0, "Should be initialized empty"
    assert tuple(table._region_df.columns) == AnnotationTable.columns

    start = pd.Timestamp("2022-06-17")
    end = pd.Timestamp("2022-06-19")
    regions = {"TIME": (start, end)}
    table.add_annotation(regions, spec=spec, id=100, test_description="A test")

    d = {"region": "range", "dim": "TIME", "value": (start, end), "_id": 100}
    expected = pd.DataFrame([d]).astype({"_id": object})
    pd.testing.assert_frame_equal(table._region_df, expected)

    d = {
        "start[TIME]": {100: start},
        "end[TIME]": {100: end},
        "test_description": {100: "A test"},
    }
    expected = pd.DataFrame(d)
    expected.index.name = "id"
    pd.testing.assert_frame_equal(table.get_dataframe(), expected)


def test_table_multiple_kdim() -> None:
    spec = {
        "TIME": {"type": np.datetime64, "region": "range"},
        "x": {"type": float, "region": "point"},
    }
    table = AnnotationTable()
    table.load(primary_key_name="id", fields=["test_description"])

    start = pd.Timestamp("2022-06-17")
    end = pd.Timestamp("2022-06-19")
    regions = {"x": 1, "TIME": (start, end)}
    table.add_annotation(regions, spec=spec, id=100, test_description="A test")

    d = {
        "region": {0: "point", 1: "range"},
        "dim": {0: "x", 1: "TIME"},
        "value": {0: 1, 1: (start, end)},
        "_id": {0: 100, 1: 100},
    }
    expected = pd.DataFrame(d).astype({"_id": object})
    pd.testing.assert_frame_equal(table._region_df, expected)

    d = {
        "start[TIME]": {100: start},
        "end[TIME]": {100: end},
        "point[x]": {100: 1},
        "test_description": {100: "A test"},
    }
    expected = pd.DataFrame(d)
    expected.index.name = "id"
    pd.testing.assert_frame_equal(table.get_dataframe(), expected)


def test_table_multiple_kdim_and_annotations() -> None:
    spec = {
        "TIME": {"type": np.datetime64, "region": "range"},
        "x": {"type": float, "region": "point"},
    }
    table = AnnotationTable()
    table.load(primary_key_name="id", fields=["test_description"])

    start = pd.Timestamp("2022-06-17")
    end = pd.Timestamp("2022-06-19")
    regions = {"x": 1, "TIME": (start, end)}
    table.add_annotation(regions, spec=spec, id=100, test_description="101")
    regions = {"x": 2}
    table.add_annotation(regions, spec=spec, id=101, test_description="102")

    d = {
        "region": {0: "point", 1: "range", 2: "point"},
        "dim": {0: "x", 1: "TIME", 2: "x"},
        "value": {0: 1, 1: (start, end), 2: 2},
        "_id": {0: 100, 1: 100, 2: 101},
    }
    expected = pd.DataFrame(d).astype({"_id": object})
    pd.testing.assert_frame_equal(table._region_df, expected)

    d = {
        "start[TIME]": {100: start, 101: pd.NaT},
        "end[TIME]": {100: end, 101: pd.NaT},
        "point[x]": {100: 1, 101: 2},
        "test_description": {100: "101", 101: "102"},
    }
    expected = pd.DataFrame(d)
    expected.index.name = "id"
    pd.testing.assert_frame_equal(table.get_dataframe(), expected)
