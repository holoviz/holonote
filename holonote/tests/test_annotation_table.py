from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest

from holonote.annotate import AnnotationTable

if TYPE_CHECKING:
    from holonote.annotate.typing import SpecDict


def _init_table(spec: SpecDict) -> AnnotationTable:
    table = AnnotationTable()
    table.load(primary_key_name="id", fields=["test_description"])
    assert len(table._region_df) == 0, "Should be initialized empty"
    assert tuple(table._region_df.columns) == AnnotationTable.columns
    return table


def _time_range():
    return (pd.Timestamp("2022-06-17"), pd.Timestamp("2022-06-19"))


def test_table_single_kdim() -> None:
    spec = {"TIME": {"type": np.datetime64, "region": "range"}}
    table = _init_table(spec)

    start, end = _time_range()
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
    pd.testing.assert_frame_equal(table.get_dataframe(spec=spec), expected)


def test_table_multiple_kdim() -> None:
    spec = {
        "TIME": {"type": np.datetime64, "region": "range"},
        "x": {"type": float, "region": "point"},
    }
    table = _init_table(spec)

    start, end = _time_range()
    regions = {"x": 1, "TIME": (start, end)}
    table.add_annotation(regions, spec=spec, id=100, test_description="A test")

    d = {
        "region": ["point", "range"],
        "dim": ["x", "TIME"],
        "value": [1, (start, end)],
        "_id": [100, 100],
    }
    expected = pd.DataFrame(d).astype({"_id": object})
    pd.testing.assert_frame_equal(table._region_df, expected)

    d = {
        "start[TIME]": {100: start},
        "end[TIME]": {100: end},
        "point[x]": {100: 1.0},
        "test_description": {100: "A test"},
    }
    expected = pd.DataFrame(d)
    expected.index.name = "id"
    pd.testing.assert_frame_equal(table.get_dataframe(spec=spec), expected)


def test_table_multiple_kdim_and_annotations() -> None:
    spec = {
        "TIME": {"type": np.datetime64, "region": "range"},
        "x": {"type": float, "region": "point"},
    }
    table = _init_table(spec)

    start, end = _time_range()
    regions = {"x": 1, "TIME": (start, end)}
    table.add_annotation(regions, spec=spec, id=100, test_description="101")
    regions = {"x": 2}
    table.add_annotation(regions, spec=spec, id=101, test_description="102")

    d = {
        "region": ["point", "range", "point"],
        "dim": ["x", "TIME", "x"],
        "value": [1, (start, end), 2],
        "_id": [100, 100, 101],
    }
    expected = pd.DataFrame(d).astype({"_id": object})
    pd.testing.assert_frame_equal(table._region_df, expected)

    d = {
        "start[TIME]": {100: start, 101: pd.NaT},
        "end[TIME]": {100: end, 101: pd.NaT},
        "point[x]": {100: 1.0, 101: 2.0},
        "test_description": {100: "101", 101: "102"},
    }
    expected = pd.DataFrame(d)
    expected.index.name = "id"
    pd.testing.assert_frame_equal(table.get_dataframe(spec=spec), expected)


def test_only_adding_one_dim_with_multiple_dimensions() -> None:
    spec = {
        "TIME": {"type": np.datetime64, "region": "range"},
        "x": {"type": float, "region": "point"},
    }
    table = _init_table(spec)

    regions = {"x": 1}
    table.add_annotation(regions, spec=spec, id=100, test_description="101")
    regions = {"x": 2}
    table.add_annotation(regions, spec=spec, id=101, test_description="102")

    d = {
        "region": ["point", "point"],
        "dim": ["x", "x"],
        "value": [1, 2],
        "_id": [100, 101],
    }
    expected = pd.DataFrame(d).astype({"_id": object, "value": object})
    pd.testing.assert_frame_equal(table._region_df, expected)

    d = {
        "start[TIME]": {100: pd.NaT, 101: pd.NaT},
        "end[TIME]": {100: pd.NaT, 101: pd.NaT},
        "point[x]": {100: 1.0, 101: 2.0},
        "test_description": {100: "101", 101: "102"},
    }
    expected = pd.DataFrame(d)
    expected.index.name = "id"
    pd.testing.assert_frame_equal(table.get_dataframe(spec=spec), expected)


def test_nodata_multiple_dimension() -> None:
    spec = {
        "TIME": {"type": np.datetime64, "region": "range"},
        "x": {"type": float, "region": "point"},
    }
    table = _init_table(spec)

    assert table._region_df.empty
    assert tuple(table._region_df.columns) == AnnotationTable.columns

    d = {
        "start[TIME]": [pd.NaT],
        "end[TIME]": [pd.NaT],
        "point[x]": [0.0],
        "test_description": [""],
    }
    expected = pd.DataFrame(d).drop(index=0)
    expected.index.name = "id"
    output = table.get_dataframe(spec=spec)

    assert output.empty
    pd.testing.assert_series_equal(output.dtypes, expected.dtypes)
    pd.testing.assert_index_equal(output.columns, expected.columns)


@pytest.mark.parametrize("dim", ["x", "TIME"])
def test_table_multiple_kdim_and_annotations_with_selected_dims(dim) -> None:
    spec = {
        "TIME": {"type": np.datetime64, "region": "range"},
        "x": {"type": float, "region": "point"},
    }
    table = _init_table(spec)

    start, end = _time_range()
    regions = {"x": 1, "TIME": (start, end)}
    table.add_annotation(regions, spec=spec, id=100, test_description="101")
    regions = {"x": 2}
    table.add_annotation(regions, spec=spec, id=101, test_description="102")

    d = {
        "region": ["point", "range", "point"],
        "dim": ["x", "TIME", "x"],
        "value": [1, (start, end), 2],
        "_id": [100, 100, 101],
    }
    expected = pd.DataFrame(d).astype({"_id": object})
    pd.testing.assert_frame_equal(table._region_df, expected)

    d = {
        "start[TIME]": {100: start, 101: pd.NaT},
        "end[TIME]": {100: end, 101: pd.NaT},
        "point[x]": {100: 1.0, 101: 2.0},
        "test_description": {100: "101", 101: "102"},
    }
    d = {k: v for k, v in d.items() if "[" not in k or dim in k}
    expected = pd.DataFrame(d)
    expected.index.name = "id"
    pd.testing.assert_frame_equal(table.get_dataframe(spec=spec, dims=[dim]), expected)


@pytest.mark.parametrize("dim", ["x", "TIME"])
def test_nodata_multiple_dimension_with_selected_dims(dim) -> None:
    spec = {
        "TIME": {"type": np.datetime64, "region": "range"},
        "x": {"type": float, "region": "point"},
    }
    table = _init_table(spec)

    assert table._region_df.empty

    d = {
        "start[TIME]": [pd.NaT],
        "end[TIME]": [pd.NaT],
        "point[x]": [0.0],
        "test_description": [""],
    }
    d = {k: v for k, v in d.items() if "[" not in k or dim in k}
    expected = pd.DataFrame(d).drop(index=0)
    expected.index.name = "id"
    output = table.get_dataframe(spec=spec, dims=[dim])

    assert output.empty
    pd.testing.assert_series_equal(output.dtypes, expected.dtypes)
    pd.testing.assert_index_equal(output.columns, expected.columns)


def test_table_multiple_kdim_with_wrong_dims() -> None:
    spec = {
        "TIME": {"type": np.datetime64, "region": "range"},
        "x": {"type": float, "region": "point"},
    }
    table = _init_table(spec)

    msg = r"Dimension\(s\) 'bad1' not in the spec"
    with pytest.raises(ValueError, match=msg):
        table.get_dataframe(spec=spec, dims=["bad1"])

    with pytest.raises(ValueError, match=msg):
        table.get_dataframe(spec=spec, dims=["TIME", "bad1"])

    msg = r"Dimension\(s\) 'bad1', 'bad2' not in the spec"
    with pytest.raises(ValueError, match=msg):
        table.get_dataframe(spec=spec, dims=["bad1", "bad2"])

    assert False  # noqa
