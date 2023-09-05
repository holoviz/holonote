from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.mark.parametrize(
    "define_kwargs",
    [
        {},
        {"TIME": "time", "description": "description"},
        {"TIME": "time"},
        {"TIME": "time", "description": "description", "index": True},
        {"TIME": "time", "index": True},
    ],
)
def test_define_annotations_range1d(annotator_range1d, define_kwargs):
    annotator = annotator_range1d
    annotator_table = annotator.annotation_table
    assert annotator_table._region_df.empty
    assert annotator_table._field_df.empty

    # Create data
    time = pd.date_range("2020-01-01", periods=3, freq="D")
    descriptions = ["Ann0", "Ann1", "Ann2"]
    if define_kwargs:
        data = pd.DataFrame({"time": time, "description": descriptions})
    else:
        data = pd.DataFrame({"TIME": time, "description": descriptions})

    # Define annotations
    annotator.define_annotations(data, **define_kwargs)

    # Validate it
    rdf, fdf = annotator_table._region_df, annotator_table._field_df
    assert (rdf["value"].to_list() == time).all()
    assert (rdf["dim"] == "TIME").all()
    assert (rdf["region"] == "range").all()
    assert fdf["description"].to_list() == ["Ann0", "Ann1", "Ann2"]
    assert fdf.index.unique().size == 3
    assert (rdf["_id"] == fdf.index).all()

    if "index" in define_kwargs:
        assert (fdf.index == ["0", "1", "2"]).all()
    else:
        assert (fdf.index != ["0", "1", "2"]).all()


@pytest.mark.parametrize(
    "define_kwargs",
    [
        {"x": ("start", "end"), "description": "description"},
        {"x": ("start", "end")},
        {"x": ("start", "end"), "description": "description", "index": True},
        {"x": ("start", "end"), "index": True},
    ],
)
def test_define_annotations_range2d(annotator_range2d, define_kwargs):
    annotator = annotator_range2d
    annotator_table = annotator.annotation_table
    assert annotator_table._region_df.empty
    assert annotator_table._field_df.empty

    # Create data
    starts = np.arange(3)
    ends = starts + 2
    descriptions = ["Ann0", "Ann1", "Ann2"]
    data = pd.DataFrame({"start": starts, "end": ends, "description": descriptions})

    # Define annotations
    annotator.define_annotations(data, **define_kwargs)

    # Validate it
    rdf, fdf = annotator_table._region_df, annotator_table._field_df
    assert rdf["value"].to_list() == [(0, 2), (1, 3), (2, 4)]
    assert (rdf["dim"] == "x").all()
    assert (rdf["region"] == "range").all()
    assert fdf["description"].to_list() == ["Ann0", "Ann1", "Ann2"]
    assert fdf.index.unique().size == 3
    assert (rdf["_id"] == fdf.index).all()

    if "index" in define_kwargs:
        assert (fdf.index == ["0", "1", "2"]).all()
    else:
        assert (fdf.index != ["0", "1", "2"]).all()
