from __future__ import annotations

import contextlib
from typing import Iterator

import holoviews as hv
import numpy as np
import pandas as pd
import pytest

from holonote.annotate import Annotator, SQLiteDB, UUIDHexStringKey


@pytest.fixture()
def conn_sqlite_uuid(tmp_path) -> Iterator[SQLiteDB]:
    conn = SQLiteDB(filename=str(tmp_path / "test.db"), primary_key=UUIDHexStringKey())
    yield conn
    with contextlib.suppress(Exception):
        conn.cursor.close()
    with contextlib.suppress(Exception):
        conn.con.close()


@pytest.fixture()
def annotator_range1d(conn_sqlite_uuid) -> Annotator:
    anno = Annotator(
        {"TIME": np.datetime64},
        fields=["description"],
        connector=conn_sqlite_uuid,
    )
    return anno


@pytest.fixture()
def annotator_point1d(conn_sqlite_uuid) -> Annotator:
    anno = Annotator(
        {"TIME": (np.datetime64, "point")},
        fields=["description"],
        connector=conn_sqlite_uuid,
    )
    return anno


@pytest.fixture()
def annotator_range2d(conn_sqlite_uuid) -> Annotator:
    anno = Annotator(
        {"x": float, "y": float},
        fields=["description"],
        connector=conn_sqlite_uuid,
    )
    return anno


@pytest.fixture()
def annotator_point2d(conn_sqlite_uuid) -> Annotator:
    anno = Annotator(
        {"x": (float, "point"), "y": (float, "point")},
        fields=["description"],
        connector=conn_sqlite_uuid,
    )
    return anno


@pytest.fixture()
def multiple_region_annotator(conn_sqlite_uuid) -> Annotator:
    return Annotator(
        {"TIME": np.datetime64},
        fields=["description"],
        # region_types=["Point", "Range"],
        connector=conn_sqlite_uuid,
    )


@pytest.fixture()
def multiple_annotators(conn_sqlite_uuid) -> Annotator:
    return Annotator(
        {"TIME": np.datetime64, "x": float, "y": float},
        fields=["description"],
        connector=conn_sqlite_uuid,
    )


@pytest.fixture()
def multiple_fields_annotator(conn_sqlite_uuid) -> Annotator:
    conn_sqlite_uuid.fields = ["field1", "field2"]
    anno = Annotator(
        {"TIME": np.datetime64},
        fields=["field1", "field2"],
        connector=conn_sqlite_uuid,
    )
    return anno


@pytest.fixture()
def element_range1d() -> hv.Curve:
    time = pd.date_range("2020-01-01", "2020-01-10", freq="D").to_numpy()
    return hv.Curve(time, kdims=["TIME"])


@pytest.fixture()
def element_range2d() -> hv.Image:
    x = np.arange(10)
    xy = x[:, np.newaxis] * x
    return hv.Image(xy, kdims=["x", "y"])
