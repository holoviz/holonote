from __future__ import annotations

import contextlib
from collections.abc import Iterator

import holoviews as hv
import numpy as np
import pandas as pd
import pytest

from holonote.annotate import Annotator, SQLiteDB, UUIDHexStringKey

optional_markers = {
    "benchmark": {
        "help": "Run benchmarks",
        "marker-descr": "Benchmark test marker",
        "skip-reason": "Test only runs with the --benchmark option.",
    },
}


def pytest_addoption(parser):
    for marker, info in optional_markers.items():
        parser.addoption(f"--{marker}", action="store_true", default=False, help=info["help"])


def pytest_configure(config):
    for marker, info in optional_markers.items():
        config.addinivalue_line("markers", "{}: {}".format(marker, info["marker-descr"]))


def pytest_collection_modifyitems(config, items):
    skipped, selected = [], []
    markers = [m for m in optional_markers if config.getoption(f"--{m}")]
    empty = not markers
    for item in items:
        if empty and any(m in item.keywords for m in optional_markers):
            skipped.append(item)
        elif empty or (not empty and any(m in item.keywords for m in markers)):
            selected.append(item)
        else:
            skipped.append(item)

    config.hook.pytest_deselected(items=skipped)
    items[:] = selected


@pytest.fixture
def conn_sqlite_uuid(tmp_path) -> Iterator[SQLiteDB]:
    conn = SQLiteDB(filename=str(tmp_path / "test.db"), primary_key=UUIDHexStringKey())
    yield conn
    with contextlib.suppress(Exception):
        conn.cursor.close()
    with contextlib.suppress(Exception):
        conn.con.close()


@pytest.fixture
def annotator_range1d(conn_sqlite_uuid) -> Annotator:
    anno = Annotator(
        {"TIME": np.datetime64},
        fields=["description"],
        connector=conn_sqlite_uuid,
    )
    return anno


@pytest.fixture
def annotator_point1d(conn_sqlite_uuid) -> Annotator:
    anno = Annotator(
        {"TIME": (np.datetime64, "point")},
        fields=["description"],
        connector=conn_sqlite_uuid,
    )
    return anno


@pytest.fixture
def annotator_range2d(conn_sqlite_uuid) -> Annotator:
    anno = Annotator(
        {"x": float, "y": float},
        fields=["description"],
        connector=conn_sqlite_uuid,
    )
    return anno


@pytest.fixture
def annotator_point2d(conn_sqlite_uuid) -> Annotator:
    anno = Annotator(
        {"x": (float, "point"), "y": (float, "point")},
        fields=["description"],
        connector=conn_sqlite_uuid,
    )
    return anno


@pytest.fixture
def multiple_region_annotator(conn_sqlite_uuid) -> Annotator:
    return Annotator(
        {"TIME": np.datetime64},
        fields=["description"],
        # region_types=["Point", "Range"],
        connector=conn_sqlite_uuid,
    )


@pytest.fixture
def multiple_annotators(conn_sqlite_uuid) -> Annotator:
    return Annotator(
        {"TIME": np.datetime64, "x": float, "y": float},
        fields=["description"],
        connector=conn_sqlite_uuid,
    )


@pytest.fixture
def multiple_fields_annotator(conn_sqlite_uuid) -> Annotator:
    conn_sqlite_uuid.fields = ["field1", "field2"]
    anno = Annotator(
        {"TIME": np.datetime64},
        fields=["field1", "field2"],
        connector=conn_sqlite_uuid,
    )
    return anno


@pytest.fixture
def element_range1d() -> hv.Curve:
    time = pd.date_range("2020-01-01", "2020-01-10", freq="D").to_numpy()
    return hv.Curve(time, kdims=["TIME"])


@pytest.fixture
def element_range2d() -> hv.Image:
    x = np.arange(10)
    xy = x[:, np.newaxis] * x
    return hv.Image(xy, kdims=["x", "y"])


@pytest.fixture
def cat_annotator(conn_sqlite_uuid) -> Annotator:
    # Initialize annotator
    annotator = Annotator(
        {"x": float},
        fields=["description", "category"],
        connector=conn_sqlite_uuid,
        groupby="category",
    )
    # Add data to annotator
    data = {
        "category": ["A", "B", "A", "C", "B"],
        "start_number": [1, 6, 11, 16, 21],
        "end_number": [5, 10, 15, 20, 25],
        "description": list("ABCDE"),
    }
    annotator.define_annotations(pd.DataFrame(data), x=("start_number", "end_number"))
    # Setup display
    annotator.get_display("x")
    return annotator


@pytest.fixture()
def cat_annotator_no_data(conn_sqlite_uuid) -> Annotator:
    # Initialize annotator
    annotator = Annotator(
        {"x": float},
        fields=["description", "category"],
        connector=conn_sqlite_uuid,
        groupby="category",
    )
    # Setup display
    annotator.get_display("x")
    return annotator
