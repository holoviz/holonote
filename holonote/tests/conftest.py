from __future__ import annotations

from typing import Iterator

import numpy as np
import pytest

from holonote.annotate import Annotator, SQLiteDB, UUIDHexStringKey


@pytest.fixture()
def conn_sqlite_uuid(tmp_path) -> Iterator[SQLiteDB]:
    conn = SQLiteDB(filename=str(tmp_path / "test.db"), primary_key=UUIDHexStringKey())
    yield conn
    try:
        conn.cursor.close()
    except Exception:
        pass
    try:
        conn.con.close()
    except Exception:
        pass


@pytest.fixture()
def annotator_range1d(conn_sqlite_uuid) -> Annotator:
    anno = Annotator(
        {"TIME": np.datetime64},
        fields=["description"],
        region_types=["Range"],
        connector=conn_sqlite_uuid,
    )
    return anno


@pytest.fixture()
def annotator_point1d(conn_sqlite_uuid) -> Annotator:
    anno = Annotator(
        {"TIME": (np.datetime64, 'single')},
        fields=["description"],
        region_types=["Point"],
        connector=conn_sqlite_uuid,
    )
    return anno


@pytest.fixture()
def annotator_range2d(conn_sqlite_uuid) -> Annotator:
    anno = Annotator(
        {"x": float, "y": float},
        fields=["description"],
        region_types=["Range"],
        connector=conn_sqlite_uuid,
    )
    return anno


@pytest.fixture()
def annotator_point2d(conn_sqlite_uuid) -> Annotator:
    anno = Annotator(
        {"x": (float, "single"), "y": (float, "single")},
        fields=["description"],
        region_types=["Point"],
        connector=conn_sqlite_uuid,
    )
    return anno


@pytest.fixture()
def multiple_region_annotator(conn_sqlite_uuid) -> Annotator:
    return Annotator(
        {"TIME": np.datetime64},
        fields=["description"],
        region_types=["Point", "Range"],
        connector=conn_sqlite_uuid,
    )


@pytest.fixture()
def multiple_annotators(conn_sqlite_uuid) -> dict[str, Annotator | SQLiteDB]:
    annotator_range1d = Annotator(
        {"TIME": np.datetime64},
        fields=["description"],
        region_types=["Range"],
        connector=conn_sqlite_uuid,
        init=False,
    )
    annotator_range2d = Annotator(
        {"x": float, "y": float},
        fields=["description"],
        region_types=["Range"],
        connector=conn_sqlite_uuid,
    )

    annotator_range1d.load()
    annotator_range2d.load()
    output = {
        "annotation1d": annotator_range1d,
        "annotation2d": annotator_range2d,
        "conn": conn_sqlite_uuid,
    }
    return output


@pytest.fixture()
def multiple_fields_annotator(conn_sqlite_uuid) -> Annotator:
    conn_sqlite_uuid.fields = ["field1", "field2"]
    anno = Annotator(
        {"TIME": np.datetime64},
        fields=["field1", "field2"],
        connector=conn_sqlite_uuid,
    )
    return anno
