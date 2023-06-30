import numpy as np
import pytest

from holonote.annotate import Annotator, SQLiteDB, UUIDHexStringKey


@pytest.fixture()
def conn_sqlite_uuid(tmp_path):
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
def annotator_range1d(conn_sqlite_uuid):
    anno = Annotator(
        {"TIME": np.datetime64},
        fields=["description"],
        region_types=["Range"],
        connector=conn_sqlite_uuid,
    )
    return anno


@pytest.fixture()
def annotator_point1d(conn_sqlite_uuid):
    anno = Annotator(
        {"TIME": np.datetime64},
        fields=["description"],
        region_types=["Point"],
        connector=conn_sqlite_uuid,
    )
    return anno


@pytest.fixture()
def annotator_range2d(conn_sqlite_uuid):
    anno = Annotator(
        {"x": float, "y": float},
        fields=["description"],
        region_types=["Range"],
        connector=conn_sqlite_uuid,
    )
    return anno


@pytest.fixture()
def annotator_point2d(conn_sqlite_uuid):
    anno = Annotator(
        {"x": float, "y": float},
        fields=["description"],
        region_types=["Point"],
        connector=conn_sqlite_uuid,
    )
    return anno


@pytest.fixture()
def multiple_region_annotator(conn_sqlite_uuid):
    anno = Annotator(
        {"TIME": np.datetime64},
        fields=["description"],
        region_types=["Point", "Range"],
        connector=conn_sqlite_uuid,
    )
    return anno


@pytest.fixture()
def multiple_annotators(conn_sqlite_uuid):
    range1d_anno = Annotator(
        {"TIME": np.datetime64},
        fields=["description"],
        region_types=["Range"],
        connector=conn_sqlite_uuid,
    )
    range2d_anno = Annotator(
        {"A": np.float64, "B": np.float64},
        fields=["description"],
        region_types=["Range"],
        connector=conn_sqlite_uuid,
    )

    output = {
        "annotation1d": range1d_anno,
        "annotation2d": range2d_anno,
        "conn": conn_sqlite_uuid,
    }
    return output


@pytest.fixture()
def multiple_fields_annotator(conn_sqlite_uuid):
    conn_sqlite_uuid.fields = ["field1", "field2"]
    anno = Annotator(
        {"TIME": np.datetime64},
        fields=["field1", "field2"],
        connector=conn_sqlite_uuid,
    )
    return anno
