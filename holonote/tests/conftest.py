import pytest

from holonote.annotate import SQLiteDB, UUIDHexStringKey


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
