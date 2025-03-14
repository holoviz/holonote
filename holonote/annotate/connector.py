from __future__ import annotations

import datetime as dt
import hashlib
import os
import sqlite3
import sys
import uuid
from contextlib import contextmanager
from functools import cache
from pathlib import Path
from shutil import copyfile
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import param

if TYPE_CHECKING:
    from .typing import SpecDict


@cache
def _sqlite_adapters() -> None:
    # Most of the following code has been copied here from in Python 3.11:
    # `sqlite3.dbapi2.register_adapters_and_converters`.
    # Including minor modifications to source code to pass linting
    # https://docs.python.org/3/license.html#psf-license
    # Extra adapters have been added for dt.time, numpy.datetime64, and pd.Timestamp.

    def adapt_date(val):
        return val.isoformat()

    def adapt_datetime(val):
        return val.isoformat(" ")

    def adapt_time(t) -> str:
        return f"{t.hour:02d}:{t.minute:02d}:{t.second:02d}.{t.microsecond:06d}"

    def adapt_np_datetime64(val):
        return np.datetime_as_string(val, unit="us").replace("T", " ")

    def convert_date(val):
        return dt.date(*map(int, val.split(b"-")))

    def convert_timestamp(val):
        datepart, timepart = val.split(b" ")
        year, month, day = map(int, datepart.split(b"-"))
        timepart_full = timepart.split(b".")
        hours, minutes, seconds = map(int, timepart_full[0].split(b":"))
        microseconds = int(f"{timepart_full[1].decode():0<6.6}") if len(timepart_full) == 2 else 0

        val = dt.datetime(year, month, day, hours, minutes, seconds, microseconds)
        return val

    if sys.version_info >= (3, 12):
        # Python 3.12 has removed datetime support from sqlite3
        # https://github.com/python/cpython/pull/93095
        sqlite3.register_adapter(dt.date, adapt_date)
        sqlite3.register_adapter(dt.datetime, adapt_datetime)
        sqlite3.register_converter("date", convert_date)
        sqlite3.register_converter("timestamp", convert_timestamp)

    sqlite3.register_adapter(dt.time, adapt_time)
    sqlite3.register_adapter(np.datetime64, adapt_np_datetime64)
    sqlite3.register_adapter(pd.Timestamp, adapt_datetime)


def _get_valid_sqlite_name(name: object):
    # See https://stackoverflow.com/questions/6514274/how-do-you-escape-strings\
    # -for-sqlite-table-column-names-in-python
    # Ensure the string can be encoded as UTF-8.
    # Ensure the string does not include any NUL characters.
    # Replace all " with "".
    # Wrap the entire thing in double quotes.
    # Inspired from pandas.io.sql._get_valid_sqlite_name

    try:
        uname = str(name).encode("utf-8", "strict").decode("utf-8")
    except UnicodeError as err:
        msg = f"Cannot convert identifier to UTF-8: '{name}'"
        raise ValueError(msg) from err

    if not len(uname):
        msg = "Empty table or column name specified"
        raise ValueError(msg)

    nul_index = uname.find("\x00")
    if nul_index >= 0:
        msg = "SQLite identifier cannot contain NULs"
        raise ValueError(msg)
    return '"' + uname.replace('"', '""') + '"'


class PrimaryKey(param.Parameterized):
    """
    Generator of the primary key used to keep track of annotations in
    HoloViews.

    The generated key is used to reference annotations until they are
    committed, at which point they may 1) be inserted in the database as
    the primary key value (policy='insert') 2) are checked against the
    primary key value chosen by the database which is expected to match
    in most cases.

    In real situations where the key is chosen by the database, the key
    generated will *not* always match the actual key assigned. The
    policy parameter decides the resulting behavior in these cases.
    """

    field_name = param.String(default="id", allow_None=False)

    policy = param.ObjectSelector(
        default="ignore-mismatch",
        objects=["insert", "ignore-mismatch", "warn-mismatch", "error-mismatch"],
    )

    schema = param.String(default="INTEGER PRIMARY KEY", constant=True, allow_None=False)

    connector_class = param.String(default="SQLiteDB")

    def __call__(self, connector, key_list=None):
        """
        The key list is the current list of index values that are
        outstanding (i.e. have not been committed).
        """
        raise NotImplementedError

    def validate(self, database_key, local_key):
        if self.policy in ["insert", "ignore-mismatch"]:
            return

        if database_key != local_key:
            print("MISMATCH")

    def cast(self, value):
        "Cast a user supplied value to a known supported type"
        raise NotImplementedError


class AutoIncrementKey(PrimaryKey):
    """
    AUTOINCREMENT needed to prevent reuse of ids from deleted rows:
    https://www.sqlite.org/autoinc.html
    """

    field_name = param.String(default="id", allow_None=False)

    policy = param.ObjectSelector(
        default="ignore-mismatch",
        objects=["insert", "ignore-mismatch", "warn-mismatch", "error-mismatch"],
    )

    schema = param.String(
        default="INTEGER PRIMARY KEY AUTOINCREMENT", constant=True, allow_None=False
    )

    def __call__(self, connector, key_list=None):
        key_list_max = max(key_list) if key_list else 0
        connector_max = connector.max_rowid()
        connector_max = 0 if connector_max is None else connector_max
        max_rowid = max(key_list_max, connector_max)
        return max_rowid + 1

    def cast(self, value):
        return int(value)  # e.g. np.int64 from a pandas index won't work!


class UUIDHexStringKey(PrimaryKey):  # Probably the better default
    """
    Example of 'insert' policy where the generated primary key value can
    be inserted in the database as uuids are independent and not
    expected to clash.
    """

    field_name = param.String(default="uuid", allow_None=False)

    policy = param.ObjectSelector(
        default="insert", objects=["insert", "ignore-mismatch", "warn-mismatch", "error-mismatch"]
    )

    schema = param.String("TEXT PRIMARY KEY", constant=True, allow_None=False)

    length = param.Integer(default=32, bounds=(4, 32))

    def __call__(self, connector, key_list=None):
        return uuid.uuid4().hex[: self.length]

    def cast(self, value):
        return str(value)


class UUIDBinaryKey(PrimaryKey):
    """
    Example of 'insert' policy where the generated primary key value can
    be inserted in the database as uuids are independent and not
    expected to clash.
    """

    field_name = param.String(default="uuid", allow_None=False)

    policy = param.ObjectSelector(
        default="insert", objects=["insert", "ignore-mismatch", "warn-mismatch", "error-mismatch"]
    )

    schema = param.String("BINARY PRIMARY KEY", constant=True, allow_None=False)

    def __call__(self, connector, key_list=None):
        return uuid.uuid4().bytes

    def cast(self, value):
        return bytes(value)


class WidgetKey(PrimaryKey):
    """
    Placeholder for a concept where the user can insert a primary key
    value via a widget.
    """


class Connector(param.Parameterized):
    """
    Base class that support the auto-generated default schema
    """

    primary_key = param.Parameter(default=UUIDHexStringKey(), allow_None=False)

    commit_hook = param.Parameter(default=None, doc="Callback, applies default schema if None")

    fields = param.List(default=None, doc="List of column names for domain-specific fields")

    transforms = param.Dict(
        default={
            "insert": lambda x: x,
            "update": lambda x: x,
            "delete": lambda x: x,
            "save": lambda x: x,
            "load": lambda x: x,
        }
    )

    operation_mapping = {}  # Mapping from operation type to corresponding connector method

    # iterate on all the possible types
    type_mapping = {
        bool: "BOOLEAN",
        str: "TEXT",
        float: "REAL",
        int: "INTEGER",
        np.datetime64: "TIMESTAMP",
        dt.date: "TIMESTAMP",
        dt.datetime: "TIMESTAMP",
        dt.time: "TIME",
        pd.Timedelta: "INTEGER",
        pd.Timestamp: "TIMESTAMP",
        np.dtype("datetime64[ns]"): "TIMESTAMP",
        np.dtype("<M8"): "TIMESTAMP",
        np.float64: "REAL",
        np.float32: "REAL",
        np.float16: "REAL",
        np.int8: "INTEGER",
        np.int16: "INTEGER",
        np.int32: "INTEGER",
        np.int64: "INTEGER",
        np.uint8: "INTEGER",
        np.uint16: "INTEGER",
        np.uint32: "INTEGER",
    }

    @classmethod
    def field_value_to_type(cls, value):
        if isinstance(value, list):
            assert all(isinstance(el, str) for el in value), "Only string enums supported"
            return str
        elif hasattr(value, "dtype"):
            return value.dtype
        elif type(value) in cls.type_mapping:
            return type(value)
        elif isinstance(value, param.Parameter) and value.default is not None:
            return type(value.default)
        else:
            msg = f"Connector cannot handle type {type(value)!s}"
            raise TypeError(msg)

    @classmethod
    def schema_from_field_values(cls, fields):
        "Given a dictionary of fields fields and values return the field schemas"
        return {
            name: cls.type_mapping[cls.field_value_to_type(value)]
            for name, value in fields.items()
        }

    @classmethod
    def schema_from_field_types(cls, fields):
        "Given a dictionary of fields fields and values return the field schemas"
        return {name: cls.type_mapping[value] for name, value in fields.items()}

    @classmethod
    def expand_region_column_schema(cls, region_types, kdim_dtypes):
        region_type_column_prefixes = {
            "Range": ("start", "end"),
            "Point": ("point",),
            "point": ("point",),
            "range": ("start", "end"),
        }
        fields = {}
        for region_type in region_types:
            for column in region_type_column_prefixes[region_type]:
                for kdim, kdim_dtype in kdim_dtypes.items():
                    fields[f"{column}_{kdim}"] = cls.type_mapping[kdim_dtype]
        return fields

    @classmethod
    def generate_schema(cls, primary_key, all_region_types, all_kdim_dtypes, field_types):
        schemas = {primary_key.field_name: primary_key.schema}
        region_schemas = {}
        for region_types, kdim_dtypes in zip(all_region_types, all_kdim_dtypes):
            region_schemas = dict(
                region_schemas, **cls.expand_region_column_schema(region_types, kdim_dtypes)
            )
        schemas = dict(schemas, **region_schemas)
        return dict(schemas, **cls.schema_from_field_types(field_types))

    def _incompatible_schema_check(self, expected_keys, columns, fields, region_type):
        msg_prefix = (
            "Unable to read annotations that were stored with a "
            "schema inconsistent with the current settings:"
        )
        msg_suffix = f"Columns found: {columns}"

        missing_field_columns = set(fields) - set(columns)
        if missing_field_columns:
            msg = f"{msg_prefix} Missing field columns {missing_field_columns}. {msg_suffix}"
            raise Exception(msg)

        non_field_columns = set(columns) - set(fields)
        missing_region_columns = set(expected_keys) - non_field_columns
        if missing_region_columns:
            msg = f"{msg_prefix} Missing {region_type!r} region columns {missing_region_columns}. {msg_suffix}"
            raise Exception(msg)

    def _create_column_schema(self, spec: SpecDict, fields: list[str]) -> None:
        field_dtypes = dict.fromkeys(fields, str)  # FIXME - generalize
        all_region_types = [{v["region"] for v in spec.values()}]
        all_kdim_dtypes = [{k: v["type"] for k, v in spec.items()}]
        schema = self.generate_schema(
            self.primary_key, all_region_types, all_kdim_dtypes, field_dtypes
        )
        self.column_schema = schema


class _SQLiteDB(Connector):
    """
    Simple example of a Connector without dependencies, using sqlite3.

    column_schema is a dictionary from column name to SQL CREATE TABLE type declaration.
    """

    filename = param.String(default="annotations.db")

    table_name = param.String(
        default=None,
        allow_None=True,
        doc="""
        The SQL table name to connect to in the database. If None, an
        automatically generated table name will be used.""",
    )

    column_schema = param.Dict(default={})

    operation_mapping = {
        "insert": "add_row",
        "delete": "delete_row",
        "save": "add_rows",
        "update": "update_row",
    }

    _tablename_prefix = "annotations"

    def __init__(self, column_schema=None, connect=True, **params):
        """
        First key in column_schema is assumed to the primary key field if not explicitly specified.
        """
        if column_schema is None:
            column_schema = {}

        params["column_schema"] = column_schema
        self.con = None
        super().__init__(**params)

        if connect:
            self._initialize(column_schema, create_table=False)

    @contextmanager
    def run_transaction(self):
        cur = self.con.cursor()
        try:
            yield cur
            self.con.commit()
        except Exception:
            self.con.rollback()
            raise
        finally:
            cur.close()

    def execute(self, query, *args) -> None:
        with self.run_transaction() as cursor:
            cursor.execute(query, *args)

    def close(self):
        self.con.close()
        self.con = None

    def _initialize(self, column_schema, create_table=True):
        _sqlite_adapters()
        if self.con is None:
            self.con = self._create_database_connection()
        if create_table:
            if self.table_name is None:
                self.table_name = self._generate_table_name(column_schema)
            self.create_table(column_schema=column_schema)

    def _create_database_connection(self) -> sqlite3.Connection:
        return sqlite3.connect(
            self.filename, detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES
        )

    @property
    def _safe_table_name(self) -> str:
        return _get_valid_sqlite_name(self.table_name)

    def _generate_table_name(self, column_schema) -> str:
        "Given the column_schema outputs a deterministic table name"
        h = hashlib.new("md5")
        for item in sorted(column_schema):
            h.update(item.encode("utf-8"))
            h.update(column_schema[item].encode("utf-8"))
        return f"{self._tablename_prefix}_{h.hexdigest()[:8]}"

    @property
    def uninitialized(self):
        if self.con is not None:
            return self.table_name not in self.get_tables()
        return True

    def max_rowid(self):
        with self.run_transaction() as cursor:
            return cursor.execute(f"SELECT max(ROWID) from {self._safe_table_name}").fetchone()[0]

    def initialize(self, column_schema):
        self.column_schema = column_schema
        self._initialize(column_schema)

    def load_dataframe(self):
        raw_df = pd.read_sql_query(f"SELECT * FROM {self._safe_table_name}", self.con)
        return raw_df.set_index(self.primary_key.field_name)

    def get_tables(self):
        with self.run_transaction() as cursor:
            res = cursor.execute("SELECT name FROM sqlite_master").fetchmany()
        return [el[0] for el in res]

    def create_table(self, column_schema=None):
        column_schema = column_schema if column_schema else self.column_schema
        column_spec = ",\n".join(
            [f"{_get_valid_sqlite_name(name)} {spec}" for name, spec in column_schema.items()]
        )
        query = f"CREATE TABLE IF NOT EXISTS {self._safe_table_name} ({column_spec});"
        self.execute(query)

    def delete_table(self):
        self.execute(f"DROP TABLE IF EXISTS {self._safe_table_name}")

    def add_rows(self, field_list):  # Used execute_many
        for field in field_list:
            self.add_row(**field)

    def add_row(self, **fields):
        columns, parameters = zip(*fields.items())

        if self.primary_key.policy != "insert":
            columns = columns[1:]
            parameters = parameters[1:]

        placeholders = ", ".join(["?"] * len(parameters))
        column_str = ", ".join(map(_get_valid_sqlite_name, columns))
        query = f"INSERT INTO {self._safe_table_name} ({column_str}) VALUES({placeholders});"
        with self.run_transaction() as cursor:
            cursor.execute(query, parameters)
            self.primary_key.validate(cursor.lastrowid, fields[self.primary_key.field_name])

    def delete_all_rows(self):
        "Obviously a destructive operation!"
        self.execute(f"DELETE FROM {self._safe_table_name};")

    def delete_row(self, id_val):
        self.execute(
            f"DELETE FROM {self._safe_table_name} WHERE {self.primary_key.field_name} = ?",
            (self.primary_key.cast(id_val),),
        )

    def update_row(self, **updates):  # updates as a dictionary OR remove posarg?
        assert self.primary_key.field_name in updates
        id_val = updates.pop(self.primary_key.field_name)
        set_updates = ", ".join('"' + k + '"' + " = ?" for k in updates)
        query = f'UPDATE {self._safe_table_name} SET {set_updates} WHERE "{self.primary_key.field_name}" = ?;'
        self.execute(query, [*updates.values(), id_val])


class _SQLiteDBJupyterLite(_SQLiteDB):
    # Sqlite don't work in JupyterLite environment:
    # https://github.com/jupyterlite/pyodide-kernel/issues/35

    def __init__(self, column_schema=None, connect=True, **params) -> None:
        self.tmp_folder = Path("/tmp/holonote") / os.getcwd().removeprefix("/drive")
        self.tmp_folder.mkdir(parents=True, exist_ok=True)
        self.tmp_file = self.tmp_folder / "annotations.db"

        if os.path.exists(self.filename):
            copyfile(self.filename, self.tmp_file)

        super().__init__(column_schema, connect, **params)

    def _create_database_connection(self) -> sqlite3.Connection:
        class CopyConnection(sqlite3.Connection):
            def commit(_self):
                super().commit()
                copyfile(self.tmp_file, self.filename)

        con = sqlite3.connect(
            self.tmp_file,
            factory=CopyConnection,
            detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES,
        )
        return con


if "_pyodide" in sys.modules:
    js = __import__("js")

    # JupyterLite does not have document
    SQLiteDB = _SQLiteDB if hasattr(js, "document") else _SQLiteDBJupyterLite

else:

    class SQLiteDB(_SQLiteDB):
        pass
