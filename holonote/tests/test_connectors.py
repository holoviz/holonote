from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from holonote.annotate import (
    AutoIncrementKey,
    Connector,
    UUIDBinaryKey,
    UUIDHexStringKey,
)


@pytest.fixture(params=[UUIDHexStringKey, AutoIncrementKey, UUIDBinaryKey])
def database(conn_sqlite_uuid, request):
    # Change the primary key type
    conn_sqlite_uuid.primary_key = request.param(field_name='uuid')
    fields = {
        'uuid': conn_sqlite_uuid.primary_key.schema,
        'description': 'TEXT',
        'start':'TIMESTAMP',
        'end': 'TIMESTAMP'
    }
    conn_sqlite_uuid.initialize(fields)
    return conn_sqlite_uuid


class TestConnector:
    "Tests for classmethods on the base class"

    def test_fields_from_metadata_literals(self):
        fields = Connector.schema_from_field_values({'A':3, 'B':'string', 'C':False})
        assert fields == {'A': 'INTEGER', 'B': 'TEXT', 'C': 'BOOLEAN'}

    def test_schema_from_value_datetime(self):
        datetime_type = Connector.field_value_to_type(np.datetime64('NaT'))
        assert Connector.type_mapping[datetime_type] == 'TIMESTAMP'

    def test_expand_range_region_column_schema_datetime(self):
        result = Connector.expand_region_column_schema(['Range'], {'xdim':np.datetime64})
        assert result == {'start_xdim': 'TIMESTAMP', 'end_xdim': 'TIMESTAMP'}

    def test_expand_range_region_column_schema_datetimes(self):
        result = Connector.expand_region_column_schema(['Range'],
                                                       {'xdim':np.datetime64,
                                                        'ydim':int})
        expected = {
            'start_xdim': 'TIMESTAMP', 'end_xdim': 'TIMESTAMP',
            'start_ydim': 'INTEGER', 'end_ydim': 'INTEGER'
        }
        assert result == expected

    def test_generate_schema(self):
        region_types = [['Range']]
        kdim_dtypes = [{'xdim':np.datetime64, 'ydim':int}]
        result = Connector.generate_schema(AutoIncrementKey(), region_types, kdim_dtypes,
                                           {'description':str})
        expected = {
            'id': 'INTEGER PRIMARY KEY AUTOINCREMENT', 'start_xdim': 'TIMESTAMP',
            'start_ydim': 'INTEGER', 'end_xdim': 'TIMESTAMP',
            'end_ydim': 'INTEGER', 'description': 'TEXT'
        }
        assert result == expected


class TestSQLiteDB:

    def test_setup(self, database):
        assert database.con is not None

    def test_initialized(self, database):
        assert not database.uninitialized

    def test_add_row(self, database, request):
        id1 = database.primary_key(database)
        start = pd.Timestamp('2022-06-01')
        end = pd.Timestamp('2022-06-03')
        description = 'A description'
        insertion = {"uuid": id1, 'description':description, 'start':start, 'end':end}
        df = pd.DataFrame({"uuid":pd.Series([id1], dtype=object),
                           'description':[description], 'start':[start], 'end':[end]}).set_index("uuid")
        database.add_row(**insertion)
        pd.testing.assert_frame_equal(database.load_dataframe(), df)

    def test_add_three_rows_delete_one(self, database):
        id1 = database.primary_key(database)
        insertion1 = {'uuid': id1,
                     'description':'A description',
                     'start':pd.Timestamp('2022-06-01'),
                     'end':pd.Timestamp('2022-06-03')}

        id2 = database.primary_key(database, [id1])
        insertion2 = {'uuid': id2,
                     'description':'A 2nd description',
                     'start':pd.Timestamp('2024-06-01'),
                     'end':pd.Timestamp('2024-06-03')}

        id3 = database.primary_key(database, [id2])
        insertion3 = {'uuid': id3,
                     'description':'A 3rd description',
                     'start':pd.Timestamp('2026-06-01'),
                     'end':pd.Timestamp('2026-06-03')}

        df_data = {'uuid': pd.Series([insertion1['uuid'], insertion3['uuid']], dtype=object),
                   'description':[insertion1['description'], insertion3['description']],
                   'start':[insertion1['start'], insertion3['start']],
                   'end':[insertion1['end'], insertion3['end']]}
        df = pd.DataFrame(df_data).set_index('uuid')
        database.add_row(**insertion1)
        database.add_row(**insertion2)
        database.add_row(**insertion3)
        database.delete_row(id2)
        pd.testing.assert_frame_equal(database.load_dataframe(), df)
