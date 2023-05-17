import sys
import unittest

import numpy as np
import pandas as pd
from holonote.annotate import Connector, SQLiteDB, AutoIncrementKey, UUIDHexStringKey, UUIDBinaryKey


filename = ':memory:'


class TestConnector(unittest.TestCase):
    "Tests for classmethods on the base class"

    def test_fields_from_metadata_literals(self):
        fields = Connector.schema_from_field_values({'A':3, 'B':'string', 'C':False})
        self.assertEqual(fields,{'A': 'INTEGER', 'B': 'TEXT', 'C': 'BOOLEAN'})

    def test_schema_from_value_datetime(self):
        datetime_type = Connector.field_value_to_type(np.datetime64('NaT'))
        self.assertEqual(Connector.type_mapping[datetime_type],'TIMESTAMP')

    def test_expand_range_region_column_schema_datetime(self):
        result = Connector.expand_region_column_schema(['Range'], {'xdim':np.datetime64})
        self.assertEqual(result, {'start_xdim':'TIMESTAMP', 'end_xdim':'TIMESTAMP'})

    def test_expand_range_region_column_schema_datetimes(self):
        result = Connector.expand_region_column_schema(['Range'],
                                                       {'xdim':np.datetime64,
                                                        'ydim':int})
        self.assertEqual(result,{'start_xdim':'TIMESTAMP',
                                 'end_xdim':'TIMESTAMP',
                                 'start_ydim':'INTEGER',
                                 'end_ydim':'INTEGER'})

    def test_generate_schema(self):
        region_types = [['Range']]
        kdim_dtypes = [{'xdim':np.datetime64, 'ydim':int}]
        result = Connector.generate_schema(AutoIncrementKey(), region_types, kdim_dtypes,
                                           {'description':str})
        self.assertEqual(result, {'id': 'INTEGER PRIMARY KEY AUTOINCREMENT',
                                  'start_xdim': 'TIMESTAMP',
                                  'start_ydim': 'INTEGER',
                                  'end_xdim': 'TIMESTAMP',
                                  'end_ydim': 'INTEGER',
                                  'description': 'TEXT'})


class TestSQLiteUUIDHexKey(unittest.TestCase):
    """
    Example using policy = 'insert'
    """

    def setUp(self):
        self.db = SQLiteDB(filename=filename, primary_key=UUIDHexStringKey())

        fields = {'uuid': self.db.primary_key.schema,
                  'description': 'TEXT',
                  'start':'TIMESTAMP',
                  'end': 'TIMESTAMP'}
        self.db.initialize(fields)


    def tearDown(self):
        self.db.con.close()

    def test_setup(self):
        self.assertTrue(self.db.con is not None)

    def test_add_row(self):
        id1 = self.db.primary_key(self.db)
        start = pd.Timestamp('2022-06-01')
        end = pd.Timestamp('2022-06-03')
        description = 'A description'
        insertion = {'uuid': id1, 'description':description, 'start':start, 'end':end}
        df = pd.DataFrame({'uuid':pd.Series([id1], dtype=object),
                           'description':[description], 'start':[start], 'end':[end]}).set_index('uuid')
        self.db.add_row(**insertion)
        pd.testing.assert_frame_equal(self.db.load_dataframe(), df)


    def test_add_three_rows_delete_one(self):
        id1 = self.db.primary_key(self.db)
        insertion1 = {'uuid': id1,
                     'description':f'A description',
                     'start':pd.Timestamp('2022-06-01'),
                     'end':pd.Timestamp('2022-06-03')}

        id2 = self.db.primary_key(self.db)
        insertion2 = {'uuid': id2,
                     'description':f'A 2nd description',
                     'start':pd.Timestamp('2024-06-01'),
                     'end':pd.Timestamp('2024-06-03')}

        id3 = self.db.primary_key(self.db)
        insertion3 = {'uuid': id3,
                     'description':f'A 3rd description',
                     'start':pd.Timestamp('2026-06-01'),
                     'end':pd.Timestamp('2026-06-03')}

        df_data = {'uuid': pd.Series([insertion1['uuid'], insertion3['uuid']], dtype=object),
                   'description':[insertion1['description'], insertion3['description']],
                   'start':[insertion1['start'], insertion3['start']],
                   'end':[insertion1['end'], insertion3['end']]}
        df = pd.DataFrame(df_data).set_index('uuid')
        self.db.add_row(**insertion1)
        self.db.add_row(**insertion2)
        self.db.add_row(**insertion3)
        self.db.delete_row(id2)
        pd.testing.assert_frame_equal(self.db.load_dataframe(), df)


class TestSQLiteDBAutoIncrementKey(unittest.TestCase):

    def setUp(self):
        self.db = SQLiteDB(filename=filename, primary_key=AutoIncrementKey())

        fields = {'id': self.db.primary_key.schema,
                  'description': 'TEXT',
                  'start':'TIMESTAMP',
                  'end': 'TIMESTAMP'}
        self.db.initialize(fields)

    def tearDown(self):
        self.db.con.close()

    def test_setup(self):
        self.assertTrue(self.db.con is not None)

    def test_columns(self):
        self.assertEqual(self.db.columns,('id', 'description', 'start', 'end'))

    def test_add_row(self):
        id1 = self.db.primary_key(self.db)
        insertion = {'id': id1,
                     'description':f'A description',
                     'start':pd.Timestamp('2022-06-01'),
                     'end':pd.Timestamp('2022-06-03')}

        self.db.add_row(**insertion)
        df = pd.DataFrame([insertion]).set_index('id')
        pd.testing.assert_frame_equal(self.db.load_dataframe(), df)

    def test_add_row_mismatch(self):
        insertion = {'id': 200,  # Will not match autoincrement rowid
                     'description':f'A description',
                     'start':pd.Timestamp('2022-06-01'),
                     'end':pd.Timestamp('2022-06-03')}

        insertion_mismatched_id = insertion.copy()
        df = pd.DataFrame([insertion_mismatched_id]).set_index('id')
        self.db.add_row(**insertion)
        self.assertFalse(self.db.load_dataframe().equals(df))


class TestSQLiteUUIDBinaryKey(unittest.TestCase):
    """
    Example using policy = 'insert'
    """

    def setUp(self):
        self.db = SQLiteDB(filename=filename, primary_key=UUIDBinaryKey())

        fields = {'uuid': self.db.primary_key.schema,
                  'description': 'TEXT',
                  'start':'TIMESTAMP',
                  'end': 'TIMESTAMP'}
        self.db.initialize(fields)


    def tearDown(self):
        self.db.con.close()

    def test_setup(self):
        self.assertTrue(self.db.con is not None)

    def test_add_row(self):
        id1 = self.db.primary_key(self.db)
        insertion = {'uuid': id1,
                     'description':f'A description',
                     'start':pd.Timestamp('2022-06-01'),
                     'end':pd.Timestamp('2022-06-03')}
        df = pd.DataFrame([insertion]).set_index('uuid')
        self.db.add_row(**insertion)
        pd.testing.assert_frame_equal(self.db.load_dataframe(), df)
