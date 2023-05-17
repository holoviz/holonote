import os
import uuid
import sqlite3
import datetime as dt

import param
import pandas as pd
import numpy as np

try:
    import sqlalchemy
except:
    sqlalchemy = None

from .table import AnnotationTable


class PrimaryKey(param.Parameterized):
    """
    Generator of the primary key used to keep track of annotations in
    HoloViews.

    The generated key is used to reference annotations until they are
    comitted, at which point they may 1) be inserted in the database as
    the primary key value (policy='insert') 2) are checked against the
    primary key value chosen by the database which is expected to match
    in most cases.

    In real situations where the key is chosen by the databse, the key
    generated will *not* always match the actual key assigned. The
    policy parameter decides the resulting behavior in these cases.
    """

    field_name = param.String(default='id', allow_None=False)

    policy = param.ObjectSelector(default='ignore-mismatch',
                                  objects=['insert', 'ignore-mismatch', 'warn-mismatch', 'error-mismatch'])

    schema = param.String(default='INTEGER PRIMARY KEY',
                          constant=True, allow_None=False)

    connector_class = param.String(default='SQLiteDB')

    def __call__(self, connector, key_list=[]):
        """
        The key list is the current list of index values that are
        outstanding (i.e. have not been comitted).
        """
        raise NotImplementedError

    def validate(self, database_key, local_key):
        if self.policy in ['insert', 'ignore-mismatch']:
            return

        if database_key != local_key:
            print('MISMATCH')

    def cast(self, value):
        "Cast a user supplied value to a known supported type"
        raise NotImplementedError


class AutoIncrementKey(PrimaryKey):
    """
    AUTOINCREMENT needed to prevent reuse of ids from deleted rows:
    https://www.sqlite.org/autoinc.html
    """

    field_name = param.String(default='id', allow_None=False)

    policy = param.ObjectSelector(default='ignore-mismatch',
                                  objects=['insert', 'ignore-mismatch', 'warn-mismatch', 'error-mismatch'])

    schema = param.String(default='INTEGER PRIMARY KEY AUTOINCREMENT',
                          constant=True, allow_None=False)

    def __call__(self, connector, key_list=[]):
        key_list_max = max(key_list) if key_list else 0
        connector_max = connector.max_rowid()
        connector_max = 0 if connector_max is None else connector_max
        max_rowid = max(key_list_max, connector_max)
        return max_rowid + 1

    def cast(self, value):
        return int(value)  # e.g. np.int64 from a pandas index won't work!


class UUIDHexStringKey(PrimaryKey): # Probably the better default
    """
    Example of 'insert' policy where the generated primary key value can
    be inserted in the database as uuids are independent and not
    expected to clash.
    """

    field_name = param.String(default='uuid', allow_None=False)

    policy = param.ObjectSelector(default='insert',
                                  objects=['insert', 'ignore-mismatch', 'warn-mismatch', 'error-mismatch'])

    schema = param.String('TEXT PRIMARY KEY', constant=True, allow_None=False)

    length = param.Integer(default=32, bounds=(4,32))



    def __call__(self, connector, key_list=[]):
        return uuid.uuid4().hex[:self.length]

    def cast(self, value):
        return str(value)


class UUIDBinaryKey(PrimaryKey):
    """
    Example of 'insert' policy where the generated primary key value can
    be inserted in the database as uuids are independent and not
    expected to clash.
    """

    field_name = param.String(default='uuid', allow_None=False)

    policy = param.ObjectSelector(default='insert',
                                  objects=['insert', 'ignore-mismatch', 'warn-mismatch', 'error-mismatch'])

    schema = param.String('BINARY PRIMARY KEY', constant=True, allow_None=False)

    def __call__(self, connector, key_list=[]):
        return uuid.uuid4().bytes


class WidgetKey(PrimaryKey):
    """
    Placeholder for a concept where the user can insert a primary key
    value via a widget.
    """
    pass



class Connector(param.Parameterized):
    """
    Base class that support the auto-generated default schema
    """

    primary_key = param.Parameter(default=UUIDHexStringKey(), allow_None=False)

    commit_hook = param.Parameter(default=None, doc='Callback, applies default schema if None')

    fields = param.List(default=['description'], doc='''
      List of column names for domain-specific fields''')

    transforms = param.Dict(default={'insert':lambda x: x,
                                     'update':lambda x: x ,
                                     'delete':lambda x: x ,
                                     'save': lambda x: x,
                                     'load': lambda x: x})

    operation_mapping = {} # Mapping from operation type to corresponding connector method

    type_mapping = {
        bool: 'BOOLEAN',
        str: 'TEXT',
        float: 'REAL',
        int: 'INTEGER',
        np.datetime64: 'TIMESTAMP',
        dt.date: 'TIMESTAMP',
        dt.datetime: 'TIMESTAMP',
        param.Integer: 'INTEGER',
        param.Number: 'REAL',
        param.String: 'TEXT',
        param.Boolean: 'BOOLEAN',
        np.dtype('datetime64[ns]'): 'TIMESTAMP',
        np.dtype('<M8'):'TIMESTAMP',
        np.float64: 'REAL',
        }

    def __init__(self, **params):
        self.annotation_table = None
        super().__init__(**params)

    @classmethod
    def field_value_to_type(cls, value):
        if isinstance(value, list):
            assert all([isinstance(el, str) for el in value]), 'Only string enums supported'
            return str
        elif hasattr(value, 'dtype'):
            return  value.dtype
        elif type(value) in cls.type_mapping:
            return type(value)
        elif isinstance(value, param.Parameter) and value.default is not None:
            return type(value.default)
        else:
            raise Exception(f'Connector cannot handle type {str(type(value))}')

    @classmethod
    def schema_from_field_values(cls, fields):
        "Given a dictionary of fields fields and values return the field schemas"
        return {name:cls.type_mapping[cls.field_value_to_type(value)] for name, value in fields.items()}

    @classmethod
    def schema_from_field_types(cls, fields):
        "Given a dictionary of fields fields and values return the field schemas"
        return {name:cls.type_mapping[value] for name, value in fields.items()}

    @classmethod
    def expand_region_column_schema(cls, region_types, kdim_dtypes):
        region_type_column_prefixes = {'Range':('start', 'end'), 'Point':('point',)}
        fields = {}
        for region_type in region_types:
            for column in region_type_column_prefixes[region_type]:
                for kdim, kdim_dtype in kdim_dtypes.items():
                    fields[f'{column}_{kdim}'] = cls.type_mapping[kdim_dtype]
        return fields

    @classmethod
    def generate_schema(cls, primary_key, all_region_types, all_kdim_dtypes, field_types):
        schemas = {primary_key.field_name:primary_key.schema}
        region_schemas = {}
        for region_types, kdim_dtypes in zip(all_region_types, all_kdim_dtypes):
            region_schemas = dict(region_schemas, **cls.expand_region_column_schema(region_types, kdim_dtypes))
        schemas = dict(schemas, **region_schemas)
        return dict(schemas, **cls.schema_from_field_types(field_types))


    def _initialize_annotation_table(self):
        if self.annotation_table is None:
            self.annotation_table = AnnotationTable()

    def commit(self):
        "Applies the commit hook to the connector"

        if self.uninitialized:
            self.initialize_table()

        if self.commit_hook is not None:
            self.commit_hook()
        else:
            self.commit_default_schema()

    def commit_default_schema(self):
        for commit in self.annotation_table.commits():
            operation = commit['operation']
            kwargs = self.transforms[operation](commit['kwargs'])
            getattr(self,self.operation_mapping[operation])(**kwargs)

        for annotator in self.annotation_table._annotators.values():
            annotator.annotation_table.clear_edits()


    def _incompatible_schema_check(self, expected_keys, columns, fields, region_type):
        msg_prefix = ("Unable to read annotations that were stored with a "
                      "schema inconsistent with the current settings: ")
        msg_suffix = f"Columns found: {columns}"

        missing_field_columns = set(fields) - set(columns)
        if missing_field_columns:
            raise Exception(msg_prefix + f'Missing field columns {missing_field_columns}. ' + msg_suffix)

        non_field_columns = set(columns) - set(fields)
        missing_region_columns = set(expected_keys) - non_field_columns
        if missing_region_columns:
            raise Exception(msg_prefix
                            + f'Missing {repr(region_type)} region columns {missing_region_columns}. '
                            + msg_suffix)

    def load_annotation_table(self, annotation_table, fields):
        df = self.transforms['load'](self.load_dataframe())
        fields_df = df[fields].copy()
        annotation_table.define_fields(fields_df, {ind:ind for ind in fields_df.index})
        all_region_types = [an.region_types for an in annotation_table._annotators.values()]
        all_kdim_dtypes = [an.kdim_dtypes for an in annotation_table._annotators.values()]
        for region_types, kdim_dtypes in zip(all_region_types, all_kdim_dtypes):
            assert all(el in ['Range', 'Point'] for el in region_types)
            for region_type in region_types:
                if len(kdim_dtypes)==1:
                    kdim = list(kdim_dtypes.keys())[0]
                    if region_type == 'Range':
                        expected_keys = [f'start_{kdim}', f'end_{kdim}']
                        self._incompatible_schema_check(expected_keys, list(df.columns), fields, region_type)
                        annotation_table.define_ranges(kdim, df[f'start_{kdim}'], df[f'end_{kdim}'])
                    elif region_type == 'Point':
                        self._incompatible_schema_check([f'point_{kdim}'], list(df.columns), fields, region_type)
                        annotation_table.define_points(kdim, df[f'point_{kdim}'])
                elif len(kdim_dtypes)==2:
                    kdim1, kdim2 = list(kdim_dtypes.keys())
                    if region_type == 'Range':
                        self._incompatible_schema_check([f'start_{kdim1}', f'end_{kdim1}',
                                                         f'start_{kdim2}', f'end_{kdim2}'],
                                                        list(df.columns), fields, region_type)
                        annotation_table.define_ranges([kdim1, kdim2],
                                                       df[f'start_{kdim1}'], df[f'end_{kdim1}'],
                                                       df[f'start_{kdim2}'], df[f'end_{kdim2}'])
                    elif region_type == 'Point':
                        self._incompatible_schema_check([f'point_{kdim1}', f'point_{kdim2}'],
                                                        list(df.columns), fields, region_type)
                        annotation_table.define_points([kdim1, kdim2], df[f'point_{kdim1}'], df[f'point_{kdim2}'])
        annotation_table.clear_edits()

    def add_annotation(self, **fields):
        "Primary key specification is optional. Used to works across Annotation instances."
        if self.primary_key.field_name not in fields:
            index_val = self.primary_key(self, list(self.annotation_table._field_df.index))
            fields[self.primary_key.field_name] = index_val
        self.annotation_table.add_annotation('annotator-regions', **fields)


        for annotator in self.annotation_table._annotators.values():
            annotator.refresh(clear=True)



class SQLiteDB(Connector):
    """
    Simple example of a Connector without dependencies, using sqlite3.

    column_schema is a dictionary from column name to SQL CREATE TABLE type declaration.
    """

    filename = param.String(default='annotations.db')

    table_name = param.String(default=None, allow_None=True, doc='''
       The SQL table name to connect to in the database. If None, an
       automatically generated table name will be used. Note that
       auto-generated names may only be used by connectors generated
       automatically by a single annotator instance.''')

    column_schema = param.Dict(default={})

    operation_mapping = {'insert':'add_row',
                         'delete':'delete_row',
                         'save':'add_rows',
                         'update':'update_row'}

    _auto_table_prefix = 'annotations_'

    def __init__(self, column_schema={}, connect=True, **params):
        """
        First key in column_schema is assumed to the primary key field if not explicitly specified.
        """
        params['column_schema'] = column_schema
        self.con, self.cursor = None, None
        super().__init__(**params)


        if connect:
            self._initialize(column_schema, create_table=False)


    def _initialize(self, column_schema, create_table=True):
        if self.con is None:
            self.con = sqlite3.connect(self.filename, detect_types=sqlite3.PARSE_DECLTYPES |
                                       sqlite3.PARSE_COLNAMES)
            self.cursor = self.con.cursor()
        if create_table:
            self.create_table(column_schema=column_schema)

        super()._initialize_annotation_table()

    @property
    def uninitialized(self):
        if self.con is not None:
            return self.table_name not in self.get_tables()
        return True

    @property
    def columns(self):
        "Return names of columns"
        result = self.cursor.execute("PRAGMA table_info('%s')" % self.table_name).fetchall()
        return list(zip(*result))[1]

    def max_rowid(self):
        return self.cursor.execute(f"SELECT max(ROWID) from {self.table_name}").fetchone()[0]

    def initialize(self, column_schema):
        self.column_schema = column_schema
        self._initialize(column_schema)

    def load_dataframe(self):
        uninitialized = (self.cursor is None)
        if uninitialized:
            self._initialize({}, create_table=False)


        raw_df = pd.read_sql_query(f'SELECT * FROM {self.table_name}', self.con)
        # dtype={self.primary_key.field_name:self.primary_key.dtype})
        df = raw_df.set_index(self.primary_key.field_name)
        if uninitialized:
            self.con, self.cursor = None, None
        return df

    def get_tables(self):
        res = self.cursor.execute("SELECT name FROM sqlite_master")
        return [el[0] for el in res.fetchall() if el[0].startswith(self._auto_table_prefix)]

    def create_table(self, column_schema=None):
        column_schema = column_schema if column_schema else self.column_schema
        column_spec = ',\n'.join(['{name} {spec}'.format(name=name, spec=spec)
                                  for name, spec in column_schema.items()])
        create_table_sql = f'CREATE TABLE IF NOT EXISTS {self.table_name} (' + column_spec +  ');'
        self.cursor.execute(create_table_sql)
        self.con.commit()

    def initialize_table(self):
        field_dtypes = {col:str for col in self.fields} # FIXME - generalize
        all_region_types = [an.region_types for an in self.annotation_table._annotators.values()]
        all_kdim_dtypes = [an.kdim_dtypes for an in self.annotation_table._annotators.values()]
        schema = self.generate_schema(self.primary_key, all_region_types, all_kdim_dtypes, field_dtypes)
        self.create_table(schema)

    def delete_table(self):
        self.cursor.execute(f"DROP TABLE IF EXISTS {self.table_name}")
        self.con.commit()

    def add_rows(self, field_list): # Used execute_many
        for field in field_list:
            self.add_row(**field)

    def add_row(self, **fields):
        # Note, missing fields will be set as NULL
        columns = self.columns
        field_values = [fields.get(col,None) for col in self.columns]
        field_values = [pd.to_datetime(el) if isinstance(el, np.datetime64) else el for el in field_values]
        field_values = [el.to_pydatetime() if isinstance(el, pd.Timestamp) else el for el in field_values]

        if self.primary_key.policy != 'insert':
           field_values = field_values[1:]
           columns = columns[1:]

        placeholders = ', '.join(['?'] * len(field_values))
        self.cursor.execute(f"INSERT INTO {self.table_name} {str(columns)} VALUES({placeholders});", field_values)
        self.primary_key.validate(self.cursor.lastrowid, fields[self.primary_key.field_name])
        self.con.commit()

    def delete_all_rows(self):
        "Obviously a destructive operation!"
        self.cursor.execute(f"DELETE FROM {self.table_name};")
        self.con.commit()

    def delete_row(self, id_val):
        self.cursor.execute(f"DELETE FROM {self.table_name} WHERE {self.primary_key.field_name} = ?",
                            (self.primary_key.cast(id_val),))
        self.con.commit()

    def update_row(self, **updates): # updates as a dictionary OR remove posarg?
        assert self.primary_key.field_name in updates
        id_val = updates.pop(self.primary_key.field_name)
        set_updates = ', '.join('\"' + k + '\"' + " = ?" for k in updates.keys())
        query = f"UPDATE {self.table_name} SET " + set_updates + f" WHERE \"{self.primary_key.field_name}\" = ?;"
        self.cursor.execute(query, list(updates.values()) + [id_val])
        self.con.commit()
