
import unittest
import numpy as np
import pandas as pd

import holoviews as hv
from holonote.annotate import AnnotationTable
from holonote.annotate import Annotator
from holonote.annotate import SQLiteDB, UUIDHexStringKey


class TestMultipleRegion1DAnnotator(unittest.TestCase):

    def setUp(self):
        assert Annotator.connector_class is SQLiteDB, 'Expecting default SQLite connector'
        Annotator.connector_class.filename = ':memory:'
        Annotator.connector_class.primary_key = UUIDHexStringKey()
        self.annotator = Annotator({'TIME': np.datetime64}, fields=['description'],
                                   region_types=['Point', 'Range'])

    def tearDown(self):
        self.annotator.connector.cursor.close()
        self.annotator.connector.con.close()
        del self.annotator

    def test_point_range_commit_insertion(self):
        descriptions = ['A point insertion', 'A range insertion']
        timestamp = np.datetime64('2022-06-06')
        self.annotator.set_point(timestamp)
        self.annotator.add_annotation(description=descriptions[0])

        start, end = np.datetime64('2022-06-06'), np.datetime64('2022-06-08')
        self.annotator.set_range(start, end)
        self.annotator.add_annotation(description=descriptions[1])

        self.annotator.commit()

        # FIXME! Index order is inverted?
        df = pd.DataFrame({'uuid': pd.Series(self.annotator.df.index[::-1], dtype=object),
                           'point_TIME':[timestamp, pd.NaT],
                           'start_TIME':[pd.NaT, start],
                           'end_TIME': [pd.NaT, end],
                           'description':descriptions}
                           ).set_index('uuid')

        sql_df = self.annotator.connector.load_dataframe()
        pd.testing.assert_frame_equal(sql_df, df)


class TestMultiplePlotAnnotator(unittest.TestCase):

    def setUp(self):
        assert Annotator.connector_class is SQLiteDB, 'Expecting default SQLite connector'
        Annotator.connector_class.filename = ':memory:'
        Annotator.connector_class.primary_key = UUIDHexStringKey()

        self.connector = SQLiteDB()
        xvals, yvals  = np.linspace(-4, 0, 202), np.linspace(4, 0, 202)
        xs, ys = np.meshgrid(xvals, yvals)
        image = hv.Image(np.sin(ys*xs), kdims=['A', 'B'])
        self.image_annotator = Annotator(image, connector=self.connector,
                                         fields=['description'], region_types=['Range'])

        curve = hv.Curve((np.arange('2005-02', '2005-03', dtype='datetime64[D]'), range(28)), kdims=['TIME'])
        self.curve_annotator = Annotator(curve, connector=self.connector,
                                         fields=['description'], region_types=['Range'])

    def test_element_kdim_dtypes(self):
        assert self.image_annotator.kdim_dtypes == {'A': np.float64, 'B': np.float64}
        assert self.curve_annotator.kdim_dtypes == {'TIME': np.datetime64}

    def test_multiplot_add_annotation(self):
        self.image_annotator.set_range(-0.25, 0.25, -0.1, 0.1)
        self.curve_annotator.set_range(np.datetime64('2005-02-13'), np.datetime64('2005-02-16'))
        self.connector.add_annotation(description='Multi-plot annotation')



    def tearDown(self):
        self.connector.cursor.close()
        self.connector.con.close()
        del self.image_annotator


class TestAnnotatorMultipleStringFields(unittest.TestCase):

    def setUp(self):
        assert Annotator.connector_class is SQLiteDB, 'Expecting default SQLite connector'
        Annotator.connector_class.filename = ':memory:'
        Annotator.connector_class.primary_key = UUIDHexStringKey()
        self.annotator = Annotator({'TIME': np.datetime64}, fields=['field1', 'field2'])


    def test_insertion_values(self):
        start, end = np.datetime64('2022-06-06'), np.datetime64('2022-06-08')
        self.annotator.set_range(start, end)
        self.annotator.add_annotation(field1='A test field', field2='Another test field')
        commits = self.annotator.annotation_table.commits()
        kwargs = commits[0]['kwargs']
        assert len(commits)==1, 'Only one insertion commit made'
        assert 'uuid' in kwargs.keys(), 'Expected uuid primary key in kwargs'
        kwargs.pop('uuid')
        assert kwargs == dict(field1='A test field', field2='Another test field', start_TIME=start, end_TIME=end)


    def test_commit_insertion(self):
        start, end  = np.datetime64('2022-06-06'), np.datetime64('2022-06-08')
        field1 = 'A test field'
        field2 = 'Another test field'
        self.annotator.set_range(start, end)
        self.annotator.add_annotation(field1=field1, field2=field2)
        self.annotator.commit()

        df = pd.DataFrame({'uuid': pd.Series(self.annotator.df.index[0], dtype=object),
                           'start_TIME':[start],
                           'end_TIME':[end],
                           'field1':[field1],
                           'field2':[field2]}
                           ).set_index('uuid')

        sql_df = self.annotator.connector.load_dataframe()
        pd.testing.assert_frame_equal(sql_df, df)


    def test_commit_update(self):
        start1, end1  = np.datetime64('2022-06-06'), np.datetime64('2022-06-08')
        start2, end2  = np.datetime64('2023-06-06'), np.datetime64('2023-06-08')
        self.annotator.set_range(start1, end1)
        self.annotator.add_annotation(field1='Field 1.1', field2='Field 1.2')
        self.annotator.set_range(start2, end2)
        self.annotator.add_annotation(field1='Field 2.1', field2='Field 2.2')
        self.annotator.commit()
        self.annotator.update_annotation_fields(self.annotator.df.index[0], field1='NEW Field 1.1')
        self.annotator.commit()
        sql_df = self.annotator.connector.load_dataframe()
        assert set(sql_df['field1']) == set(['NEW Field 1.1', 'Field 2.1'])
