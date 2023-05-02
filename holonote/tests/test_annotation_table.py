import unittest
import numpy as np
import pandas as pd

from holonote.annotate import AnnotationTable



class TestBasicTableLoad(unittest.TestCase):

    def setUp(self):
        self.table = AnnotationTable()

        # Load some metadata and region data

    def test_table_region_df(self):
        self.table.load(primary_key_name='id', fields=['test_description'])
        assert len(self.table._region_df) == 0, 'Should be initialized empty'
        assert tuple(self.table._region_df.columns) == AnnotationTable.columns

        start = pd.Timestamp('2022-06-17 18:32:48.623476')
        end = pd.Timestamp('2022-06-19 04:44:09.306402')
        regions = [ {'region_type': 'Range',
                     'value': (start, end),
                     'dim1': 'TIME',
                     'dim2': None} ]
        self.table.add_annotation(regions, id=100, test_description='A test')


        expected = pd.DataFrame([{'region_type':'Range', 'dim1':'TIME', 'dim2': None,
                                  'value':(start, end), '_id':100}]).astype({'_id':object})

        pd.testing.assert_frame_equal(self.table._region_df, expected)


# Test other metadata field
