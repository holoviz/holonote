from __future__ import annotations

import pandas as pd

from holonote.annotate import AnnotationTable


def test_table_region_df():
    table = AnnotationTable()
    table.load(primary_key_name='id', fields=['test_description'])
    assert len(table._region_df) == 0, 'Should be initialized empty'
    assert tuple(table._region_df.columns) == AnnotationTable.columns

    start = pd.Timestamp('2022-06-17 18:32:48.623476')
    end = pd.Timestamp('2022-06-19 04:44:09.306402')
    regions = [ {'region_type': 'Range',
                    'value': (start, end),
                    'dim1': 'TIME',
                    'dim2': None} ]
    table.add_annotation(regions, id=100, test_description='A test')


    expected = pd.DataFrame([{'region_type':'Range', 'dim1':'TIME', 'dim2': None,
                                'value':(start, end), '_id':100}]).astype({'_id':object})

    pd.testing.assert_frame_equal(table._region_df, expected)
