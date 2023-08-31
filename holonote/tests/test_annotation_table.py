from __future__ import annotations

import numpy as np
import pandas as pd

from holonote.annotate import AnnotationTable


def test_table_region_df():
    spec = {"TIME": {"type": np.datetime64, "region": "range"}}
    table = AnnotationTable()
    table.load(primary_key_name='id', fields=['test_description'])
    assert len(table._region_df) == 0, 'Should be initialized empty'
    assert tuple(table._region_df.columns) == AnnotationTable.columns

    start = pd.Timestamp('2022-06-17 18:32:48.623476')
    end = pd.Timestamp('2022-06-19 04:44:09.306402')
    regions = {"TIME": (start, end)}
    table.add_annotation(regions, spec=spec, id=100, test_description='A test')

    d = {'region': 'range', 'dim':'TIME', 'value': (start, end), '_id':100}
    expected = pd.DataFrame([d]).astype({'_id':object})

    pd.testing.assert_frame_equal(table._region_df, expected)
