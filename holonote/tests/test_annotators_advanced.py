import holoviews as hv
import numpy as np
import pandas as pd

from holonote.annotate import Annotator


def test_multipoint_range_commit_insertion(multiple_region_annotator):
    descriptions = ['A point insertion', 'A range insertion']
    timestamp = np.datetime64('2022-06-06')
    multiple_region_annotator.set_point(timestamp)
    multiple_region_annotator.add_annotation(description=descriptions[0])

    start, end = np.datetime64('2022-06-06'), np.datetime64('2022-06-08')
    multiple_region_annotator.set_range(start, end)
    multiple_region_annotator.add_annotation(description=descriptions[1])

    multiple_region_annotator.commit()

    # FIXME! Index order is inverted?
    df = pd.DataFrame({'uuid': pd.Series(multiple_region_annotator.df.index[::-1], dtype=object),
                        'point_TIME':[timestamp, pd.NaT],
                        'start_TIME':[pd.NaT, start],
                        'end_TIME': [pd.NaT, end],
                        'description':descriptions}
                        ).set_index('uuid')

    sql_df = multiple_region_annotator.connector.load_dataframe()
    pd.testing.assert_frame_equal(sql_df, df)


def test_infer_kdim_dtype_image():
    xvals, yvals  = np.linspace(-4, 0, 202), np.linspace(4, 0, 202)
    xs, ys = np.meshgrid(xvals, yvals)
    image = hv.Image(np.sin(ys*xs), kdims=['A', 'B'])
    assert Annotator._infer_kdim_dtypes(image) == {'A': np.float64, 'B': np.float64}


def test_infer_kdim_dtype_curve():
    curve = hv.Curve((np.arange('2005-02', '2005-03', dtype='datetime64[D]'), range(28)), kdims=['TIME'])
    assert Annotator._infer_kdim_dtypes(curve) == {'TIME': np.datetime64}


def test_multiplot_add_annotation(multiple_annotators):
    multiple_annotators["annotation1d"].set_range(np.datetime64('2005-02-13'), np.datetime64('2005-02-16'))
    multiple_annotators["annotation2d"].set_range(-0.25, 0.25, -0.1, 0.1)
    multiple_annotators["conn"].add_annotation(description='Multi-plot annotation')


class TestAnnotatorMultipleStringFields:

    def test_insertion_values(self, multiple_fields_annotator):
        start, end = np.datetime64('2022-06-06'), np.datetime64('2022-06-08')
        multiple_fields_annotator.set_range(start, end)
        multiple_fields_annotator.add_annotation(field1='A test field', field2='Another test field')
        commits = multiple_fields_annotator.annotation_table.commits()
        kwargs = commits[0]['kwargs']
        assert len(commits)==1, 'Only one insertion commit made'
        assert 'uuid' in kwargs.keys(), 'Expected uuid primary key in kwargs'
        kwargs.pop('uuid')
        assert kwargs == dict(field1='A test field', field2='Another test field', start_TIME=start, end_TIME=end)


    def test_commit_insertion(self, multiple_fields_annotator):
        start, end  = np.datetime64('2022-06-06'), np.datetime64('2022-06-08')
        field1 = 'A test field'
        field2 = 'Another test field'
        multiple_fields_annotator.set_range(start, end)
        multiple_fields_annotator.add_annotation(field1=field1, field2=field2)
        multiple_fields_annotator.commit()

        df = pd.DataFrame({'uuid': pd.Series(multiple_fields_annotator.df.index[0], dtype=object),
                           'start_TIME':[start],
                           'end_TIME':[end],
                           'field1':[field1],
                           'field2':[field2]}
                           ).set_index('uuid')

        sql_df = multiple_fields_annotator.connector.load_dataframe()
        pd.testing.assert_frame_equal(sql_df, df)


    def test_commit_update(self, multiple_fields_annotator):
        start1, end1  = np.datetime64('2022-06-06'), np.datetime64('2022-06-08')
        start2, end2  = np.datetime64('2023-06-06'), np.datetime64('2023-06-08')
        multiple_fields_annotator.set_range(start1, end1)
        multiple_fields_annotator.add_annotation(field1='Field 1.1', field2='Field 1.2')
        multiple_fields_annotator.set_range(start2, end2)
        multiple_fields_annotator.add_annotation(field1='Field 2.1', field2='Field 2.2')
        multiple_fields_annotator.commit()
        multiple_fields_annotator.update_annotation_fields(multiple_fields_annotator.df.index[0], field1='NEW Field 1.1')
        multiple_fields_annotator.commit()
        sql_df = multiple_fields_annotator.connector.load_dataframe()
        assert set(sql_df['field1']) == {'NEW Field 1.1', 'Field 2.1'}
