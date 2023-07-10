# TODO:

# * (after refactor) annotators -> annotator, connectors -> connector [ ]

# TESTS

# Schema error (needs file or connect in memory??)
# .snapshot() and .revert_to_snapshot()

import uuid

import numpy as np
import pandas as pd
import pytest


class TestBasicRange1DAnnotator:
    def test_point_insertion_exception(self, annotator_range1d):
        timestamp = np.datetime64('2022-06-06')
        expected_msg = r"Point region types not enabled as region_types=\['Range'\]"
        with pytest.raises(ValueError, match=expected_msg):
            annotator_range1d.set_point(timestamp)

    def test_insertion_edit_table_columns(self, annotator_range1d):
        annotator_range1d.set_range(np.datetime64('2022-06-06'), np.datetime64('2022-06-08'))
        annotator_range1d.add_annotation(description='A test annotation!')
        commits = annotator_range1d.commit(return_commits=True)
        assert len(commits)==1, 'Only one insertion commit made '
        annotator_range1d.commit(return_commits=True)
        assert commits[0]['operation'] == 'insert'
        assert set(commits[0]['kwargs'].keys()) == set(annotator_range1d.connector.columns)

    def test_range_insertion_values(self, annotator_range1d) -> None:
        start, end = np.datetime64('2022-06-06'), np.datetime64('2022-06-08')
        annotator_range1d.set_range(start, end)
        annotator_range1d.add_annotation(description='A test annotation!')
        commits = annotator_range1d.commit(return_commits=True)
        assert len(commits)==1, 'Only one insertion commit made'
        kwargs = commits[0]['kwargs']
        assert 'uuid' in kwargs.keys(), 'Expected uuid primary key in kwargs'
        kwargs.pop('uuid')
        assert kwargs, dict(description='A test annotation!', start_TIME=start, end_TIME=end)

    def test_range_commit_insertion(self, annotator_range1d):
        start, end  = np.datetime64('2022-06-06'), np.datetime64('2022-06-08')
        description = 'A test annotation!'
        annotator_range1d.set_range(start, end)
        annotator_range1d.add_annotation(description=description)
        annotator_range1d.commit(return_commits=True)

        df = pd.DataFrame({'uuid': pd.Series(annotator_range1d.df.index[0], dtype=object),
                           'start_TIME':[start],
                           'end_TIME':[end],
                           'description':[description]}
                           ).set_index('uuid')

        sql_df = annotator_range1d.connector.load_dataframe()
        pd.testing.assert_frame_equal(sql_df, df)


    def test_range_addition_deletion_by_uuid(self, annotator_range1d):
        start1, end1  = np.datetime64('2022-06-06'), np.datetime64('2022-06-08')
        start2, end2  = np.datetime64('2023-06-06'), np.datetime64('2023-06-08')
        start3, end3  = np.datetime64('2024-06-06'), np.datetime64('2024-06-08')
        annotator_range1d.set_range(start1, end1)
        annotator_range1d.add_annotation(description='Annotation 1')
        annotator_range1d.set_range(start2, end2)
        annotator_range1d.add_annotation(description='Annotation 2', uuid='08286429')
        annotator_range1d.set_range(start3, end3)
        annotator_range1d.add_annotation(description='Annotation 3')
        annotator_range1d.commit(return_commits=True)
        sql_df = annotator_range1d.connector.load_dataframe()
        assert set(sql_df['description']) ==set(['Annotation 1', 'Annotation 2', 'Annotation 3'])
        deletion_index = sql_df.index[1]
        annotator_range1d.delete_annotation(deletion_index)
        annotator_range1d.commit(return_commits=True)
        sql_df = annotator_range1d.connector.load_dataframe()
        assert set(sql_df['description']) == set(['Annotation 1', 'Annotation 3'])


    def test_range_define_preserved_index_mismatch(self, annotator_range1d):
        starts = [np.datetime64('2022-06-%.2d' % d) for d in  range(6,15, 4)]
        ends = [np.datetime64('2022-06-%.2d' % (d+2)) for d in  range(6,15, 4)]
        descriptions = ['Annotation %d' % d for d in [1,2,3]]
        annotation_id = [uuid.uuid4().hex[:8] for d in [1,2,3]]

        data = pd.DataFrame({'uuid':annotation_id, 'start':starts, 'end':ends, 'description':descriptions}).set_index('uuid')
        annotator_range1d.define_fields(data[['description']], preserve_index=True)
        annotator_range1d.define_ranges(data['start'].iloc[:2], data['end'].iloc[:2])
        msg = f"Following annotations have no associated region: {{{annotation_id[2]!r}}}"
        with pytest.raises(ValueError, match=msg):
            annotator_range1d.commit(return_commits=True)

    def test_range_define_auto_index_mismatch(self, annotator_range1d):
        starts = [np.datetime64('2022-06-%.2d' % d) for d in  range(6,15, 4)]
        ends = [np.datetime64('2022-06-%.2d' % (d+2)) for d in  range(6,15, 4)]
        descriptions = ['Annotation %d' % d for d in [1,2,3]]
        annotation_id = [uuid.uuid4().hex[:8] for d in [1,2,3]]

        data = pd.DataFrame({'uuid':annotation_id, 'start':starts,
                             'end':ends, 'description':descriptions}).set_index('uuid')
        annotator_range1d.define_fields(data[['description']], preserve_index=False)
        annotator_range1d.define_ranges(data['start'].iloc[:2], data['end'].iloc[:2])
        with pytest.raises(ValueError,
                           match="Following annotations have no associated region:"):
            annotator_range1d.commit(return_commits=True)

    def test_range_define_unassigned_indices(self, annotator_range1d):
        starts = [np.datetime64('2022-06-%.2d' % d) for d in  range(6,15, 4)]
        ends = [np.datetime64('2022-06-%.2d' % (d+2)) for d in  range(6,15, 4)]
        descriptions = ['Annotation %d' % d for d in [1,2,3]]
        annotation_id1 = [uuid.uuid4().hex[:8] for d in [1,2,3]]
        mismatched = [uuid.uuid4().hex[:8] for d in [1,2]]
        annotation_id2 = [*mismatched, annotation_id1[2]]

        data1 = pd.DataFrame({'uuid':annotation_id1, 'start':starts,
                              'end':ends, 'description':descriptions}).set_index('uuid')
        data2 = pd.DataFrame({'uuid':annotation_id2, 'start':starts,
                              'end':ends, 'description':descriptions}).set_index('uuid')

        annotator_range1d.define_fields(data1[['description']])
        with pytest.raises(KeyError, match=str(mismatched)):
            annotator_range1d.define_ranges(data2['start'], data2['end'])


class TestBasicRange2DAnnotator:

    def test_point_insertion_exception(self, annotator_range2d):
        x,y = 0.5,0.5
        expected_msg = r"Point region types not enabled as region_types=\['Range'\]"
        with pytest.raises(ValueError, match=expected_msg):
            annotator_range2d.set_point(x,y)

    def test_insertion_edit_table_columns(self, annotator_range2d):
        annotator_range2d.set_range(-0.25, 0.25, -0.1, 0.1)
        annotator_range2d.add_annotation(description='A test annotation!')
        commits = annotator_range2d.commit(return_commits=True)
        assert len(commits)==1, 'Only one insertion commit made '
        assert commits[0]['operation'] == 'insert'
        assert set(commits[0]['kwargs'].keys()) == set(annotator_range2d.connector.columns)

    def test_range_insertion_values(self, annotator_range2d):
        startx, endx, starty, endy = -0.25, 0.25, -0.1, 0.1
        annotator_range2d.set_range(startx, endx, starty, endy)
        annotator_range2d.add_annotation(description='A test annotation!')
        commits = annotator_range2d.commit(return_commits=True)
        assert len(commits)==1, 'Only one insertion commit made'
        kwargs = commits[0]['kwargs']
        assert 'uuid' in kwargs.keys(), 'Expected uuid primary key in kwargs'
        kwargs.pop('uuid')
        assert kwargs ==  dict(description='A test annotation!',
                                      start_x=startx, end_x=endx, start_y=starty, end_y=endy)

    def test_range_commit_insertion(self, annotator_range2d):
        startx, endx, starty, endy = -0.25, 0.25, -0.1, 0.1
        description = 'A test annotation!'
        annotator_range2d.set_range(startx, endx, starty, endy)
        annotator_range2d.add_annotation(description=description)
        annotator_range2d.commit(return_commits=True)

        df = pd.DataFrame({'uuid': pd.Series(annotator_range2d.df.index[0], dtype=object),
                           'start_x':[startx],
                           'start_y':[starty],
                           'end_x':[endx],
                           'end_y':[endy],
                           'description':[description]}
                           ).set_index('uuid')

        sql_df = annotator_range2d.connector.load_dataframe()
        pd.testing.assert_frame_equal(sql_df, df)


    def test_range_addition_deletion_by_uuid(self, annotator_range2d):
        startx1, endx1, starty1, endy1 = -0.251, 0.251, -0.11, 0.11
        startx2, endx2, starty2, endy2 = -0.252, 0.252, -0.12, 0.12
        startx3, endx3, starty3, endy3 = -0.253, 0.253, -0.13, 0.13
        annotator_range2d.set_range(startx1, endx1, starty1, endy1)
        annotator_range2d.add_annotation(description='Annotation 1')
        annotator_range2d.set_range(startx2, endx2, starty2, endy2)
        annotator_range2d.add_annotation(description='Annotation 2', uuid='08286429')
        annotator_range2d.set_range(startx3, endx3, starty3, endy3)
        annotator_range2d.add_annotation(description='Annotation 3')
        annotator_range2d.commit(return_commits=True)
        sql_df = annotator_range2d.connector.load_dataframe()
        assert set(sql_df['description']) == set(['Annotation 1', 'Annotation 2', 'Annotation 3'])
        deletion_index = sql_df.index[1]
        annotator_range2d.delete_annotation(deletion_index)
        annotator_range2d.commit(return_commits=True)
        sql_df = annotator_range2d.connector.load_dataframe()
        assert set(sql_df['description']) == set(['Annotation 1', 'Annotation 3'])


    def test_range_define_preserved_index_mismatch(self, annotator_range2d):
        xstarts, xends = [-0.3, -0.2, -0.1], [0.3, 0.2, 0.1]
        ystarts, yends = [-0.35, -0.25, -0.15], [0.35, 0.25, 0.15]
        descriptions = ['Annotation %d' % d for d in [1,2,3]]
        annotation_id = [uuid.uuid4().hex[:8] for d in [1,2,3]]

        data = pd.DataFrame({'uuid':annotation_id, 'xstart':xstarts, 'xend':xends,
                             'ystart':ystarts, 'yend':yends,
                             'description':descriptions}).set_index('uuid')
        annotator_range2d.define_fields(data[['description']], preserve_index=True)
        annotator_range2d.define_ranges(data['xstart'].iloc[:2], data['xend'].iloc[:2],
                                     data['ystart'].iloc[:2], data['yend'].iloc[:2])

        msg = f"Following annotations have no associated region: {{{annotation_id[2]!r}}}"
        with pytest.raises(ValueError, match=msg):
            annotator_range2d.commit(return_commits=True)

    def test_range_define_auto_index_mismatch(self, annotator_range2d):
        xstarts, xends = [-0.3, -0.2, -0.1], [0.3, 0.2, 0.1]
        ystarts, yends = [-0.35, -0.25, -0.15], [0.35, 0.25, 0.15]
        descriptions = ['Annotation %d' % d for d in [1,2,3]]
        annotation_id = [uuid.uuid4().hex[:8] for d in [1,2,3]]
        data = pd.DataFrame({'uuid':annotation_id, 'xstart':xstarts, 'xend':xends,
                             'ystart':ystarts, 'yend':yends,
                             'description':descriptions}).set_index('uuid')
        annotator_range2d.define_fields(data[['description']], preserve_index=False)
        annotator_range2d.define_ranges(data['xstart'].iloc[:2], data['xend'].iloc[:2],
                                     data['ystart'].iloc[:2], data['yend'].iloc[:2])
        msg = "Following annotations have no associated region:"
        with pytest.raises(ValueError, match=msg):
            annotator_range2d.commit(return_commits=True)

    def test_range_define_unassigned_indices(self, annotator_range2d):
        xstarts, xends = [-0.3, -0.2, -0.1], [0.3, 0.2, 0.1]
        ystarts, yends = [-0.35, -0.25, -0.15], [0.35, 0.25, 0.15]
        descriptions = ['Annotation %d' % d for d in [1,2,3]]
        annotation_id1 = [uuid.uuid4().hex[:8] for d in [1,2,3]]
        mismatched = [uuid.uuid4().hex[:8] for d in [1,2]]
        annotation_id2 = [*mismatched, annotation_id1[2]]

        data1 = pd.DataFrame({'uuid':annotation_id1, 'xstart':xstarts, 'xend':xends,
                             'ystart':ystarts, 'yend':yends,
                             'description':descriptions}).set_index('uuid')
        data2 = pd.DataFrame({'uuid':annotation_id2, 'xstart':xstarts, 'xend':xends,
                             'ystart':ystarts, 'yend':yends,
                             'description':descriptions}).set_index('uuid')

        annotator_range2d.define_fields(data1[['description']])
        with pytest.raises(KeyError, match=str(mismatched)):
            annotator_range2d.define_ranges(data2['xstart'], data2['xend'],
                                         data2['ystart'], data2['yend'])


class TestBasicPoint1DAnnotator:

    def test_insertion_edit_table_columns(self, annotator_point1d):
        annotator_point1d.set_point(np.datetime64('2022-06-06'))
        annotator_point1d.add_annotation(description='A test annotation!')
        commits = annotator_point1d.commit(return_commits=True)
        assert len(commits)==1, 'Only one insertion commit made '
        annotator_point1d.commit(return_commits=True)
        assert commits[0]['operation'] == 'insert'
        assert set(commits[0]['kwargs'].keys()) == set(annotator_point1d.connector.columns)

    def test_range_insertion_exception(self, annotator_point1d):
        start, end = np.datetime64('2022-06-06'), np.datetime64('2022-06-08')
        msg = r"Range region types not enabled as region_types=\['Point'\]"
        with pytest.raises(ValueError, match=msg):
            annotator_point1d.set_range(start, end)

    def test_point_insertion_values(self, annotator_point1d):
        timestamp = np.datetime64('2022-06-06')
        annotator_point1d.set_point(timestamp)
        annotator_point1d.add_annotation(description='A test annotation!')
        commits = annotator_point1d.commit(return_commits=True)
        assert len(commits)==1, 'Only one insertion commit made'
        kwargs = commits[0]['kwargs']
        assert 'uuid' in kwargs.keys(), 'Expected uuid primary key in kwargs'
        kwargs.pop('uuid')
        assert kwargs == dict(description='A test annotation!', point_TIME=timestamp)

    def test_point_commit_insertion(self, annotator_point1d):
        timestamp = np.datetime64('2022-06-06')
        description = 'A test annotation!'
        annotator_point1d.set_point(timestamp)
        annotator_point1d.add_annotation(description=description)
        annotator_point1d.commit(return_commits=True)

        df = pd.DataFrame({'uuid': pd.Series(annotator_point1d.df.index[0], dtype=object),
                           'point_TIME':[timestamp],
                           'description':[description]}
                           ).set_index('uuid')

        sql_df = annotator_point1d.connector.load_dataframe()
        pd.testing.assert_frame_equal(sql_df, df)


    def test_point_addition_deletion_by_uuid(self, annotator_point1d):
        ts1  = np.datetime64('2022-06-06')
        ts2  = np.datetime64('2023-06-06')
        ts3  = np.datetime64('2024-06-06')
        annotator_point1d.set_point(ts1)
        annotator_point1d.add_annotation(description='Annotation 1')
        annotator_point1d.set_point(ts2)
        annotator_point1d.add_annotation(description='Annotation 2', uuid='08286429')
        annotator_point1d.set_point(ts3)
        annotator_point1d.add_annotation(description='Annotation 3')
        annotator_point1d.commit(return_commits=True)
        sql_df = annotator_point1d.connector.load_dataframe()
        assert set(sql_df['description']) == set(['Annotation 1', 'Annotation 2', 'Annotation 3'])
        deletion_index = sql_df.index[1]
        annotator_point1d.delete_annotation(deletion_index)
        annotator_point1d.commit(return_commits=True)
        sql_df = annotator_point1d.connector.load_dataframe()
        assert set(sql_df['description']) == set(['Annotation 1', 'Annotation 3'])

    def test_point_define_preserved_index_mismatch(self, annotator_point1d):
        timestamps = [np.datetime64('2022-06-%.2d' % d) for d in  range(6,15, 4)]
        descriptions = ['Annotation %d' % d for d in [1,2,3]]
        annotation_id = [uuid.uuid4().hex[:8] for d in [1,2,3]]

        data = pd.DataFrame({'uuid':annotation_id, 'timestamps':timestamps,
                             'description':descriptions}).set_index('uuid')
        annotator_point1d.define_fields(data[['description']], preserve_index=True)
        annotator_point1d.define_points(data['timestamps'].iloc[:2])
        msg = f"Following annotations have no associated region: {{{annotation_id[2]!r}}}"
        with pytest.raises(ValueError, match=msg):
            annotator_point1d.commit(return_commits=True)

    def test_point_define_auto_index_mismatch(self, annotator_point1d):
        timestamps = [np.datetime64('2022-06-%.2d' % d) for d in  range(6,15, 4)]
        descriptions = ['Annotation %d' % d for d in [1,2,3]]
        annotation_id = [uuid.uuid4().hex[:8] for d in [1,2,3]]

        data = pd.DataFrame({'uuid':annotation_id, 'timestamps':timestamps,
                             'description':descriptions}).set_index('uuid')
        annotator_point1d.define_fields(data[['description']], preserve_index=False)
        annotator_point1d.define_points(data['timestamps'].iloc[:2])
        with pytest.raises(ValueError, match="Following annotations have no associated region:"):
            annotator_point1d.commit(return_commits=True)

    def test_point_define_unassigned_indices(self, annotator_point1d):
        timestamps = [np.datetime64('2022-06-%.2d' % d) for d in  range(6,15, 4)]
        descriptions = ['Annotation %d' % d for d in [1,2,3]]
        annotation_id1 = [uuid.uuid4().hex[:8] for d in [1,2,3]]
        mismatched = [uuid.uuid4().hex[:8] for d in [1,2]]
        annotation_id2 = [*mismatched, annotation_id1[2]]

        data1 = pd.DataFrame({'uuid':annotation_id1, 'timestamps':timestamps,
                              'description':descriptions}).set_index('uuid')
        data2 = pd.DataFrame({'uuid':annotation_id2, 'timestamps':timestamps,
                              'description':descriptions}).set_index('uuid')

        annotator_point1d.define_fields(data1[['description']])
        with pytest.raises(KeyError, match=str(mismatched)):
            annotator_point1d.define_points(data2['timestamps'])


class TestBasicPoint2DAnnotator:

    def test_insertion_edit_table_columns(self, annotator_point2d):
        annotator_point2d.set_point(-0.25, 0.1)
        annotator_point2d.add_annotation(description='A test annotation!')
        commits = annotator_point2d.commit(return_commits=True)
        assert len(commits)==1, 'Only one insertion commit made '
        assert commits[0]['operation'] == 'insert'
        assert set(commits[0]['kwargs'].keys()) == set(annotator_point2d.connector.columns)

    def test_range_insertion_exception(self, annotator_point2d):
        x1,x2,y1,y2 = -0.25,0.25, -0.3, 0.3
        expected_msg = r"Range region types not enabled as region_types=\['Point'\]"
        with pytest.raises(ValueError, match=expected_msg):
            annotator_point2d.set_range(x1,x2,y1,y2)

    def test_point_insertion_values(self, annotator_point2d):
        x,y = 0.5, 0.3
        annotator_point2d.set_point(x,y)
        annotator_point2d.add_annotation(description='A test annotation!')
        commits = annotator_point2d.commit(return_commits=True)
        assert len(commits)==1, 'Only one insertion commit made'
        kwargs = commits[0]['kwargs']
        assert 'uuid' in kwargs.keys(), 'Expected uuid primary key in kwargs'
        kwargs.pop('uuid')
        assert kwargs == dict(description='A test annotation!', point_x=x, point_y=y)

    def test_point_commit_insertion(self, annotator_point2d):
        x, y = 0.5, 0.3
        description = 'A test annotation!'
        annotator_point2d.set_point(x,y)
        annotator_point2d.add_annotation(description=description)
        annotator_point2d.commit(return_commits=True)

        df = pd.DataFrame({'uuid': pd.Series(annotator_point2d.df.index[0], dtype=object),
                           'point_x':[x],
                           'point_y':[y],
                           'description':[description]}
                           ).set_index('uuid')

        sql_df = annotator_point2d.connector.load_dataframe()
        pd.testing.assert_frame_equal(sql_df, df)


    def test_point_addition_deletion_by_uuid(self, annotator_point2d):
        x1, y1  = 0.2,0.2
        x2, y2  = 0.3,0.3
        x3, y3  = 0.4,0.4
        annotator_point2d.set_point(x1, y1)
        annotator_point2d.add_annotation(description='Annotation 1')
        annotator_point2d.set_point(x2, y2)
        annotator_point2d.add_annotation(description='Annotation 2', uuid='08286429')
        annotator_point2d.set_point(x3, y3)
        annotator_point2d.add_annotation(description='Annotation 3')
        annotator_point2d.commit(return_commits=True)
        sql_df = annotator_point2d.connector.load_dataframe()
        assert set(sql_df['description']) == set(['Annotation 1', 'Annotation 2', 'Annotation 3'])
        deletion_index = sql_df.index[1]
        annotator_point2d.delete_annotation(deletion_index)
        annotator_point2d.commit(return_commits=True)
        sql_df = annotator_point2d.connector.load_dataframe()
        assert set(sql_df['description']) == set(['Annotation 1', 'Annotation 3'])

    def test_point_define_preserved_index_mismatch(self, annotator_point2d):
        xs, ys  = [-0.1,-0.2,-0.3], [0.1,0.2,0.3]
        descriptions = ['Annotation %d' % d for d in [1,2,3]]
        annotation_id = [uuid.uuid4().hex[:8] for d in [1,2,3]]

        data = pd.DataFrame({'uuid':annotation_id, 'xs':xs, 'ys':ys,
                             'description':descriptions}).set_index('uuid')
        annotator_point2d.define_fields(data[['description']], preserve_index=True)
        annotator_point2d.define_points(data['xs'].iloc[:2], data['ys'].iloc[:2])
        msg = f"Following annotations have no associated region: {{{annotation_id[2]!r}}}"
        with pytest.raises(ValueError, match=msg):
            annotator_point2d.commit(return_commits=True)

    def test_point_define_auto_index_mismatch(self, annotator_point2d):
        xs, ys  = [-0.1,-0.2,-0.3], [0.1,0.2,0.3]
        descriptions = ['Annotation %d' % d for d in [1,2,3]]
        annotation_id = [uuid.uuid4().hex[:8] for d in [1,2,3]]

        data = pd.DataFrame({'uuid':annotation_id, 'xs':xs, 'ys':ys,
                             'description':descriptions}).set_index('uuid')
        annotator_point2d.define_fields(data[['description']], preserve_index=False)
        annotator_point2d.define_points(data['xs'].iloc[:2], data['ys'].iloc[:2])
        msg = "Following annotations have no associated region:"
        with pytest.raises(ValueError, match=msg):
            annotator_point2d.commit(return_commits=True)

    def test_point_define_unassigned_indices(self, annotator_point2d):
        xs, ys  = [-0.1,-0.2,-0.3], [0.1,0.2,0.3]
        descriptions = ['Annotation %d' % d for d in [1,2,3]]
        annotation_id1 = [uuid.uuid4().hex[:8] for d in [1,2,3]]
        mismatched = [uuid.uuid4().hex[:8] for d in [1,2]]
        annotation_id2 = [*mismatched, annotation_id1[2]]

        data1 = pd.DataFrame({'uuid':annotation_id1, 'xs':xs, 'ys':ys,
                              'description':descriptions}).set_index('uuid')
        data2 = pd.DataFrame({'uuid':annotation_id2, 'xs':xs, 'ys':ys,
                              'description':descriptions}).set_index('uuid')

        annotator_point2d.define_fields(data1[['description']])
        with pytest.raises(KeyError, match=str(mismatched)):
            annotator_point2d.define_points(data2['xs'], data2['ys'])
