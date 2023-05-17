from holonote.annotate import AnnotationTable
# TODO:

# * (after refactor) annotators -> annotator, connectors -> connector [ ]

# TESTS

# Schema error (needs file or connect in memory??)
# .snapshot() and .revert_to_snapshot()

import uuid
import unittest
import numpy as np
import pandas as pd

import holoviews as hv
from holonote.annotate import Annotator
from holonote.annotate import SQLiteDB, UUIDHexStringKey

class TestBasicRange1DAnnotator(unittest.TestCase):

    def setUp(self):
        assert Annotator.connector_class is SQLiteDB, 'Expecting default SQLite connector'
        Annotator.connector_class.filename = ':memory:'
        Annotator.connector_class.primary_key = UUIDHexStringKey()
        self.annotator = Annotator({'TIME': np.datetime64}, fields=['description'], region_types=['Range'])

    def tearDown(self):
        self.annotator.connector.cursor.close()
        self.annotator.connector.con.close()
        del self.annotator

    def test_point_insertion_exception(self):
        timestamp = np.datetime64('2022-06-06')
        with self.assertRaises(ValueError) as cm:
            self.annotator.set_point(timestamp)

        expected_msg = "Point region types not enabled as region_types=['Range']"
        self.assertEqual(str(cm.exception), expected_msg)

    def test_insertion_edit_table_columns(self):
        self.annotator.set_range(np.datetime64('2022-06-06'), np.datetime64('2022-06-08'))
        self.annotator.add_annotation(description='A test annotation!')
        commits = self.annotator.annotation_table.commits()
        assert len(commits)==1, 'Only one insertion commit made '
        self.annotator.commit()
        self.assertEqual(commits[0]['operation'],'insert')
        self.assertEqual(set(commits[0]['kwargs'].keys()),
                         set(self.annotator.connector.columns))

    def test_range_insertion_values(self):
        start, end = np.datetime64('2022-06-06'), np.datetime64('2022-06-08')
        self.annotator.set_range(start, end)
        self.annotator.add_annotation(description='A test annotation!')
        commits = self.annotator.annotation_table.commits()
        assert len(commits)==1, 'Only one insertion commit made'
        kwargs = commits[0]['kwargs']
        assert 'uuid' in kwargs.keys(), 'Expected uuid primary key in kwargs'
        kwargs.pop('uuid')
        self.assertEqual(kwargs, dict(description='A test annotation!',
                                      start_TIME=start, end_TIME=end))

    def test_range_commit_insertion(self):
        start, end  = np.datetime64('2022-06-06'), np.datetime64('2022-06-08')
        description = 'A test annotation!'
        self.annotator.set_range(start, end)
        self.annotator.add_annotation(description=description)
        self.annotator.commit()

        df = pd.DataFrame({'uuid': pd.Series(self.annotator.df.index[0], dtype=object),
                           'start_TIME':[start],
                           'end_TIME':[end],
                           'description':[description]}
                           ).set_index('uuid')

        sql_df = self.annotator.connector.load_dataframe()
        pd.testing.assert_frame_equal(sql_df, df)


    def test_range_addition_deletion_by_uuid(self):
        start1, end1  = np.datetime64('2022-06-06'), np.datetime64('2022-06-08')
        start2, end2  = np.datetime64('2023-06-06'), np.datetime64('2023-06-08')
        start3, end3  = np.datetime64('2024-06-06'), np.datetime64('2024-06-08')
        self.annotator.set_range(start1, end1)
        self.annotator.add_annotation(description='Annotation 1')
        self.annotator.set_range(start2, end2)
        self.annotator.add_annotation(description='Annotation 2', uuid='08286429')
        self.annotator.set_range(start3, end3)
        self.annotator.add_annotation(description='Annotation 3')
        self.annotator.commit()
        sql_df = self.annotator.connector.load_dataframe()
        self.assertEqual(set(sql_df['description']), set(['Annotation 1', 'Annotation 2', 'Annotation 3']))
        deletion_index = sql_df.index[1]
        self.annotator.delete_annotation(deletion_index)
        self.annotator.commit()
        sql_df = self.annotator.connector.load_dataframe()
        self.assertEqual(set(sql_df['description']), set(['Annotation 1', 'Annotation 3']))


    def test_range_define_preserved_index_mismatch(self):
        starts = [np.datetime64('2022-06-%.2d' % d) for d in  range(6,15, 4)]
        ends = [np.datetime64('2022-06-%.2d' % (d+2)) for d in  range(6,15, 4)]
        descriptions = ['Annotation %d' % d for d in [1,2,3]]
        annotation_id = [uuid.uuid4().hex[:8] for d in [1,2,3]]

        data = pd.DataFrame({'uuid':annotation_id, 'start':starts, 'end':ends, 'description':descriptions}).set_index('uuid')
        self.annotator.define_fields(data[['description']], preserve_index=True)
        self.annotator.define_ranges(data['start'].iloc[:2], data['end'].iloc[:2])
        with self.assertRaisesRegex(ValueError,
                           f"Following annotations have no associated region: {{{repr(annotation_id[2])}}}"):
            self.annotator.commit()

    def test_range_define_auto_index_mismatch(self):
        starts = [np.datetime64('2022-06-%.2d' % d) for d in  range(6,15, 4)]
        ends = [np.datetime64('2022-06-%.2d' % (d+2)) for d in  range(6,15, 4)]
        descriptions = ['Annotation %d' % d for d in [1,2,3]]
        annotation_id = [uuid.uuid4().hex[:8] for d in [1,2,3]]

        data = pd.DataFrame({'uuid':annotation_id, 'start':starts,
                             'end':ends, 'description':descriptions}).set_index('uuid')
        self.annotator.define_fields(data[['description']], preserve_index=False)
        self.annotator.define_ranges(data['start'].iloc[:2], data['end'].iloc[:2])
        with self.assertRaisesRegex(ValueError,
                           "Following annotations have no associated region:"):
            self.annotator.commit()

    def test_range_define_unassigned_indices(self):
        starts = [np.datetime64('2022-06-%.2d' % d) for d in  range(6,15, 4)]
        ends = [np.datetime64('2022-06-%.2d' % (d+2)) for d in  range(6,15, 4)]
        descriptions = ['Annotation %d' % d for d in [1,2,3]]
        annotation_id1 = [uuid.uuid4().hex[:8] for d in [1,2,3]]
        mismatched = [uuid.uuid4().hex[:8] for d in [1,2]]
        annotation_id2 = mismatched + [annotation_id1[2]]

        data1 = pd.DataFrame({'uuid':annotation_id1, 'start':starts,
                              'end':ends, 'description':descriptions}).set_index('uuid')
        data2 = pd.DataFrame({'uuid':annotation_id2, 'start':starts,
                              'end':ends, 'description':descriptions}).set_index('uuid')

        self.annotator.define_fields(data1[['description']])
        with self.assertRaises(KeyError) as cm:
            self.annotator.define_ranges(data2['start'], data2['end'])
        assert f'{mismatched}' in str(cm.exception)


class TestBasicRange2DAnnotator(unittest.TestCase):

    def setUp(self):
        assert Annotator.connector_class is SQLiteDB, 'Expecting default SQLite connector'
        Annotator.connector_class.filename = ':memory:'
        Annotator.connector_class.primary_key = UUIDHexStringKey()
        self.annotator = Annotator({'x': float, 'y':float},
                                   fields=['description'], region_types=['Range'])

    def tearDown(self):
        self.annotator.connector.cursor.close()
        self.annotator.connector.con.close()
        del self.annotator

    def test_point_insertion_exception(self):
        x,y = 0.5,0.5
        with self.assertRaises(ValueError) as cm:
            self.annotator.set_point(x,y)

        expected_msg = "Point region types not enabled as region_types=['Range']"
        self.assertEqual(str(cm.exception), expected_msg)

    def test_insertion_edit_table_columns(self):
        self.annotator.set_range(-0.25, 0.25, -0.1, 0.1)
        self.annotator.add_annotation(description='A test annotation!')
        commits = self.annotator.annotation_table.commits()
        assert len(commits)==1, 'Only one insertion commit made '
        self.annotator.commit()
        self.assertEqual(commits[0]['operation'],'insert')
        self.assertEqual(set(commits[0]['kwargs'].keys()),
                         set(self.annotator.connector.columns))

    def test_range_insertion_values(self):
        startx, endx, starty, endy = -0.25, 0.25, -0.1, 0.1
        self.annotator.set_range(startx, endx, starty, endy)
        self.annotator.add_annotation(description='A test annotation!')
        commits = self.annotator.annotation_table.commits()
        assert len(commits)==1, 'Only one insertion commit made'
        kwargs = commits[0]['kwargs']
        assert 'uuid' in kwargs.keys(), 'Expected uuid primary key in kwargs'
        kwargs.pop('uuid')
        self.assertEqual(kwargs, dict(description='A test annotation!',
                                      start_x=startx, end_x=endx, start_y=starty, end_y=endy))

    def test_range_commit_insertion(self):
        startx, endx, starty, endy = -0.25, 0.25, -0.1, 0.1
        description = 'A test annotation!'
        self.annotator.set_range(startx, endx, starty, endy)
        self.annotator.add_annotation(description=description)
        self.annotator.commit()

        df = pd.DataFrame({'uuid': pd.Series(self.annotator.df.index[0], dtype=object),
                           'start_x':[startx],
                           'start_y':[starty],
                           'end_x':[endx],
                           'end_y':[endy],
                           'description':[description]}
                           ).set_index('uuid')

        sql_df = self.annotator.connector.load_dataframe()
        pd.testing.assert_frame_equal(sql_df, df)


    def test_range_addition_deletion_by_uuid(self):
        startx1, endx1, starty1, endy1 = -0.251, 0.251, -0.11, 0.11
        startx2, endx2, starty2, endy2 = -0.252, 0.252, -0.12, 0.12
        startx3, endx3, starty3, endy3 = -0.253, 0.253, -0.13, 0.13
        self.annotator.set_range(startx1, endx1, starty1, endy1)
        self.annotator.add_annotation(description='Annotation 1')
        self.annotator.set_range(startx2, endx2, starty2, endy2)
        self.annotator.add_annotation(description='Annotation 2', uuid='08286429')
        self.annotator.set_range(startx3, endx3, starty3, endy3)
        self.annotator.add_annotation(description='Annotation 3')
        self.annotator.commit()
        sql_df = self.annotator.connector.load_dataframe()
        self.assertEqual(set(sql_df['description']), set(['Annotation 1', 'Annotation 2', 'Annotation 3']))
        deletion_index = sql_df.index[1]
        self.annotator.delete_annotation(deletion_index)
        self.annotator.commit()
        sql_df = self.annotator.connector.load_dataframe()
        self.assertEqual(set(sql_df['description']), set(['Annotation 1', 'Annotation 3']))


    def test_range_define_preserved_index_mismatch(self):
        xstarts, xends = [-0.3, -0.2, -0.1], [0.3, 0.2, 0.1]
        ystarts, yends = [-0.35, -0.25, -0.15], [0.35, 0.25, 0.15]
        descriptions = ['Annotation %d' % d for d in [1,2,3]]
        annotation_id = [uuid.uuid4().hex[:8] for d in [1,2,3]]

        data = pd.DataFrame({'uuid':annotation_id, 'xstart':xstarts, 'xend':xends,
                             'ystart':ystarts, 'yend':yends,
                             'description':descriptions}).set_index('uuid')
        self.annotator.define_fields(data[['description']], preserve_index=True)
        self.annotator.define_ranges(data['xstart'].iloc[:2], data['xend'].iloc[:2],
                                     data['ystart'].iloc[:2], data['yend'].iloc[:2])
        with self.assertRaisesRegex(ValueError,
                           f"Following annotations have no associated region: {{{repr(annotation_id[2])}}}"):
            self.annotator.commit()

    def test_range_define_auto_index_mismatch(self):
        xstarts, xends = [-0.3, -0.2, -0.1], [0.3, 0.2, 0.1]
        ystarts, yends = [-0.35, -0.25, -0.15], [0.35, 0.25, 0.15]
        descriptions = ['Annotation %d' % d for d in [1,2,3]]
        annotation_id = [uuid.uuid4().hex[:8] for d in [1,2,3]]
        data = pd.DataFrame({'uuid':annotation_id, 'xstart':xstarts, 'xend':xends,
                             'ystart':ystarts, 'yend':yends,
                             'description':descriptions}).set_index('uuid')
        self.annotator.define_fields(data[['description']], preserve_index=False)
        self.annotator.define_ranges(data['xstart'].iloc[:2], data['xend'].iloc[:2],
                                     data['ystart'].iloc[:2], data['yend'].iloc[:2])
        with self.assertRaisesRegex(ValueError,
                           "Following annotations have no associated region:"):
            self.annotator.commit()

    def test_range_define_unassigned_indices(self):
        xstarts, xends = [-0.3, -0.2, -0.1], [0.3, 0.2, 0.1]
        ystarts, yends = [-0.35, -0.25, -0.15], [0.35, 0.25, 0.15]
        descriptions = ['Annotation %d' % d for d in [1,2,3]]
        annotation_id1 = [uuid.uuid4().hex[:8] for d in [1,2,3]]
        mismatched = [uuid.uuid4().hex[:8] for d in [1,2]]
        annotation_id2 = mismatched + [annotation_id1[2]]

        data1 = pd.DataFrame({'uuid':annotation_id1, 'xstart':xstarts, 'xend':xends,
                             'ystart':ystarts, 'yend':yends,
                             'description':descriptions}).set_index('uuid')
        data2 = pd.DataFrame({'uuid':annotation_id2, 'xstart':xstarts, 'xend':xends,
                             'ystart':ystarts, 'yend':yends,
                             'description':descriptions}).set_index('uuid')

        self.annotator.define_fields(data1[['description']])
        with self.assertRaises(KeyError) as cm:
            self.annotator.define_ranges(data2['xstart'], data2['xend'],
                                         data2['ystart'], data2['yend'])
        assert f'{mismatched}' in str(cm.exception)



class TestBasicPoint1DAnnotator(unittest.TestCase):

    def setUp(self):
        assert Annotator.connector_class is SQLiteDB, 'Expecting default SQLite connector'
        Annotator.connector_class.filename = ':memory:'
        Annotator.connector_class.primary_key = UUIDHexStringKey()
        self.annotator = Annotator({'TIME': np.datetime64}, fields=['description'], region_types=['Point'])

    def tearDown(self):
        self.annotator.connector.cursor.close()
        self.annotator.connector.con.close()
        del self.annotator

    def test_insertion_edit_table_columns(self):
        self.annotator.set_point(np.datetime64('2022-06-06'))
        self.annotator.add_annotation(description='A test annotation!')
        commits = self.annotator.annotation_table.commits()
        assert len(commits)==1, 'Only one insertion commit made '
        self.annotator.commit()
        self.assertEqual(commits[0]['operation'],'insert')
        self.assertEqual(set(commits[0]['kwargs'].keys()),
                         set(self.annotator.connector.columns))

    def test_range_insertion_exception(self):
        start, end = np.datetime64('2022-06-06'), np.datetime64('2022-06-08')
        with self.assertRaises(ValueError) as cm:
            self.annotator.set_range(start, end)
        expected_msg = "Range region types not enabled as region_types=['Point']"
        self.assertEqual(str(cm.exception), expected_msg)

    def test_point_insertion_values(self):
        timestamp = np.datetime64('2022-06-06')
        self.annotator.set_point(timestamp)
        self.annotator.add_annotation(description='A test annotation!')
        commits = self.annotator.annotation_table.commits()
        assert len(commits)==1, 'Only one insertion commit made'
        kwargs = commits[0]['kwargs']
        assert 'uuid' in kwargs.keys(), 'Expected uuid primary key in kwargs'
        kwargs.pop('uuid')
        self.assertEqual(kwargs, dict(description='A test annotation!', point_TIME=timestamp))

    def test_point_commit_insertion(self):
        timestamp = np.datetime64('2022-06-06')
        description = 'A test annotation!'
        self.annotator.set_point(timestamp)
        self.annotator.add_annotation(description=description)
        self.annotator.commit()

        df = pd.DataFrame({'uuid': pd.Series(self.annotator.df.index[0], dtype=object),
                           'point_TIME':[timestamp],
                           'description':[description]}
                           ).set_index('uuid')

        sql_df = self.annotator.connector.load_dataframe()
        pd.testing.assert_frame_equal(sql_df, df)


    def test_point_addition_deletion_by_uuid(self):
        ts1  = np.datetime64('2022-06-06')
        ts2  = np.datetime64('2023-06-06')
        ts3  = np.datetime64('2024-06-06')
        self.annotator.set_point(ts1)
        self.annotator.add_annotation(description='Annotation 1')
        self.annotator.set_point(ts2)
        self.annotator.add_annotation(description='Annotation 2', uuid='08286429')
        self.annotator.set_point(ts3)
        self.annotator.add_annotation(description='Annotation 3')
        self.annotator.commit()
        sql_df = self.annotator.connector.load_dataframe()
        self.assertEqual(set(sql_df['description']), set(['Annotation 1', 'Annotation 2', 'Annotation 3']))
        deletion_index = sql_df.index[1]
        self.annotator.delete_annotation(deletion_index)
        self.annotator.commit()
        sql_df = self.annotator.connector.load_dataframe()
        self.assertEqual(set(sql_df['description']), set(['Annotation 1', 'Annotation 3']))

    def test_point_define_preserved_index_mismatch(self):
        timestamps = [np.datetime64('2022-06-%.2d' % d) for d in  range(6,15, 4)]
        descriptions = ['Annotation %d' % d for d in [1,2,3]]
        annotation_id = [uuid.uuid4().hex[:8] for d in [1,2,3]]

        data = pd.DataFrame({'uuid':annotation_id, 'timestamps':timestamps,
                             'description':descriptions}).set_index('uuid')
        self.annotator.define_fields(data[['description']], preserve_index=True)
        self.annotator.define_points(data['timestamps'].iloc[:2])
        with self.assertRaisesRegex(ValueError,
                           f"Following annotations have no associated region: {{{repr(annotation_id[2])}}}"):
            self.annotator.commit()

    def test_point_define_auto_index_mismatch(self):
        timestamps = [np.datetime64('2022-06-%.2d' % d) for d in  range(6,15, 4)]
        descriptions = ['Annotation %d' % d for d in [1,2,3]]
        annotation_id = [uuid.uuid4().hex[:8] for d in [1,2,3]]

        data = pd.DataFrame({'uuid':annotation_id, 'timestamps':timestamps,
                             'description':descriptions}).set_index('uuid')
        self.annotator.define_fields(data[['description']], preserve_index=False)
        self.annotator.define_points(data['timestamps'].iloc[:2])
        with self.assertRaisesRegex(ValueError,
                           "Following annotations have no associated region:"):
            self.annotator.commit()

    def test_point_define_unassigned_indices(self):
        timestamps = [np.datetime64('2022-06-%.2d' % d) for d in  range(6,15, 4)]
        descriptions = ['Annotation %d' % d for d in [1,2,3]]
        annotation_id1 = [uuid.uuid4().hex[:8] for d in [1,2,3]]
        mismatched = [uuid.uuid4().hex[:8] for d in [1,2]]
        annotation_id2 = mismatched + [annotation_id1[2]]

        data1 = pd.DataFrame({'uuid':annotation_id1, 'timestamps':timestamps,
                              'description':descriptions}).set_index('uuid')
        data2 = pd.DataFrame({'uuid':annotation_id2, 'timestamps':timestamps,
                              'description':descriptions}).set_index('uuid')

        self.annotator.define_fields(data1[['description']])
        with self.assertRaises(KeyError) as cm:
            self.annotator.define_points(data2['timestamps'])
        assert f'{mismatched}' in str(cm.exception)



class TestBasicPoint2DAnnotator(unittest.TestCase):

    def setUp(self):
        assert Annotator.connector_class is SQLiteDB, 'Expecting default SQLite connector'
        Annotator.connector_class.filename = ':memory:'
        Annotator.connector_class.primary_key = UUIDHexStringKey()
        self.annotator = Annotator({'x': float, 'y':float}, fields=['description'], region_types=['Point'])

    def tearDown(self):
        self.annotator.connector.cursor.close()
        self.annotator.connector.con.close()
        del self.annotator

    def test_insertion_edit_table_columns(self):
        self.annotator.set_point(-0.25, 0.1)
        self.annotator.add_annotation(description='A test annotation!')
        commits = self.annotator.annotation_table.commits()
        assert len(commits)==1, 'Only one insertion commit made '
        self.annotator.commit()
        self.assertEqual(commits[0]['operation'],'insert')
        self.assertEqual(set(commits[0]['kwargs'].keys()),
                         set(self.annotator.connector.columns))

    def test_range_insertion_exception(self):
        x1,x2,y1,y2 = -0.25,0.25, -0.3, 0.3
        with self.assertRaises(ValueError) as cm:
            self.annotator.set_range(x1,x2,y1,y2)
        expected_msg = "Range region types not enabled as region_types=['Point']"
        self.assertEqual(str(cm.exception), expected_msg)

    def test_point_insertion_values(self):
        x,y = 0.5, 0.3
        self.annotator.set_point(x,y)
        self.annotator.add_annotation(description='A test annotation!')
        commits = self.annotator.annotation_table.commits()
        assert len(commits)==1, 'Only one insertion commit made'
        kwargs = commits[0]['kwargs']
        assert 'uuid' in kwargs.keys(), 'Expected uuid primary key in kwargs'
        kwargs.pop('uuid')
        self.assertEqual(kwargs, dict(description='A test annotation!', point_x=x, point_y=y))

    def test_point_commit_insertion(self):
        x, y = 0.5, 0.3
        description = 'A test annotation!'
        self.annotator.set_point(x,y)
        self.annotator.add_annotation(description=description)
        self.annotator.commit()

        df = pd.DataFrame({'uuid': pd.Series(self.annotator.df.index[0], dtype=object),
                           'point_x':[x],
                           'point_y':[y],
                           'description':[description]}
                           ).set_index('uuid')

        sql_df = self.annotator.connector.load_dataframe()
        pd.testing.assert_frame_equal(sql_df, df)


    def test_point_addition_deletion_by_uuid(self):
        x1, y1  = 0.2,0.2
        x2, y2  = 0.3,0.3
        x3, y3  = 0.4,0.4
        self.annotator.set_point(x1, y1)
        self.annotator.add_annotation(description='Annotation 1')
        self.annotator.set_point(x2, y2)
        self.annotator.add_annotation(description='Annotation 2', uuid='08286429')
        self.annotator.set_point(x3, y3)
        self.annotator.add_annotation(description='Annotation 3')
        self.annotator.commit()
        sql_df = self.annotator.connector.load_dataframe()
        self.assertEqual(set(sql_df['description']), set(['Annotation 1', 'Annotation 2', 'Annotation 3']))
        deletion_index = sql_df.index[1]
        self.annotator.delete_annotation(deletion_index)
        self.annotator.commit()
        sql_df = self.annotator.connector.load_dataframe()
        self.assertEqual(set(sql_df['description']), set(['Annotation 1', 'Annotation 3']))

    def test_point_define_preserved_index_mismatch(self):
        xs, ys  = [-0.1,-0.2,-0.3], [0.1,0.2,0.3]
        descriptions = ['Annotation %d' % d for d in [1,2,3]]
        annotation_id = [uuid.uuid4().hex[:8] for d in [1,2,3]]

        data = pd.DataFrame({'uuid':annotation_id, 'xs':xs, 'ys':ys,
                             'description':descriptions}).set_index('uuid')
        self.annotator.define_fields(data[['description']], preserve_index=True)
        self.annotator.define_points(data['xs'].iloc[:2], data['ys'].iloc[:2])
        with self.assertRaisesRegex(ValueError,
                           f"Following annotations have no associated region: {{{repr(annotation_id[2])}}}"):
            self.annotator.commit()

    def test_point_define_auto_index_mismatch(self):
        xs, ys  = [-0.1,-0.2,-0.3], [0.1,0.2,0.3]
        descriptions = ['Annotation %d' % d for d in [1,2,3]]
        annotation_id = [uuid.uuid4().hex[:8] for d in [1,2,3]]

        data = pd.DataFrame({'uuid':annotation_id, 'xs':xs, 'ys':ys,
                             'description':descriptions}).set_index('uuid')
        self.annotator.define_fields(data[['description']], preserve_index=False)
        self.annotator.define_points(data['xs'].iloc[:2], data['ys'].iloc[:2])
        with self.assertRaisesRegex(ValueError,
                           "Following annotations have no associated region:"):
            self.annotator.commit()

    def test_point_define_unassigned_indices(self):
        xs, ys  = [-0.1,-0.2,-0.3], [0.1,0.2,0.3]
        descriptions = ['Annotation %d' % d for d in [1,2,3]]
        annotation_id1 = [uuid.uuid4().hex[:8] for d in [1,2,3]]
        mismatched = [uuid.uuid4().hex[:8] for d in [1,2]]
        annotation_id2 = mismatched + [annotation_id1[2]]

        data1 = pd.DataFrame({'uuid':annotation_id1, 'xs':xs, 'ys':ys,
                              'description':descriptions}).set_index('uuid')
        data2 = pd.DataFrame({'uuid':annotation_id2, 'xs':xs, 'ys':ys,
                              'description':descriptions}).set_index('uuid')

        self.annotator.define_fields(data1[['description']])
        with self.assertRaises(KeyError) as cm:
            self.annotator.define_points(data2['xs'], data2['ys'])
        assert f'{mismatched}' in str(cm.exception)



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
        self.connector = SQLiteDB(table_name='annotations')
        xvals, yvals  = np.linspace(-4, 0, 202), np.linspace(4, 0, 202)
        xs, ys = np.meshgrid(xvals, yvals)
        image = hv.Image(np.sin(ys*xs), kdims=['A', 'B'])
        self.image_annotator = Annotator(image, connector=self.connector,
                                         fields=['description'], region_types=['Range'])

        curve = hv.Curve((np.arange('2005-02', '2005-03', dtype='datetime64[D]'), range(28)), kdims=['TIME'])
        self.curve_annotator = Annotator(curve, connector=self.connector,
                                         fields=['description'], region_types=['Range'])


    def test_element_kdim_dtypes(self):
        self.assertEqual(self.image_annotator.kdim_dtypes, {'A':np.float64 , 'B':np.float64})
        self.assertEqual(self.curve_annotator.kdim_dtypes, {'TIME': np.datetime64})

    def test_multiplot_add_annotation(self):
        self.image_annotator.set_range(-0.25, 0.25, -0.1, 0.1)
        self.curve_annotator.set_range(np.datetime64('2005-02-13'), np.datetime64('2005-02-16'))
        self.connector.add_annotation(description='Multi-plot annotation')


    def test_table_name_required(self):
        assert Annotator.connector_class is SQLiteDB, 'Expecting default SQLite connector'
        Annotator.connector_class.filename = ':memory:'
        Annotator.connector_class.primary_key = UUIDHexStringKey()
        connector = SQLiteDB()
        xvals, yvals  = np.linspace(-4, 0, 202), np.linspace(4, 0, 202)
        xs, ys = np.meshgrid(xvals, yvals)
        image = hv.Image(np.sin(ys*xs), kdims=['A', 'B'])
        image_annotator = Annotator(image, connector=connector,
                                    fields=['description'], region_types=['Range'])

        curve = hv.Curve((np.arange('2005-02', '2005-03', dtype='datetime64[D]'), range(28)), kdims=['TIME'])


        with self.assertRaisesRegex(Exception,
                            "A table_name must be set in the connector when shared across multiple annotators"):
            curve_annotator = Annotator(curve, connector=connector,
                                        fields=['description'], region_types=['Range'])

        connector.cursor.close()
        connector.con.close()

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
        self.assertEqual(kwargs, dict(field1='A test field', field2='Another test field',
                                      start_TIME=start, end_TIME=end))


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
        start3, end3  = np.datetime64('2024-06-06'), np.datetime64('2024-06-08')
        self.annotator.set_range(start1, end1)
        self.annotator.add_annotation(field1='Field 1.1', field2='Field 1.2')
        self.annotator.set_range(start2, end2)
        self.annotator.add_annotation(field1='Field 2.1', field2='Field 2.2')
        self.annotator.commit()
        self.annotator.update_annotation_fields(self.annotator.df.index[0], field1='NEW Field 1.1')
        self.annotator.commit()
        sql_df = self.annotator.connector.load_dataframe()
        self.assertEqual(set(sql_df['field1']), set(['NEW Field 1.1', 'Field 2.1']))
