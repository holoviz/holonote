from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from holonote.annotate import Annotator


class TestBasicRange1DAnnotator:
    @pytest.mark.skip("Need to add validation to set_regions")
    def test_point_insertion_exception(self, annotator_range1d):
        timestamp = np.datetime64("2022-06-06")
        expected_msg = "Only 'point' region allowed for 'set_regions'"
        with pytest.raises(ValueError, match=expected_msg):
            annotator_range1d.set_regions(TIME=timestamp)

    def test_insertion_edit_table_columns(self, annotator_range1d):
        annotator_range1d.set_regions(
            TIME=(np.datetime64("2022-06-06"), np.datetime64("2022-06-08"))
        )
        annotator_range1d.add_annotation(description="A test annotation!")
        commits = annotator_range1d.commit(return_commits=True)
        assert len(commits) == 1, "Only one insertion commit made "
        annotator_range1d.commit(return_commits=True)
        assert commits[0]["operation"] == "insert"
        assert set(commits[0]["kwargs"].keys()) == set(annotator_range1d.connector.columns)

    def test_range_insertion_values(self, annotator_range1d) -> None:
        start, end = np.datetime64("2022-06-06"), np.datetime64("2022-06-08")
        annotator_range1d.set_regions(TIME=(start, end))
        annotator_range1d.add_annotation(description="A test annotation!")
        commits = annotator_range1d.commit(return_commits=True)
        assert len(commits) == 1, "Only one insertion commit made"
        kwargs = commits[0]["kwargs"]
        assert "uuid" in kwargs, "Expected uuid primary key in kwargs"
        kwargs.pop("uuid")
        assert kwargs, {"description": "A test annotation!", "start_TIME": start, "end_TIME": end}

    def test_range_commit_insertion(self, annotator_range1d):
        start, end = np.datetime64("2022-06-06"), np.datetime64("2022-06-08")
        description = "A test annotation!"
        annotator_range1d.set_regions(TIME=(start, end))
        annotator_range1d.add_annotation(description=description)
        annotator_range1d.commit(return_commits=True)

        df = pd.DataFrame(
            {
                "uuid": pd.Series(annotator_range1d.df.index[0], dtype=object),
                "start_TIME": [start],
                "end_TIME": [end],
                "description": [description],
            }
        ).set_index("uuid")

        sql_df = annotator_range1d.connector.load_dataframe()
        pd.testing.assert_frame_equal(sql_df, df)

    def test_range_addition_deletion_by_uuid(self, annotator_range1d):
        start1, end1 = np.datetime64("2022-06-06"), np.datetime64("2022-06-08")
        start2, end2 = np.datetime64("2023-06-06"), np.datetime64("2023-06-08")
        start3, end3 = np.datetime64("2024-06-06"), np.datetime64("2024-06-08")
        annotator_range1d.set_regions(TIME=(start1, end1))
        annotator_range1d.add_annotation(description="Annotation 1")
        annotator_range1d.set_regions(TIME=(start2, end2))
        annotator_range1d.add_annotation(description="Annotation 2", uuid="08286429")
        annotator_range1d.set_regions(TIME=(start3, end3))
        annotator_range1d.add_annotation(description="Annotation 3")
        annotator_range1d.commit(return_commits=True)
        sql_df = annotator_range1d.connector.load_dataframe()
        assert set(sql_df["description"]) == {"Annotation 1", "Annotation 2", "Annotation 3"}
        deletion_index = sql_df.index[1]
        annotator_range1d.delete_annotation(deletion_index)
        annotator_range1d.commit(return_commits=True)
        sql_df = annotator_range1d.connector.load_dataframe()
        assert set(sql_df["description"]) == {"Annotation 1", "Annotation 3"}


class TestBasicRange2DAnnotator:
    @pytest.mark.skip("Need to add validation to set_regions")
    def test_point_insertion_exception(self, annotator_range2d):
        x, y = 0.5, 0.5
        expected_msg = "Only 'point' region allowed for 'set_regions'"
        with pytest.raises(ValueError, match=expected_msg):
            annotator_range2d.set_regions(x=x, y=y)

    def test_insertion_edit_table_columns(self, annotator_range2d):
        annotator_range2d.set_regions(x=(-0.25, 0.25), y=(-0.1, 0.1))
        annotator_range2d.add_annotation(description="A test annotation!")
        commits = annotator_range2d.commit(return_commits=True)
        assert len(commits) == 1, "Only one insertion commit made "
        assert commits[0]["operation"] == "insert"
        assert set(commits[0]["kwargs"].keys()) == set(annotator_range2d.connector.columns)

    def test_range_insertion_values(self, annotator_range2d):
        startx, endx, starty, endy = -0.25, 0.25, -0.1, 0.1
        annotator_range2d.set_regions(x=(startx, endx), y=(starty, endy))
        annotator_range2d.add_annotation(description="A test annotation!")
        commits = annotator_range2d.commit(return_commits=True)
        assert len(commits) == 1, "Only one insertion commit made"
        kwargs = commits[0]["kwargs"]
        assert "uuid" in kwargs, "Expected uuid primary key in kwargs"
        kwargs.pop("uuid")
        assert kwargs == {
            "description": "A test annotation!",
            "start_x": startx,
            "end_x": endx,
            "start_y": starty,
            "end_y": endy,
        }

    def test_range_commit_insertion(self, annotator_range2d):
        startx, endx, starty, endy = -0.25, 0.25, -0.1, 0.1
        description = "A test annotation!"
        annotator_range2d.set_regions(x=(startx, endx), y=(starty, endy))
        annotator_range2d.add_annotation(description=description)
        annotator_range2d.commit(return_commits=True)

        df = pd.DataFrame(
            {
                "uuid": pd.Series(annotator_range2d.df.index[0], dtype=object),
                "start_x": [startx],
                "start_y": [starty],
                "end_x": [endx],
                "end_y": [endy],
                "description": [description],
            }
        ).set_index("uuid")

        sql_df = annotator_range2d.connector.load_dataframe()
        pd.testing.assert_frame_equal(sql_df, df)

    def test_range_addition_deletion_by_uuid(self, annotator_range2d):
        startx1, endx1, starty1, endy1 = -0.251, 0.251, -0.11, 0.11
        startx2, endx2, starty2, endy2 = -0.252, 0.252, -0.12, 0.12
        startx3, endx3, starty3, endy3 = -0.253, 0.253, -0.13, 0.13
        annotator_range2d.set_regions(x=(startx1, endx1), y=(starty1, endy1))
        annotator_range2d.add_annotation(description="Annotation 1")
        annotator_range2d.set_regions(x=(startx2, endx2), y=(starty2, endy2))
        annotator_range2d.add_annotation(description="Annotation 2", uuid="08286429")
        annotator_range2d.set_regions(x=(startx3, endx3), y=(starty3, endy3))
        annotator_range2d.add_annotation(description="Annotation 3")
        annotator_range2d.commit(return_commits=True)
        sql_df = annotator_range2d.connector.load_dataframe()
        assert set(sql_df["description"]) == {"Annotation 1", "Annotation 2", "Annotation 3"}
        deletion_index = sql_df.index[1]
        annotator_range2d.delete_annotation(deletion_index)
        annotator_range2d.commit(return_commits=True)
        sql_df = annotator_range2d.connector.load_dataframe()
        assert set(sql_df["description"]) == {"Annotation 1", "Annotation 3"}


class TestBasicPoint1DAnnotator:
    def test_insertion_edit_table_columns(self, annotator_point1d):
        annotator_point1d.set_regions(TIME=np.datetime64("2022-06-06"))
        annotator_point1d.add_annotation(description="A test annotation!")
        commits = annotator_point1d.commit(return_commits=True)
        assert len(commits) == 1, "Only one insertion commit made "
        annotator_point1d.commit(return_commits=True)
        assert commits[0]["operation"] == "insert"
        assert set(commits[0]["kwargs"].keys()) == set(annotator_point1d.connector.columns)

    @pytest.mark.skip("Need to add validation to set_regions")
    def test_range_insertion_exception(self, annotator_point1d):
        start, end = np.datetime64("2022-06-06"), np.datetime64("2022-06-08")
        msg = "Only 'range' region allowed for 'set_range'"
        with pytest.raises(ValueError, match=msg):
            annotator_point1d.set_regions(TIME=(start, end))

    def test_point_insertion_values(self, annotator_point1d):
        timestamp = np.datetime64("2022-06-06")
        annotator_point1d.set_regions(TIME=timestamp)
        annotator_point1d.add_annotation(description="A test annotation!")
        commits = annotator_point1d.commit(return_commits=True)
        assert len(commits) == 1, "Only one insertion commit made"
        kwargs = commits[0]["kwargs"]
        assert "uuid" in kwargs, "Expected uuid primary key in kwargs"
        kwargs.pop("uuid")
        assert kwargs == {"description": "A test annotation!", "point_TIME": timestamp}

    def test_point_commit_insertion(self, annotator_point1d):
        timestamp = np.datetime64("2022-06-06")
        description = "A test annotation!"
        annotator_point1d.set_regions(TIME=timestamp)
        annotator_point1d.add_annotation(description=description)
        annotator_point1d.commit(return_commits=True)

        df = pd.DataFrame(
            {
                "uuid": pd.Series(annotator_point1d.df.index[0], dtype=object),
                "point_TIME": [timestamp],
                "description": [description],
            }
        ).set_index("uuid")

        sql_df = annotator_point1d.connector.load_dataframe()
        pd.testing.assert_frame_equal(sql_df, df)

    def test_point_addition_deletion_by_uuid(self, annotator_point1d):
        ts1 = np.datetime64("2022-06-06")
        ts2 = np.datetime64("2023-06-06")
        ts3 = np.datetime64("2024-06-06")
        annotator_point1d.set_regions(TIME=ts1)
        annotator_point1d.add_annotation(description="Annotation 1")
        annotator_point1d.set_regions(TIME=ts2)
        annotator_point1d.add_annotation(description="Annotation 2", uuid="08286429")
        annotator_point1d.set_regions(TIME=ts3)
        annotator_point1d.add_annotation(description="Annotation 3")
        annotator_point1d.commit(return_commits=True)
        sql_df = annotator_point1d.connector.load_dataframe()
        assert set(sql_df["description"]) == {"Annotation 1", "Annotation 2", "Annotation 3"}
        deletion_index = sql_df.index[1]
        annotator_point1d.delete_annotation(deletion_index)
        annotator_point1d.commit(return_commits=True)
        sql_df = annotator_point1d.connector.load_dataframe()
        assert set(sql_df["description"]) == {"Annotation 1", "Annotation 3"}


class TestBasicPoint2DAnnotator:
    def test_insertion_edit_table_columns(self, annotator_point2d):
        annotator_point2d.set_regions(x=-0.25, y=0.1)
        annotator_point2d.add_annotation(description="A test annotation!")
        commits = annotator_point2d.commit(return_commits=True)
        assert len(commits) == 1, "Only one insertion commit made "
        assert commits[0]["operation"] == "insert"
        assert set(commits[0]["kwargs"].keys()) == set(annotator_point2d.connector.columns)

    @pytest.mark.skip("Need to add validation to set_regions")
    def test_range_insertion_exception(self, annotator_point2d):
        x1, x2, y1, y2 = -0.25, 0.25, -0.3, 0.3
        expected_msg = "Only 'range' region allowed for 'set_range'"
        with pytest.raises(ValueError, match=expected_msg):
            annotator_point2d.set_regions(x=(x1, x2), y=(y1, y2))

    def test_point_insertion_values(self, annotator_point2d):
        x, y = 0.5, 0.3
        annotator_point2d.set_regions(x=x, y=y)
        annotator_point2d.add_annotation(description="A test annotation!")
        commits = annotator_point2d.commit(return_commits=True)
        assert len(commits) == 1, "Only one insertion commit made"
        kwargs = commits[0]["kwargs"]
        assert "uuid" in kwargs, "Expected uuid primary key in kwargs"
        kwargs.pop("uuid")
        assert kwargs == {"description": "A test annotation!", "point_x": x, "point_y": y}

    def test_point_commit_insertion(self, annotator_point2d):
        x, y = 0.5, 0.3
        description = "A test annotation!"
        annotator_point2d.set_regions(x=x, y=y)
        annotator_point2d.add_annotation(description=description)
        annotator_point2d.commit(return_commits=True)

        df = pd.DataFrame(
            {
                "uuid": pd.Series(annotator_point2d.df.index[0], dtype=object),
                "point_x": [x],
                "point_y": [y],
                "description": [description],
            }
        ).set_index("uuid")

        sql_df = annotator_point2d.connector.load_dataframe()
        pd.testing.assert_frame_equal(sql_df, df)

    def test_point_addition_deletion_by_uuid(self, annotator_point2d):
        x1, y1 = 0.2, 0.2
        x2, y2 = 0.3, 0.3
        x3, y3 = 0.4, 0.4
        annotator_point2d.set_regions(x=x1, y=y1)
        annotator_point2d.add_annotation(description="Annotation 1")
        annotator_point2d.set_regions(x=x2, y=y2)
        annotator_point2d.add_annotation(description="Annotation 2", uuid="08286429")
        annotator_point2d.set_regions(x=x3, y=y3)
        annotator_point2d.add_annotation(description="Annotation 3")
        annotator_point2d.commit(return_commits=True)
        sql_df = annotator_point2d.connector.load_dataframe()
        assert set(sql_df["description"]) == {"Annotation 1", "Annotation 2", "Annotation 3"}
        deletion_index = sql_df.index[1]
        annotator_point2d.delete_annotation(deletion_index)
        annotator_point2d.commit(return_commits=True)
        sql_df = annotator_point2d.connector.load_dataframe()
        assert set(sql_df["description"]) == {"Annotation 1", "Annotation 3"}


@pytest.mark.parametrize("fields", [["test"], ["test1", "test2"]])
def test_connector_use_annotator_fields(conn_sqlite_uuid, fields):
    annotator = Annotator({"TIME": float}, connector=conn_sqlite_uuid, fields=fields)

    assert annotator.fields == fields
    assert annotator.connector.fields == fields


def test_connector_use_annotator_fields_default(conn_sqlite_uuid):
    annotator = Annotator({"TIME": float}, connector=conn_sqlite_uuid)

    assert annotator.fields == ["description"]
    assert annotator.connector.fields == ["description"]


def test_annotator_default_region(conn_sqlite_uuid):
    annotator = Annotator({"TIME": float}, connector=conn_sqlite_uuid, default_region="point")

    assert annotator.spec["TIME"]["region"] == "point"


def test_fields_with_spaces(conn_sqlite_uuid):
    fields = ["A Field with space"]
    annotator = Annotator({"A": float, "B": float}, fields=fields, connector=conn_sqlite_uuid)
    assert annotator.annotation_table._field_df.columns == fields


def test_static_fields(conn_sqlite_uuid):
    static_fields = {"static": "1"}
    annotator = Annotator({"TIME": float}, static_fields=static_fields, connector=conn_sqlite_uuid)

    assert annotator.static_fields == static_fields
    assert "static" in annotator.annotation_table._field_df.columns
    assert "static" in annotator.df
    assert annotator.connector.fields == ["description", "static"]

    # Create new entry
    start, end = np.datetime64("2022-06-06"), np.datetime64("2022-06-08")
    annotator.set_regions(TIME=(start, end))
    annotator.add_annotation(description="test")
    output = annotator.df
    assert output["static"].iloc[0] == "1"
    assert output["description"].iloc[0] == "test"
    assert output.shape[0] == 1

    # Update entry
    annotator.update_annotation_fields(output.index[0], description="test2")
    output = annotator.df
    assert output["static"].iloc[0] == "1"
    assert output["description"].iloc[0] == "test2"
    assert output.shape[0] == 1


@pytest.mark.parametrize("dtype", [dt.datetime, dt.date])
def test_spec_datetime_date(conn_sqlite_uuid, dtype):
    annotator = Annotator(spec={"time": dtype}, connector=conn_sqlite_uuid)

    output = annotator.df.dtypes
    assert output["start[time]"] == np.dtype("datetime64[ns]")
    assert output["end[time]"] == np.dtype("datetime64[ns]")


def test_multiple_regions_one_first(multiple_annotators, conn_sqlite_uuid):
    annotator = multiple_annotators

    annotator.set_regions(TIME=(pd.Timestamp("2022-06-06"), pd.Timestamp("2022-06-08")))
    annotator.add_annotation(description="Only time")
    annotator.commit()

    annotator.set_regions(x=(-0.25, 0.25), y=(-0.1, 0.1))
    annotator.add_annotation(description="x and y")
    annotator.commit()
