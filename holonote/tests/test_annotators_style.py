import holoviews as hv
import pandas as pd
import pytest

from holonote.annotate import Annotator, Style
from holonote.tests.util import get_editor, get_indicator


@pytest.fixture()
def cat_annotator(conn_sqlite_uuid) -> Annotator:
    # Initialize annotator
    annotator = Annotator(
        {"x": float},
        fields=["description", "category"],
        connector=conn_sqlite_uuid,
        groupby="category",
    )
    # Add data to annotator
    data = {
        "category": ["A", "B", "A", "C", "B"],
        "start_number": [1, 6, 11, 16, 21],
        "end_number": [5, 10, 15, 20, 25],
        "description": list("ABCDE"),
    }
    annotator.define_annotations(pd.DataFrame(data), x=("start_number", "end_number"))
    # Setup display
    annotator.get_display("x")
    return annotator


def compare_style(cat_annotator, categories):
    style = cat_annotator.style
    indicator = get_indicator(cat_annotator, hv.VSpans)
    assert indicator.opts["color"] == cat_annotator.style.color
    expected_dim = hv.dim("uuid").categorize(categories=categories, default=style.alpha)
    assert str(indicator.opts["alpha"]) == str(expected_dim)

    editor = get_editor(cat_annotator, hv.VSpan)
    assert editor.opts["color"] == style.edit_color
    assert editor.opts["alpha"] == style.edit_alpha


def test_style_accessor(cat_annotator) -> None:
    assert isinstance(cat_annotator.style, Style)


def test_style_default(cat_annotator) -> None:
    compare_style(cat_annotator, {})


def test_style_default_select(cat_annotator) -> None:
    style = cat_annotator.style

    # Select the first value
    cat_annotator.select_by_index(0)
    compare_style(cat_annotator, {0: style.selection_alpha})

    # remove it again
    cat_annotator.select_by_index()
    compare_style(cat_annotator, {})
