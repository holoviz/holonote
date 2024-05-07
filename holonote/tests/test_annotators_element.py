from __future__ import annotations

import holoviews as hv
import pytest
from holoviews.operation.datashader import rasterize

from holonote.tests.util import get_editor, get_indicator


def get_editor_data(annotator, element_type, kdims=None):
    el = get_editor(annotator, element_type, kdims)
    return getattr(el, "data", None)


def get_indicator_data(annotator, element_type, kdims=None):
    for el in get_indicator(annotator, element_type, kdims):
        yield el.data


def test_set_regions_range1d(annotator_range1d) -> None:
    annotator = annotator_range1d
    annotator.get_display("TIME")

    # No regions has been set
    output = get_editor_data(annotator, hv.VSpan)
    expected = [None, None]
    assert output == expected

    # Setting regions
    annotator.set_regions(TIME=(-0.25, 0.25))
    output = get_editor_data(annotator, hv.VSpan)
    expected = [-0.25, 0.25]
    assert output == expected

    # Adding annotation and remove selection regions.
    annotator.add_annotation(description="Test")
    output = get_editor_data(annotator, hv.VSpan)
    expected = [None, None]
    assert output == expected

    output = next(get_indicator_data(annotator, hv.Rectangles))
    output1 = output.iloc[0][["start[TIME]", "end[TIME]"]].tolist()
    expected1 = [-0.25, 0.25]
    assert output1 == expected1
    output2 = output.iloc[0]["description"]
    expected2 = "Test"
    assert output2 == expected2


def test_edit_streams(annotator_range1d) -> None:
    annotator = annotator_range1d
    edit_streams = annotator.get_display("TIME").edit_streams
    assert len(edit_streams) == 3
    assert "box_select" in edit_streams
    assert "tap" not in edit_streams
    assert "lasso_select" not in edit_streams


def test_set_regions_range2d(annotator_range2d) -> None:
    annotator = annotator_range2d
    annotator.get_display("x", "y")

    # No regions has been set
    output = get_editor_data(annotator, hv.Rectangles)
    assert output.empty

    # Setting regions
    annotator.set_regions(x=(-0.25, 0.25), y=(-0.25, 0.25))
    output = get_editor_data(annotator, hv.Rectangles).iloc[0].to_list()
    expected = [-0.25, -0.25, 0.25, 0.25]
    assert output == expected

    # Adding annotation and remove selection regions.
    annotator.add_annotation(description="Test")
    output = get_editor_data(annotator, hv.Rectangles)
    assert output.empty

    output = next(get_indicator_data(annotator, hv.Rectangles))
    output1 = output.iloc[0][["start[x]", "start[y]", "end[x]", "end[y]"]].tolist()
    expected1 = [-0.25, -0.25, 0.25, 0.25]
    assert output1 == expected1
    output2 = output.iloc[0]["description"]
    expected2 = "Test"
    assert output2 == expected2


def test_set_regions_multiple(multiple_annotators):
    annotator = multiple_annotators
    annotator.get_display("TIME")
    annotator.get_display("x", "y")

    # No regions has been set
    # Time annotation
    output = get_editor_data(annotator, hv.VSpan, "TIME")
    expected = [None, None]
    assert output == expected
    # xy annotation
    output = get_editor_data(annotator, hv.Rectangles, ("x", "y"))
    assert output.empty

    # Setting regions
    annotator.set_regions(TIME=(-0.25, 0.25), x=(-0.25, 0.25), y=(-0.25, 0.25))
    # Time annotation
    output = get_editor_data(annotator, hv.VSpan, "TIME")
    expected = [-0.25, 0.25]
    assert output == expected
    # xy annotation
    output = get_editor_data(annotator, hv.Rectangles, ("x", "y")).iloc[0].to_list()
    expected = [-0.25, -0.25, 0.25, 0.25]
    assert output == expected

    # Adding annotation and remove selection regions.
    annotator.add_annotation(description="Test")
    # Time annotation
    output = get_editor_data(annotator, hv.VSpan, "TIME")
    expected = [None, None]
    assert output == expected

    output = next(get_indicator_data(annotator, hv.Rectangles, "TIME"))
    output1 = output.iloc[0][["start[TIME]", "end[TIME]"]].tolist()
    expected1 = [-0.25, 0.25]
    assert output1 == expected1
    output2 = output.iloc[0]["description"]
    expected2 = "Test"
    assert output2 == expected2

    # xy annotation
    output = get_editor_data(annotator, hv.Rectangles, ("x", "y"))
    assert output.empty

    output = next(get_indicator_data(annotator, hv.Rectangles, ("x", "y")))
    output1 = output.iloc[0][["start[x]", "start[y]", "end[x]", "end[y]"]].tolist()
    expected1 = [-0.25, -0.25, 0.25, 0.25]
    assert output1 == expected1
    output2 = output.iloc[0]["description"]
    expected2 = "Test"
    assert output2 == expected2


def test_editable_enabled(annotator_range1d):
    annotator_range1d.get_display("TIME")
    assert annotator_range1d._displays

    annotator_range1d.editable_enabled = False
    for display in annotator_range1d._displays.values():
        assert not display.editable_enabled

    annotator_range1d.editable_enabled = True
    for display in annotator_range1d._displays.values():
        assert display.editable_enabled


def test_selection_enabled(annotator_range1d, element_range1d):
    annotator_range1d.get_display("TIME")
    assert annotator_range1d._displays

    annotator_range1d.selection_enabled = False
    for display in annotator_range1d._displays.values():
        assert not display.selection_enabled

    annotator_range1d.selection_enabled = True
    for display in annotator_range1d._displays.values():
        assert display.selection_enabled


def test_groupby(cat_annotator):
    cat_annotator.groupby = "category"
    iter_indicator = get_indicator(cat_annotator, hv.VSpans)
    indicator = next(iter_indicator)
    assert indicator.data.shape == (2, 5)
    assert (indicator.data["category"] == "A").all()

    indicator = next(iter_indicator)
    assert indicator.data.shape == (2, 5)
    assert (indicator.data["category"] == "B").all()

    indicator = next(iter_indicator)
    assert indicator.data.shape == (1, 5)
    assert (indicator.data["category"] == "C").all()

    with pytest.raises(StopIteration):
        next(iter_indicator)


def test_groupby_visible(cat_annotator):
    cat_annotator.groupby = "category"
    cat_annotator.visible = ["A"]
    iter_indicator = get_indicator(cat_annotator, hv.VSpans)
    indicator = next(iter_indicator)
    assert indicator.data.shape == (2, 5)
    assert (indicator.data["category"] == "A").all()

    with pytest.raises(StopIteration):
        next(iter_indicator)


def test_multiply_overlay(annotator_range1d):
    el1 = hv.Curve([], kdims=["TIME"])
    el2 = hv.Curve([], kdims=["TIME"])
    el = el1 * el2
    assert isinstance(el, hv.Overlay)

    el = el * annotator_range1d
    hv.render(el)
    assert isinstance(el.last, hv.Overlay)

    el.opts(width=100)
    hv.render(el)
    assert el.last.opts.get().kwargs.get("width") == 100


def test_multiply_ndoverlay(annotator_range1d):
    el1 = hv.Curve([], kdims=["TIME"])
    el2 = hv.Curve([], kdims=["TIME"])
    el = hv.NdOverlay({1: el1, 2: el2})
    assert isinstance(el, hv.NdOverlay)

    el = el * annotator_range1d
    hv.render(el)
    assert isinstance(el.last, hv.Overlay)

    el.opts(width=100)
    hv.render(el)
    assert el.last.opts.get().kwargs.get("width") == 100


def test_multiply_layout(annotator_range1d):
    el1 = hv.Curve([], kdims=["TIME"])
    el2 = hv.Curve([], kdims=["TIME"])
    el = el1 + el2
    assert isinstance(el, hv.Layout)

    el = el * annotator_range1d
    hv.render(el)
    assert isinstance(el, hv.Layout)

    el.opts(width=100)
    hv.render(el)
    assert el.opts.get().kwargs.get("width") == 100


def test_multiply_dynamicmap(annotator_range1d):
    # rasterize creates a dynamicmap
    el1 = rasterize(hv.Curve([], kdims=["TIME"]))
    assert isinstance(el1, hv.DynamicMap)

    el = el1 * annotator_range1d
    hv.render(el)
    # an overlay including a dynamicmap is still a dynamicmap
    assert isinstance(el, hv.DynamicMap)


def test_multiply_dynamicmap_layout(annotator_range1d):
    el1 = rasterize(hv.Curve([], kdims=["TIME"]))
    el2 = hv.Curve([], kdims=["TIME"])
    el = el1 + el2
    assert isinstance(el, hv.Layout)

    el = el * annotator_range1d
    hv.render(el)
    assert isinstance(el, hv.Layout)
