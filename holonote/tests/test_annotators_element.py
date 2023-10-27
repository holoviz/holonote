from __future__ import annotations

import holoviews as hv

bk_renderer = hv.renderer("bokeh")


def _get_display(annotator, kdims=None) -> hv.Element:
    if kdims is None:
        kdims = next(iter(annotator._displays))
    kdims = (kdims,) if isinstance(kdims, str) else tuple(kdims)
    return annotator.get_display(*kdims)


def get_region_editor_data(
    annotator,
    element_type,
    kdims=None,
):
    el = _get_display(annotator, kdims).region_editor()
    for e in el.last.traverse():
        if isinstance(e, element_type):
            return e.data


def get_indicators_data(annotator, element_type, kdims=None):
    si = _get_display(annotator, kdims).static_indicators
    return next(iter(si.data.values())).data


def test_set_regions_range1d(annotator_range1d, element_range1d) -> None:
    annotator = annotator_range1d
    element = element_range1d
    annotator_element = annotator * element
    bk_renderer.get_plot(annotator_element)

    # No regions has been set
    output = get_region_editor_data(annotator, hv.VSpan)
    expected = [None, None]
    assert output == expected

    # Setting regions
    annotator.set_regions(TIME=(-0.25, 0.25))
    output = get_region_editor_data(annotator, hv.VSpan)
    expected = [-0.25, 0.25]
    assert output == expected

    # Adding annotation and remove selection regions.
    annotator.add_annotation(description="Test")
    output = get_region_editor_data(annotator, hv.VSpan)
    expected = [None, None]
    assert output == expected

    output = get_indicators_data(annotator, hv.Rectangles)
    output1 = output.loc[0, ["start[TIME]", "end[TIME]"]].tolist()
    expected1 = [-0.25, 0.25]
    assert output1 == expected1
    output2 = output.loc[0, "description"]
    expected2 = "Test"
    assert output2 == expected2


def test_set_regions_range2d(annotator_range2d, element_range2d) -> None:
    annotator = annotator_range2d
    element = element_range2d
    annotator_element = annotator * element
    bk_renderer.get_plot(annotator_element)

    # No regions has been set
    output = get_region_editor_data(annotator, hv.Rectangles)
    assert output.empty

    # Setting regions
    annotator.set_regions(x=(-0.25, 0.25), y=(-0.25, 0.25))
    output = get_region_editor_data(annotator, hv.Rectangles).iloc[0].to_list()
    expected = [-0.25, -0.25, 0.25, 0.25]
    assert output == expected

    # Adding annotation and remove selection regions.
    annotator.add_annotation(description="Test")
    output = get_region_editor_data(annotator, hv.Rectangles)
    assert output.empty

    output = get_indicators_data(annotator, hv.Rectangles)
    output1 = output.loc[0, ["start[x]", "start[y]", "end[x]", "end[y]"]].tolist()
    expected1 = [-0.25, -0.25, 0.25, 0.25]
    assert output1 == expected1
    output2 = output.loc[0, "description"]
    expected2 = "Test"
    assert output2 == expected2


def test_set_regions_multiple(multiple_annotators, element_range1d, element_range2d):
    annotator = multiple_annotators
    time_annotation = annotator * element_range1d
    xy_annotation = annotator * element_range2d
    annotator_element = time_annotation + xy_annotation
    bk_renderer.get_plot(annotator_element)

    # No regions has been set
    # Time annotation
    output = get_region_editor_data(annotator, hv.VSpan, "TIME")
    expected = [None, None]
    assert output == expected
    # xy annotation
    output = get_region_editor_data(annotator, hv.Rectangles, ("x", "y"))
    assert output.empty

    # Setting regions
    annotator.set_regions(TIME=(-0.25, 0.25), x=(-0.25, 0.25), y=(-0.25, 0.25))
    # Time annotation
    output = get_region_editor_data(annotator, hv.VSpan, "TIME")
    expected = [-0.25, 0.25]
    assert output == expected
    # xy annotation
    output = get_region_editor_data(annotator, hv.Rectangles, ("x", "y")).iloc[0].to_list()
    expected = [-0.25, -0.25, 0.25, 0.25]
    assert output == expected

    # Adding annotation and remove selection regions.
    annotator.add_annotation(description="Test")
    # Time annotation
    output = get_region_editor_data(annotator, hv.VSpan, "TIME")
    expected = [None, None]
    assert output == expected

    output = get_indicators_data(annotator, hv.Rectangles, "TIME")
    output1 = output.loc[0, ["start[TIME]", "end[TIME]"]].tolist()
    expected1 = [-0.25, 0.25]
    assert output1 == expected1
    output2 = output.loc[0, "description"]
    expected2 = "Test"
    assert output2 == expected2

    # xy annotation
    output = get_region_editor_data(annotator, hv.Rectangles, ("x", "y"))
    assert output.empty

    output = get_indicators_data(annotator, hv.Rectangles, ("x", "y"))
    output1 = output.loc[0, ["start[x]", "start[y]", "end[x]", "end[y]"]].tolist()
    expected1 = [-0.25, -0.25, 0.25, 0.25]
    assert output1 == expected1
    output2 = output.loc[0, "description"]
    expected2 = "Test"
    assert output2 == expected2


def test_editable_enabled(annotator_range1d, element_range1d):
    annotator_range1d * element_range1d

    annotator_range1d.editable_enabled = False
    for display in annotator_range1d._displays.values():
        assert not display.editable_enabled

    annotator_range1d.editable_enabled = True
    for display in annotator_range1d._displays.values():
        assert display.editable_enabled


def test_selection_enabled(annotator_range1d, element_range1d):
    annotator_range1d * element_range1d

    annotator_range1d.selection_enabled = False
    for display in annotator_range1d._displays.values():
        assert not display.selection_enabled

    annotator_range1d.selection_enabled = True
    for display in annotator_range1d._displays.values():
        assert display.selection_enabled
