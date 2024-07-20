import holoviews as hv
import pandas as pd
import pytest

from holonote.annotate import Style
from holonote.annotate.display import _default_color
from holonote.app.panel import PanelWidgets
from holonote.tests.util import get_editor, get_indicator


def compare_indicator_color(indicator, style):
    if isinstance(indicator.opts["color"], hv.dim):
        if isinstance(style.color, hv.dim):
            assert str(indicator.opts["color"]) == str(style.color)
        elif style.color is None and style._colormap:
            assert dict(zip(style._groupby[1], _default_color)) == style._colormap
        else:
            expected_dim = hv.dim("__selected__").categorize(
                categories={True: style.selection_color}, default=style.color
            )
            assert str(indicator.opts["color"]) == str(expected_dim)
    else:
        assert indicator.opts["color"] == style.color


def compare_style(cat_annotator):
    style = cat_annotator.style
    indicator = next(get_indicator(cat_annotator, hv.VSpans))
    compare_indicator_color(indicator, style)
    expected_dim = hv.dim("__selected__").categorize(
        categories={True: style.selection_alpha}, default=style.alpha
    )
    assert str(indicator.opts["alpha"]) == str(expected_dim)

    editor = get_editor(cat_annotator, hv.VSpan)
    assert editor.opts["color"] == style.edit_color
    assert editor.opts["alpha"] == style.edit_alpha


def get_selected_indicator_data(annotator) -> pd.Series:
    df = pd.concat([i.data for i in get_indicator(annotator, hv.VLines)])
    return df["__selected__"]


def test_color_undefined_resorting_nodatainit(cat_annotator_no_data):
    annotator = cat_annotator_no_data
    panel_widgets = PanelWidgets(annotator)

    # Add a new annotation type 'A'

    annotator.set_regions(x=(0, 1))
    annotator.add_annotation(category="A")
    annotator.commit()

    compare_style(cat_annotator_no_data)
    visible_options = panel_widgets.visible_widget.options
    colormap = panel_widgets.colormap

    assert "A" in visible_options, "New annotation type 'A' should be in visible options"
    assert (
        colormap["A"] == _default_color[0]
    ), f"Expected default color for 'A', but got {colormap['A']}"

    # Add a new annotation type 'C'

    annotator.set_regions(x=(0, 1))
    annotator.add_annotation(category="C")
    annotator.commit()

    compare_style(annotator)
    visible_options = panel_widgets.visible_widget.options
    colormap = panel_widgets.colormap

    assert "C" in visible_options, "New annotation type 'C' should be in visible options"
    assert (
        colormap["C"] == _default_color[1]
    ), f"Expected default color for 'C', but got {colormap['C']}"

    # Add a new annotation type 'B' which resorts the order (A-B-C) of the options and therefore colormap

    annotator.set_regions(x=(0, 1))
    annotator.add_annotation(category="B")
    annotator.commit()

    compare_style(annotator)
    visible_options = panel_widgets.visible_widget.options
    colormap = panel_widgets.colormap

    assert "B" in visible_options, "New annotation type 'B' should be in visible options"
    assert (
        colormap["B"] == _default_color[1]
    ), f"Expected default color for 'B', but got {colormap['B']}"
    assert (
        colormap["C"] == _default_color[2]
    ), f"Expected default color for 'C', but got {colormap['C']}"

    # Remove the annotation type 'B', which again resorts the order of the options and assigned colormap

    b_index = annotator.df[annotator.df["category"] == "B"].index[0]
    annotator.delete_annotation(b_index)
    annotator.commit()

    compare_style(annotator)
    visible_options = panel_widgets.visible_widget.options
    colormap = panel_widgets.colormap

    assert (
        "B" not in visible_options
    ), "Removed annotation type 'B' should not be in visible options"
    assert (
        colormap["A"] == _default_color[0]
    ), f"Expected default color for 'A', but got {colormap['A']}"
    assert (
        colormap["C"] == _default_color[1]
    ), f"Expected default color for 'C', but got {colormap['C']}"


def test_color_dim_defined(cat_annotator):
    annotator = cat_annotator
    color_dim = hv.dim("category").categorize(
        categories={"A": "purple", "B": "orange", "C": "green"}, default="grey"
    )
    annotator.style.color = color_dim
    panel_widgets = PanelWidgets(annotator)

    compare_style(annotator)

    # Add a new annotation type 'D'
    annotator.set_regions(x=(3, 4))
    annotator.add_annotation(category="D")
    annotator.commit()

    compare_style(annotator)
    visible_options = panel_widgets.visible_widget.options
    colormap = panel_widgets.colormap

    assert "D" in visible_options, "New annotation type 'D' should be in visible options"
    assert (
        colormap["D"] == "grey"
    ), f"Expected default color 'grey' for 'D', but got {colormap['D']}"

    # Delete annotation type 'C', which only has a single data entry
    c_index = annotator.df[annotator.df["category"] == "C"].index[0]
    annotator.delete_annotation(c_index)
    annotator.commit()

    compare_style(annotator)
    visible_options = panel_widgets.visible_widget.options
    colormap = panel_widgets.colormap

    assert (
        "C" not in visible_options
    ), "Removed annotation type 'C' should not be in visible options"

    # Add annotation type 'C' again
    annotator.set_regions(x=(0, 1))
    annotator.add_annotation(category="C")
    annotator.commit()

    compare_style(annotator)
    visible_options = panel_widgets.visible_widget.options
    colormap = panel_widgets.colormap

    assert colormap["C"] == "green", f"Expected color 'green' for 'C', but got {colormap['C']}"


def test_style_accessor(cat_annotator) -> None:
    assert isinstance(cat_annotator.style, Style)


def test_style_default(cat_annotator) -> None:
    compare_style(cat_annotator)


def test_style_default_select(cat_annotator) -> None:
    # Select the first value
    index = cat_annotator.df.index[0]
    cat_annotator.select_by_index(index)
    compare_style(cat_annotator)
    sel_data = get_selected_indicator_data(cat_annotator)
    assert sel_data[index]
    assert sel_data.sum() == 1

    # remove it again
    cat_annotator.select_by_index()
    compare_style(cat_annotator)
    sel_data = get_selected_indicator_data(cat_annotator)
    assert sel_data.sum() == 0


def test_style_change_color_alpha(cat_annotator) -> None:
    style = cat_annotator.style
    style.color = "blue"
    style.alpha = 0.1

    compare_style(cat_annotator)

    style.edit_color = "yellow"
    style.edit_alpha = 1
    cat_annotator.set_regions(x=(0, 1))  # To update plot
    compare_style(cat_annotator)


def test_style_color_dim(cat_annotator):
    style = cat_annotator.style
    style.color = hv.dim("category").categorize(
        categories={"B": "red", "A": "blue", "C": "green"}, default="grey"
    )
    compare_style(cat_annotator)


@pytest.mark.xfail(
    reason="hv.dim is not supported for selection.color in the current implementation"
)
def test_style_selection_color(cat_annotator):
    style = cat_annotator.style
    style.selection_color = "blue"
    compare_style(cat_annotator)
    sel_data = get_selected_indicator_data(cat_annotator)
    assert sel_data.sum() == 0


#     # Select the first value
#     index = cat_annotator.df.index[0]
#     cat_annotator.select_by_index(index)
#     compare_style(cat_annotator)
#     sel_data = get_selected_indicator_data(cat_annotator)
#     assert sel_data[index]
#     assert sel_data.sum() == 1

#     # remove it again
#     cat_annotator.select_by_index()
#     compare_style(cat_annotator)
#     sel_data = get_selected_indicator_data(cat_annotator)
#     assert sel_data.sum() == 0


# def test_style_error_color_dim_and_selection(cat_annotator):
#     style = cat_annotator.style
#     style.color = hv.dim("category").categorize(
#         categories={"B": "red", "A": "blue", "C": "green"}, default="grey"
#     )
#     style.selection_color = "blue"
#     msg = "'Style.color' cannot be a `hv.dim` / `None` when 'Style.selection_color' is not None"
#     with pytest.raises(ValueError, match=msg):
#         compare_style(cat_annotator)


# def test_style_opts(cat_annotator):
#     cat_annotator.style.opts = {"line_width": 2}
#     compare_style(cat_annotator)

#     indicator = next(get_indicator(cat_annotator, hv.VSpans))
#     assert indicator.opts["line_width"] == 2

#     editor = get_editor(cat_annotator, hv.VSpan)
#     assert "line_width" not in editor.opts.get().kwargs

#     cat_annotator.style.edit_opts = {"line_width": 3}
#     cat_annotator.set_regions(x=(0, 1))  # To update plot
#     editor = get_editor(cat_annotator, hv.VSpan)
#     assert editor.opts["line_width"] == 3


# @pytest.mark.parametrize("opts", [{"alpha": 0.5}, {"color": "red"}], ids=["alpha", "color"])
# def test_style_opts_warn(cat_annotator, opts):
#     msg = "Color and alpha opts should be set directly on the style object"
#     with pytest.raises(UserWarning, match=msg):
#         cat_annotator.style.opts = opts


# def test_style_reset(cat_annotator) -> None:
#     style = cat_annotator.style
#     style.color = "blue"
#     compare_style(cat_annotator)

#     style.reset()
#     assert style.color == Style.color
#     compare_style(cat_annotator)


# def test_groupby_color_change(cat_annotator) -> None:
#     cat_annotator.groupby = "category"
#     cat_annotator.visible = ["A", "B", "C"]

#     indicators = hv.render(cat_annotator.get_display("x").static_indicators()).renderers
#     color_cycle = cat_annotator.style._colormap.values()
#     for indicator, expected_color in zip(indicators, color_cycle):
#         assert indicator.glyph.fill_color == expected_color
