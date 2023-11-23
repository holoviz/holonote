import holoviews as hv
import pandas as pd
import pytest

from holonote.annotate import Style
from holonote.tests.util import get_editor, get_indicator


def compare_indicator_color(indicator, style):
    if isinstance(indicator.opts["color"], hv.dim):
        if isinstance(style.color, hv.dim):
            assert str(indicator.opts["color"]) == str(style.color)
        else:
            style_color = (
                style.color.values[0] if isinstance(style.color, hv.Cycle) else style.color
            )
            expected_dim = hv.dim("__selected__").categorize(
                categories={True: style.selection_color}, default=style_color
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


def test_style_selection_color(cat_annotator):
    style = cat_annotator.style
    style.selection_color = "blue"
    compare_style(cat_annotator)
    sel_data = get_selected_indicator_data(cat_annotator)
    assert sel_data.sum() == 0

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


def test_style_error_color_dim_and_selection(cat_annotator):
    style = cat_annotator.style
    style.color = hv.dim("category").categorize(
        categories={"B": "red", "A": "blue", "C": "green"}, default="grey"
    )
    style.selection_color = "blue"
    msg = r"'Style\.color' cannot be a `hv.dim` when 'Style.selection_color' is not None"
    with pytest.raises(ValueError, match=msg):
        compare_style(cat_annotator)


def test_style_opts(cat_annotator):
    cat_annotator.style.opts = {"line_width": 2}
    compare_style(cat_annotator)

    indicator = next(get_indicator(cat_annotator, hv.VSpans))
    assert indicator.opts["line_width"] == 2

    editor = get_editor(cat_annotator, hv.VSpan)
    assert "line_width" not in editor.opts.get().kwargs

    cat_annotator.style.edit_opts = {"line_width": 3}
    cat_annotator.set_regions(x=(0, 1))  # To update plot
    editor = get_editor(cat_annotator, hv.VSpan)
    assert editor.opts["line_width"] == 3


@pytest.mark.parametrize("opts", [{"alpha": 0.5}, {"color": "red"}], ids=["alpha", "color"])
def test_style_opts_warn(cat_annotator, opts):
    msg = "Color and alpha opts should be set directly on the style object"
    with pytest.raises(UserWarning, match=msg):
        cat_annotator.style.opts = opts


def test_style_reset(cat_annotator) -> None:
    style = cat_annotator.style
    style.color = "blue"
    compare_style(cat_annotator)

    style.reset()
    assert style.color == Style.color
    compare_style(cat_annotator)


def test_groupby_color_change(cat_annotator) -> None:
    cat_annotator.groupby = "category"
    cat_annotator.visible = ["A", "B", "C"]

    indicators = hv.render(cat_annotator.get_display("x").static_indicators()).renderers
    color_cycle = cat_annotator.style.color.values
    for indicator, expected_color in zip(indicators, color_cycle):
        assert indicator.glyph.fill_color == expected_color
