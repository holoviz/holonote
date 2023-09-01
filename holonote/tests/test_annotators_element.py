from __future__ import annotations

import holoviews as hv

hv.extension("bokeh")


def get_data(annotation_element, element_type):
    hv.renderer("bokeh").get_plot(annotation_element)
    for e in annotation_element.last.values():
        if isinstance(e, element_type) and not e.data.empty:
            return e.data


def test_set_regions_rectangles(annotator_range2d, element_range2d) -> None:
    annotator = annotator_range2d
    element = element_range2d
    anno_el = annotator * element

    # No regions has been set.
    assert get_data(anno_el, hv.Rectangles) is None

    annotator.set_regions(x=(-0.25, 0.25), y=(-0.25, 0.25))
    assert (get_data(anno_el, hv.Rectangles).values == [-0.25, -0.25, 0.25, 0.25]).all()

    # annotator.add_annotation(description="Test")
    # assert get_data(anno_el, hv.Rectangles) is None
